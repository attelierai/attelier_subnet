"""Private Comfy research node: compare delivered media to trusted VAE tensors.

Metrics are observations, not pass/fail decisions; thresholds must be frozen on
honest calibration before held-out media is scored. No miner metadata is proof.
"""

import hashlib
import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def bounded_audio_samples(media: Path, rate: int, allowed_samples: int) -> np.ndarray:
    """Decode a bounded window and retain one exact sentinel sample per channel."""
    raw = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(media),
            "-map",
            "0:a:0",
            "-ac",
            "2",
            "-ar",
            str(rate),
            # A whole AAC frame of headroom keeps decoding bounded. asetpts
            # removes container PTS gaps before output -t; atrim defines the
            # exact sentinel instead of float -t rounding.
            "-t",
            str((allowed_samples + 1024) / rate),
            "-af",
            f"asetpts=N/SR/TB,atrim=end_sample={allowed_samples + 1}",
            "-f",
            "f32le",
            "-",
        ],
        check=True,
        capture_output=True,
        timeout=90,
    ).stdout
    if len(raw) % (2 * np.dtype("<f4").itemsize):
        raise ValueError("Delivered audio bytes differ")
    samples = np.frombuffer(raw, dtype="<f4").reshape(-1, 2)
    if len(samples) > allowed_samples:
        raise ValueError("Delivered audio length differs")
    return samples


def stream_number(stream: dict, field: str) -> float:
    try:
        value = float(stream[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Delivered {field} differs") from exc
    if not math.isfinite(value):
        raise ValueError(f"Delivered {field} differs")
    return value


class H3TraceMediaAudit:
    CATEGORY = "attelier/trace"
    RETURN_TYPES = ("STRING",)
    FUNCTION = "measure"
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "audio": ("AUDIO",),
                "media_path": ("STRING", {"default": "/workspace/forge/delivered.mp4"}),
                "result_path": ("STRING", {"default": "/workspace/forge/media-audit.json"}),
            }
        }

    def measure(self, images, audio, media_path, result_path):
        import torch

        media, result = Path(media_path), Path(result_path)
        if not media.is_absolute() or not result.is_absolute() or result.exists():
            raise ValueError("Fresh absolute evidence paths required")
        if not media.is_file() or media.stat().st_size > 100 * 1024**2:
            raise ValueError("Bounded delivered media required")
        if tuple(images.shape) != (124, 480, 832, 3):
            raise ValueError("This experiment supports only the 5s/480p profile")
        if not torch.isfinite(images).all():
            raise ValueError("Finite RGB VAE output required")
        count, height, width, _ = images.shape
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-of", "json", str(media)],
            check=True,
            capture_output=True,
            timeout=20,
        )
        streams = json.loads(probe.stdout)["streams"]
        videos = [s for s in streams if s["codec_type"] == "video"]
        audios = [s for s in streams if s["codec_type"] == "audio"]
        if len(videos) != 1 or len(audios) != 1:
            raise ValueError("Exactly one video and audio stream required")
        video = videos[0]
        profile = {
            "codec_name": "h264",
            "pix_fmt": "yuv420p",
            "color_range": "tv",
            "color_space": "bt709",
            "color_transfer": "iec61966-2-1",
            "color_primaries": "bt709",
        }
        if any(video.get(key) != value for key, value in profile.items()):
            raise ValueError("Delivered encoding differs from pinned Comfy profile")
        if (video["height"], video["width"]) != (height, width):
            raise ValueError("Delivered video dimensions differ")
        if video["avg_frame_rate"] != "24/1":
            raise ValueError("Delivered video frame rate differs")
        video_start = stream_number(video, "start_time")
        audio_start = stream_number(audios[0], "start_time")
        video_duration = stream_number(video, "duration")
        audio_duration = stream_number(audios[0], "duration")
        if (
            abs(video_start) > 1 / 32000
            or abs(audio_start) > 1 / 32000
            or abs(video_duration - 124 / 24) > 1 / 24
            or abs(audio_duration - 165334 / 32000) > 1024 / 32000
        ):
            raise ValueError("Delivered stream timing differs")
        before = digest(media)
        raw = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(media),
                "-map",
                "0:v:0",
                "-frames:v",
                str(count + 1),
                "-fps_mode",
                "passthrough",
                "-pix_fmt",
                "rgb24",
                "-f",
                "rawvideo",
                "-",
            ],
            check=True,
            capture_output=True,
            timeout=90,
        ).stdout
        if len(raw) != count * height * width * 3:
            raise ValueError("Delivered frame count differs")
        decoded = np.frombuffer(raw, dtype=np.uint8).reshape(count, height, width, 3)
        tensor_hash = hashlib.sha256(
            canonical({"shape": list(images.shape), "dtype": str(images.dtype)})
        )
        rms, mae = [], []
        for index in range(count):
            frame = images[index].detach().cpu().contiguous()
            tensor_hash.update(frame.view(torch.uint8).numpy().tobytes())
            encoder_input = (frame * 255).clamp(0, 255).byte().numpy()
            error = (decoded[index].astype(np.float64) - encoder_input.astype(np.float64)) / 255
            rms.append(float(np.sqrt(np.mean(error * error))))
            mae.append(float(np.mean(np.abs(error))))
        waveform = audio["waveform"].detach().cpu().float().numpy()
        rate = int(audio["sample_rate"])
        if (
            rate != 32000
            or audios[0].get("codec_name") != "aac"
            or int(audios[0].get("sample_rate", 0)) != 32000
            or audios[0].get("channels") != 2
        ):
            raise ValueError("Delivered audio encoding differs from pinned Comfy profile")
        if waveform.shape != (1, 2, 165600) or not np.isfinite(waveform).all():
            raise ValueError("Finite stereo audio required")
        source_samples = waveform.shape[2]
        # Pinned VideoFromComponents.save_to crops audio to the video duration
        # before AAC encoding. Comparing the uncropped latent decode to AAC
        # padding would falsely charge an honest encoder for the removed tail.
        waveform = waveform[:, :, : math.ceil((rate / 24) * count)]
        expected = waveform.shape[2]
        samples = bounded_audio_samples(media, rate, expected + 4096)
        if len(samples) < expected:
            raise ValueError("Delivered audio length differs")
        error = samples[:expected].astype(np.float64) - waveform[0].T
        audio_rms = float(np.sqrt(np.mean(error * error)))
        if digest(media) != before:
            raise ValueError("Delivered media changed during audit")
        receipt = {
            "schema": "forge/h3-media-audit/v3",
            "status": "MEASURED",
            "mediaSha256": before,
            "trustedVideoTensorSha256": tensor_hash.hexdigest(),
            "video": {
                "frames": count,
                "meanFrameRmse": float(np.mean(rms)),
                "maxFrameRmse": max(rms),
                "meanFrameMae": float(np.mean(mae)),
                "canonicalSha256": hashlib.sha256(raw).hexdigest(),
            },
            "audio": {
                "rms": audio_rms,
                "relativeRms": audio_rms
                / max(float(np.sqrt(np.mean(waveform.astype(np.float64) ** 2))), 1e-6),
                "expectedSamples": expected,
                "sourceSamples": source_samples,
                "trimPolicy": "pinned_comfy_video_duration",
                "deliveredSamples": len(samples),
            },
            "limits": [
                "codec distortion requires calibration",
                "pinned media profile",
                "trusted VAE and final latent required",
                "not standalone model proof",
            ],
        }
        result.parent.mkdir(parents=True, exist_ok=True)
        with result.open("xb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(canonical(receipt) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        return (json.dumps(receipt),)


NODE_CLASS_MAPPINGS = {"H3TraceMediaAudit": H3TraceMediaAudit}
NODE_DISPLAY_NAME_MAPPINGS = {"H3TraceMediaAudit": "H3 Delivered Media Audit (Experiment)"}
