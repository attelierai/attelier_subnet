"""ComfyUI custom nodes for a bounded, private H3 trace experiment.

This is experiment instrumentation, not a proof of remote execution.  It is
installed into an isolated Comfy custom_nodes directory by h3_trace_capture.py.
The graph controller supplies public hashes and opaque lineage identifiers; raw
prompts, weights and tensors stay in the experiment directory on the GPU host.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import comfy.model_management
import comfy.nested_tensor
import comfy.sample
import comfy.utils
import latent_preview
import torch

SCHEMA = "forge/inference-trace-capture/v1"
DEFAULT_MAX_BYTES = 20 * 1024 * 1024 * 1024
_PACKED_LAYOUT_TAG = "__forge_packed_layout_v1__"
_UUID_TAG = "__forge_uuid_v1__"
_PACKED_LAYOUT_FIELDS = (
    "seq_len",
    "position_ids",
    "img_pos",
    "img_update",
    "audio_pos",
    "audio_update",
    "signature",
    "segments",
)


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _digest(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


class _TraceRun:
    def __init__(self, root: str, run_id: str, metadata: dict[str, str]):
        self.root = Path(root).resolve()
        if not self.root.is_absolute() or not run_id or "/" in run_id or ".." in run_id:
            raise ValueError("trace root/run id invalid")
        self.directory = self.root / run_id
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.directory, 0o700)
        self.manifest_path = self.directory / "manifest.json"
        if self.manifest_path.exists():
            self.manifest = json.loads(self.manifest_path.read_text())
            expected = {"schema": SCHEMA, "runId": run_id, **metadata}
            if {key: self.manifest.get(key) for key in expected} != expected:
                raise ValueError("trace manifest identity changed")
        else:
            self.manifest = {
                "schema": SCHEMA,
                "runId": run_id,
                **metadata,
                "trace": [],
                "forward": [],
                "finalLatent": None,
                "decoded": {},
                "output": None,
                "storedBytes": 0,
                "createdAt": time.time(),
            }
            self._flush()

    def _flush(self) -> None:
        fd, name = tempfile.mkstemp(prefix=".manifest-", dir=self.directory)
        try:
            with os.fdopen(fd, "wb") as target:
                target.write(_canonical(self.manifest))
                target.flush()
                os.fsync(target.fileno())
            os.chmod(name, 0o600)
            os.replace(name, self.manifest_path)
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def tensor(self, kind: str, tensor: torch.Tensor, *, step: int | None = None) -> dict[str, Any]:
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("trace value is not tensor")
        safe_kind = kind.replace("/", "_")
        suffix = "final" if step is None else f"step-{step:03d}"
        relative = f"tensors/{suffix}-{safe_kind}.pt"
        target = self.directory / relative
        target.parent.mkdir(mode=0o700, exist_ok=True)
        if not target.exists():
            maximum = int(os.environ.get("H3_TRACE_MAX_BYTES", DEFAULT_MAX_BYTES))
            estimate = int(tensor.numel() * tensor.element_size()) + 1024 * 1024
            if maximum <= 0 or int(self.manifest.get("storedBytes", 0)) + estimate > maximum:
                raise ValueError("trace storage limit would be exceeded")
            fd, name = tempfile.mkstemp(prefix=".tensor-", dir=target.parent)
            try:
                # PyTorch's zip writer rejects some anonymous/hidden paths;
                # writing through the open descriptor is portable on its
                # supported CUDA/Python runtimes.
                with os.fdopen(fd, "wb") as stream:
                    torch.save(tensor.detach().to("cpu").contiguous(), stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.chmod(name, 0o600)
                os.replace(name, target)
            finally:
                if os.path.exists(name):
                    os.unlink(name)
            self.manifest["storedBytes"] = (
                int(self.manifest.get("storedBytes", 0)) + target.stat().st_size
            )
        return {
            "tensorPath": relative,
            "sha256": _digest(target),
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "bytes": target.stat().st_size,
        }

    def latent(self, kind: str, value: Any, *, step: int | None = None) -> list[dict[str, Any]]:
        if isinstance(value, torch.Tensor):
            return [{"component": "tensor", **self.tensor(kind, value, step=step)}]
        if getattr(value, "is_nested", False):
            names = ("video", "audio")
            return [
                {
                    "component": names[index] if index < len(names) else f"part-{index}",
                    **self.tensor(f"{kind}-{index}", child, step=step),
                }
                for index, child in enumerate(value.unbind())
            ]
        raise ValueError("unsupported latent representation")

    def sample_step(self, step: int, total: int, sigma: float, state: Any, denoised: Any) -> None:
        record = {
            "step": step,
            "totalSteps": total,
            "sigma": sigma,
            # This state is captured by Comfy's sampler callback after model
            # evaluation. It is evidence for comparison, not a claimed atomic
            # denoising transition.
            "stateCallback": self.latent("state", state, step=step),
            "denoised": self.latent("denoised", denoised, step=step),
        }
        self.manifest["trace"].append(record)
        self._flush()

    def final_latent(self, value: Any) -> None:
        self.manifest["finalLatent"] = self.latent("final-latent", value)
        self._flush()

    def forward(self, index: int, params: dict[str, Any], prediction: torch.Tensor) -> None:
        """Persist real model-call inputs and the paired prediction.

        This wrapper sits at Comfy's ``model_function_wrapper`` boundary, before
        ``BaseModel.apply_model``. Unlike a sampler callback, its ``input`` is
        exactly the tensor supplied to the DiT call.  The sanitized conditioning
        is retained privately only to permit a trusted fixed-input replay.
        """
        name = f"forward/{index:03d}.pt"
        target = self.directory / name
        target.parent.mkdir(mode=0o700, exist_ok=True)
        if not target.exists():
            payload = {
                "input": _cpu_copy(params["input"]),
                "timestep": _cpu_copy(params["timestep"]),
                "conditioning": _cpu_copy(params["c"]),
                "prediction": _cpu_copy(prediction),
            }
            estimate = _tensor_bytes(payload) + 1024 * 1024
            maximum = int(os.environ.get("H3_TRACE_MAX_BYTES", DEFAULT_MAX_BYTES))
            if maximum <= 0 or int(self.manifest.get("storedBytes", 0)) + estimate > maximum:
                raise ValueError("trace storage limit would be exceeded")
            fd, temporary = tempfile.mkstemp(prefix=".forward-", dir=target.parent)
            try:
                with os.fdopen(fd, "wb") as stream:
                    torch.save(payload, stream)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.chmod(temporary, 0o600)
                os.replace(temporary, target)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            self.manifest["storedBytes"] = (
                int(self.manifest.get("storedBytes", 0)) + target.stat().st_size
            )
        self.manifest["forward"].append(
            {
                "call": index,
                "tensorPath": name,
                "sha256": _digest(target),
                "bytes": target.stat().st_size,
            }
        )
        self._flush()

    def decoded(self, kind: str, value: Any) -> None:
        if isinstance(value, dict):
            value = value.get("waveform")
        if not isinstance(value, torch.Tensor):
            raise ValueError("decoded trace value is not tensor")
        self.manifest["decoded"][kind] = self.tensor(f"decoded-{kind}", value)
        self._flush()

    def decoded_images(self, images: torch.Tensor) -> None:
        """Record a whole decoded-frame digest and only a few inspectable frames.

        A full 5s H3 decode can be hundreds of MiB. Keeping it would turn a
        trace audit into an unbounded storage attack. The digest covers every
        raw decoded frame in order; the three persisted frames are diagnostic
        material only and never treated as independent observations.
        """
        if not isinstance(images, torch.Tensor) or images.ndim != 4:
            raise ValueError("expected decoded image tensor [frames,height,width,channels]")
        digest = hashlib.sha256(
            _canonical({"shape": list(images.shape), "dtype": str(images.dtype)})
        )
        count = int(images.shape[0])
        for index in range(count):
            frame = images[index].detach().to("cpu").contiguous()
            digest.update(frame.view(torch.uint8).numpy().tobytes())
        selected = sorted({0, count // 2, count - 1})
        self.manifest["decoded"]["videoFrames"] = {
            "rawTensorSha256": digest.hexdigest(),
            "shape": list(images.shape),
            "dtype": str(images.dtype).removeprefix("torch."),
            "selectedFrames": [
                {"frame": index, **self.tensor(f"decoded-video-frame-{index:03d}", images[index])}
                for index in selected
            ],
        }
        self._flush()


def _run(
    root,
    run_id,
    split,
    condition,
    lineage_id,
    input_hash,
    model_manifest_hash,
    workflow_hash,
    runtime_id,
):
    return _TraceRun(
        root,
        run_id,
        {
            "split": split,
            "condition": condition,
            "lineageId": lineage_id,
            "inputHash": input_hash,
            "capture": {
                "modelManifestHash": model_manifest_hash,
                "workflowHash": workflow_hash,
                "runtime": runtime_id,
            },
        },
    )


def _cpu_copy(value: Any, path: str = "conditioning") -> Any:
    """Make a tensor-only replay bundle safe to load with ``weights_only``.

    H3 text-to-video's conditioning is tensors, lists, dictionaries and scalar
    scheduler metadata. A custom object would make replay ambiguous, so capture
    stops rather than pickle an executable object into the evidence archive.
    """
    if isinstance(value, torch.Tensor):
        return value.detach().to("cpu").contiguous()
    if isinstance(value, uuid.UUID):
        return {_UUID_TAG: str(value)}
    if (
        type(value).__module__ == "comfy.ldm.minimax.model"
        and type(value).__name__ == "PackedLayout"
    ):
        return {
            _PACKED_LAYOUT_TAG: {
                field: _cpu_copy(getattr(value, field), f"{path}.{field}")
                for field in _PACKED_LAYOUT_FIELDS
            }
        }
    if isinstance(value, dict):
        return {str(key): _cpu_copy(item, f"{path}.{key}") for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_cpu_copy(item, f"{path}[{index}]") for index, item in enumerate(value)]
    if value is None or isinstance(value, (bool, float, int, str)):
        return value
    kind = f"{type(value).__module__}.{type(value).__name__}"
    raise ValueError(f"unsupported replay conditioning value at {path}: {kind}")


def _tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel() * value.element_size())
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, list):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _unsupported_types(
    value: Any, path: str = "conditioning", found: list[str] | None = None
) -> list[str]:
    """Report only structural types before an expensive trace capture aborts."""
    found = [] if found is None else found
    if (
        len(found) >= 20
        or isinstance(value, (torch.Tensor, uuid.UUID, bool, float, int, str))
        or value is None
    ):
        return found
    if (
        type(value).__module__ == "comfy.ldm.minimax.model"
        and type(value).__name__ == "PackedLayout"
    ):
        for field in _PACKED_LAYOUT_FIELDS:
            _unsupported_types(getattr(value, field), f"{path}.{field}", found)
        return found
    if isinstance(value, dict):
        for key, item in value.items():
            _unsupported_types(item, f"{path}.{key}", found)
        return found
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _unsupported_types(item, f"{path}[{index}]", found)
        return found
    found.append(f"{path}:{type(value).__module__}.{type(value).__name__}")
    return found


class _ForwardCapture:
    def __init__(self, trace: _TraceRun, selected: set[int], delegate, skip_model_step: int = -1):
        self.trace, self.selected, self.delegate, self.calls = trace, selected, delegate, 0
        self.skip_model_step = skip_model_step
        self.previous_prediction: torch.Tensor | None = None

    def close(self) -> None:
        """Release the one-step research cache before downstream VAE work."""
        self.previous_prediction = None

    def __call__(self, model_function, params):
        index = self.calls
        self.calls += 1
        if index in self.selected:
            unsupported = _unsupported_types(params["c"])
            if unsupported:
                raise ValueError(
                    "unsupported replay conditioning values: " + "; ".join(unsupported)
                )
        if index == self.skip_model_step:
            if index < 1 or self.previous_prediction is None:
                raise ValueError("skip_model_step requires a preceding model prediction")
            # A clone makes the deliberately wrong returned prediction isolated
            # from sampler/model mutation. It is never persisted on the GPU
            # beyond this sampler invocation.
            result = self.previous_prediction.detach().clone()
        else:
            result = (
                self.delegate(model_function, params)
                if self.delegate is not None
                else model_function(params["input"], params["timestep"], **params["c"])
            )
        if index in self.selected:
            self.trace.forward(index, params, result)
        if self.skip_model_step >= 1:
            # Keep only one detached prediction for this controlled experiment.
            self.previous_prediction = result.detach().clone()
        return result


class H3TraceSamplerCustomAdvanced:
    """A copy of Comfy's sampler node with its documented callback retained."""

    CATEGORY = "attelier/trace"
    RETURN_TYPES = ("LATENT", "LATENT")
    RETURN_NAMES = ("output", "denoised_output")
    FUNCTION = "sample"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "noise": ("NOISE",),
                "guider": ("GUIDER",),
                "sampler": ("SAMPLER",),
                "sigmas": ("SIGMAS",),
                "latent_image": ("LATENT",),
                "trace_root": (
                    "STRING",
                    {"default": "/workspace/forge/trace-experiment", "multiline": False},
                ),
                "run_id": ("STRING", {"default": "replace-at-submission", "multiline": False}),
                "split": (["calibration", "holdout"],),
                "condition": (
                    [
                        "honest_repeat",
                        "steps_10",
                        "steps_3",
                        "alt_checkpoint",
                        "lora",
                        "wrong_output",
                        "padded_steps_10",
                        "sidecar_honest_wrong_output",
                    ],
                ),
                "lineage_id": ("STRING", {"default": "replace-at-submission", "multiline": False}),
                "input_hash": ("STRING", {"default": "replace-at-submission", "multiline": False}),
                "model_manifest_hash": (
                    "STRING",
                    {"default": "replace-at-submission", "multiline": False},
                ),
                "workflow_hash": (
                    "STRING",
                    {"default": "replace-at-submission", "multiline": False},
                ),
                "runtime_id": ("STRING", {"default": "replace-at-submission", "multiline": False}),
            },
            "optional": {
                "skip_model_step": (
                    "INT",
                    {"default": -1, "min": -1, "max": 100000, "step": 1},
                ),
                # Production trusted replay needs every DiT invocation.  The
                # historical experiment intentionally kept 0/mid/last only to
                # stay inside its storage budget, so preserve that default.
                "capture_all_forwards": ("BOOLEAN", {"default": False}),
            },
        }

    def sample(
        self,
        noise,
        guider,
        sampler,
        sigmas,
        latent_image,
        trace_root,
        run_id,
        split,
        condition,
        lineage_id,
        input_hash,
        model_manifest_hash,
        workflow_hash,
        runtime_id,
        skip_model_step=-1,
        capture_all_forwards=False,
    ):
        trace = _run(
            trace_root,
            run_id,
            split,
            condition,
            lineage_id,
            input_hash,
            model_manifest_hash,
            workflow_hash,
            runtime_id,
        )
        if isinstance(skip_model_step, bool) or not isinstance(skip_model_step, int):
            raise ValueError("skip_model_step must be an integer")
        if not isinstance(capture_all_forwards, bool):
            raise ValueError("capture_all_forwards must be boolean")
        # A ModelPatcher clone preserves the approved weights and all graph
        # patches, while making the wrapper private to this execution.
        patched = guider.model_patcher.clone()
        previous_wrapper = patched.model_options.get("model_function_wrapper")
        declared_steps = int(sigmas.shape[-1] - 1)
        if skip_model_step != -1 and not 1 <= skip_model_step < declared_steps:
            raise ValueError("skip_model_step must be -1 or a noninitial declared sampler step")
        forward_capture = _ForwardCapture(
            trace,
            set(range(declared_steps))
            if capture_all_forwards
            else {0, declared_steps // 2, max(0, declared_steps - 1)},
            previous_wrapper,
            skip_model_step,
        )
        patched.set_model_unet_function_wrapper(forward_capture)
        guider.model_patcher = patched
        # CFGGuider caches model_options at construction. Keep it on the same
        # clone or the sampler receives the original options and never calls
        # our model-function wrapper.
        guider.model_options = patched.model_options
        latent = latent_image.copy()
        samples_in = comfy.sample.fix_empty_latent_channels(
            guider.model_patcher,
            latent["samples"],
            latent.get("downscale_ratio_spacial"),
            latent.get("downscale_ratio_temporal"),
        )
        latent["samples"] = samples_in
        noise_value = noise.generate_noise(latent)
        trace.manifest["inputNoise"] = trace.latent("input-noise", noise_value)
        trace.manifest["inputLatent"] = trace.latent("input-latent", samples_in)
        trace.manifest["declaredSteps"] = declared_steps
        trace._flush()
        mask = latent.get("noise_mask")
        x0 = {}
        preview = latent_preview.prepare_callback(guider.model_patcher, sigmas.shape[-1] - 1, x0)

        def callback(step, denoised, state, total):
            trace.sample_step(step, total, float(sigmas[step].item()), state, denoised)
            preview(step, denoised, state, total)

        try:
            output = guider.sample(
                noise_value,
                samples_in,
                sampler,
                sigmas,
                denoise_mask=mask,
                callback=callback,
                disable_pbar=not comfy.utils.PROGRESS_BAR_ENABLED,
                seed=noise.seed,
            )
        finally:
            forward_capture.close()
        output = output.to(comfy.model_management.intermediate_device())
        trace.final_latent(output)
        trace.manifest["callbackSteps"] = len(trace.manifest["trace"])
        trace._flush()
        out = latent.copy()
        out.pop("downscale_ratio_spacial", None)
        out.pop("downscale_ratio_temporal", None)
        out["samples"] = output
        if "x0" not in x0:
            return (out, out)
        denoised = x0["x0"]
        if output.is_nested and not denoised.is_nested:
            shapes = [item.shape for item in output.unbind()]
            denoised = comfy.nested_tensor.NestedTensor(
                comfy.utils.unpack_latents(denoised, shapes)
            )
        denoised = guider.model_patcher.model.process_latent_out(denoised.cpu())
        denoised_out = latent.copy()
        denoised_out["samples"] = denoised
        return (out, denoised_out)


class _H3TraceDecoded:
    CATEGORY = "attelier/trace"
    FUNCTION = "record"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trace_root": ("STRING", {"default": "/workspace/forge/trace-experiment"}),
                "run_id": ("STRING", {"default": "replace-at-submission"}),
                "split": (["calibration", "holdout"],),
                "condition": (
                    [
                        "honest_repeat",
                        "steps_10",
                        "steps_3",
                        "alt_checkpoint",
                        "lora",
                        "wrong_output",
                        "padded_steps_10",
                        "sidecar_honest_wrong_output",
                    ],
                ),
                "lineage_id": ("STRING", {"default": "replace-at-submission"}),
                "input_hash": ("STRING", {"default": "replace-at-submission"}),
                "model_manifest_hash": ("STRING", {"default": "replace-at-submission"}),
                "workflow_hash": ("STRING", {"default": "replace-at-submission"}),
                "runtime_id": ("STRING", {"default": "replace-at-submission"}),
            }
        }

    def _trace(self, values):
        return _run(*values)


class H3TraceImages(_H3TraceDecoded):
    RETURN_TYPES = ("IMAGE",)

    @classmethod
    def INPUT_TYPES(cls):
        values = super().INPUT_TYPES()
        values["required"] = {"images": ("IMAGE",), **values["required"]}
        return values

    def record(self, images, **kwargs):
        self._trace(
            (
                kwargs.pop("trace_root"),
                kwargs.pop("run_id"),
                kwargs.pop("split"),
                kwargs.pop("condition"),
                kwargs.pop("lineage_id"),
                kwargs.pop("input_hash"),
                kwargs.pop("model_manifest_hash"),
                kwargs.pop("workflow_hash"),
                kwargs.pop("runtime_id"),
            )
        ).decoded_images(images)
        return (images,)


class H3TraceAudio(_H3TraceDecoded):
    RETURN_TYPES = ("AUDIO",)

    @classmethod
    def INPUT_TYPES(cls):
        values = super().INPUT_TYPES()
        values["required"] = {"audio": ("AUDIO",), **values["required"]}
        return values

    def record(self, audio, **kwargs):
        self._trace(
            (
                kwargs.pop("trace_root"),
                kwargs.pop("run_id"),
                kwargs.pop("split"),
                kwargs.pop("condition"),
                kwargs.pop("lineage_id"),
                kwargs.pop("input_hash"),
                kwargs.pop("model_manifest_hash"),
                kwargs.pop("workflow_hash"),
                kwargs.pop("runtime_id"),
            )
        ).decoded("audioWaveform", audio)
        return (audio,)


def _tensor_digest(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().to("cpu").contiguous()
    digest = hashlib.sha256(_canonical({"shape": list(cpu.shape), "dtype": str(cpu.dtype)}))
    digest.update(cpu.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _to_device(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        if set(value) == {_UUID_TAG}:
            return uuid.UUID(str(value[_UUID_TAG]))
        if set(value) == {_PACKED_LAYOUT_TAG}:
            state = value[_PACKED_LAYOUT_TAG]
            if not isinstance(state, dict) or set(state) != set(_PACKED_LAYOUT_FIELDS):
                raise ValueError("invalid PackedLayout replay record")
            from comfy.ldm.minimax.model import PackedLayout

            layout = PackedLayout.__new__(PackedLayout)
            for field in _PACKED_LAYOUT_FIELDS:
                # The original layout stores its lookup tables on CPU; MiniMax
                # moves only the tables it needs inside its own forward.
                setattr(layout, field, _to_device(state[field], torch.device("cpu")))
            return layout
        return {key: _to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_device(item, device) for item in value]
    if value is None or isinstance(value, (bool, float, int, str)):
        return value
    raise ValueError(f"unsupported replay value: {type(value).__name__}")


class H3TraceForwardReplay:
    """Re-evaluate a real captured DiT call with the connected MODEL.

    The caller loads either the approved model or an alternative checkpoint in
    this separate graph, holding the input, timestep and conditioning fixed.
    """

    CATEGORY = "attelier/trace"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    FUNCTION = "replay"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "bundle_path": ("STRING", {"default": "/workspace/forge/trace/forward/000.pt"}),
                "result_path": ("STRING", {"default": "/workspace/forge/trace/replay.json"}),
                "candidate_label": ("STRING", {"default": "approved-model"}),
            }
        }

    def replay(self, model, bundle_path, result_path, candidate_label):
        comfy.model_management.load_model_gpu(model)
        return self._replay_loaded(model, bundle_path, result_path, candidate_label)

    def _replay_loaded(self, model, bundle_path, result_path, candidate_label):
        bundle = Path(bundle_path).resolve()
        result = Path(result_path).resolve()
        if not bundle.is_file() or not bundle.is_absolute() or not result.is_absolute():
            raise ValueError("absolute replay bundle/result paths required")
        if result.exists():
            raise FileExistsError("replay result already exists")
        payload = torch.load(bundle, map_location="cpu", weights_only=True)
        if set(payload) != {"input", "timestep", "conditioning", "prediction"}:
            raise ValueError("trace forward bundle shape invalid")
        if not isinstance(payload["input"], torch.Tensor) or not isinstance(
            payload["prediction"], torch.Tensor
        ):
            raise ValueError("trace forward tensors missing")
        with torch.inference_mode():
            actual = model.model.apply_model(
                _to_device(payload["input"], model.load_device),
                _to_device(payload["timestep"], model.load_device),
                **_to_device(payload["conditioning"], model.load_device),
            )
        expected = payload["prediction"]
        if not isinstance(actual, torch.Tensor) or tuple(actual.shape) != tuple(expected.shape):
            outcome = {"status": "INCONCLUSIVE", "reason": "prediction_shape_mismatch"}
        else:
            difference = actual.detach().to("cpu").float() - expected.float()
            rms = difference.square().mean().sqrt().item()
            reference_rms = expected.float().square().mean().sqrt().item()
            outcome = {
                "status": "MEASURED",
                "maxAbs": difference.abs().max().item(),
                "rms": rms,
                "relativeRms": rms / max(reference_rms, 1e-12),
                "expectedSha256": _tensor_digest(expected),
                "actualSha256": _tensor_digest(actual),
            }
        receipt = {
            "schema": "forge/inference-forward-replay/v1",
            "candidateLabel": candidate_label,
            "bundleSha256": _digest(bundle),
            "outcome": outcome,
        }
        result.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = result.with_suffix(".partial")
        temporary.write_bytes(_canonical(receipt))
        os.chmod(temporary, 0o600)
        os.replace(temporary, result)
        return (json.dumps(receipt, sort_keys=True),)


class H3TraceForwardReplayBatch:
    """Replay up to 64 fixed-input calls without repeatedly loading one UNET."""

    CATEGORY = "attelier/trace"
    OUTPUT_NODE = True
    RETURN_TYPES = ("STRING",)
    FUNCTION = "replay_batch"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "requests_json": ("STRING", {"default": "[]", "multiline": True}),
            }
        }

    def replay_batch(self, model, requests_json):
        requests = json.loads(requests_json)
        if not isinstance(requests, list) or not 1 <= len(requests) <= 64:
            raise ValueError("replay batch must contain 1..64 requests")
        required = {"bundle_path", "result_path", "candidate_label"}
        bundles, results = set(), set()
        for request in requests:
            if not isinstance(request, dict) or set(request) != required:
                raise ValueError("invalid replay batch request")
            bundle, result = Path(request["bundle_path"]), Path(request["result_path"])
            if (
                not bundle.is_absolute()
                or not result.is_absolute()
                or not bundle.is_file()
                or result.exists()
            ):
                raise ValueError("replay batch paths invalid or result already exists")
            bundle_key, result_key = str(bundle.resolve()), str(result.resolve())
            if (
                bundle_key in bundles
                or result_key in results
                or not isinstance(request["candidate_label"], str)
            ):
                raise ValueError("replay batch has duplicate paths or invalid label")
            bundles.add(bundle_key)
            results.add(result_key)
        # One ModelPatcher load; all calls below reuse its resident weights.
        comfy.model_management.load_model_gpu(model)
        single = H3TraceForwardReplay()
        receipts = [json.loads(single._replay_loaded(model, **request)[0]) for request in requests]
        return (
            json.dumps(
                {"schema": "forge/inference-forward-replay-batch/v1", "receipts": receipts},
                sort_keys=True,
            ),
        )


class H3TraceFinalLatentLoader:
    """Restore a captured H3 AV final latent for a VAE-only audit graph."""

    CATEGORY = "attelier/trace"
    RETURN_TYPES = ("LATENT",)
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "manifest_path": (
                    "STRING",
                    {"default": "/workspace/forge/trace-experiment/run/manifest.json"},
                )
            }
        }

    def load(self, manifest_path):
        manifest = Path(manifest_path).resolve()
        if not manifest.is_file() or not manifest.is_absolute():
            raise ValueError("absolute trace manifest path required")
        document = json.loads(manifest.read_text())
        if document.get("schema") != SCHEMA:
            raise ValueError("unsupported trace manifest")
        descriptors = document.get("finalLatent")
        if (
            not isinstance(descriptors, list)
            or len(descriptors) != 2
            or {entry.get("component") for entry in descriptors}
            != {
                "video",
                "audio",
            }
        ):
            raise ValueError("trace final latent must contain video/audio components")
        components = {}
        root = manifest.parent.resolve()
        for descriptor in descriptors:
            required = {"component", "tensorPath", "sha256", "shape", "dtype", "bytes"}
            if not isinstance(descriptor, dict) or not required <= set(descriptor):
                raise ValueError("trace final latent descriptor invalid")
            relative = Path(str(descriptor["tensorPath"]))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("trace final latent path unsafe")
            tensor_path = (root / relative).resolve()
            try:
                tensor_path.relative_to(root)
            except ValueError as error:
                raise ValueError("trace final latent path escapes manifest") from error
            if not tensor_path.is_file() or _digest(tensor_path) != descriptor["sha256"]:
                raise ValueError("trace final latent custody mismatch")
            if tensor_path.stat().st_size != descriptor["bytes"]:
                raise ValueError("trace final latent byte count mismatch")
            value = torch.load(tensor_path, map_location="cpu", weights_only=True)
            if not isinstance(value, torch.Tensor):
                raise ValueError("trace final latent tensor invalid")
            if (
                list(value.shape) != descriptor["shape"]
                or str(value.dtype).removeprefix("torch.") != descriptor["dtype"]
            ):
                raise ValueError("trace final latent tensor metadata mismatch")
            components[descriptor["component"]] = value
        return (
            {
                "samples": comfy.nested_tensor.NestedTensor(
                    (components["video"], components["audio"])
                )
            },
        )


class H3TraceSeal:
    """Make a complete sampler trace an explicit Comfy graph completion point.

    It deliberately does not decode or save media.  A worker capture graph can
    terminate at this node, preventing a miner-provided MP4 from becoming an
    implicit success artifact.  The validator's renderer later consumes the
    receipt-bound final latent under its own graph.
    """

    CATEGORY = "attelier/trace"
    OUTPUT_NODE = True
    RETURN_TYPES = ()
    FUNCTION = "seal"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # This edge is essential: OUTPUT_NODE execution otherwise has
                # no dependency on the sampler and Comfy can seal an empty
                # manifest before node 10 ever runs.
                "samples": ("LATENT",),
                "trace_root": ("STRING", {"default": "/workspace/forge/trace-experiment"}),
                "run_id": ("STRING", {"default": "replace-at-submission"}),
            }
        }

    def seal(self, samples, trace_root, run_id):
        if not isinstance(samples, dict) or "samples" not in samples:
            raise ValueError("trace seal latent input invalid")
        root, name = Path(trace_root).resolve(), str(run_id)
        if not root.is_absolute() or not name or Path(name).name != name or ".." in name:
            raise ValueError("trace root/run id invalid before seal")
        manifest = root / name / "manifest.json"
        if not manifest.is_file():
            raise ValueError("trace manifest missing before seal")
        document = json.loads(manifest.read_text())
        if document.get("schema") != SCHEMA or document.get("runId") != name:
            raise ValueError("trace manifest identity invalid before seal")
        steps = document.get("declaredSteps")
        if (
            steps != 20
            or document.get("callbackSteps") != steps
            or len(document.get("trace", [])) != steps
            or len(document.get("forward", [])) != steps
            or not document.get("finalLatent")
        ):
            raise ValueError("trace incomplete before seal")
        sealed = root / name / "sealed.json"
        if sealed.exists():
            raise FileExistsError("trace already sealed")
        temporary = sealed.with_suffix(".partial")
        receipt = {
            "schema": "forge/inference-trace-seal/v1",
            "runId": name,
            "manifestSha256": _digest(manifest),
            "steps": steps,
        }
        with temporary.open("xb") as stream:
            os.fchmod(stream.fileno(), 0o600)
            stream.write(_canonical(receipt))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, sealed)
        return ()


NODE_CLASS_MAPPINGS = {
    "H3TraceSamplerCustomAdvanced": H3TraceSamplerCustomAdvanced,
    "H3TraceImages": H3TraceImages,
    "H3TraceAudio": H3TraceAudio,
    "H3TraceForwardReplay": H3TraceForwardReplay,
    "H3TraceForwardReplayBatch": H3TraceForwardReplayBatch,
    "H3TraceFinalLatentLoader": H3TraceFinalLatentLoader,
    "H3TraceSeal": H3TraceSeal,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "H3TraceSamplerCustomAdvanced": "H3 Trace Sampler (Experiment)",
    "H3TraceImages": "H3 Trace Decoded Video (Experiment)",
    "H3TraceAudio": "H3 Trace Decoded Audio (Experiment)",
    "H3TraceForwardReplay": "H3 Trace Fixed-Input Forward Replay (Experiment)",
    "H3TraceForwardReplayBatch": "H3 Trace Fixed-Input Forward Replay Batch (Experiment)",
    "H3TraceFinalLatentLoader": "H3 Trace Final Latent Loader (Experiment)",
    "H3TraceSeal": "H3 Trace Complete Capture Seal",
}
