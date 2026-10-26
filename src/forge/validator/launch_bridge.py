"""Concrete, fail-closed bridge from V1 execution receipts to consensus.

The V1 policy is deliberately smaller than the historical MVP 32/128 route.
This module does not weaken that route or accept its old receipts.  It reads
only an operator-owned, read-only receipt directory: model submissions cannot
provide rows or a claimed policy result.  That local custody is the remaining
honest-validator trust boundary; it is not remote GPU attestation.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from forge.chain.reward_publication import WeightIntent, testnet_weight_intent
from forge.evaluation import launch_policy
from forge.evaluation.evaluator_qualification import replay_qualification_package
from forge.evaluation.launch_reference import replay_bootstrap_reference
from forge.evaluation.synthetic_prompts import launch_panel
from forge.models.artifact_manifest import ModelManifest
from forge.protocol.admission_checkpoint import SignedAdmissionHistory, verify_signed_history
from forge.protocol.consensus_types import (
    CertificateError,
    FinalizedCompetitionState,
    LocalEvaluationReport,
    LocalVerdict,
    RoundCertificate,
    RoundPlan,
)
from forge.protocol.hashing import hash_canonical, sha256_file
from forge.protocol.ports import Verifier
from forge.protocol.preconsensus import CertificateVerifier, apply_certificate
from forge.protocol.store import Store

BUNDLE_SCHEMA = "forge/launch-execution-bundle/v1"
BUNDLE_SCHEMA_V2 = "forge/launch-execution-bundle/v2"
SCORE_SCHEMA = "forge/launch-score/v1"
PANEL_SCHEMA = "forge/panel/v1"
TIMING_SCHEMAS = {"forge/paired-timing/v1", "forge/screen-timing/v1"}


class LaunchBridgeError(ValueError):
    """The provisional launch route lacks an authority-bearing binding."""


def _require_entropy_panel(round_plan: RoundPlan, policy_plan: Mapping[str, Any]) -> None:
    """Bind the public V1 prompt panel to post-close entropy in every entry path."""

    inputs = round_plan.prompt_seed_inputs
    entropy_block, entropy_hash = inputs.get("entropy_block"), inputs.get("entropy_hash")
    if (
        type(entropy_block) is not int
        or entropy_block <= round_plan.snapshot_block
        or entropy_block >= round_plan.commit_deadline_block
    ):
        raise LaunchBridgeError("launch entropy block is outside the frozen round window")
    _sha(entropy_hash, "launch entropy hash")
    if launch_policy.policy_for(policy_plan) == "v2":
        launch_policy.validate_plan(
            policy_plan, expected_entropy_hash=entropy_hash, expected_round_id=round_plan.round_id
        )
        return
    expected_prompts = launch_panel(entropy_hash, round_id=round_plan.round_id)
    if policy_plan.get("prompts") != expected_prompts:
        raise LaunchBridgeError("launch prompts differ from the public entropy panel")
    timing = [
        prompt_id
        for prompt_id in sorted(
            (prompt["prompt_id"] for prompt in expected_prompts),
            key=lambda prompt_id: hash_canonical(
                "forge/launch-v1-timing/v1", entropy_hash, round_plan.round_id, prompt_id
            ),
        )[:8]
    ]
    if policy_plan.get("timing_prompt_ids") != timing:
        raise LaunchBridgeError("launch timing prompts differ from the public entropy panel")


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or set(value) - set("0123456789abcdef"):
        raise LaunchBridgeError(f"{name} must be a lowercase SHA-256")
    return value


class LocalLaunchReceiptStore:
    """Read immutable local execution receipts by content hash.

    The directory must be owned and populated by validator operations, not a
    miner upload endpoint.  A hash proves the stored bytes did not change; it
    cannot itself prove that a GPU executed them, hence callers must not mount
    an untrusted submission directory here.
    """

    def __init__(self, root: str | Path):
        self._input_root = Path(root)
        self.root = self._input_root.resolve()

    def read(self, kind: str, digest: str) -> dict[str, Any]:
        _sha(digest, f"{kind} receipt hash")
        if kind not in {
            "bundles",
            "panels",
            "timing",
            "scores",
            "diversity",
            "qualification",
            "capacity",
            "custody",
            "burn",
            "bootstrap",
        }:
            raise LaunchBridgeError("unsupported local receipt kind")
        raw_directory = self.root / kind
        if (
            self._input_root.is_symlink()
            or raw_directory.parent != self.root
            or raw_directory.is_symlink()
            or not raw_directory.is_dir()
        ):
            raise LaunchBridgeError("trusted local receipt directory is unsafe")
        directory = raw_directory.resolve()
        named = directory / f"{digest}.json"
        candidates = [named] if named.is_file() else list(directory.glob("*.json"))
        path = next(
            (
                item
                for item in candidates
                if item.is_file()
                and not item.is_symlink()
                and item.parent == raw_directory
                and sha256_file(item) == digest
            ),
            None,
        )
        if path is None:
            raise LaunchBridgeError(f"trusted local {kind} receipt is absent or altered")
        try:
            value = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise LaunchBridgeError(f"trusted local {kind} receipt is invalid JSON") from error
        if not isinstance(value, dict):
            raise LaunchBridgeError(f"trusted local {kind} receipt must be an object")
        return value


@dataclass(frozen=True)
class LaunchV1Capacity:
    """Measured V1 capacity, intentionally separate from MVP 32/128 rules."""

    receipt_hash: str


def _genesis_anchor(store: Store, state: FinalizedCompetitionState) -> str:
    """Recover the immutable baseline from the persisted finalized-state chain."""

    current = state
    payload = store.state_payload(current.state_hash())
    if payload is None:
        if current.state_version != 0:
            raise LaunchBridgeError("missing finalized-state history for a non-genesis parent")
    else:
        try:
            stored = FinalizedCompetitionState.from_dict(payload)
        except (CertificateError, KeyError, TypeError, ValueError) as error:
            raise LaunchBridgeError("stored finalized state is invalid") from error
        if stored != current:
            raise LaunchBridgeError("supplied parent differs from stored finalized state")
    while current.state_version > 0:
        parent_payload = store.state_payload(current.previous_state_hash)
        if parent_payload is None:
            raise LaunchBridgeError("finalized-state history is incomplete")
        try:
            parent = FinalizedCompetitionState.from_dict(parent_payload)
        except (CertificateError, KeyError, TypeError, ValueError) as error:
            raise LaunchBridgeError("stored ancestor state is invalid") from error
        if (
            parent.challenge_id != current.challenge_id
            or parent.state_version + 1 != current.state_version
            or parent.state_hash() != current.previous_state_hash
        ):
            raise LaunchBridgeError("finalized-state ancestry is inconsistent")
        current = parent
    expected_genesis = FinalizedCompetitionState.genesis(
        current.challenge_id, current.current_king_fingerprint, current.activation_block
    )
    if current != expected_genesis:
        raise LaunchBridgeError("finalized-state root is not a valid genesis")
    return current.current_king_fingerprint


class LaunchExecutionReplayer:
    """Rebuild V1 policy rows from raw validator-owned receipts only."""

    def __init__(self, receipts: LocalLaunchReceiptStore):
        self.receipts = receipts

    @staticmethod
    def _require_binding(
        bundle: Mapping[str, Any], round_plan: RoundPlan, policy_plan: Mapping[str, Any]
    ) -> None:
        required = {
            "round_plan_hash": round_plan.plan_hash(),
            "parent_state_hash": round_plan.parent_state_hash,
            "admission_history_root": round_plan.prompt_seed_inputs.get("admission_history_root"),
            "prompt_history_root": round_plan.prompt_seed_inputs.get("prompt_history_root"),
            "launch_plan_id": policy_plan.get("plan_id"),
        }
        if any(
            not isinstance(value, str) or bundle.get(key) != value
            for key, value in required.items()
        ):
            raise LaunchBridgeError("execution bundle does not bind the frozen round histories")
        if bundle.get("models") != policy_plan.get("models"):
            raise LaunchBridgeError("execution bundle model binding differs from launch plan")
        if (
            policy_plan["models"]["candidate"] != round_plan.challenger_fingerprint
            or policy_plan["models"]["king"] != round_plan.king_fingerprint
            or round_plan.prompt_seed_inputs.get("launch_plan_id") != policy_plan["plan_id"]
            or round_plan.prompt_seed_inputs.get("anchor_fingerprint")
            != policy_plan["models"]["anchor"]
        ):
            raise LaunchBridgeError("launch models are not bound to the certified round")

    @staticmethod
    def _panel_videos(
        panel: Mapping[str, Any],
        digest: str,
        policy_plan: Mapping[str, Any],
        contract_hash: str,
    ) -> dict[tuple[str, str, int], str]:
        if panel.get("schema") != PANEL_SCHEMA or panel.get("geometry") != policy_plan["geometry"]:
            raise LaunchBridgeError("source panel schema or geometry is invalid")
        if (
            panel.get("contract_hash") != contract_hash
            or panel.get("admission") != "public_snapshot"
        ):
            raise LaunchBridgeError("source panel is not a public validator execution")
        runtime = panel.get("runtime", {})
        if (
            panel.get("runtime_id") != policy_plan["runtime"]["runtime_id"]
            or hash_canonical("forge/launch-hardware/v1", panel.get("hardware"))
            != policy_plan["runtime"]["hardware_id"]
            or hash_canonical("forge/launch-dependencies/v1", runtime.get("dependencies"))
            != policy_plan["runtime"]["dependency_lock_hash"]
        ):
            raise LaunchBridgeError("source panel runtime differs from launch plan")
        manifest = panel.get("manifest_hash")
        try:
            manifest_object = ModelManifest.from_dict(panel.get("manifest"))
        except (TypeError, ValueError) as error:
            raise LaunchBridgeError("source panel manifest is invalid") from error
        if manifest_object.fingerprint() != manifest:
            raise LaunchBridgeError("source panel manifest hash changed")
        if manifest not in set(policy_plan["models"].values()) or not isinstance(
            panel.get("observations"), list
        ):
            raise LaunchBridgeError("source panel does not bind a launch model")
        prompt_map = {item["prompt_id"]: item for item in policy_plan["prompts"]}
        actual_prompts = {
            item.get("prompt_id"): item
            for item in panel.get("prompts", [])
            if isinstance(item, Mapping)
        }
        if (
            not actual_prompts
            or len(actual_prompts) != len(panel.get("prompts", []))
            or not set(actual_prompts).issubset(prompt_map)
        ):
            raise LaunchBridgeError("source panel prompt subset is invalid")
        for prompt_id, actual in actual_prompts.items():
            prompt = prompt_map[prompt_id]
            if actual.get("text") != prompt["text"] or actual.get("video_seeds") != [
                prompt["seed"]
            ]:
                raise LaunchBridgeError("source panel prompt/seed differs from launch plan")
        videos: dict[tuple[str, str, int], str] = {}
        for item in panel["observations"]:
            if not isinstance(item, Mapping) or item.get("status") != "OK":
                raise LaunchBridgeError("source panel has a missing execution")
            prompt_id, seed, video = (
                item.get("prompt_id"),
                item.get("seed"),
                item.get("video_sha256"),
            )
            if (
                not isinstance(prompt_id, str)
                or type(seed) is not int
                or prompt_map.get(prompt_id, {}).get("seed") != seed
            ):
                raise LaunchBridgeError("source panel observation identity is invalid")
            _sha(video, "source panel video hash")
            key = (manifest, prompt_id, seed)
            if key in videos:
                raise LaunchBridgeError("source panel duplicated a video")
            videos[key] = video
        if {(pid, seed) for _, pid, seed in videos} != {
            (prompt_id, prompt_map[prompt_id]["seed"]) for prompt_id in actual_prompts
        }:
            raise LaunchBridgeError(
                "source panel observation coverage differs from its prompt subset"
            )
        return videos

    def _score_rows(
        self,
        hashes: list[Any],
        policy_plan: Mapping[str, Any],
        panels: Mapping[str, dict[tuple[str, str, int], str]],
    ) -> dict[tuple[str, str, int], dict[str, Any]]:
        rows: dict[tuple[str, str, int], dict[str, Any]] = {}
        expected_evaluator = policy_plan["evaluator"]
        for digest in hashes:
            score = self.receipts.read("scores", _sha(digest, "score receipt hash"))
            required = {
                "schema",
                "plan_id",
                "prompt_id",
                "prompt_text",
                "seed",
                "manifest_hash",
                "video_sha256",
                "evaluator_id",
                "spec_hash",
                "geometry",
                "result",
                "source_report_sha256",
            }
            if (
                set(score) != required
                or score["schema"] != SCORE_SCHEMA
                or score["plan_id"] != policy_plan["plan_id"]
            ):
                raise LaunchBridgeError("score receipt schema or launch plan changed")
            if (
                score["evaluator_id"] != expected_evaluator["evaluator_id"]
                or score["spec_hash"] != expected_evaluator["spec_hash"]
                or score["geometry"] != policy_plan["geometry"]
            ):
                raise LaunchBridgeError("score evaluator or geometry differs from launch plan")
            manifest, prompt_id, seed = score["manifest_hash"], score["prompt_id"], score["seed"]
            if manifest not in set(policy_plan["models"].values()) or type(seed) is not int:
                raise LaunchBridgeError("score receipt model or seed is foreign")
            source = _sha(score["source_report_sha256"], "score source report hash")
            video = _sha(score["video_sha256"], "score video hash")
            if source not in panels or panels[source].get((manifest, prompt_id, seed)) != video:
                raise LaunchBridgeError("score is not derived from a supplied raw panel video")
            result = score["result"]
            if (
                not isinstance(result, Mapping)
                or set(result)
                != {"scores", "diagnostics", "elapsed_seconds", "peak_cuda_bytes", "video_sha256"}
                or result.get("video_sha256") != video
            ):
                raise LaunchBridgeError("score result provenance is invalid")
            expected_prompt = next(
                (p for p in policy_plan["prompts"] if p["prompt_id"] == prompt_id), None
            )
            if (
                expected_prompt is None
                or expected_prompt["seed"] != seed
                or expected_prompt["text"] != score["prompt_text"]
            ):
                raise LaunchBridgeError("score prompt text/seed differs from launch plan")
            key = (manifest, prompt_id, seed)
            row = {
                "video_sha256": video,
                "scores": result["scores"],
                "diagnostics": result["diagnostics"],
            }
            if key in rows:
                raise LaunchBridgeError("duplicate trusted score receipt")
            rows[key] = row
        return rows

    def _timing_rows(
        self,
        hashes: list[Any],
        policy_plan: Mapping[str, Any],
        contract_hash: str,
        panels: Mapping[str, dict[tuple[str, str, int], str]],
    ) -> dict[str, list[float]]:
        rows: dict[str, list[float]] = {}
        runtime = policy_plan["runtime"]
        seen_slots: set[str] = set()
        for digest in hashes:
            timing = self.receipts.read("timing", _sha(digest, "timing receipt hash"))
            if (
                timing.get("schema") not in TIMING_SCHEMAS
                or timing.get("frozen_plan_id") != policy_plan["plan_id"]
                or timing.get("contract_hash") != contract_hash
            ):
                raise LaunchBridgeError("timing receipt lacks the frozen launch plan")
            if (
                timing.get("reference_manifest_hash") != policy_plan["models"]["king"]
                or timing.get("candidate_manifest_hash") != policy_plan["models"]["candidate"]
                or timing.get("runtime_id") != runtime["runtime_id"]
                or timing.get("runtime_id")
                != hash_canonical("forge/runtime/v1", timing.get("runtime"))
                or timing.get("geometry") != policy_plan["geometry"]
            ):
                raise LaunchBridgeError("timing receipt bindings differ from launch plan")
            hardware = hash_canonical("forge/launch-hardware/v1", timing.get("hardware"))
            dependencies = hash_canonical(
                "forge/launch-dependencies/v1", timing.get("runtime", {}).get("dependencies")
            )
            if (
                hardware != runtime["hardware_id"]
                or dependencies != runtime["dependency_lock_hash"]
            ):
                raise LaunchBridgeError(
                    "timing hardware or dependency lock differs from launch plan"
                )
            video_links = timing.get("report_video_sha256")
            if (
                not isinstance(video_links, Mapping)
                or set(video_links)
                != {
                    "reference",
                    "candidate",
                }
                or not all(isinstance(value, Mapping) for value in video_links.values())
            ):
                raise LaunchBridgeError("timing receipt lacks panel video links")
            groups: dict[str, list[Mapping[str, Any]]] = {}
            seed_by_prompt = {item["prompt_id"]: item["seed"] for item in policy_plan["prompts"]}
            is_abba = (
                timing.get("schema") == "forge/paired-timing/v1"
                and timing.get("design", {}).get("name") == "confirmation_abba"
            )
            is_ab = (
                timing.get("schema") == "forge/screen-timing/v1"
                and timing.get("design", {}).get("name") == "screening_ab"
            )
            if not is_abba and not is_ab:
                raise LaunchBridgeError("timing receipt has an unrecognized timing design")
            for item in timing.get("measurements", []):
                if not isinstance(item, Mapping) or item.get("status") != "OK":
                    raise LaunchBridgeError("timing receipt has a failed execution")
                prompt_id = item.get("prompt_id")
                if (
                    prompt_id not in seed_by_prompt
                    or item.get("seed") != seed_by_prompt[prompt_id]
                    or item.get("pair_id") != f"{prompt_id}:{seed_by_prompt[prompt_id]}"
                    or item.get("gpu_synchronized") is not True
                ):
                    raise LaunchBridgeError("timing receipt has a foreign or unsynchronized row")
                pair_key = f"{prompt_id}:{seed_by_prompt[prompt_id]}"
                for side, manifest in (
                    ("reference", policy_plan["models"]["king"]),
                    ("candidate", policy_plan["models"]["candidate"]),
                ):
                    video = video_links[side].get(pair_key)
                    if not isinstance(video, str) or not any(
                        panel.get((manifest, prompt_id, seed_by_prompt[prompt_id])) == video
                        for panel in panels.values()
                    ):
                        raise LaunchBridgeError("timing receipt is not linked to panel videos")
                groups.setdefault(prompt_id, []).append(item)
            if (
                not groups
                or (is_abba and not set(groups).issubset(policy_plan["timing_prompt_ids"]))
                or (is_ab and set(groups) & set(policy_plan["timing_prompt_ids"]))
            ):
                raise LaunchBridgeError(
                    "timing receipt uses the wrong prompt subset for its design"
                )
            for prompt_id, values in groups.items():
                ordered = sorted(values, key=lambda item: item.get("run_order", -1))
                expected_sides = (
                    ["reference", "candidate", "candidate", "reference"]
                    if is_abba
                    else ["reference", "candidate"]
                )
                if (
                    len(ordered) != len(expected_sides)
                    or [item.get("run_order") for item in ordered]
                    != list(range(len(expected_sides)))
                    or [item.get("side") for item in ordered] != expected_sides
                ):
                    raise LaunchBridgeError(
                        "timing receipt does not match its frozen AB/ABBA design"
                    )
                times = []
                for item in ordered:
                    value = item.get("full_pipeline_seconds", item.get("wall_time_seconds"))
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value)
                        or value <= 0
                    ):
                        raise LaunchBridgeError("timing receipt measurement is invalid")
                    times.append(float(value))
                if prompt_id in seen_slots:
                    raise LaunchBridgeError("duplicate trusted timing receipt")
                seen_slots.add(prompt_id)
                if is_abba:
                    rows[prompt_id] = times
        if seen_slots != {item["prompt_id"] for item in policy_plan["prompts"]}:
            raise LaunchBridgeError("timing ABBA/AB shards do not cover the frozen quality panel")
        if set(rows) != set(policy_plan["timing_prompt_ids"]):
            raise LaunchBridgeError("timing ABBA shards do not cover the frozen speed panel")
        return rows

    def _diversity_rows(
        self, receipts: list[str], policy_plan: Mapping[str, Any], contract_hash: str
    ) -> list[dict[str, Any]]:
        """Import V2 decoded-pixel diversity receipts from trusted local custody."""
        found: dict[str, dict[str, Any]] = {}
        for digest in receipts:
            receipt = self.receipts.read("diversity", _sha(digest, "diversity receipt hash"))
            body = {key: value for key, value in receipt.items() if key != "receipt_id"}
            if (
                receipt.get("schema") != "forge/launch-diversity/v2"
                or receipt.get("plan_id") != policy_plan["plan_id"]
                or receipt.get("receipt_id") != hash_canonical("forge/launch-diversity/v2", body)
                or not isinstance(receipt.get("rows"), list)
            ):
                raise LaunchBridgeError("diversity receipt is invalid or unbound")
            source = _sha(receipt.get("source_report_sha256"), "diversity source report hash")
            panel = self.receipts.read("panels", source)
            if (
                panel.get("schema") != PANEL_SCHEMA
                or panel.get("admission") != "public_snapshot"
                or panel.get("contract_hash") != contract_hash
                or panel.get("geometry") != policy_plan["geometry"]
                or panel.get("runtime_id") != policy_plan["runtime"]["runtime_id"]
                or panel.get("manifest_hash") != policy_plan["models"]["candidate"]
                or hash_canonical("forge/launch-hardware/v1", panel.get("hardware"))
                != policy_plan["runtime"]["hardware_id"]
                or hash_canonical(
                    "forge/launch-dependencies/v1", panel.get("runtime", {}).get("dependencies")
                )
                != policy_plan["runtime"]["dependency_lock_hash"]
            ):
                raise LaunchBridgeError(
                    "diversity source panel differs from frozen candidate runtime"
                )
            actual_prompts = {
                item.get("prompt_id"): item
                for item in panel.get("prompts", [])
                if isinstance(item, Mapping)
            }
            observations: dict[tuple[str, int], str] = {}
            for item in panel.get("observations", []):
                if not isinstance(item, Mapping) or item.get("status") != "OK":
                    raise LaunchBridgeError("diversity source panel has incomplete execution")
                key = (item.get("prompt_id"), item.get("seed"))
                video = item.get("video_sha256")
                if key in observations or not isinstance(key[0], str) or type(key[1]) is not int:
                    raise LaunchBridgeError("diversity source panel observation is invalid")
                observations[key] = _sha(video, "diversity source video hash")
            for row in receipt["rows"]:
                if not isinstance(row, Mapping):
                    raise LaunchBridgeError("diversity receipt row is invalid")
                prompt_id = row.get("prompt_id")
                if prompt_id in found:
                    raise LaunchBridgeError("duplicate trusted diversity prompt")
                try:
                    seeds = launch_policy.diversity_seeds(policy_plan, prompt_id)
                except (TypeError, ValueError) as error:
                    raise LaunchBridgeError("diversity row prompt is not frozen") from error
                prompt = next(
                    item for item in policy_plan["prompts"] if item["prompt_id"] == prompt_id
                )
                if (
                    set(row)
                    != {
                        "type",
                        "plan_id",
                        "prompt_id",
                        "seeds",
                        "source_video_sha256",
                        "decoded_video_sha256",
                    }
                    or row.get("type") != "diversity"
                    or row.get("plan_id") != policy_plan["plan_id"]
                    or row.get("seeds") != seeds
                    or actual_prompts.get(prompt_id, {}).get("text") != prompt["text"]
                    or actual_prompts.get(prompt_id, {}).get("video_seeds") != seeds
                    or row.get("source_video_sha256")
                    != [observations.get((prompt_id, seed)) for seed in seeds]
                    or not isinstance(row.get("decoded_video_sha256"), list)
                    or len(row["decoded_video_sha256"]) != 3
                ):
                    raise LaunchBridgeError(
                        "diversity row is not bound to its source candidate videos"
                    )
                found[prompt_id] = {
                    "type": "diversity",
                    "plan_id": policy_plan["plan_id"],
                    "prompt_id": prompt_id,
                    "decoded_video_sha256": row["decoded_video_sha256"],
                }
        if set(found) != set(policy_plan["diversity_prompt_ids"]):
            raise LaunchBridgeError("diversity receipts do not cover frozen prompts")
        return list(found.values())

    def replay_receipts(
        self,
        *,
        policy_plan: Mapping[str, Any],
        contract_hash: str,
        panel_receipts: list[Any],
        timing_receipts: list[Any],
        score_receipts: list[Any],
        diversity_receipts: list[Any] | None = None,
        qualification_receipt: str | None = None,
    ) -> dict[str, Any]:
        """Rebuild policy rows from trusted custody without a consensus round.

        Callers still bind a policy to their own authority context before this
        method.  This common receipt graph deliberately retains every raw
        panel, score, timing, diversity and V2 qualification check used by a
        signed consensus bundle.
        """
        policy = dict(policy_plan)
        version = launch_policy.policy_for(policy)
        if version == "v1":
            launch_policy._check_plan(policy)
            if diversity_receipts is not None or qualification_receipt is not None:
                raise LaunchBridgeError("V1 receipt replay does not accept V2 evidence")
        else:
            launch_policy._check_plan_v2(policy)
            if not isinstance(diversity_receipts, list) or qualification_receipt is None:
                raise LaunchBridgeError("V2 receipt replay requires diversity and qualification")
            package = self.receipts.read(
                "qualification", _sha(qualification_receipt, "qualification receipt hash")
            )
            if replay_qualification_package(package) != policy["evaluator_qualification"]:
                raise LaunchBridgeError("V2 qualification receipt is not trusted or qualified")
        if (
            not isinstance(contract_hash, str)
            or not all(
                isinstance(value, list)
                for value in (panel_receipts, timing_receipts, score_receipts)
            )
            or len(set(panel_receipts)) != len(panel_receipts)
        ):
            raise LaunchBridgeError("execution receipt lists are invalid")
        panels = {
            digest: self._panel_videos(
                self.receipts.read("panels", _sha(digest, "panel receipt hash")),
                digest,
                policy,
                contract_hash,
            )
            for digest in panel_receipts
        }
        expected_slots = {(prompt["prompt_id"], prompt["seed"]) for prompt in policy["prompts"]}
        for manifest in policy["models"].values():
            slots: set[tuple[str, int]] = set()
            for panel in panels.values():
                for observed_manifest, prompt_id, seed in panel:
                    if observed_manifest != manifest:
                        continue
                    if (prompt_id, seed) in slots:
                        raise LaunchBridgeError("panel shards duplicate a model prompt")
                    slots.add((prompt_id, seed))
            if slots != expected_slots:
                raise LaunchBridgeError("panel shards do not cover a frozen launch model")
        scores = self._score_rows(score_receipts, policy, panels)
        timings = self._timing_rows(timing_receipts, policy, contract_hash, panels)
        rows: list[dict[str, Any]] = []
        for prompt in policy["prompts"]:
            prompt_id, seed = prompt["prompt_id"], prompt["seed"]
            candidate = scores[(policy["models"]["candidate"], prompt_id, seed)]
            for reference in ("king", "anchor"):
                rows.append(
                    {
                        "type": "quality",
                        "plan_id": policy["plan_id"],
                        "prompt_id": prompt_id,
                        "reference": reference,
                        "model_ids": {
                            "reference": policy["models"][reference],
                            "candidate": policy["models"]["candidate"],
                        },
                        "evaluator_id": policy["evaluator"]["evaluator_id"],
                        "geometry": policy["geometry"],
                        "reference_result": scores[(policy["models"][reference], prompt_id, seed)],
                        "candidate_result": candidate,
                    }
                )
        for prompt_id in policy["timing_prompt_ids"]:
            row: dict[str, Any] = {
                "type": "timing",
                "plan_id": policy["plan_id"],
                "prompt_id": prompt_id,
                "times": timings[prompt_id],
            }
            if version == "v1":
                row.update(
                    {
                        "model_ids": {
                            "reference": policy["models"]["king"],
                            "candidate": policy["models"]["candidate"],
                        },
                        "runtime_id": policy["runtime"]["runtime_id"],
                        "hardware_id": policy["runtime"]["hardware_id"],
                        "geometry": policy["geometry"],
                    }
                )
            rows.append(row)
        if version == "v2":
            rows.extend(self._diversity_rows(diversity_receipts or [], policy, contract_hash))
        return {"rows": rows, "result": launch_policy.evaluate(policy, rows)}

    def replay(
        self, bundle_hash: str, round_plan: RoundPlan, *, validator_hotkey: str | None = None
    ) -> dict[str, Any]:
        try:
            bundle = self.receipts.read("bundles", _sha(bundle_hash, "execution bundle hash"))
            required = {
                "schema",
                "validator_hotkey",
                "round_plan_hash",
                "parent_state_hash",
                "admission_history_root",
                "prompt_history_root",
                "launch_plan_id",
                "models",
                "launch_plan",
                "panel_receipts",
                "timing_receipts",
                "score_receipts",
            }
            v2_required = required | {"diversity_receipts", "qualification_receipt"}
            if (
                (set(bundle) != required and set(bundle) != v2_required)
                or bundle["schema"] not in {BUNDLE_SCHEMA, BUNDLE_SCHEMA_V2}
                or not isinstance(bundle["launch_plan"], dict)
            ):
                raise LaunchBridgeError("execution bundle schema is invalid")
            if validator_hotkey is not None and bundle["validator_hotkey"] != validator_hotkey:
                raise LaunchBridgeError("execution bundle belongs to another validator")
            policy_plan = bundle["launch_plan"]
            version = launch_policy.policy_for(policy_plan)
            if version == "v1":
                launch_policy._check_plan(policy_plan)
                if bundle["schema"] != BUNDLE_SCHEMA:
                    raise LaunchBridgeError("V1 plan requires V1 execution bundle")
            else:
                launch_policy._check_plan_v2(policy_plan)
                if bundle["schema"] != BUNDLE_SCHEMA_V2:
                    raise LaunchBridgeError("V2 plan requires V2 execution bundle")
                package = self.receipts.read(
                    "qualification",
                    _sha(bundle["qualification_receipt"], "qualification receipt hash"),
                )
                qualification = replay_qualification_package(package)
                if qualification != policy_plan["evaluator_qualification"]:
                    raise LaunchBridgeError("V2 qualification receipt is not trusted or qualified")
            _require_entropy_panel(round_plan, policy_plan)
            self._require_binding(bundle, round_plan, policy_plan)
            reconstructed = self.replay_receipts(
                policy_plan=policy_plan,
                contract_hash=round_plan.contract_hash,
                panel_receipts=bundle["panel_receipts"],
                timing_receipts=bundle["timing_receipts"],
                score_receipts=bundle["score_receipts"],
                diversity_receipts=bundle.get("diversity_receipts"),
                qualification_receipt=bundle.get("qualification_receipt"),
            )
            rows, result = reconstructed["rows"], reconstructed["result"]
            return {
                "status": result["status"],
                "launch_plan_id": policy_plan["plan_id"],
                "result_hash": hash_canonical(f"forge/launch-policy-result/{version}", result),
                "bundle_hash": bundle_hash,
                "result": result,
                "rows": rows,
            }
        except (KeyError, TypeError, ValueError, LaunchBridgeError) as error:
            return {"status": "INCOMPLETE", "reasons": [str(error)], "bundle_hash": bundle_hash}


def launch_local_report(
    plan: RoundPlan, validator_hotkey: str, bundle_hash: str, replayer: LaunchExecutionReplayer
) -> LocalEvaluationReport:
    if validator_hotkey not in plan.committee:
        raise CertificateError("local validator is outside the frozen committee")
    replay = replayer.replay(bundle_hash, plan, validator_hotkey=validator_hotkey)
    verdict = (
        LocalVerdict.PROMOTE
        if replay.get("status") in {"PROMOTE_ELIGIBLE", "PASS"}
        else (
            LocalVerdict.KEEP
            if replay.get("status") in {"FAIL", "INCONCLUSIVE"}
            else LocalVerdict.INCOMPLETE
        )
    )
    return LocalEvaluationReport(
        plan.network,
        plan.netuid,
        plan.round_id,
        validator_hotkey,
        plan.plan_hash(),
        plan.parent_state_hash,
        plan.contract_hash,
        plan.king_fingerprint,
        plan.challenger_fingerprint,
        verdict,
        {
            "launch_bundle_hash": bundle_hash,
            "launch_result_hash": replay.get("result_hash"),
            "launch_status": replay.get("status"),
        },
    )


class LaunchCertificateVerifier(CertificateVerifier):
    """Replay every signed V1 report from local execution evidence before promotion."""

    def __init__(
        self,
        signature_verifier: Verifier,
        *,
        replayer: LaunchExecutionReplayer,
        store: Store | None = None,
    ):
        super().__init__(signature_verifier, store)
        self.replayer = replayer

    def verify(
        self, certificate: RoundCertificate, plan: RoundPlan, state: FinalizedCompetitionState
    ) -> None:
        if self.store is None:
            raise CertificateError("V1 certificate verification requires finalized-state history")
        anchor = _genesis_anchor(self.store, state)
        if plan.prompt_seed_inputs.get("anchor_fingerprint") != anchor:
            if not plan.prompt_seed_inputs.get("bootstrap_receipt_sha256"):
                raise CertificateError("V1 plan anchor differs from the immutable genesis baseline")
            receipt = self.replayer.receipts.read(
                "bootstrap",
                _sha(plan.prompt_seed_inputs["bootstrap_receipt_sha256"], "bootstrap receipt hash"),
            )
            if replay_bootstrap_reference(
                receipt,
                contract_hash=plan.contract_hash,
                genesis_reference_hash=anchor,
            ) != plan.prompt_seed_inputs.get("anchor_fingerprint"):
                raise CertificateError("V2 bootstrap reference differs from the signed round")
        CertificateVerifier(self.signature_verifier).verify(certificate, plan, state)
        for reveal in certificate.reveals:
            report = reveal.report
            digest = report.metrics.get("launch_bundle_hash")
            if not isinstance(digest, str) or report.verdict == LocalVerdict.INCOMPLETE:
                raise CertificateError("signed launch report lacks trusted execution bundle")
            replay = launch_local_report(plan, report.validator_hotkey, digest, self.replayer)
            if (
                replay.verdict != report.verdict
                or replay.metrics.get("launch_result_hash")
                != report.metrics.get("launch_result_hash")
                or replay.metrics.get("launch_status") != report.metrics.get("launch_status")
                or replay.verdict == LocalVerdict.INCOMPLETE
            ):
                raise CertificateError(
                    "trusted V1 receipt replay does not reproduce signed verdict"
                )
        super().verify(certificate, plan, state)


class LaunchV1Bridge:
    """Open/certify V1 rounds without enabling legacy MVP capacity assumptions."""

    def __init__(
        self, store: Store, signature_verifier: Verifier, receipts: LocalLaunchReceiptStore
    ):
        self.store, self.signature_verifier, self.receipts = store, signature_verifier, receipts

    def open_round(
        self,
        plan: RoundPlan,
        state: FinalizedCompetitionState,
        signed_history: SignedAdmissionHistory,
        launch_plan: Mapping[str, Any],
        capacity: LaunchV1Capacity,
    ) -> bool:
        if launch_policy.policy_for(launch_plan) == "v2":
            launch_policy._check_plan_v2(dict(launch_plan))
        else:
            launch_policy._check_plan(dict(launch_plan))
        _require_entropy_panel(plan, launch_plan)
        history = signed_history.history
        genesis_anchor = _genesis_anchor(self.store, state)
        if (
            history.network != plan.network
            or history.netuid != plan.netuid
            or history.through_block < plan.snapshot_block
            or history.history_root != plan.prompt_seed_inputs.get("admission_history_root")
            or not verify_signed_history(
                history,
                signed_history,
                trusted_committee=plan.committee,
                expected_certified_state_hash=plan.parent_state_hash,
                verifier=self.signature_verifier,
            )
        ):
            raise LaunchBridgeError("signed admission history is not bound to this V1 round")
        if (
            plan.parent_state_hash != state.state_hash()
            or plan.challenge_id != state.challenge_id
            or plan.prompt_seed_inputs.get("launch_plan_id") != launch_plan["plan_id"]
            or launch_plan["models"]["candidate"] != plan.challenger_fingerprint
            or launch_plan["models"]["king"] != plan.king_fingerprint
            or launch_plan["models"]["anchor"] != plan.prompt_seed_inputs.get("anchor_fingerprint")
        ):
            raise LaunchBridgeError("V1 plan does not extend the certified state")
        if launch_policy.policy_for(launch_plan) != "v2":
            if launch_plan["models"]["anchor"] != genesis_anchor:
                raise LaunchBridgeError("V1 plan does not extend the certified state anchor")
        elif launch_plan["models"]["anchor"] != genesis_anchor:
            receipt_hash = _sha(
                plan.prompt_seed_inputs.get("bootstrap_receipt_sha256"), "bootstrap receipt hash"
            )
            receipt = self.receipts.read("bootstrap", receipt_hash)
            if (
                replay_bootstrap_reference(
                    receipt,
                    contract_hash=plan.contract_hash,
                    genesis_reference_hash=genesis_anchor,
                )
                != launch_plan["models"]["anchor"]
            ):
                raise LaunchBridgeError("V2 imported baseline bootstrap reference is invalid")
        receipt = self.receipts.read(
            "capacity", _sha(capacity.receipt_hash, "capacity receipt hash")
        )
        expected = {
            "schema",
            "round_plan_hash",
            "launch_plan_id",
            "measured_parallelism",
            "worst_case_execution_seconds",
            "commit_delivery_seconds",
            "reveal_delivery_seconds",
            "chain_block_seconds",
        }
        if (
            set(receipt) != expected
            or receipt["schema"] != "forge/launch-capacity-receipt/v1"
            or receipt["round_plan_hash"] != plan.plan_hash()
            or receipt["launch_plan_id"] != launch_plan["plan_id"]
        ):
            raise LaunchBridgeError("V1 measured capacity receipt is not bound to the round")
        values = [
            receipt[key]
            for key in (
                "worst_case_execution_seconds",
                "commit_delivery_seconds",
                "reveal_delivery_seconds",
                "chain_block_seconds",
            )
        ]
        if (
            type(receipt["measured_parallelism"]) is not int
            or receipt["measured_parallelism"] < 1
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value <= 0
                for value in values
            )
        ):
            raise LaunchBridgeError("V1 measured capacity receipt is invalid")
        if plan.commit_deadline_block - plan.prompt_seed_inputs["entropy_block"] < math.ceil(
            (receipt["worst_case_execution_seconds"] + receipt["commit_delivery_seconds"])
            / receipt["chain_block_seconds"]
        ) or plan.reveal_deadline_block - plan.commit_deadline_block < math.ceil(
            receipt["reveal_delivery_seconds"] / receipt["chain_block_seconds"]
        ):
            raise LaunchBridgeError("V1 round deadlines do not fit measured capacity")
        persisted = self.store.latest_state_payload(state.challenge_id)
        if persisted is None:
            self.store.initialize_state(
                state_hash=state.state_hash(),
                challenge_id=state.challenge_id,
                state_version=state.state_version,
                previous_state_hash=state.previous_state_hash,
                payload=state.to_dict(),
            )
        elif FinalizedCompetitionState.from_dict(persisted).state_hash() != state.state_hash():
            raise LaunchBridgeError("durable state is not the supplied parent")
        return self.store.create_round(
            plan.round_id, plan.challenge_id, plan.plan_hash(), plan.to_dict()
        )

    def apply(
        self,
        plan: RoundPlan,
        state: FinalizedCompetitionState,
        certificate: RoundCertificate,
        verifier: LaunchCertificateVerifier,
    ) -> FinalizedCompetitionState:
        return apply_certificate(state, plan, certificate, verifier, store=self.store)

    def prepare_testnet_weight_intent(
        self,
        state: FinalizedCompetitionState,
        *,
        plan: RoundPlan,
        certificate: RoundCertificate,
        runtime_hash: str,
        registered_uids: Mapping[int, str],
        public_custody: Mapping[str, str],
        burn_hotkey: str | None,
        burn_receipt_hash: str | None,
    ) -> WeightIntent:
        """Prepare only the existing testnet intent from local custody receipts.

        This is intentionally unavailable as a mainnet publisher.  Custody and
        burn authority are resolved from the same protected local store as the
        execution graph, never from a callback supplied by a report.
        """

        verified_runtime_hash = _sha(runtime_hash, "runtime hash")
        latest_payload = self.store.latest_state_payload(state.challenge_id)
        if latest_payload is None or FinalizedCompetitionState.from_dict(latest_payload) != state:
            raise LaunchBridgeError("weight intent requires the durable finalized head")
        stored_plan = self.store.round_payload(plan.round_id)
        if stored_plan != plan.to_dict() or plan.plan_hash() != certificate.plan_hash:
            raise LaunchBridgeError("weight intent requires the stored round plan")
        stored_certificate = self.store.certificate_payload(certificate.transition_hash())
        if stored_certificate != certificate.to_dict():
            raise LaunchBridgeError("weight intent requires the stored certificate")
        if (
            certificate.network != plan.network
            or certificate.netuid != plan.netuid
            or certificate.round_id != plan.round_id
            or certificate.challenge_id != plan.challenge_id
            or certificate.parent_state_hash != plan.parent_state_hash
            or certificate.previous_king_fingerprint != plan.king_fingerprint
            or certificate.challenger_fingerprint != plan.challenger_fingerprint
            or certificate.challenger_hotkey != plan.challenger_hotkey
            or certificate.activation_block != plan.activation_block
        ):
            raise LaunchBridgeError(
                "weight intent certificate context differs from the stored plan"
            )

        def custody_ok(fingerprint: str, digest: str) -> bool:
            try:
                receipt = self.receipts.read("custody", digest)
                return receipt == {
                    "schema": "forge/launch-public-custody/v1",
                    "fingerprint": fingerprint,
                    "publicly_downloadable": True,
                }
            except (LaunchBridgeError, ValueError):
                return False

        def burn_ok(hotkey: str) -> str | None:
            if burn_receipt_hash is None:
                return None
            try:
                receipt = self.receipts.read("burn", burn_receipt_hash)
                if receipt == {
                    "schema": "forge/launch-burn-qualification/v1",
                    "hotkey": hotkey,
                    "runtime_hash": verified_runtime_hash,
                    "network": plan.network,
                    "netuid": plan.netuid,
                }:
                    return burn_receipt_hash
            except (LaunchBridgeError, ValueError):
                pass
            return None

        if state.last_transition_hash != certificate.transition_hash():
            raise LaunchBridgeError("weight intent requires the certified V1 state")
        return testnet_weight_intent(
            state,
            network=plan.network,
            netuid=plan.netuid,
            certificate_hash=certificate.transition_hash(),
            runtime_hash=verified_runtime_hash,
            registered_uids=registered_uids,
            public_custody=public_custody,
            verify_public_custody=custody_ok,
            burn_hotkey=burn_hotkey,
            verify_burn_qualification=burn_ok,
        )
