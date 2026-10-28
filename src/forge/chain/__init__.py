"""Bittensor adapter (SDK pinned to 11.0.1). Optional: install with the `chain` extra.

On-chain content is limited to short, domain-separated commitments:

    forge:v1:sub:<artifact fingerprint>          miner submission
    forge:v1:rep:<round id>:<report commitment>  validator report commitment
    forge:v3:h:<repo>:<sha>:<role>:<steps>:<cfg>  public HF model recipe
    forge:v3:pack:<prompt pack SHA-256>           sealed prompt-pack digest
    forge:v3:eval:<evaluation record SHA-256>     signed evaluation anchor
    forge:v3:lock:<finalist lock SHA-256>          confirmation finalist lock

Report reveals, certificates and evidence live off-chain (see ``board``).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any

PREFIX = "forge:v1"
MAX_COMMITMENT_BYTES = 128  # a single Raw field


@dataclass(frozen=True)
class OnChainCommitment:
    hotkey: str
    block: int
    kind: str
    fields: tuple[str, ...]


def encode_submission(fingerprint: str) -> str:
    return _checked(f"{PREFIX}:sub:{fingerprint}")


def encode_report_commit(round_id: str, commitment: str) -> str:
    if ":" in round_id:
        raise ValueError("round ids cannot contain ':'")
    return _checked(f"{PREFIX}:rep:{round_id}:{commitment}")


def encode_prompt_pack_commit(pack_sha256: str) -> str:
    """Commit a sealed prompt-pack digest without putting prompts on chain."""
    if (
        not isinstance(pack_sha256, str)
        or len(pack_sha256) != 64
        or any(character not in "0123456789abcdef" for character in pack_sha256)
    ):
        raise ValueError("prompt-pack digest must be a lowercase SHA-256")
    return _checked(f"forge:v3:pack:{pack_sha256}")


def encode_finalist_lock_commitment(lock_sha256: str) -> str:
    if (
        not isinstance(lock_sha256, str)
        or len(lock_sha256) != 64
        or any(c not in "0123456789abcdef" for c in lock_sha256)
    ):
        raise ValueError("finalist-lock digest must be a lowercase SHA-256")
    return _checked(f"forge:v3:lock:{lock_sha256}")


def encode_evaluation_record_commitment(record_sha256: str) -> str:
    """Commit one signed off-chain evaluation record without claiming a verdict."""
    if (
        not isinstance(record_sha256, str)
        or len(record_sha256) != 64
        or any(character not in "0123456789abcdef" for character in record_sha256)
    ):
        raise ValueError("evaluation record digest must be a lowercase SHA-256")
    return _checked(f"forge:v3:eval:{record_sha256}")


def decode_commitment(hotkey: str, block: int, data: str) -> OnChainCommitment | None:
    """Parse a forge commitment; anything else on the subnet is ignored."""

    from forge.validator.submission import V3SubmissionClaim, decode_submission_commitment

    artifact = decode_submission_commitment(data)
    if isinstance(artifact, V3SubmissionClaim):
        config = dict(artifact.generation_config)
        guidance = float(config["guidance_scale"])
        guidance_token = (
            str(int(guidance))
            if guidance.is_integer()
            else f"{guidance:.1f}".rstrip("0").rstrip(".")
        )
        return OnChainCommitment(
            hotkey,
            block,
            "model_v4" if "scheduler" in config else "model_v3",
            (
                artifact.reference.backend,
                artifact.reference.repo_id,
                artifact.reference.revision,
                artifact.role,
                str(config["steps"]),
                guidance_token,
            )
            + ((config["scheduler"],) if "scheduler" in config else ()),
        )
    if artifact is not None:
        return OnChainCommitment(
            hotkey, block, "model", (artifact.backend, artifact.repo_id, artifact.revision)
        )
    parts = data.split(":")
    if len(parts) == 4 and ":".join(parts[:3]) == "forge:v3:pack":
        digest = parts[3]
        if len(digest) == 64 and all(character in "0123456789abcdef" for character in digest):
            return OnChainCommitment(hotkey, block, "pack", (digest,))
    if len(parts) == 4 and ":".join(parts[:3]) == "forge:v3:lock":
        digest = parts[3]
        if len(digest) == 64 and all(character in "0123456789abcdef" for character in digest):
            return OnChainCommitment(hotkey, block, "finalist_lock", (digest,))
    if len(parts) == 4 and ":".join(parts[:3]) == "forge:v3:eval":
        digest = parts[3]
        if len(digest) == 64 and all(character in "0123456789abcdef" for character in digest):
            return OnChainCommitment(hotkey, block, "evaluation", (digest,))
    if len(parts) < 4 or ":".join(parts[:2]) != PREFIX:
        return None
    kind, fields = parts[2], tuple(parts[3:])
    if kind == "sub" and len(fields) == 1 or kind == "rep" and len(fields) == 2:
        return OnChainCommitment(hotkey, block, kind, fields)
    return None


def _checked(text: str) -> str:
    if len(text.encode()) > MAX_COMMITMENT_BYTES:
        raise ValueError("commitment exceeds one Raw field")
    return text


class Sr25519Signer:
    """Signs with a validator hotkey. Satisfies ``ports.Signer``."""

    def __init__(self, wallet: Any):
        self._keypair = wallet.hotkey

    @property
    def validator_hotkey(self) -> str:
        return self._keypair.ss58_address

    def sign(self, message: bytes) -> bytes:
        return bytes(self._keypair.sign(message))


class Sr25519Verifier:
    """Verifies hotkey signatures from public addresses only."""

    def verify(self, validator_hotkey: str, message: bytes, signature: bytes) -> bool:
        # Receipt replay is intentionally usable on the isolated renderer image.
        # Keep the optional chain SDK at this execution boundary rather than
        # making every offline import require it.
        try:
            from bittensor.keyfiles import Keypair
        except ModuleNotFoundError as error:
            raise RuntimeError("Bittensor SDK is required for signature verification") from error
        try:
            return bool(Keypair(ss58_address=validator_hotkey).verify(message, signature))
        except ValueError:
            return False


@dataclass(frozen=True)
class WeightSubmission:
    netuid: int
    uids: list[int]
    weights: list[float]
    dry_run: bool
    result: str | None = None
    mechid: int = 0


class BittensorChain:
    """Thin wrapper over the SDK. Every read that matters is pinned to a block."""

    def __init__(self, network: str, netuid: int, *, endpoint: str | None = None):
        try:
            import bittensor as bt
        except ModuleNotFoundError as error:
            raise RuntimeError("Bittensor SDK is required for chain access") from error
        self.network = network
        self.netuid = netuid
        # The SDK accepts a WebSocket URL in its network argument. Keep our
        # protocol identity independent of that connection choice.
        rpc = endpoint if endpoint is not None else network
        if rpc.startswith(("ws://", "wss://")):
            # Pin explicit RPCs: the SDK otherwise adds public fallback and
            # archive pools when a URL maps to one of its known networks.
            self.subtensor = bt.Subtensor(network=rpc, fallback_endpoints=[], archive_endpoints=[])
        else:
            self.subtensor = bt.Subtensor(network=rpc)

    def close(self) -> None:
        self.subtensor.close()

    # --- blocks -----------------------------------------------------------

    def head(self) -> int:
        return int(self.subtensor.block)

    def finalized_height(self) -> int:
        # The SDK's _known_finalized_height cache may lag a just-finalized
        # extrinsic on this same connection. Query the finalized head itself,
        # then its immutable header, so a readback cannot use a stale snapshot.
        self.head()
        raw = self.subtensor._client._substrate.raw
        finalized_hash = self.subtensor._call(raw.get_chain_finalised_head())
        return int(self.subtensor._call(raw.get_block_number(finalized_hash)))

    def finalized_block_hash(self, height: int) -> str:
        if height > self.finalized_height():
            raise ValueError(f"block {height} is not finalized yet")
        return str(self.subtensor.read("block_info", block=height)["hash"])

    def chain_time(self, height: int) -> int:
        info = self.subtensor.read("block_info", block=height)
        return int(datetime.fromisoformat(info["timestamp"]).timestamp())

    # --- identities -------------------------------------------------------

    def _metagraph(self, height: int) -> dict[str, Any]:
        return self.subtensor.at(height).read("metagraph", netuid=self.netuid)

    def uid_snapshot(self, height: int) -> dict[int, str]:
        return dict(enumerate(self._metagraph(height)["hotkeys"]))

    def owner_snapshot(self, height: int) -> dict[int, tuple[str, str]]:
        """UID -> (hotkey, owning coldkey) at ``height``."""

        metagraph = self._metagraph(height)
        pairs = zip(metagraph["hotkeys"], metagraph["coldkeys"], strict=True)
        return {uid: (hotkey, coldkey) for uid, (hotkey, coldkey) in enumerate(pairs)}

    def committee_snapshot(self, height: int) -> dict[str, int]:
        """Validators with a permit at ``height`` and their total stake in rao (> 0)."""

        metagraph = self._metagraph(height)
        rows = zip(
            metagraph["hotkeys"],
            metagraph["validator_permit"],
            metagraph["total_stake"],
            strict=True,
        )
        return {hotkey: int(stake) for hotkey, permit, stake in rows if permit and stake > 0}

    # --- commitments ------------------------------------------------------

    def publish(self, wallet: Any, text: str) -> str:
        try:
            from bittensor import calls
        except ModuleNotFoundError as error:
            raise RuntimeError("Bittensor SDK is required for chain access") from error
        info = {"fields": [[{f"Raw{len(text.encode())}": text.encode()}]]}
        call = calls.Commitments.set_commitment(self.netuid, info)
        result = self.subtensor.submit_call(call, wallet, signer="hotkey")
        if not result.success:
            raise RuntimeError(f"commitment failed: {result.message}")
        return str(result.extrinsic_id)

    def commitment(self, hotkey: str, height: int) -> OnChainCommitment | None:
        """The commitment ``hotkey`` had in force at ``height`` (latest per hotkey)."""

        record = self.subtensor.at(height).read(
            "commitment", netuid=self.netuid, hotkey_ss58=hotkey
        )
        if record is None:
            return None
        return decode_commitment(hotkey, int(record.block), str(record.data))

    def raw_commitment(self, hotkey: str, height: int) -> tuple[int, str] | None:
        """Read the latest raw commitment for this netuid/hotkey at finality.

        Gateway M1 uses its own commitment domain, so it cannot safely use the
        legacy Forge decoder above.  This remains scoped to the instance netuid
        and to the caller's pinned finalized block.
        """
        if height > self.finalized_height():
            raise ValueError(f"block {height} is not finalized yet")
        record = self.subtensor.at(height).read(
            "commitment", netuid=self.netuid, hotkey_ss58=hotkey
        )
        if record is None:
            return None
        return int(record.block), str(record.data)

    def commitments(self, hotkeys: list[str], height: int) -> list[OnChainCommitment]:
        found = (self.commitment(hotkey, height) for hotkey in hotkeys)
        return [item for item in found if item is not None]

    def finalized_publication(
        self, hotkey: str, text: str, extrinsic_id: str, *, timeout_seconds: float = 45
    ) -> tuple[int, OnChainCommitment]:
        """Wait for RPC finality to catch up; never resubmit a transaction.

        A successful inclusion receipt and the queried finalized head can briefly
        disagree. Preserve the pending publication when the bounded read fails.
        """
        import time

        minimum = int(extrinsic_id.split("-", 1)[0])
        deadline = time.monotonic() + timeout_seconds
        while True:
            height = self.finalized_height()
            if height >= minimum:
                observed = self.commitment(hotkey, height)
                expected = decode_commitment(hotkey, observed.block if observed else 0, text)
                if observed is not None and observed == expected and observed.block >= minimum:
                    return height, observed
                raise ValueError("finalized publication differs; retain receipt, do not resubmit")
            if time.monotonic() >= deadline:
                raise TimeoutError("publication finality readback pending; do not resubmit")
            time.sleep(2)

    # --- weights ----------------------------------------------------------

    def set_weights(
        self, wallet: Any, uid_weights: dict[int, float], *, dry_run: bool = True, mechid: int = 0
    ) -> WeightSubmission:
        """Mechanism 0 is optimization; mechanism 1 is the separate inference service."""

        uids = sorted(uid_weights)
        weights = [float(uid_weights[uid]) for uid in uids]
        if not uids:
            raise ValueError("refusing to submit an empty weight vector")
        if dry_run:
            return WeightSubmission(self.netuid, uids, weights, True, mechid=mechid)
        params = {"netuid": self.netuid, "uids": uids, "weights": weights}
        if mechid:
            params["mechid"] = mechid
        result = self.subtensor.execute_tool("set_weights", params, wallet)
        if not result.success:
            raise RuntimeError(f"set_weights failed: {result.message}")
        return WeightSubmission(self.netuid, uids, weights, False, str(result.extrinsic_id), mechid)
