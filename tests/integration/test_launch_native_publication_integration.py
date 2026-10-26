"""CRV3 native reader evidence with raw-call, archive queue and events."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest
from bittensor import storage
from bittensor.keyfiles import Keypair
from bittensor.sp_core import CRYPTO_SR25519

from forge.chain.launch_native_publication import TestnetNativePublicationReader
from forge.chain.launch_publication import BittensorSdkTestnetSigner, QualifiedPublicationContext
from forge.chain.reward_publication import WeightIntent


def _sha(number: int) -> str:
    return f"{number:064x}"


def _blake(value: bytes) -> str:
    return hashlib.blake2b(value, digest_size=32).hexdigest()


class _Raw:
    def get_block_hash(self, height):
        return f"block-{height}"

    def rpc_request(self, name, params):
        return name, params


class _At:
    def __init__(self, chain, block):
        self.chain, self.block = chain, block

    def query_map(self, _item, _arguments):
        return self.chain.queues.get(self.block, [])

    def query(self, item, _arguments=None):
        if item == storage.System.Events:
            return self.chain.events.get(self.block, [])
        raise AssertionError(item)

    def read(self, name, **_kwargs):
        assert name == "metagraph"
        return {"hotkeys": ["other", "other", "other", self.chain.hotkey]}


class _Submitter:
    def __init__(self, chain):
        self.chain = chain
        self.raw = _Raw()
        self.submissions = []

    def submit(self, call, signer, **kwargs):
        self.submissions.append((call, signer, kwargs))
        if self.chain.before_submit:
            self.chain.before_submit(call, signer)
        return ("submit", call, signer)


class _Subtensor:
    def __init__(self, chain):
        self.chain = chain
        self._client = SimpleNamespace(_substrate=_Submitter(chain))

    def plan(self, _request, _wallet):
        return SimpleNamespace(
            ok=True,
            signer="hotkey",
            signer_address=self.chain.hotkey,
            call=self.chain.planned_call,
        )

    def _call(self, request):
        if isinstance(request, tuple) and request[0] == "submit":
            self.chain.finalized = 102
            if self.chain.raise_after_submit:
                raise TimeoutError("lost SDK response after broadcast")
            return SimpleNamespace(success=True, message="", extrinsic_id="100-0")
        if isinstance(request, str) and request == "block-0":
            return "0x" + _sha(1)
        name, params = request
        if name == "state_getStorage":
            return "0x" + self.chain.runtime_code.hex()
        if name == "chain_getBlock":
            block = int(params[0].rsplit("-", 1)[1])
            return {"block": {"extrinsics": self.chain.raw_extrinsics[block]}}
        raise AssertionError(request)

    def block_info(self, block):
        return self.chain.infos[block]

    def at(self, block):
        return _At(self.chain, block)

    def query(self, _item, arguments, *, block):
        assert arguments == [9, 3]
        assert block == 102
        return [(3, 65535)]


class _Chain:
    network = "test"
    netuid = 9

    def __init__(self, ciphertext, *, duplicate=False, hotkey="validator"):
        self.runtime_code = b"runtime-481"
        self.finalized = 99
        self.hotkey = hotkey
        self.planned_call = _prepared_call(ciphertext)
        self.before_submit = None
        self.raise_after_submit = False
        self.raw_extrinsics = {100: ["0x010203"], 101: [], 102: []}
        commit_hash = _blake(ciphertext)
        call = {
            "call_module": "SubtensorModule",
            "call_function": "commit_timelocked_mechanism_weights",
            "call_args": [
                {"name": "netuid", "value": 9},
                {"name": "mecid", "value": 0},
                {"name": "commit", "value": "0x" + ciphertext.hex()},
                {"name": "reveal_round", "value": 700},
                {"name": "commit_reveal_version", "value": 4},
            ],
        }
        self.infos = {
            100: SimpleNamespace(extrinsics=[{"address": hotkey, "call": call}]),
            102: {},
        }
        self.events = {
            100: [
                {
                    "module": "SubtensorModule",
                    "name": "TimelockedWeightsCommitted",
                    "attributes": [hotkey, 9, commit_hash, 700],
                }
            ],
            102: [
                {
                    "module": "SubtensorModule",
                    "name": "TimelockedWeightsRevealed",
                    "attributes": [9, hotkey],
                },
                {
                    "module": "SubtensorModule",
                    "name": "WeightsSet",
                    "attributes": [9, 3],
                },
            ],
        }
        entries = [(hotkey, 100, ciphertext, 700)]
        if duplicate:
            entries.append((hotkey, 101, b"other", 701))
        self.queues = {101: [(10, entries)]}
        self.subtensor = _Subtensor(self)

    def finalized_height(self):
        return self.finalized

    def finalized_block_hash(self, block):
        return f"block-{block}"

    def uid_snapshot(self, block):
        assert block == 102
        return {3: "burn"}


def _context(chain):
    return QualifiedPublicationContext(
        "bittensor-sdk/testnet-qualified-v1",
        "test",
        9,
        _sha(1),
        hashlib.sha256(chain.runtime_code).hexdigest(),
        0,
        _sha(2),
        _sha(3),
        _sha(4),
        None,
        _sha(5),
    )


def _intent(context):
    return WeightIntent(
        "test",
        9,
        _sha(4),
        _sha(6),
        context.runtime_hash,
        _sha(7),
        _sha(8),
        "BURN_REQUIRED",
        {3: 1.0},
        {3: "burn"},
        "fixture",
    )


def _prepared_call(ciphertext):
    return {
        "call_args": {
            "netuid": 9,
            "mecid": 0,
            "commit": ciphertext,
            "reveal_round": 700,
            "commit_reveal_version": 4,
        }
    }


def test_native_reader_proves_one_prepared_ciphertext_revealed_to_exact_vector(tmp_path):
    ciphertext = b"encrypted-crv3-commit"
    chain, context = _Chain(ciphertext), None
    context = _context(chain)
    intent = _intent(context)
    reader = TestnetNativePublicationReader(
        chain,
        validator_hotkey="validator",
        validator_uid=3,
        witness_path=tmp_path / "witness.sqlite",
    )
    reader.capture_prepared(context, intent, _prepared_call(ciphertext))
    chain.finalized = 102
    receipt = reader.receipt_from_submission(SimpleNamespace(result="100-0"), context, intent)
    # A restart obtains the durable witness and archive evidence, not a late vector guess.
    restarted = TestnetNativePublicationReader(
        chain,
        validator_hotkey="validator",
        validator_uid=3,
        witness_path=tmp_path / "witness.sqlite",
    )
    assert restarted.reconcile(context, intent, None) == receipt
    # Simulate a crash after reconcile returned but before the publisher journal
    # recorded it: the next process replays the durable receipt, not a rescan.
    after_crash = TestnetNativePublicationReader(
        chain,
        validator_hotkey="validator",
        validator_uid=3,
        witness_path=tmp_path / "witness.sqlite",
    )
    assert after_crash.reconcile(context, intent, None) == receipt
    assert restarted.reconcile(context, intent, receipt) == receipt
    finalized = restarted.finalized_weights(context, intent, receipt)
    assert finalized and finalized.finalized_block == 102
    assert finalized.weights_u16 == {3: 65535}
    assert finalized.last_update_block == 100


def test_native_reader_keeps_pending_when_same_hotkey_has_another_queued_commit(tmp_path):
    ciphertext = b"encrypted-crv3-commit"
    chain = _Chain(ciphertext, duplicate=True)
    context, intent = _context(chain), _intent(_context(chain))
    reader = TestnetNativePublicationReader(
        chain,
        validator_hotkey="validator",
        validator_uid=3,
        witness_path=tmp_path / "witness.sqlite",
    )
    reader.capture_prepared(context, intent, _prepared_call(ciphertext))
    chain.finalized = 102
    receipt = reader.receipt_from_submission(SimpleNamespace(result="100-0"), context, intent)
    assert reader.finalized_weights(context, intent, receipt) is None


def test_sdk_signer_persists_exact_planned_call_before_lost_response(tmp_path):
    ciphertext = b"encrypted-crv3-commit"
    hotkey = Keypair.create_from_mnemonic(Keypair.generate_mnemonic(), CRYPTO_SR25519)
    chain = _Chain(ciphertext, hotkey=hotkey.ss58_address)
    context, intent = _context(chain), _intent(_context(chain))
    reader = TestnetNativePublicationReader(
        chain,
        validator_hotkey=hotkey.ss58_address,
        validator_uid=3,
        witness_path=tmp_path / "witness.sqlite",
    )

    def witness_exists_before_submit(call, signer):
        witness = reader.witnesses.get(intent.intent_hash())
        assert call is chain.planned_call
        assert signer is hotkey
        assert witness.ciphertext_hash == _blake(ciphertext)

    chain.before_submit = witness_exists_before_submit
    chain.raise_after_submit = True
    signer = BittensorSdkTestnetSigner(chain, hotkey, reader)
    with pytest.raises(TimeoutError, match="lost SDK response"):
        signer.submit(context, intent)
    submitted = chain.subtensor._client._substrate.submissions
    assert len(submitted) == 1 and submitted[0][0] is chain.planned_call

    restarted = TestnetNativePublicationReader(
        chain,
        validator_hotkey=hotkey.ss58_address,
        validator_uid=3,
        witness_path=tmp_path / "witness.sqlite",
    )
    recovered = restarted.reconcile(context, intent, None)
    assert recovered and recovered.submitted_block == 100
