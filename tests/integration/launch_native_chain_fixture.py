"""External-chain boundary fixture shared by native launch publication tests.

It deliberately models the SDK/raw RPC and archive interfaces, while all
validator/certificate code remains real. ``runtime_code`` is supplied by the
test (normally the official v481 WASM held in memory).
"""

from __future__ import annotations

import hashlib
from types import SimpleNamespace
from typing import Any

from bittensor import storage


def _blake(value: bytes) -> str:
    return hashlib.blake2b(value, digest_size=32).hexdigest()


class _Raw:
    def get_block_hash(self, height: int) -> str:
        return f"block-{height}"

    def rpc_request(self, name: str, params: list[Any]):
        return name, params


class _At:
    def __init__(self, chain: CompositeChain, block: int):
        self.chain, self.block = chain, block

    def query_map(self, _item: Any, _arguments: Any):
        return self.chain.queues.get(self.block, [])

    def query(self, item: Any, arguments: Any = None):
        if item == storage.System.Events:
            return self.chain.events.get(self.block, [])
        if item == storage.SubtensorModule.RecycleOrBurn:
            assert arguments == [self.chain.netuid]
            return {"Burn": None}
        if item == storage.SubtensorModule.SubnetOwnerHotkey:
            assert arguments == [self.chain.netuid]
            return self.chain.fallback_hotkey
        raise AssertionError(f"unexpected storage query {item}")

    def read(self, name: str, **_kwargs: Any):
        assert name == "metagraph"
        return {"hotkeys": [self.chain.hotkeys_by_uid.get(uid, "") for uid in range(8)]}


class _Submitter:
    def __init__(self, chain: CompositeChain):
        self.chain, self.raw, self.submissions = chain, _Raw(), []

    def submit(self, call: Any, signer: Any, **kwargs: Any):
        self.submissions.append((call, signer, kwargs))
        self.chain._submit(call, signer)
        return ("submit", call, signer)


class _Subtensor:
    def __init__(self, chain: CompositeChain):
        self.chain = chain
        self._client = SimpleNamespace(_substrate=_Submitter(chain))

    def plan(self, _request: Any, wallet: Any):
        hotkey = wallet.hotkey.ss58_address
        ciphertext = hashlib.sha256(f"cipher:{hotkey}:{self.chain.finalized}".encode()).digest()
        return SimpleNamespace(
            ok=True,
            signer="hotkey",
            signer_address=hotkey,
            call={
                "call_args": {
                    "netuid": self.chain.netuid,
                    "mecid": 0,
                    "commit": ciphertext,
                    "reveal_round": self.chain.finalized + 10,
                    "commit_reveal_version": 4,
                }
            },
        )

    def _call(self, request: Any):
        if isinstance(request, tuple) and request[0] == "submit":
            return SimpleNamespace(
                success=True, message="", extrinsic_id=f"{self.chain.finalized}-0"
            )
        if request == "block-0":
            return "0x" + self.chain.genesis_hash
        name, params = request
        if name == "state_getStorage":
            return "0x" + self.chain.runtime_code.hex()
        if name == "chain_getBlock":
            block = int(params[0].rsplit("-", 1)[1])
            return {"block": {"extrinsics": self.chain.raw_extrinsics.get(block, [])}}
        if name == "state_getRuntimeVersion":
            return {"specName": "node-subtensor", "specVersion": 481}
        raise AssertionError(f"unexpected raw request {request}")

    def block_info(self, block: int):
        return self.chain.infos.get(block, SimpleNamespace(extrinsics=[]))

    def at(self, block: int):
        return _At(self.chain, block)

    def query(self, _item: Any, arguments: list[int], *, block: int):
        uid = arguments[1]
        assert arguments[0] == self.chain.netuid
        assert uid in {0, 3}
        return [(7, 65535)]


class CompositeChain:
    network = "test"

    def __init__(
        self, wallets: dict[str, Any], runtime_code: bytes, genesis_hash: str, netuid: int = 578
    ):
        if len(wallets) != 2 or not runtime_code or len(genesis_hash) != 64:
            raise ValueError("fixture requires two wallets, WASM bytes and a SHA-256 genesis")
        self.netuid, self.runtime_code, self.genesis_hash = (
            netuid,
            bytes(runtime_code),
            genesis_hash,
        )
        self.finalized = 100
        self.wallets = dict(wallets)
        addresses = [wallet.hotkey.ss58_address for wallet in wallets.values()]
        self.hotkeys_by_uid = {0: addresses[0], 3: addresses[1], 7: "burn"}
        self.fallback_hotkey = "burn"
        self.queues: dict[int, list[Any]] = {}
        self.events: dict[int, list[Any]] = {}
        self.raw_extrinsics: dict[int, list[Any]] = {}
        self.infos: dict[int, Any] = {}
        self.pending: dict[str, tuple[int, bytes, int]] = {}
        self.submission_indexes: dict[str, int] = {}
        self.subtensor = _Subtensor(self)

    def finalized_height(self) -> int:
        return self.finalized

    def finalized_block_hash(self, block: int) -> str:
        return f"block-{block}"

    def uid_snapshot(self, _block: int) -> dict[int, str]:
        return dict(self.hotkeys_by_uid)

    def committee_snapshot(self, _block: int) -> dict[str, int]:
        return {hotkey: 1 for uid, hotkey in self.hotkeys_by_uid.items() if uid in {0, 3}}

    def _submit(self, call: Any, signer: Any) -> None:
        args = call["call_args"]
        ciphertext = bytes(args["commit"])
        hotkey = getattr(signer, "ss58_address", None) or signer.hotkey.ss58_address
        # capture_prepared observes ``finalized`` first; inclusion must be a
        # later finalized block so the native witness has an unambiguous cursor.
        block, reveal = self.finalized + 1, int(args["reveal_round"])
        self.finalized = block
        self.pending[hotkey] = (block, ciphertext, reveal)
        self.raw_extrinsics.setdefault(block, []).append("0x010203")
        self.infos[block] = SimpleNamespace(
            extrinsics=[
                {
                    "address": hotkey,
                    "call": {
                        "call_module": "SubtensorModule",
                        "call_function": "commit_timelocked_mechanism_weights",
                        "call_args": [
                            {"name": "netuid", "value": self.netuid},
                            {"name": "mecid", "value": 0},
                            {"name": "commit", "value": "0x" + ciphertext.hex()},
                            {"name": "reveal_round", "value": reveal},
                            {"name": "commit_reveal_version", "value": 4},
                        ],
                    },
                }
            ]
        )
        self.events[block] = [
            {
                "module": "SubtensorModule",
                "name": "TimelockedWeightsCommitted",
                "attributes": [hotkey, self.netuid, _blake(ciphertext), reveal],
            }
        ]
        self.queues[block + 1] = [(0, [(hotkey, block, ciphertext, reveal)])]

    def reveal(self, label: str) -> None:
        hotkey = self.wallets[label].hotkey.ss58_address
        block, _ciphertext, _reveal = self.pending[hotkey]
        self.finalized = max(self.finalized + 1, block + 2)
        self.queues[self.finalized - 1] = [
            (0, [(key, *value) for key, value in self.pending.items()])
        ]
        uid = next(uid for uid, value in self.hotkeys_by_uid.items() if value == hotkey)
        self.events[self.finalized] = [
            {
                "module": "SubtensorModule",
                "name": "TimelockedWeightsRevealed",
                "attributes": [self.netuid, hotkey],
            },
            {"module": "SubtensorModule", "name": "WeightsSet", "attributes": [self.netuid, uid]},
        ]
        del self.pending[hotkey]
