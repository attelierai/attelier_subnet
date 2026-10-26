# Native burn reviews: testnet runtimes 483 and 484

Runtime 484 was independently exercised at finalized testnet block 8,287,680,
`0x75da69788584f9fe9b65cbe772e57ae4419bceda339896785c07ab7f587bfbc7`.
The exact 2,677,869-byte deployed WASM has SHA-256
`8ba7928018328ba9e08d6089b0d92c841967801733ff4cd3c4965f105881e681`.
All four controlled cases described below passed with the same expected deltas.
The [484 receipt](testnet-484-burn.json), SHA-256
`85ec1532e4002fa5238239df5dacc006698082b998cedc1a1de99fa876fb48f2`,
binds that executable and origin. Use `scripts/review_testnet_484_burn.mjs` and
that origin block with the reproduction procedure below. This is also an exact
executable review, without attributing an unverified upstream source commit.
No GPU work or signed upstream operation is part of either runtime review.

The published receipts are explicitly redacted copies: account addresses are
replaced consistently with opaque labels. Runtime hashes, block hashes, measured
values and assertions are unchanged. Each copy records the original receipt hash;
its displayed SHA-256 identifies the redacted bytes. These copies document the
observed behavior, not a new execution or a signature over redacted data.

## Runtime 483

The fallback allowlist accepts the exact deployed WASM SHA-256
`1016a0f8c23aa1cfb88a37c723fd0440bfe7b8141490b77ae793dfd7be7540eb`.
This is a behavioral review of the executable, not a verified association with
an upstream source commit. Merely advertising version 483 does not qualify code.

The [receipt](testnet-483-burn.json), SHA-256
`6e1e38f8d19333aff66237bac281def451ac66e1e7d952970eec6379b3d5e621`,
records four completed local-fork blocks. Each case seeds 1,000,000,000 alpha base
units of pending miner emission, disables fresh block emission and zeroes the
pending owner, validator and root budgets. Both mechanisms target the same UID.
The owner-associated hotkey list is cleared in the fixture to isolate the explicit
`SubnetOwnerHotkey` route. These are controlled local inputs, not chain transactions.

| Case | Burned | Recycled | Target stake increase |
| --- | ---: | ---: | ---: |
| Burn, owner UID 0 | 1,000,000,000 | 0 | 0 |
| Recycle, owner UID 0 | 0 | 1,000,000,000 | 0 |
| Burn, owner UID 7 | 1,000,000,000 | 0 | 0 |
| Burn mode, nonowner UID 8 | 0 | 0 | 1,000,000,000 |

All four cases require drained pending emission, the expected incentive vector
and an advanced epoch. Recycling reduces subnet alpha outstanding by the recycled
amount; burning does not. `MinerBurned` is 1.0 for both owner cases and 0 for the
nonowner control, so that indicator alone is not evidence of burning.

The base snapshot is finalized testnet block 8,283,526,
`0x3bfc41da6ab5ce1bf445d9af41f6cebcdae1599eee71686af7efe5ffb8cf238b`.
The executor is [Chopsticks](https://github.com/AcalaNetwork/chopsticks) 1.5.2
with Node 22.22.1. No wallet, signed extrinsic, paid resource or upstream write is
used. Unknown custom host imports are allowed at instantiation but trap if called;
they are not replaced with successful mock results. Signature mocking is disabled
for the tested runtime initialization. Chopsticks supplies the local block's
timestamp inherent. This review does not exercise signed publication, CRV3 reveal,
all runtime paths, network consensus or mainnet.

## Reproduce locally

Use a fresh scratch directory, install the pinned executor there, and copy
`scripts/review_testnet_483_burn.mjs` into that directory so Node can resolve the
executor's dependencies. Keep the public archive RPC read-only and bind the fork
to loopback. No runtime or chain database needs to be saved locally.

```sh
npm install --ignore-scripts --no-audit --no-fund --save-exact @acala-network/chopsticks@1.5.2
LOG_LEVEL=warn timeout --signal=TERM 900s ./node_modules/.bin/chopsticks \
  --endpoint wss://test.finney.opentensor.ai:443 \
  --block 0x3bfc41da6ab5ce1bf445d9af41f6cebcdae1599eee71686af7efe5ffb8cf238b \
  --host 127.0.0.1 --port 18349 --build-block-mode Manual \
  --allow-unresolved-imports=true --mock-signature-host=false \
  --offchain-worker=false --process-queued-messages=false \
  --max-memory-block-count 4 --runtime-log-level 1 --rpc-timeout 20000
```

In a second terminal, from that scratch directory, run
`node review_testnet_483_burn.mjs`. Require exit 0 and all four cases `PASS` in
`runtime-burn-receipt.json`. The first block can take several minutes because the
fork lazily fetches archive state. New local block hashes and starting counter
values need not equal the saved receipt; the assertions and exact WASM hash must.

Production qualification still re-reads the actual finalized `:code`, testnet
genesis, burn mode and owner-to-UID mapping. Any changed executable or recycle
setting fails closed before wallet/worker access. Historical 481 proofs retain
their v1 source-bound format; 483 and 484 proofs use v2 with an explicit behavioral review.
