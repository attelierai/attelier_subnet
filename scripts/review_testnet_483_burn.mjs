import { ApiPromise, WsProvider } from '@polkadot/api';
import { createHash } from 'node:crypto';
import { writeFileSync } from 'node:fs';
import assert from 'node:assert/strict';

const timer = setTimeout(() => process.exit(2), 600_000);
const provider = new WsProvider('ws://127.0.0.1:18349', false, {}, 500_000);
await provider.connect();
const api = await ApiPromise.create({ provider, noInitWarn: true });
const out = {
  schema: 'forge/runtime-burn-controlled-fork/v1',
  local_only: true, upstream_writes: false, signed_extrinsics: 0,
  signature_mock: false, unresolved_imports: 'trap if called',
  origin_block: '0x3bfc41da6ab5ce1bf445d9af41f6cebcdae1599eee71686af7efe5ffb8cf238b',
  cases: [],
};
const save = () => writeFileSync('runtime-burn-receipt.json', JSON.stringify(out, null, 2) + '\n');
try {
  const code = await api.rpc.state.getStorage('0x3a636f6465');
  out.runtime_sha256 = createHash('sha256').update(code.unwrap().toU8a(true)).digest('hex');
  assert.equal(out.runtime_sha256, '1016a0f8c23aa1cfb88a37c723fd0440bfe7b8141490b77ae793dfd7be7540eb');
  out.runtime = api.runtimeVersion.specVersion.toNumber();
  const q = api.query.subtensorModule;
  const count = (await q.subnetworkN(578)).toNumber();
  const mechanisms = (await q.mechanismCountCurrent(578)).toNumber();
  const keys = await Promise.all(Array.from({ length: count }, (_, i) => q.keys(578, i).then(x => x.toString())));
  const owner = (await q.subnetOwner(578)).toString();
  out.neurons = count; out.mechanisms = mechanisms;
  async function snapshot() {
    const r = {};
    for (const name of ['minerBurned', 'subnetAlphaOut', 'pendingServerEmission', 'pendingValidatorEmission', 'pendingRootAlphaDivs', 'pendingOwnerCut', 'lastEpochBlock', 'firstEmissionBlockNumber', 'subnetOwnerHotkey', 'emission', 'incentive']) r[name] = (await q[name](578)).toJSON();
    for (const name of ['totalAlphaIssuance', 'alphaBurned', 'alphaRecycled']) r[name] = (await api.query.alphaAssets[name](578)).toJSON();
    r.stake = Object.fromEntries(await Promise.all(keys.map(async (key, uid) => [uid, (await q.totalHotkeyAlpha(key, 578)).toJSON()])));
    return r;
  }
  for (const scenario of [
    { name: 'burn-owner-uid0', mode: 'Burn', ownerUid: 0, targetUid: 0 },
    { name: 'recycle-owner-uid0', mode: 'Recycle', ownerUid: 0, targetUid: 0 },
    { name: 'burn-owner-uid7', mode: 'Burn', ownerUid: 7, targetUid: 7 },
    { name: 'pay-nonowner-uid8', mode: 'Burn', ownerUid: 7, targetUid: 8 },
  ]) {
    const height = (await api.rpc.chain.getHeader()).number.toNumber();
    const weights = [], lastUpdate = [];
    for (let m = 0; m < mechanisms; m++) {
      const index = 578 + m * 4096;
      lastUpdate.push([[index], Array(count).fill(height)]);
      for (let uid = 0; uid < count; uid++) weights.push([[index, uid], [[scenario.targetUid, 65535]]]);
    }
    const seed = { SubtensorModule: {
      $removePrefix: ['NetworksAdded', 'Weights', 'Bonds'],
      NetworksAdded: [[[578], true]], RecycleOrBurn: [[[578], scenario.mode]],
      // Removing this optional entry disables fresh per-block emission, while
      // the existing subnet can still drain the known pending epoch budget.
      FirstEmissionBlockNumber: [[[578], null]],
      PendingServerEmission: [[[578], 1_000_000_000]],
      PendingValidatorEmission: [[[578], 0]], PendingRootAlphaDivs: [[[578], 0]],
      PendingOwnerCut: [[[578], 0]], PendingEpochAt: [[[578], height + 1]],
      SubnetOwnerHotkey: [[[578], keys[scenario.ownerUid]]],
      // Isolate the explicit owner target from other owner-associated hotkeys.
      OwnedHotkeys: [[[owner], []]],
      Weights: weights, LastUpdate: lastUpdate,
    } };
    await provider.send('dev_setStorage', [seed]);
    const before = await snapshot();
    const c = { ...scenario, seed, before };
    out.cases.push(c); save();
    c.block = await provider.send('dev_newBlock', [{}]);
    c.after = await snapshot(); save();
    const a = c.after;
    assert.equal(before.firstEmissionBlockNumber, null);
    assert.equal(before.pendingServerEmission, 1_000_000_000);
    assert.equal(before.pendingValidatorEmission + before.pendingRootAlphaDivs + before.pendingOwnerCut, 0);
    assert.equal(a.lastEpochBlock, height + 1);
    assert.equal(a.pendingServerEmission, 0);
    assert.equal(a.incentive[scenario.targetUid], 65535);
    const burn = a.alphaBurned - before.alphaBurned;
    const recycle = a.alphaRecycled - before.alphaRecycled;
    c.deltas = { burned: burn, recycled: recycle, alpha_out: a.subnetAlphaOut - before.subnetAlphaOut,
      target_stake: a.stake[scenario.targetUid] - before.stake[scenario.targetUid] };
    if (scenario.targetUid === scenario.ownerUid) {
      assert.equal(a.minerBurned.bits, 4294967296);
      assert.equal(c.deltas.target_stake, 0);
      assert.equal(burn, scenario.mode === 'Burn' ? 1_000_000_000 : 0);
      assert.equal(recycle, scenario.mode === 'Recycle' ? 1_000_000_000 : 0);
      assert.equal(c.deltas.alpha_out, scenario.mode === 'Recycle' ? -1_000_000_000 : 0);
    } else {
      assert.equal(a.minerBurned.bits, 0);
      assert.equal(burn, 0); assert.equal(recycle, 0);
      assert.equal(c.deltas.alpha_out, 0);
      assert.equal(c.deltas.target_stake, 1_000_000_000);
    }
    c.status = 'PASS'; save();
    console.log(JSON.stringify({ name: c.name, status: c.status, deltas: c.deltas }));
  }
  out.status = 'PASS';
} catch (error) {
  out.status = 'FAIL'; out.error = String(error); process.exitCode = 1;
  console.error(error);
} finally {
  save(); await api.disconnect(); clearTimeout(timer);
}
