// GC-counted allocation measurement (PERF-ALLOC-AUDIT, backlog item 33). Method as climate/m1 alloc-diag.mjs:
// bytes freed by every GC event inside the measured loop (v8.GCProfiler) plus the non-negative net heap growth,
// divided by the call count. That is a conservative upper bound (profiler bookkeeping included), which is why every
// harness also measures a no-op control in the same loop shape that must read ~0 B/call.
// Usage from a harness: await runCases(name, {real: loopFn, control: loopFn}, callsPerLoop, loopsPerMeasure, extra)
import v8 from 'node:v8';
import assert from 'node:assert/strict';
import {pathToFileURL, fileURLToPath} from 'node:url';

export const REPO = fileURLToPath(new URL('../../../', import.meta.url));
export const src = f => pathToFileURL(`${REPO}/src/${f}`).href;
const gc = globalThis.gc;
if (!gc) throw new Error('run with node --expose-gc');

export function count(loop, loops, callsPerLoop) {
  gc(); gc();
  const heap0 = process.memoryUsage().heapUsed, prof = new v8.GCProfiler(); prof.start();
  for (let r = 0; r < loops; r++) loop();
  const profile = prof.stop(), heap1 = process.memoryUsage().heapUsed;
  assert.ok(Array.isArray(profile?.statistics), 'GCProfiler statistics are unavailable');
  const stats = profile.statistics;
  let freed = 0, scavenges = 0;
  for (const s of stats) {
    const before = s.beforeGC?.heapStatistics?.usedHeapSize, after = s.afterGC?.heapStatistics?.usedHeapSize;
    assert.ok(Number.isFinite(before) && Number.isFinite(after), 'GCProfiler heap sizes are unavailable');
    freed += Math.max(0, before - after);
    if (/scavenge/i.test(s.gcType ?? '')) scavenges++;
  }
  const calls = loops * callsPerLoop, growth = heap1 - heap0;
  return {calls, gcEvents: stats.length, scavenges, heapGrowthBytes: growth, freedByGcBytes: freed,
    bytesPerCall: +((Math.max(0, growth) + freed) / calls).toFixed(3),
    // What the old heap-growth checks would have seen for the same loop.
    heapGrowthPerCall: +(growth / calls).toFixed(3)};
}

/** Warm both loops, then measure control, real, control (alternating). */
export function runCases(name, cases, callsPerLoop, loops, extra = {}) {
  const warm = Number(process.env.ALLOC_WARM) || Math.max(3, Math.ceil(loops / 3));
  for (let w = 0; w < warm; w++) for (const fn of Object.values(cases)) fn();
  // Throwaway first measurement: absorbs one-time GCProfiler / first-use heap growth (round-1 first controls read 0.2-18 B).
  count(cases.control, 1, callsPerLoop);
  const out = {name, node: process.version, callsPerLoop, loops};
  const order = ['control', ...Object.keys(cases).filter(k => k !== 'control'), 'control'];
  const seen = {};
  for (const k of order) {
    const r = count(cases[k], loops, callsPerLoop);
    const key = seen[k] ? `${k}2` : k; seen[k] = true;
    out[key] = r;
  }
  Object.assign(out, extra);
  console.log('RESULT ' + JSON.stringify(out));
  assert.ok(out.control.bytesPerCall < 1 && out.control2.bytesPerCall < 1,
    `${name}: invalid allocation measurement, no-op control must be < 1 B/call`);
  assert.ok(out.real.bytesPerCall < 4, `${name}: ${out.real.bytesPerCall} B/call exceeds the 4 B regression ceiling`);
  assert.ok(extra.sink !== false, `${name}: non-finite sink`);
  return out;
}

// The extracted Kit keeps the current-source allocation gate. Atlas's historical
// baseline comparison remains in Atlas; its Git revision is not a Kit revision.
export const TREE = process.env.ALLOC_TREE || 'wt';
assert.equal(TREE, 'wt', 'historical Atlas ALLOC_TREE=base is only supported in Atlas');
export const mod = src;

// Driver design (phase 2): each measured case is a per-sample BODY function body(i) (its own function literal, so its
// inner call sites stay monomorphic), called from one shared driver loop whose call site is made megamorphic first, so
// no body is inlined into the driver. Every body is then optimized as a normal (non-OSR) function by invocation count,
// as in a consumer's per-sample function; the driver passes only a Smi, so its call boundary allocates nothing.
// Why: in a single long loop function, the measured code runs as OSR-compiled code, and V8's OSR code boxes some loop
// values that a normal compile keeps unboxed (diagnostic x-look6.mjs: the same body read 3.1-3.6 B/call in an OSR loop
// and 0.004 B/call optimized normally).
export function drive(body, n) { for (let i = 0; i < n; i++) body(i); }
let megamorphic = false;
export function makeDriverMegamorphic() {
  if (megamorphic) return; megamorphic = true;
  const sink = new Float64Array(1);
  const ds = [i => { sink[0] += i; }, i => { sink[0] -= i; }, i => { sink[0] *= 1; }, i => { sink[0] = i; }, i => { sink[0] += 2 * i; }, i => { sink[0] += i & 3; }];
  for (let r = 0; r < 200; r++) for (const d of ds) drive(d, 50);
}
/** runCases over bodies: each case loop is drive(body, callsPerLoop). */
export function runBodies(name, bodies, callsPerLoop, loops, extra = {}) {
  makeDriverMegamorphic();
  const cases = {};
  for (const [k, b] of Object.entries(bodies)) cases[k] = () => drive(b, callsPerLoop);
  return runCases(name, cases, callsPerLoop, loops, extra);
}
