import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  floraAt,
  buildFloraGeometryData,
  floraGeometryToBufferGeometry,
  growthSkeletonToGeometryData,
  generateProceduralTreeSkeleton,
  FLORA_KIND_NAMES,
  FLORA_LOD_COUNT,
  toMyr,
} from '../src/flora-field.ts';
import { detCos, sine, detPow, hypotDet } from '../src/det-math.ts';

const SEED = 'FLORA-GOLDEN-001';
const BODY = 'sector/1,0,0/system/2/planet/1';

function hash32(value) {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < value.length; i++) { h ^= value.charCodeAt(i); h = Math.imul(h, 16777619); }
  h ^= h >>> 16; h = Math.imul(h, 0x7feb352d); h ^= h >>> 15; h = Math.imul(h, 0x846ca68b);
  return (h ^ (h >>> 16)) >>> 0;
}
function unit(key) { return hash32(key) / 0x1_0000_0000; }

/** >=100 pseudo-random unit-sphere directions, deterministic across runs. */
function sampleDirections(count, salt) {
  const directions = [];
  for (let i = 0; i < count; i++) {
    const key = `${salt}|${i}`;
    const z = unit(`${key}|z`) * 2 - 1;
    const theta = unit(`${key}|theta`) * Math.PI * 2;
    const radius = Math.sqrt(Math.max(0, 1 - z * z));
    directions.push([radius * Math.cos(theta), z, radius * Math.sin(theta)]);
  }
  return directions;
}

function fixedBiome(overrides = {}) {
  return {
    weights: [0.1, 0.6, 0.25, 0.05],
    forestPotential: true,
    moisture: 0.6,
    slopeDeg: 8,
    ocean: false,
    ...overrides,
  };
}
const COMPLEX_GATE = { stage: 'complex' };
const MICROBIAL_GATE = { stage: 'microbial' };
const SAMPLE_DIRECTIONS = sampleDirections(120, 'flora-field-directions');

test('floraAt is deterministic over >=100 (address, direction) samples', () => {
  const bodies = ['sector/0,0,0/system/1/planet/0', 'sector/9,-3,2/system/4/planet/2'];
  let sampleCount = 0;
  for (const body of bodies) {
    for (const direction of SAMPLE_DIRECTIONS) {
      const biome = fixedBiome();
      const a = floraAt(SEED, body, direction, biome, COMPLEX_GATE, toMyr(1200));
      const b = floraAt(SEED, body, direction, biome, COMPLEX_GATE, toMyr(1200));
      assert.deepEqual(a, b, `mismatch for ${body} @ ${direction.join(',')}`);
      sampleCount++;
    }
  }
  assert.ok(sampleCount >= 100, 'sanity: sampled at least 100 (address,direction) pairs');
});

test('floraAt does not depend on visit order or prior queries', () => {
  const biome = fixedBiome();
  const baseline = SAMPLE_DIRECTIONS.map((direction) => floraAt(SEED, BODY, direction, biome, COMPLEX_GATE, toMyr(500)));
  // Query in reverse order, interleaved with unrelated bodies/directions, to
  // simulate arbitrary streaming/visit order.
  const shuffled = [...SAMPLE_DIRECTIONS].reverse();
  for (const direction of shuffled) {
    floraAt(SEED, 'sector/5,5,5/system/9/planet/9', direction, biome, COMPLEX_GATE, toMyr(999));
  }
  const revisited = SAMPLE_DIRECTIONS.map((direction) => floraAt(SEED, BODY, direction, biome, COMPLEX_GATE, toMyr(500)));
  assert.deepEqual(revisited, baseline);
});

test('t-seek independence: evaluating floraAt directly at a t matches evaluating it after visiting many other t values first', () => {
  const biome = fixedBiome();
  const direction = SAMPLE_DIRECTIONS[7];
  const direct = floraAt(SEED, BODY, direction, biome, COMPLEX_GATE, toMyr(4321));
  for (let t = 0; t < 5000; t += 137) floraAt(SEED, BODY, direction, biome, COMPLEX_GATE, toMyr(t));
  const afterStepping = floraAt(SEED, BODY, direction, biome, COMPLEX_GATE, toMyr(4321));
  assert.deepEqual(afterStepping, direct);
  // Discrete fields (presence/kind/instanceSeed) must be IDENTICAL across a
  // wide t range - only the continuous `scale` pulse may vary.
  // 19999 (not 1e6/1_000_000, which predates the toMyr range fix below and
  // is actually raw-seconds-shaped, not a plausible Myr magnitude) is the
  // widest in-range sample available near MYR_MAX, keeping this test's
  // "wide t range" intent while respecting the now-enforced plausible
  // domain - see the dedicated toMyr-rejection test for the out-of-range case.
  const acrossTime = [0, 10, 5000, 19999, -250].map((t) => floraAt(SEED, BODY, direction, biome, COMPLEX_GATE, toMyr(t)));
  for (const sample of acrossTime) {
    if (!sample.present) { assert.equal(direct.present, false); continue; }
    assert.equal(sample.kind, direct.kind);
    assert.equal(sample.instanceSeed, direct.instanceSeed);
    assert.equal(sample.rotationY, direct.rotationY);
  }
});

test('biosphere stage gates flora presence and kind', () => {
  const biome = fixedBiome();
  let sterileCount = 0, unformedCount = 0, complexCount = 0, microbialTreeCount = 0, microbialPresentCount = 0;
  for (const direction of SAMPLE_DIRECTIONS) {
    const sterile = floraAt(SEED, BODY, direction, biome, { stage: 'sterile' }, toMyr(100));
    const unformed = floraAt(SEED, BODY, direction, biome, { stage: 'unformed' }, toMyr(100));
    const complex = floraAt(SEED, BODY, direction, biome, COMPLEX_GATE, toMyr(100));
    const microbial = floraAt(SEED, BODY, direction, biome, MICROBIAL_GATE, toMyr(100));
    if (sterile.present) sterileCount++;
    if (unformed.present) unformedCount++;
    if (complex.present) complexCount++;
    if (microbial.present) {
      microbialPresentCount++;
      if (microbial.kind === 'tree') microbialTreeCount++;
    }
  }
  assert.equal(sterileCount, 0, 'sterile stage must have zero flora');
  assert.equal(unformedCount, 0, 'unformed stage must have zero flora');
  assert.ok(complexCount > 0, 'complex stage should populate at least some samples');
  assert.equal(microbialTreeCount, 0, 'microbial stage must never place trees');
  assert.ok(microbialPresentCount <= complexCount, 'microbial density must not exceed complex density');
});

test('an empty availableKinds gate (all vegetation lineages extinct) yields no flora even at complex stage', () => {
  const biome = fixedBiome();
  for (const direction of SAMPLE_DIRECTIONS.slice(0, 20)) {
    const result = floraAt(SEED, BODY, direction, biome, { stage: 'complex', availableKinds: [] }, toMyr(100));
    assert.equal(result.present, false);
  }
});

test('ocean and steep-slope points never carry flora', () => {
  const oceanBiome = fixedBiome({ ocean: true });
  const steepBiome = fixedBiome({ slopeDeg: 60 });
  for (const direction of SAMPLE_DIRECTIONS.slice(0, 30)) {
    assert.equal(floraAt(SEED, BODY, direction, oceanBiome, COMPLEX_GATE, toMyr(0)).present, false);
    assert.equal(floraAt(SEED, BODY, direction, steepBiome, COMPLEX_GATE, toMyr(0)).present, false);
  }
});

test('geometry builders construct valid triangle data for every kind and LOD without a GL context', () => {
  for (const kind of FLORA_KIND_NAMES) {
    for (let lod = 0; lod < FLORA_LOD_COUNT; lod++) {
      const data = buildFloraGeometryData(kind, lod, 0x1234 + lod);
      assert.ok(data.triangleCount > 0, `${kind} LOD${lod} produced no triangles`);
      assert.equal(data.positions.length / 3, data.normals.length / 3);
      assert.equal(data.positions.length / 3, data.colors.length / 3);
      assert.equal(data.positions.length / 3, data.windStiffness.length);
      assert.equal(data.indices.length, data.triangleCount * 3);
      for (const value of data.positions) assert.ok(Number.isFinite(value));
      const bufferGeometry = floraGeometryToBufferGeometry(data);
      assert.equal(bufferGeometry.getAttribute('position').count, data.positions.length / 3);
      bufferGeometry.dispose();
    }
  }
});

test('woody tree and flower triangles face the side their vertex normals describe', () => {
  const expectedTriangles = { tree: [491, 133, 131], flower: [39, 26, 23] };
  for (const kind of ['tree', 'flower']) {
    for (let lod = 0; lod < FLORA_LOD_COUNT; lod++) {
      const data = buildFloraGeometryData(kind, lod, 0x1234 + lod);
      assert.equal(data.triangleCount, expectedTriangles[kind][lod], `${kind} LOD${lod} topology/count stays unchanged`);
      for (let triangle = 0; triangle < data.indices.length; triangle += 3) {
        const ia = data.indices[triangle] * 3;
        const ib = data.indices[triangle + 1] * 3;
        const ic = data.indices[triangle + 2] * 3;
        const ux = data.positions[ib] - data.positions[ia];
        const uy = data.positions[ib + 1] - data.positions[ia + 1];
        const uz = data.positions[ib + 2] - data.positions[ia + 2];
        const vx = data.positions[ic] - data.positions[ia];
        const vy = data.positions[ic + 1] - data.positions[ia + 1];
        const vz = data.positions[ic + 2] - data.positions[ia + 2];
        const nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
        const area = Math.hypot(nx, ny, nz);
        assert.ok(area > 1e-12, `${kind} LOD${lod} triangle ${triangle / 3} is nondegenerate`);
        const normalDot = [ia, ib, ic].reduce((sum, offset) =>
          sum + nx * data.normals[offset] + ny * data.normals[offset + 1] + nz * data.normals[offset + 2], 0) / (3 * area);
        assert.ok(normalDot > 0.1, `${kind} LOD${lod} triangle ${triangle / 3} winding agrees with its normals (${normalDot})`);
      }
    }
  }
});

test('space-colonization tree growth is deterministic and produces a usable skeleton', () => {
  const a = generateProceduralTreeSkeleton({ seed: 777 });
  const b = generateProceduralTreeSkeleton({ seed: 777 });
  assert.deepEqual(a, b);
  assert.ok(a.nodeCount > 5);
  assert.equal(a.signature, b.signature);
  const other = generateProceduralTreeSkeleton({ seed: 778 });
  assert.notEqual(other.signature, a.signature);
  for (const preset of ['broadleaf', 'windswept', 'spire']) {
    const skeleton = generateProceduralTreeSkeleton({ preset, seed: 42 });
    const geometry = growthSkeletonToGeometryData(skeleton);
    assert.ok(geometry.triangleCount > 0, `${preset} produced no triangles`);
  }
});

test('ulp-perturbation of platform transcendentals leaves floraAt discrete decisions unchanged', () => {
  // Scope note (mirrors tests/golden/perturbation.test.mjs): floraAt's
  // presence/kind/instanceSeed decisions are derived only from the
  // Math.imul-based hashU32/hash32 mix, never from Math.sin/cos/pow/atan2/
  // tan/cbrt/exp/log, so they must be exactly invariant. `floraAt` itself
  // never calls `generateProceduralTreeSkeleton`, so the tree-growth
  // skeleton is out of scope for THIS test - see the dedicated test below
  // ("det-math routing makes generateProceduralTreeSkeleton's topology...")
  // for its own perturbation coverage. The plant archetype geometry builders
  // (buildTreeArchetype and friends) DO use platform transcendentals for
  // ordinary continuous shape on an already-finalized skeleton (like this
  // codebase's existing gas-giant band color and orbital position
  // exemptions) and are intentionally out of scope everywhere.
  const names = ['sin', 'cos', 'exp', 'log', 'pow', 'atan2', 'tan', 'cbrt'];
  const originals = {};
  for (const name of names) originals[name] = Math[name];
  const nudge = (x) => (Number.isFinite(x) ? x + Math.abs(x || 1e-300) * Number.EPSILON : x);
  for (const name of names) { const original = originals[name]; Math[name] = (...args) => nudge(original(...args)); }
  let perturbed;
  try {
    perturbed = SAMPLE_DIRECTIONS.map((direction) => floraAt(SEED, BODY, direction, fixedBiome(), COMPLEX_GATE, toMyr(500)));
  } finally {
    for (const name of names) Math[name] = originals[name];
  }
  const baseline = SAMPLE_DIRECTIONS.map((direction) => floraAt(SEED, BODY, direction, fixedBiome(), COMPLEX_GATE, toMyr(500)));
  assert.equal(perturbed.length, baseline.length);
  for (let i = 0; i < baseline.length; i++) {
    assert.equal(perturbed[i].present, baseline[i].present, `presence changed at sample ${i}`);
    if (baseline[i].present) {
      assert.equal(perturbed[i].kind, baseline[i].kind, `kind changed at sample ${i}`);
      assert.equal(perturbed[i].instanceSeed, baseline[i].instanceSeed, `instanceSeed changed at sample ${i}`);
    }
  }
});

// ---------------------------------------------------------------------------
// FINDING 2 fix: the Myr/Seconds unit brand must be loud, not silent.
// ---------------------------------------------------------------------------

test('toMyr rejects an out-of-range magnitude (the wrong-unit failure mode the adversarial review found)', () => {
  // Legitimate values across the documented range must pass through unchanged.
  assert.equal(toMyr(0), 0);
  assert.equal(toMyr(-19_999), -19_999);
  assert.equal(toMyr(19_999), 19_999);
  // A raw seconds/Ns-scale magnitude (what the review actually observed
  // reaching flora-field.ts from fauna-runtime.ts/world-time.ts) must throw,
  // loudly - never clamp, never warn.
  assert.throws(() => toMyr(1.3e26), RangeError);
  assert.throws(() => toMyr(-1.3e26), RangeError);
  assert.throws(() => toMyr(Number.POSITIVE_INFINITY), RangeError);
  assert.throws(() => toMyr(Number.NaN), RangeError);
  // floraAt itself must refuse the same out-of-range value, not just the
  // standalone constructor - it re-validates at its own entry point so a
  // caller that bypasses TypeScript (this .mjs file included) still fails
  // loudly rather than silently computing a flickering result.
  assert.throws(() => floraAt(SEED, BODY, SAMPLE_DIRECTIONS[0], fixedBiome(), COMPLEX_GATE, 1.3e26), RangeError);
});

// ---------------------------------------------------------------------------
// FINDING 3 fix: the growth skeleton's topology and exact vertex positions
// must be provably deterministic, not empirically lucky.
// ---------------------------------------------------------------------------

test("det-math routing makes generateProceduralTreeSkeleton's topology AND exact vertex-position hash immune to platform-transcendental drift across 180 (seed, preset) combinations", () => {
  // Re-runs the adversarial review's own methodology (mirrors the
  // ulp-perturbation tests above): monkey-patch the platform transcendentals
  // that differ bitwise across real JS engines (det-math.ts's own measured
  // cross-engine audit) with a tiny ULP nudge, standing in for "a different
  // conformant engine's libm". Before this fix, the growth loop's
  // decision-feeding math called these directly, so this perturbation WOULD
  // have been able to (rarely) flip a kill-distance/influence-radius
  // decision - the original review measured 0/180 topology flips but
  // 143/180 exact vertex-position hash changes on its own sample, i.e.
  // "gated by nothing today". After routing every decision-feeding call
  // through det-math.ts (which never calls these patched globals - see
  // det-math.ts's header), both counts must be exactly 0/180: not just
  // "no topology flip in this sample" but "provably cannot drift".
  const PRESETS = ['broadleaf', 'windswept', 'spire'];
  const SEED_COUNT = 60; // 60 seeds x 3 presets = 180 combinations.
  const names = ['sin', 'cos', 'exp', 'log', 'pow', 'atan2', 'tan', 'cbrt'];
  const originals = {};
  for (const name of names) originals[name] = Math[name];
  const nudge = (x) => (Number.isFinite(x) ? x + Math.abs(x || 1e-300) * Number.EPSILON * 4 : x);

  const combos = [];
  for (let seed = 0; seed < SEED_COUNT; seed++) for (const preset of PRESETS) combos.push({ seed, preset });
  assert.equal(combos.length, 180, 'sanity: exactly 180 (seed, preset) combinations');

  const baselines = combos.map(({ seed, preset }) => generateProceduralTreeSkeleton({ seed, preset }));
  let perturbedResults;
  for (const name of names) { const original = originals[name]; Math[name] = (...args) => nudge(original(...args)); }
  try {
    perturbedResults = combos.map(({ seed, preset }) => generateProceduralTreeSkeleton({ seed, preset }));
  } finally {
    for (const name of names) Math[name] = originals[name];
  }

  let topologyMismatches = 0;
  let geometryHashMismatches = 0;
  for (let i = 0; i < combos.length; i++) {
    const base = baselines[i];
    const perturbed = perturbedResults[i];
    const topologyOf = (skeleton) => ({
      nodeCount: skeleton.nodeCount,
      iterationCount: skeleton.iterationCount,
      consumedAttractionPointCount: skeleton.consumedAttractionPointCount,
      terminalNodeCount: skeleton.terminalNodeCount,
      maxDepth: skeleton.maxDepth,
      parentIndex: skeleton.nodes.map((n) => n.parentIndex),
      depth: skeleton.nodes.map((n) => n.depth),
      childCount: skeleton.nodes.map((n) => n.childCount),
    });
    const topologyMatches = JSON.stringify(topologyOf(base)) === JSON.stringify(topologyOf(perturbed));
    if (!topologyMatches) topologyMismatches++;
    const geometryHashMatches = base.signature === perturbed.signature;
    if (!geometryHashMatches) geometryHashMismatches++;
  }

  assert.equal(topologyMismatches, 0, `expected 0/180 topology mismatches, got ${topologyMismatches}/180`);
  assert.equal(geometryHashMismatches, 0, `expected 0/180 exact vertex-position hash mismatches, got ${geometryHashMismatches}/180`);
});

test('det-math primitives used by the growth loop agree with their platform-Math counterparts to within a small tolerance (sanity check, not a determinism requirement)', () => {
  // det-math is deterministic across engines but not bit-identical to
  // Math.* (different algorithm) - this just confirms the swap didn't
  // silently break the underlying formula.
  for (let i = -20; i <= 20; i++) {
    const x = i * 0.31;
    assert.ok(Math.abs(detCos(x) - Math.cos(x)) < 1e-6, `detCos diverged at x=${x}`);
    assert.ok(Math.abs(sine(x) - Math.sin(x)) < 1e-6, `sine diverged at x=${x}`);
  }
  for (const base of [0.05, 0.3, 0.7, 1, 1.8]) {
    for (const exp of [0.1, 0.42, 0.74, 1, 2.5]) {
      assert.ok(Math.abs(detPow(base, exp) - Math.pow(base, exp)) < 1e-6, `detPow diverged at ${base}**${exp}`);
    }
  }
  assert.ok(Math.abs(hypotDet(3, 4, 0) - 5) < 1e-6);
  assert.ok(Math.abs(hypotDet(0.12, 0.34, 0.56) - Math.hypot(0.12, 0.34, 0.56)) < 1e-6);
});
