import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  buildFloraGeometryData,
  floraGeometryToBufferGeometry,
  understoryShrubSkeleton,
  FLORA_KIND_NAMES,
  FLORA_LOD_COUNT,
} from '../src/flora-field.ts';
import { FLORA_LOD_INSTANCE_BUDGET, totalFloraInstanceBudget } from '../src/flora-culling.ts';

// Varied seeds, including extremes and the golden-ratio constant.
const SEEDS = [0, 1, 42, 77, 0x1234, 31337, 123456789, 0x9e3779b9, 0xdeadbeef, 0xffffffff];
const UNDERSTORY = ['grass', 'shrub'];
const LODS = [0, 1, 2];

// Stated budget. Triangles per instance are fixed for every seed.
// Retired builders: grass 6/4/2, shrub 143/30/12 (37,760 at full capacity).
const TRIANGLES = { grass: [70, 18, 6], shrub: [284, 60, 28] };
// Every FLORA_LOD_INSTANCE_BUDGET grass+shrub slot full at its LOD.
const UNDERSTORY_TRIANGLE_CEILING = 100_000;
// Plus the unchanged flowers; ordinary play never submits flora trees.
const ORDINARY_FLORA_TRIANGLE_CEILING = 125_000;
// All six grass/shrub geometries (built once per planet).
const UNDERSTORY_BUFFER_BYTES = 40 * 1024;
// planet-renderer.ts FLORA_KIND_CULL_RADIUS_M at unit instance scale.
const CULL_RADIUS_M = { grass: 1, shrub: 2.5 };
// Retired shrub root flare reached -0.023 m.
const MIN_Y_M = { grass: -1e-6, shrub: -0.0235 };

function vertex(data, i) { return [data.positions[i * 3], data.positions[i * 3 + 1], data.positions[i * 3 + 2]]; }
function normalAt(data, i) { return [data.normals[i * 3], data.normals[i * 3 + 1], data.normals[i * 3 + 2]]; }
function sub(a, b) { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function cross(a, b) { return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]; }
function dot(a, b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function length(a) { return Math.hypot(a[0], a[1], a[2]); }
function bytesOf(data) {
  return data.positions.byteLength + data.normals.byteLength + data.uvs.byteLength + data.colors.byteLength + data.windStiffness.byteLength + data.indices.byteLength;
}
function* triangles(data) {
  for (let t = 0; t < data.indices.length; t += 3) {
    const ids = [data.indices[t], data.indices[t + 1], data.indices[t + 2]];
    const [a, b, c] = ids.map((i) => vertex(data, i));
    yield { ids, a, b, c, face: cross(sub(b, a), sub(c, a)) };
  }
}
function bounds(data) {
  let height = 0, radius = 0, minY = Infinity;
  for (let i = 0; i < data.positions.length / 3; i++) {
    const p = vertex(data, i);
    height = Math.max(height, p[1]);
    minY = Math.min(minY, p[1]);
    radius = Math.max(radius, Math.hypot(p[0], p[2]));
  }
  return { height, radius, minY };
}
/** Mean projected area of camera-facing triangles over 8 azimuths at one elevation. */
function coverage(data, elevationDeg) {
  const el = (elevationDeg * Math.PI) / 180;
  let sum = 0;
  for (let k = 0; k < 8; k++) {
    const az = (k / 8) * Math.PI * 2, view = [Math.cos(az) * Math.cos(el), Math.sin(el), Math.sin(az) * Math.cos(el)];
    for (const { face } of triangles(data)) sum += Math.max(0, dot(face, view) / 2);
  }
  return sum / 8;
}
const positionKey = (p) => p.join(',');
function edgeUses(data) {
  const uses = new Map();
  for (const { a, b, c } of triangles(data)) {
    for (const [u, v] of [[a, b], [b, c], [c, a]]) {
      const key = [positionKey(u), positionKey(v)].sort().join('|');
      const entry = uses.get(key) ?? { u, v, count: 0 };
      entry.count++;
      uses.set(key, entry);
    }
  }
  return [...uses.values()];
}
function distanceToSegment(p, a, b) {
  const ab = sub(b, a), t = Math.max(0, Math.min(1, dot(sub(p, a), ab) / dot(ab, ab)));
  return { distance: length(sub(p, [a[0] + ab[0] * t, a[1] + ab[1] * t, a[2] + ab[2] * t])), t };
}
/** Distance from `p` to a shoot axis and the drawn radius there. */
function nearestOnShoot(p, shoot) {
  let best = { distance: Infinity, radius: 0 };
  for (let i = 0; i < shoot.points.length - 1; i++) {
    const { distance, t } = distanceToSegment(p, shoot.points[i], shoot.points[i + 1]);
    if (distance < best.distance) best = { distance, radius: shoot.radii[i] + (shoot.radii[i + 1] - shoot.radii[i]) * t };
  }
  return best;
}

test('understory triangle counts are fixed per LOD for every seed and fall with distance', () => {
  for (const kind of UNDERSTORY) {
    for (const seed of SEEDS) {
      const counts = LODS.map((lod) => buildFloraGeometryData(kind, lod, seed).triangleCount);
      assert.deepEqual(counts, TRIANGLES[kind], `${kind} seed ${seed}`);
    }
    assert.ok(TRIANGLES[kind][0] > TRIANGLES[kind][1] && TRIANGLES[kind][1] > TRIANGLES[kind][2]);
  }
});

test('full-capacity flora budget stays within the stated ceilings without new capacity or draw calls', () => {
  // Capacities and draw calls are the renderer's, unchanged by geometry work.
  assert.deepEqual(FLORA_LOD_INSTANCE_BUDGET.grass, [256, 512, 1024]);
  assert.deepEqual(FLORA_LOD_INSTANCE_BUDGET.shrub, [128, 256, 512]);
  assert.equal(totalFloraInstanceBudget(), 4480);
  assert.equal(FLORA_KIND_NAMES.length * FLORA_LOD_COUNT, 12, 'one InstancedMesh per kind and LOD');

  const ceiling = (kind) => LODS.reduce((sum, lod) => sum + FLORA_LOD_INSTANCE_BUDGET[kind][lod] * Math.max(...SEEDS.map((seed) => buildFloraGeometryData(kind, lod, seed).triangleCount)), 0);
  const understory = ceiling('grass') + ceiling('shrub');
  assert.equal(understory, 99_328);
  assert.ok(understory <= UNDERSTORY_TRIANGLE_CEILING, `grass+shrub ceiling ${understory}`);
  const ordinary = understory + ceiling('flower');
  assert.ok(ordinary <= ORDINARY_FLORA_TRIANGLE_CEILING, `ordinary flora ceiling ${ordinary}`);

  let bytes = 0;
  for (const kind of UNDERSTORY) for (const lod of LODS) bytes += Math.max(...SEEDS.map((seed) => bytesOf(buildFloraGeometryData(kind, lod, seed))));
  assert.ok(bytes <= UNDERSTORY_BUFFER_BYTES, `grass+shrub typed buffers ${bytes} B`);
});

test('understory geometry keeps the renderer attribute layout with finite, bounded values', () => {
  for (const kind of UNDERSTORY) for (const lod of LODS) for (const seed of SEEDS) {
    const data = buildFloraGeometryData(kind, lod, seed), count = data.positions.length / 3, label = `${kind} LOD${lod} seed ${seed}`;
    assert.ok(data.positions instanceof Float32Array && data.normals instanceof Float32Array && data.uvs instanceof Float32Array);
    assert.ok(data.colors instanceof Float32Array && data.windStiffness instanceof Float32Array && data.indices instanceof Uint16Array);
    assert.equal(data.normals.length, count * 3, label);
    assert.equal(data.uvs.length, count * 2, label);
    assert.equal(data.colors.length, count * 3, label);
    assert.equal(data.windStiffness.length, count, label);
    assert.equal(data.indices.length, data.triangleCount * 3, label);
    assert.ok(count < 65536, label);
    for (const index of data.indices) assert.ok(index < count, label);
    for (const value of data.positions) assert.ok(Number.isFinite(value), label);
    for (let i = 0; i < count; i++) assert.ok(Math.abs(length(normalAt(data, i)) - 1) < 1e-5, `${label} normal ${i}`);
    for (const value of data.uvs) assert.ok(value >= 0 && value <= 1, label);
    for (const value of data.colors) assert.ok(value >= 0 && value <= 1, label);
    for (const value of data.windStiffness) assert.ok(Number.isFinite(value) && value >= 0 && value <= 1, label);
    const geometry = floraGeometryToBufferGeometry(data);
    assert.deepEqual(Object.keys(geometry.attributes).sort(), ['color', 'normal', 'position', 'uv', 'windStiffness']);
    geometry.dispose();
  }
});

test('understory geometry is deterministic per seed and LOD and varies across seeds', () => {
  for (const kind of UNDERSTORY) for (const lod of LODS) {
    const signatures = new Set();
    for (const seed of SEEDS) {
      const a = buildFloraGeometryData(kind, lod, seed), b = buildFloraGeometryData(kind, lod, seed);
      assert.deepEqual(a, b, `${kind} LOD${lod} seed ${seed}`);
      signatures.add(Array.from(a.positions.slice(0, 48)).join(','));
    }
    assert.equal(signatures.size, SEEDS.length, `${kind} LOD${lod} must differ between seeds`);
  }
  for (const seed of SEEDS) for (const lod of LODS) assert.deepEqual(understoryShrubSkeleton(lod, seed), understoryShrubSkeleton(lod, seed));
});

test('every understory triangle is nondegenerate and wound to agree with its vertex normals', () => {
  for (const kind of UNDERSTORY) for (const lod of LODS) for (const seed of SEEDS) {
    const data = buildFloraGeometryData(kind, lod, seed);
    for (const { ids, a, b, c, face } of triangles(data)) {
      const label = `${kind} LOD${lod} seed ${seed} triangle ${ids}`;
      assert.ok(length(face) / 2 > 1e-7, `${label} area`);
      assert.ok(Math.min(length(sub(b, a)), length(sub(c, b)), length(sub(a, c))) > 5e-4, `${label} edge`);
      for (const id of ids) assert.ok(dot(face, normalAt(data, id)) > 0, `${label} normal ${id}`);
    }
  }
});

test('thin organs are two-sided under the FrontSide material and tubes are sealed except at buried roots', () => {
  for (const seed of SEEDS) for (const lod of LODS) {
    const skeleton = understoryShrubSkeleton(lod, seed);
    for (const kind of UNDERSTORY) {
      const data = buildFloraGeometryData(kind, lod, seed), label = `${kind} LOD${lod} seed ${seed}`;
      // A back face repeats the front positions in reverse order.
      const reversed = new Set();
      for (const { a, b, c } of triangles(data)) reversed.add([c, b, a].map(positionKey).join('/'));
      let paired = 0;
      for (const { a, b, c } of triangles(data)) {
        const k = [a, b, c].map(positionKey);
        if ([k.join('/'), [k[1], k[2], k[0]].join('/'), [k[2], k[0], k[1]].join('/')].some((key) => reversed.has(key))) paired++;
      }
      const foliageTriangles = kind === 'grass' ? data.triangleCount : skeleton.foliage.length * (lod === 0 ? 8 : 4);
      assert.equal(paired, foliageTriangles, `${label} two-sided triangles`);

      // Open edges: only each stem's root ring and each twig's base ring.
      const open = edgeUses(data).filter((edge) => edge.count === 1);
      const expectedOpen = kind === 'grass' ? 0 : skeleton.stems.reduce((sum, stem) => sum + stem.sides, 0) + skeleton.twigs.reduce((sum, twig) => sum + twig.sides, 0);
      assert.equal(open.length, expectedOpen, `${label} open edges`);
      for (const { u, v } of open) for (const p of [u, v]) {
        const atRoot = p[1] <= 0.01 && Math.hypot(p[0], p[2]) <= 0.08;
        // Inside the parent's five-sided section (inscribed radius cos 36 deg).
        const insideStem = skeleton.stems.some((stem) => { const near = nearestOnShoot(p, stem); return near.distance < near.radius * 0.8; });
        assert.ok(atRoot || insideStem, `${label} exposed open edge at ${p}`);
      }
    }
  }
});

test('understory stands on its local ground origin inside its cull radius', () => {
  for (const kind of UNDERSTORY) for (const lod of LODS) for (const seed of SEEDS) {
    const data = buildFloraGeometryData(kind, lod, seed), label = `${kind} LOD${lod} seed ${seed}`;
    const { height, radius, minY } = bounds(data);
    assert.ok(minY >= MIN_Y_M[kind], `${label} minY ${minY}`);
    let rooted = 0;
    for (let i = 0; i < data.positions.length / 3; i++) {
      const p = vertex(data, i);
      assert.ok(length(p) <= CULL_RADIUS_M[kind], `${label} outside cull radius`);
      if (p[1] <= 0.01) {
        rooted++;
        assert.ok(Math.hypot(p[0], p[2]) <= 0.08, `${label} ground contact away from root ${p}`);
      }
    }
    assert.ok(rooted > 0, `${label} touches the ground`);
    if (kind === 'grass') {
      assert.ok(height >= 0.3 && height <= 0.6, `${label} height ${height}`);
      assert.ok(radius <= 0.45, `${label} radius ${radius}`);
    } else {
      assert.ok(height >= 0.6 && height <= 1.05, `${label} height ${height}`);
      assert.ok(radius >= 0.35 && radius <= 0.9, `${label} radius ${radius}`);
    }
  }
});

test('grass blades sway from a rigid base to free tips', () => {
  for (const lod of LODS) for (const seed of SEEDS) {
    const data = buildFloraGeometryData('grass', lod, seed);
    for (let i = 0; i < data.positions.length / 3; i++) if (vertex(data, i)[1] === 0) assert.equal(data.windStiffness[i], 0);
    assert.equal(Math.max(...data.windStiffness), 1);
  }
});

test('far LODs keep the near silhouette: height, spread and view coverage stay related', () => {
  for (const kind of UNDERSTORY) for (const seed of SEEDS) {
    const near = buildFloraGeometryData(kind, 0, seed), nearBounds = bounds(near);
    for (const lod of [1, 2]) {
      const far = buildFloraGeometryData(kind, lod, seed), farBounds = bounds(far), label = `${kind} LOD${lod} seed ${seed}`;
      const heightRatio = farBounds.height / nearBounds.height, radiusRatio = farBounds.radius / nearBounds.radius;
      assert.ok(heightRatio >= 0.85 && heightRatio <= 1.15, `${label} height ratio ${heightRatio}`);
      assert.ok(radiusRatio >= 0.6 && radiusRatio <= 1.1, `${label} spread ratio ${radiusRatio}`);
      for (const elevation of [5, 35, 75]) {
        const ratio = coverage(far, elevation) / coverage(near, elevation);
        assert.ok(ratio >= 0.5 && ratio <= 1.4, `${label} coverage ratio ${ratio} at ${elevation} deg`);
      }
    }
  }
});

test('shrub stems rise from the root, twigs and leaves grow from their host axes', () => {
  for (const seed of SEEDS) for (const lod of LODS) {
    const { stems, twigs, foliage } = understoryShrubSkeleton(lod, seed), label = `seed ${seed} LOD${lod}`;
    assert.equal(stems.length, 4, label);
    assert.equal(twigs.length, lod === 0 ? 8 : 0, label);
    assert.equal(foliage.length, [20, 12, 4][lod], label);
    for (const stem of stems) {
      assert.ok(Math.abs(stem.points[0][1] + 0.008) < 1e-9 && Math.hypot(stem.points[0][0], stem.points[0][2]) <= 0.04, `${label} stem root`);
      for (let i = 1; i < stem.points.length; i++) assert.ok(stem.points[i][1] > stem.points[i - 1][1], `${label} stem must keep rising`);
      for (let i = 1; i < stem.radii.length - 1; i++) assert.ok(stem.radii[i] < stem.radii[i - 1], `${label} stem must taper`);
    }
    // Roots sit on their stems' golden-angle azimuths.
    const azimuths = stems.map((stem) => Math.atan2(stem.points[0][2], stem.points[0][0])).sort((a, b) => a - b);
    const gaps = azimuths.map((a, i) => (i + 1 < azimuths.length ? azimuths[i + 1] - a : azimuths[0] + Math.PI * 2 - a));
    assert.ok(Math.max(...gaps) < 3.2, `${label} stems must spread around the crown`);
    for (const twig of twigs) {
      const parent = stems[twig.parent];
      const near = nearestOnShoot(twig.points[0], parent);
      assert.ok(near.distance < 1e-7, `${label} twig rooted on its stem axis`);
      assert.ok(Math.abs(near.radius - twig.parentRadius) < 1e-9, `${label} twig parent radius`);
      assert.ok(twig.radii[0] <= twig.parentRadius * 0.6, `${label} twig base hidden inside its stem`);
      const parentAxis = sub(parent.points[parent.points.length - 1], parent.points[0]);
      assert.ok(dot(sub(twig.points[1], twig.points[0]), parentAxis) > 0, `${label} twig grows away from the root`);
    }
    for (const organ of foliage) {
      const host = organ.host === 'twig' ? twigs[organ.hostIndex] : stems[organ.hostIndex];
      assert.ok(nearestOnShoot(organ.base, host).distance < 1e-7, `${label} foliage rooted on its host axis`);
      assert.ok(Math.abs(length(organ.direction) - 1) < 1e-6 && Math.abs(length(organ.normal) - 1) < 1e-6, label);
      assert.ok(Math.abs(dot(organ.direction, organ.normal)) < 1e-6, `${label} foliage frame`);
      const aspect = organ.length / (2 * organ.halfWidth);
      assert.ok(aspect >= (lod === 0 ? 1.5 : 0.5) && aspect <= (lod === 0 ? 2.3 : 3.4), `${label} foliage aspect ${aspect}`);
    }
  }
});

test('near shrub leaves vary in size, outline and orientation', () => {
  for (const seed of SEEDS) {
    const leaves = understoryShrubSkeleton(0, seed).foliage;
    const spread = (values) => Math.max(...values) - Math.min(...values);
    assert.ok(spread(leaves.map((leaf) => leaf.shape)) > 0.4, `seed ${seed} outlines`);
    assert.ok(spread(leaves.map((leaf) => leaf.length)) > 0.02, `seed ${seed} sizes`);
    assert.ok(spread(leaves.map((leaf) => leaf.normal[1])) > 0.15, `seed ${seed} orientations`);
    assert.equal(leaves.filter((leaf) => leaf.host === 'twig').length, 16);
  }
});
