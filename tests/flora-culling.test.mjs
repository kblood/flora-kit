import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as THREE from 'three';
import {
  cullSpheresSoA,
  packCullPlane,
  DEFAULT_FRUSTUM_PLANE_COUNT,
  FRUSTUM_PLANE_STRIDE,
  pickFloraLodBand,
  FLORA_LOD0_DISTANCE_M,
  FLORA_LOD1_DISTANCE_M,
  FLORA_VISIBLE_DISTANCE_M,
  FLORA_LOD_INSTANCE_BUDGET,
  totalFloraInstanceBudget,
  FloraLodBucket,
  FloraFieldRenderer,
  FloraCandidateBuffer,
  drawFloraCandidates,
} from '../src/flora-culling.ts';
import { buildFloraGeometryData, floraGeometryToBufferGeometry, FLORA_KIND_NAMES, FLORA_LOD_COUNT } from '../src/flora-field.ts';

/** A simple axis-aligned box frustum [-half,half]^3 in all three axes,
 * expressed as 6 inward-facing planes for cullSpheresSoA. */
function boxFrustumPlanes(half) {
  const planes = new Float32Array(DEFAULT_FRUSTUM_PLANE_COUNT * FRUSTUM_PLANE_STRIDE);
  packCullPlane(planes, 0, 1, 0, 0, half);
  packCullPlane(planes, 1, -1, 0, 0, half);
  packCullPlane(planes, 2, 0, 1, 0, half);
  packCullPlane(planes, 3, 0, -1, 0, half);
  packCullPlane(planes, 4, 0, 0, 1, half);
  packCullPlane(planes, 5, 0, 0, -1, half);
  return planes;
}

test('cullSpheresSoA keeps only spheres inside (or touching) the box frustum', () => {
  const capacity = 8;
  const centerX = new Float32Array(capacity), centerY = new Float32Array(capacity), centerZ = new Float32Array(capacity), radius = new Float32Array(capacity);
  const points = [
    [0, 0, 0, 0.1], // inside
    [9, 0, 0, 0.1], // outside +x
    [4.9, 0, 0, 0.05], // just inside +x
    [5.2, 0, 0, 0.1], // just outside +x, radius too small to reach
    [5.2, 0, 0, 0.5], // outside center but sphere reaches back in
    [0, -100, 0, 1], // far outside -y
  ];
  points.forEach((p, i) => { centerX[i] = p[0]; centerY[i] = p[1]; centerZ[i] = p[2]; radius[i] = p[3]; });
  const outIndices = new Uint32Array(capacity);
  const survivorCount = cullSpheresSoA({ centerX, centerY, centerZ, radius, count: points.length, planes: boxFrustumPlanes(5), outIndices });
  const survivors = Array.from(outIndices.slice(0, survivorCount));
  assert.deepEqual(survivors, [0, 2, 4]);
});

test('cullSpheresSoA is a pure function of its inputs (order/repetition independent)', () => {
  const capacity = 200;
  const centerX = new Float32Array(capacity), centerY = new Float32Array(capacity), centerZ = new Float32Array(capacity), radius = new Float32Array(capacity);
  for (let i = 0; i < capacity; i++) {
    centerX[i] = Math.sin(i * 0.3) * 12; centerY[i] = Math.cos(i * 0.7) * 12; centerZ[i] = Math.sin(i * 1.1) * 12; radius[i] = 0.3;
  }
  const planes = boxFrustumPlanes(6);
  const out1 = new Uint32Array(capacity), out2 = new Uint32Array(capacity);
  const count1 = cullSpheresSoA({ centerX, centerY, centerZ, radius, count: capacity, planes, outIndices: out1 });
  const count2 = cullSpheresSoA({ centerX, centerY, centerZ, radius, count: capacity, planes, outIndices: out2 });
  assert.equal(count1, count2);
  assert.deepEqual(Array.from(out1.slice(0, count1)), Array.from(out2.slice(0, count2)));
});

test('pickFloraLodBand thresholds match the documented distances', () => {
  assert.equal(pickFloraLodBand(0), 0);
  assert.equal(pickFloraLodBand(FLORA_LOD0_DISTANCE_M), 0);
  assert.equal(pickFloraLodBand(FLORA_LOD0_DISTANCE_M + 0.01), 1);
  assert.equal(pickFloraLodBand(FLORA_LOD1_DISTANCE_M), 1);
  assert.equal(pickFloraLodBand(FLORA_LOD1_DISTANCE_M + 0.01), 2);
  assert.equal(pickFloraLodBand(FLORA_VISIBLE_DISTANCE_M), 2);
  assert.equal(pickFloraLodBand(FLORA_VISIBLE_DISTANCE_M + 0.01), -1);
});

test('FloraLodBucket never writes more than its capacity and never allocates a new backing InstancedMesh', () => {
  const geometry = new THREE.BoxGeometry(1, 1, 1);
  const material = new THREE.MeshBasicMaterial();
  const capacity = 4;
  const bucket = new FloraLodBucket(geometry, material, capacity);
  const mesh = bucket.mesh;
  bucket.beginFrame();
  const identity = new THREE.Matrix4().identity().toArray();
  let accepted = 0;
  for (let i = 0; i < 10; i++) { if (bucket.tryAdd(identity)) accepted++; }
  bucket.endFrame();
  assert.equal(accepted, capacity);
  assert.equal(bucket.mesh, mesh, 'bucket must reuse the same InstancedMesh, never reallocate it');
  assert.equal(mesh.count, capacity);
  assert.equal(mesh.instanceMatrix.count, capacity, 'InstancedMesh instance-matrix storage must be sized to capacity, not to what was written');
});

test('FloraFieldRenderer exposes exactly one InstancedMesh per (kind, LOD) with the documented budget', () => {
  const renderer = new FloraFieldRenderer(
    (kind, lod) => new THREE.BoxGeometry(1, 1, 1),
    (kind, lod) => new THREE.MeshBasicMaterial(),
  );
  assert.equal(renderer.meshes.length, FLORA_KIND_NAMES.length * FLORA_LOD_COUNT);
  assert.equal(renderer.totalCapacity(), totalFloraInstanceBudget());
  for (const kind of FLORA_KIND_NAMES) {
    for (let lod = 0; lod < FLORA_LOD_COUNT; lod++) {
      const bucket = renderer.bucketFor(kind, lod);
      assert.ok(bucket, `missing bucket for ${kind} LOD${lod}`);
      assert.equal(bucket.capacity, FLORA_LOD_INSTANCE_BUDGET[kind][lod]);
    }
  }
});

test('drawFloraCandidates culls, LOD-bands, and respects per-bucket budgets with bounded instance counts', () => {
  const renderer = new FloraFieldRenderer(
    (kind, lod) => floraGeometryToBufferGeometry(buildFloraGeometryData(kind, lod, 1)),
    () => new THREE.MeshBasicMaterial(),
    { tree: [2, 2, 2], shrub: [2, 2, 2], grass: [2, 2, 2], flower: [2, 2, 2] },
  );
  const buffer = new FloraCandidateBuffer(64);
  const matrix = new THREE.Matrix4();
  // 5 grass candidates all within LOD0 range but bucket capacity is 2: only
  // 2 should be drawn, the rest silently dropped (documented degradation).
  for (let i = 0; i < 5; i++) {
    matrix.makeTranslation(i * 0.01, 0, 0);
    buffer.push('grass', i * 0.01, 0, 0, 0.2, matrix.elements);
  }
  // One tree far outside the visible range - must be dropped by LOD banding, not drawn.
  matrix.makeTranslation(500, 0, 0);
  buffer.push('tree', 500, 0, 0, 0.5, matrix.elements);
  const planes = boxFrustumPlanes(1000);
  const { survivorCount, drawnCount } = drawFloraCandidates(buffer, renderer, { x: 0, y: 0, z: 0 }, planes);
  assert.equal(survivorCount, 6, 'all 6 candidates are inside the huge frustum');
  assert.equal(drawnCount, 2, 'only 2 of 5 near grass candidates fit the LOD0 budget of 2; the far tree is beyond the visible range');
  const grassLod0 = renderer.bucketFor('grass', 0);
  assert.equal(grassLod0.instanceCount, 2);
  const treeLod2 = renderer.bucketFor('tree', 2);
  assert.equal(treeLod2.instanceCount, 0);

  // Re-run several frames: instance counts must reset each frame (bounded
  // memory, no leak/growth across frames).
  buffer.clear();
  matrix.makeTranslation(0, 0, 0);
  buffer.push('grass', 0, 0, 0, 0.2, matrix.elements);
  drawFloraCandidates(buffer, renderer, { x: 0, y: 0, z: 0 }, planes);
  assert.equal(renderer.bucketFor('grass', 0).instanceCount, 1, 'per-frame count must reset, not accumulate');
});

test('FloraCandidateBuffer.push is bounded and silently drops beyond capacity', () => {
  const buffer = new FloraCandidateBuffer(3);
  const identity = new THREE.Matrix4().identity().elements;
  assert.equal(buffer.push('grass', 0, 0, 0, 0.1, identity), true);
  assert.equal(buffer.push('grass', 1, 0, 0, 0.1, identity), true);
  assert.equal(buffer.push('grass', 2, 0, 0, 0.1, identity), true);
  assert.equal(buffer.push('grass', 3, 0, 0, 0.1, identity), false, 'must refuse once capacity is reached');
  assert.equal(buffer.count, 3);
});

test('total documented flora instance budget matches the sum of per-kind, per-LOD constants', () => {
  let expected = 0;
  for (const kind of FLORA_KIND_NAMES) for (const capacity of FLORA_LOD_INSTANCE_BUDGET[kind]) expected += capacity;
  assert.equal(totalFloraInstanceBudget(), expected);
  assert.equal(totalFloraInstanceBudget(), 4480);
});

import {checkAllocation} from './helpers/alloc-audit/check.mjs';
test('flora-culling allocation: GC bytes plus heap growth, validated no-op control', () => {
  checkAllocation('h-flora.mjs');
});

test('draw counts are borrowed per buffer and overwritten on the next frame', () => {
  const renderer = new FloraFieldRenderer(() => new THREE.BoxGeometry(), () => new THREE.MeshBasicMaterial());
  const buffer = new FloraCandidateBuffer(2), other = new FloraCandidateBuffer(2);
  const matrix = new THREE.Matrix4(), camera = {x: 0, y: 0, z: 0}, planes = boxFrustumPlanes(100);
  buffer.push('grass', 0, 0, 0, .2, matrix.elements);
  const result = drawFloraCandidates(buffer, renderer, camera, planes);
  assert.deepEqual(result, {survivorCount: 1, drawnCount: 1});
  assert.notEqual(drawFloraCandidates(other, renderer, camera, planes), result);
  buffer.clear();
  assert.equal(drawFloraCandidates(buffer, renderer, camera, planes), result);
  assert.deepEqual(result, {survivorCount: 0, drawnCount: 0});
});
