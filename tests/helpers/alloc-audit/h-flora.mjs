// drawFloraCandidates per frame (tests/flora-culling.test.mjs:110 "zero net allocation growth"; src/flora-culling.ts
// "No allocation" / "Zero-allocation"). 256 candidates, all kinds, mixed distances, default budgets. Unit: frame.
import {src, mod, runCases, TREE} from './alloc-lib.mjs';
// Bare resolution also supports a consumer's recursive submodule checkout.
import * as THREE from 'three';
const {packCullPlane, DEFAULT_FRUSTUM_PLANE_COUNT, FRUSTUM_PLANE_STRIDE, FloraFieldRenderer, FloraCandidateBuffer, drawFloraCandidates} = await import(mod('flora-culling.ts'));
const {buildFloraGeometryData, floraGeometryToBufferGeometry, FLORA_KIND_NAMES} = await import(src('flora-field.ts'));
const renderer = new FloraFieldRenderer((k, l) => floraGeometryToBufferGeometry(buildFloraGeometryData(k, l, 1)), () => new THREE.MeshBasicMaterial());
const buffer = new FloraCandidateBuffer(256), m = new THREE.Matrix4();
for (let i = 0; i < 256; i++) { const x = (i % 16) * 7 - 50, z = (i >> 4) * 9 - 70; m.makeTranslation(x, 0, z); buffer.push(FLORA_KIND_NAMES[i % FLORA_KIND_NAMES.length], x, 0, z, .5, m.elements); }
const planes = new Float32Array(DEFAULT_FRUSTUM_PLANE_COUNT * FRUSTUM_PLANE_STRIDE), h = 60;
packCullPlane(planes, 0, 1, 0, 0, h); packCullPlane(planes, 1, -1, 0, 0, h); packCullPlane(planes, 2, 0, 1, 0, h); packCullPlane(planes, 3, 0, -1, 0, h); packCullPlane(planes, 4, 0, 0, 1, h); packCullPlane(planes, 5, 0, 0, -1, h);
const cam = {x: 0, y: 1.6, z: 0}, F = 5000, acc = new Float64Array(1), res = {survivorCount: 0, drawnCount: 0};
const ctl = (b, r, c, p) => { res.survivorCount = b.count; res.drawnCount = p.length; return res; };
const seen = {survivors: 0, drawn: 0};
runCases('drawFloraCandidates@' + TREE, {
  control: () => { for (let f = 0; f < F; f++) { cam.x = (f & 15) * .1; acc[0] += ctl(buffer, renderer, cam, planes).drawnCount; } },
  real: () => { for (let f = 0; f < F; f++) { cam.x = (f & 15) * .1; const r = drawFloraCandidates(buffer, renderer, cam, planes); acc[0] += r.drawnCount; seen.survivors = r.survivorCount; seen.drawn = r.drawnCount; } },
}, F, 10, {candidatesPerFrame: buffer.count, unit: 'frame', seen, sink: Number.isFinite(acc[0])});
