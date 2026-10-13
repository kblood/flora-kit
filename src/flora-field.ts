/**
 * Planet-wide deterministic vegetation field: placement + LOD'd procedural
 * geometry, usable on ANY body address/direction (not just inside a river
 * region, unlike river-environment.ts's dressing).
 *
 * Provenance: the space-colonization tree-growth algorithm and the
 * tree/flower/grass/shrub archetype geometry builders below are ported (with
 * light renaming, no behavioural changes to the ported math) from:
 *   - C:\LLM\IWSDK\nms\shared\proceduralTree.ts (growth skeleton)
 *   - C:\LLM\IWSDK\nms\shared\heroContent.ts (archetype LOD geometry
 *     builders: buildTreeArchetype/buildFlowerArchetype/buildGrassArchetype/
 *     buildShrubArchetype and their `add*` helpers, lines ~1233-2097)
 * `nms` is a separate, read-only sibling project; nothing there was edited.
 * `floraAt` itself, `growthSkeletonToGeometryData`, and the biosphere-stage
 * gate are new: nms's flora was a fixed 50000-candidate scatter for one hero
 * demo scene, not a function of a real catalogue body/biosphere.
 *
 * Determinism contract (this module):
 *  - `floraAt` is a pure function of its arguments: same
 *    (seed, bodyAddress, direction, biome, gate, tMyr) in -> same result out,
 *    regardless of call order or what else has been queried/streamed.
 *  - Every DISCRETE decision (presence, kind, whether a kind is available) is
 *    made from the integer hash below only (Math.imul avalanche - exact and
 *    portable per docs/... determinism notes elsewhere in this codebase) and
 *    never calls Math.sin/cos/pow/atan2/tan/cbrt/exp/log, so it is immune to
 *    the cross-engine transcendental-precision drift det-math.ts documents.
 *  - `tMyr` only ever feeds a *continuous* presentation value (scale pulse);
 *    it never participates in a discrete decision, so seeking straight to a
 *    `tMyr` is always identical to having stepped there (there is no
 *    integration/accumulated state to replay).
 *  - Geometry builders (mesh vertex positions) are ordinary rendering data,
 *    not catalogue identity, so - matching the existing precedent in
 *    biosphere-geometry.ts and creature-rig.ts - they freely use plain
 *    Math.sin/cos/hypot for continuous shape, same as the ported source.
 *
 * This module does not import biosphere-model.ts, surface-biomes.ts, three
 * app singletons, or world-time.ts: the biosphere stage and biome weights
 * are passed in by the caller as plain data (see `FloraBiosphereGate` /
 * `FloraBiomeSample` below), and `tMyr` arrives as millions-of-years
 * (matching biosphere-model.ts's `dateMyr` convention - NOT seconds, and NOT
 * world-time.ts's Ns/BigInt clock). It is the branded `Myr` type declared
 * below `FLORA_LOD_COUNT`, not a plain `number`: fauna-runtime.ts's sibling
 * `t` is in SECONDS, and an adversarial review proved that a caller mixing
 * the two up produces silent wrong output with no error (see that section's
 * comment and `toMyr`'s runtime guard for the fix).
 */
import * as THREE from 'three';
import { detCos, sine, detPow, hypotDet } from './det-math.ts';

export type Vec3 = [number, number, number];

// ---------------------------------------------------------------------------
// Placement API
// ---------------------------------------------------------------------------

export type FloraKind = 'tree' | 'shrub' | 'grass' | 'flower';
export const FLORA_KIND_NAMES: readonly FloraKind[] = ['tree', 'flower', 'grass', 'shrub'];
export const FLORA_LOD_COUNT = 3;

// ---------------------------------------------------------------------------
// Branded time unit (fix for the adversarial review's FINDING 2).
//
// This module's `t` is Myr (millions of years since the atlas epoch,
// matching biosphere-model.ts's `dateMyr` - see the module header). A
// sibling module, fauna-runtime.ts, has its own `t` in SECONDS. Both were
// previously plain `number`, so a caller could silently swap them: measured
// consequence here was `floraAt`'s `scale` field going from a smooth drift
// (0.758 -> 0.731) to a flickering, non-monotonic sequence when fed a
// seconds-scale magnitude instead of Myr. A brand makes this a compile-time
// type error for any TypeScript caller; because brands are erased at
// runtime (a `.mjs` test, or any caller that bypasses the type system,
// still just sees a `number`), `toMyr` below ALSO enforces a runtime
// magnitude check that throws - never clamps, never warns - so the mistake
// is loud even when the compiler can't see it.
//
// Range and why: mirrors biosphere-model.ts's own `MIN_DATE_MYR`/
// `MAX_DATE_MYR` (-20_000..20_000, not imported - see the module header on
// why this file stays import-free of biosphere-model.ts) rather than
// inventing a new bound, since this module's own docs already say tMyr
// "matches biosphere-model.ts's dateMyr convention". +-20,000 Myr (+-20 Gyr)
// comfortably covers the ~13.8 Gyr age of the universe on either side of the
// atlas epoch while remaining ~21 orders of magnitude tighter than the
// raw-nanosecond magnitude (t=~1.3e26) that the adversarial review actually
// observed reaching this module when fauna-runtime.ts's Seconds/world-time's
// Ns-scale values were fed in by mistake - so an out-of-range tMyr is a
// reliable "wrong unit" signal, not a judgment call about cosmic history.
const MYR_MIN = -20_000;
const MYR_MAX = 20_000;

/** Branded Myr: a plain `number` at runtime, but a distinct nominal type at
 * compile time so `floraAt`/`growthPulse` cannot accept a raw number or
 * fauna-runtime.ts's `Seconds` by accident - see the section header above. */
export type Myr = number & { readonly __unit: 'Myr' };

/**
 * Constructs a branded `Myr`, throwing a loud `RangeError` if `value` is
 * outside the documented plausible range above.
 *
 * ⚠ PROVENANCE RULE for renderer/wiring code: the value passed here MUST come
 * from `world-surface.ts`'s `planetAgeMyr(planet, t)`, never from raw
 * `Ns`-to-Myr arithmetic of your own. `planetAgeMyr` *clamps* into
 * `[MIN_DATE_MYR, MAX_DATE_MYR]` on purpose (see its comment: the biosphere
 * model has no scripted extinction past that bound, so clamping the query
 * point returns exactly the "still alive" answer the uncapped math would).
 * Hand-rolled arithmetic does not clamp, and this universe is genuinely
 * travelable to 1e100 yr — a red dwarf's biosphere is legitimately alive at
 * ~1e12 yr, i.e. ~1e6 Myr — so a raw age would blow the ±20,000 Myr guard and
 * turn a LEGAL far-future seek into a `RangeError`. The guard is calibrated to
 * catch a wrong *unit* (seconds, or a ~1e26 raw-Ns magnitude), not to pass
 * judgement on cosmic history; feeding it a clamped age keeps those two
 * failure modes from being confused. Body death is gated separately and
 * correctly by engulfment/stellar-remnant checks in `biosphereStateAt`, not by
 * this range.
 *
 * `floraAt` and `growthPulse` also call
 * this at their own entry point, so even a caller that bypasses TypeScript
 * entirely (brands do not exist at runtime) still gets a thrown error
 * instead of silent drift.
 */
export function toMyr(value: number): Myr {
  if (!Number.isFinite(value) || value < MYR_MIN || value > MYR_MAX) {
    throw new RangeError(
      `toMyr: ${value} is outside the plausible tMyr range [${MYR_MIN}, ${MYR_MAX}] Myr - ` +
        'this usually means a value in the wrong time unit (e.g. raw seconds, or a raw ' +
        'world-time.ts Ns/BigInt-nanosecond magnitude) was passed where Myr was expected.',
    );
  }
  return value as Myr;
}

/** Mirrors biosphere-model.ts's `DatedBiosphere['stage']` structurally
 * (never imported - see module header). */
export type FloraStage = 'unformed' | 'forming' | 'sterile' | 'microbial' | 'complex';

export interface FloraBiosphereGate {
  stage: FloraStage;
  /** Vegetation-bearing species kinds actually alive at `tMyr` on this body,
   * as already computed by the caller from biosphere-model.ts's
   * `biosphereAt(world, dateMyr).species` (e.g. map `species.kind` for
   * `'tree'|'shrub'|'grass'` entries). Omit to allow every kind the stage
   * permits. Passing `[]` (species existing but none vegetation-bearing, or
   * all vegetation lineages currently extinct) yields no flora even at the
   * 'complex' stage - this is how an "extinct world" is expressed. */
  availableKinds?: readonly FloraKind[];
}

/** Structurally compatible with (but not imported from) surface-biomes.ts's
 * `biomeClassifier(...)` return shape. */
export interface FloraBiomeSample {
  /** [sand, vegetatedGround, rock, snow], each in [0,1], summing to ~1. */
  weights: readonly [number, number, number, number];
  forestPotential: boolean;
  moisture: number;
  slopeDeg: number;
  ocean: boolean;
}

export type FloraInstance =
  | { present: false }
  | {
      present: true;
      kind: FloraKind;
      /** Stable per-instance integer seed for `buildFloraGeometryData`/
       * `generateProceduralTreeSkeleton` variant selection - NOT a random
       * draw itself, just `hashU32` folded from the same key as everything
       * else here, so re-deriving it later (e.g. after a re-cull) reproduces
       * the identical geometry choice. */
      instanceSeed: number;
      /** Uniform scale multiplier, continuous in `tMyr` (a bounded growth
       * pulse) but never a discrete decision input. */
      scale: number;
      rotationY: number;
      /** [0,1] local visual density hint (e.g. for grass blade count
       * scaling) - continuous, not a presence gate (that is `present`). */
      density: number;
    };

/** Retained habitat decision. It contains no date or mutable simulation state.
 * Cache only while the seed, address, direction, biome and gate are unchanged. */
export type FloraRecipe = {present:false} | (Omit<Extract<FloraInstance,{present:true}>, 'scale'> & {baseScale:number;growthPhase:number});

const MAX_SLOPE_DEG = 34;
/** Fraction of the sterile 'unformed'/'forming'/'sterile' stages' flora, and
 * of 'microbial' relative to 'complex': 0 for the first three (no plants at
 * all), and a small non-zero mat density for 'microbial' (lichen/algae mats
 * only, never trees/shrubs/flowers). */
const STAGE_DENSITY_MULTIPLIER: Record<FloraStage, number> = {
  unformed: 0,
  forming: 0,
  sterile: 0,
  microbial: 0.08,
  complex: 1,
};

function hashU32(value: number, seed: number): number {
  let h = (value | 0) ^ (seed | 0);
  h = Math.imul(h ^ (h >>> 16), 0x7feb352d);
  h = Math.imul(h ^ (h >>> 15), 0x846ca68b);
  return (h ^ (h >>> 16)) >>> 0;
}
function hash32(value: string): number {
  let h = 2166136261 >>> 0;
  for (let i = 0; i < value.length; i++) {
    h ^= value.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  h ^= h >>> 16;
  h = Math.imul(h, 0x7feb352d);
  h ^= h >>> 15;
  h = Math.imul(h, 0x846ca68b);
  return (h ^ (h >>> 16)) >>> 0;
}
function random01(key: string): number {
  return hash32(key) / 0x1_0000_0000;
}
function directionKey(seed: string, bodyAddress: string, direction: Vec3, channel: string): string {
  return `${seed}|${bodyAddress}|flora|${direction[0]},${direction[1]},${direction[2]}|${channel}`;
}

/**
 * Deterministic planet-wide flora placement for a single candidate point.
 * The caller supplies `direction` (a unit vector on the body's surface
 * sphere, or any other stable per-candidate coordinate from its own
 * placement lattice - `floraAt` treats it as an opaque hash key component,
 * so the same direction always yields the same result on the same body,
 * independent of visit order or what neighbouring candidates were queried).
 *
 * `tMyr`: millions of years since the atlas epoch, matching
 * biosphere-model.ts's `dateMyr`. Only a continuous scale pulse depends on
 * it; every discrete choice (presence/kind) does not, so direct seeks and
 * stepped seeks agree exactly (see module header).
 */
export function floraAt(
  seed: string,
  bodyAddress: string,
  direction: Vec3,
  biome: FloraBiomeSample,
  gate: FloraBiosphereGate,
  tMyr: Myr,
): FloraInstance {
  // Loud on a wrong-unit value even if the TS brand was bypassed (a .mjs
  // caller, a dynamically-typed boundary, an `as Myr` cast) - see the Myr
  // section above (FINDING 2 fix).
  toMyr(tMyr);
  return floraFromRecipe(prepareFlora(seed,bodyAddress,direction,biome,gate),tMyr);
}

export function prepareFlora(seed:string,bodyAddress:string,direction:Vec3,biome:FloraBiomeSample,gate:FloraBiosphereGate):FloraRecipe {
  const stageMultiplier = STAGE_DENSITY_MULTIPLIER[gate.stage] ?? 0;
  if (stageMultiplier <= 0) return { present: false };
  if (biome.ocean) return { present: false };
  if (biome.slopeDeg >= MAX_SLOPE_DEG) return { present: false };

  const groundWeight = clamp01(biome.weights[1] ?? 0);
  if (groundWeight <= 0) return { present: false };

  const density = clamp01(groundWeight * stageMultiplier);
  const presenceRoll = random01(directionKey(seed, bodyAddress, direction, 'presence'));
  if (presenceRoll >= density) return { present: false };

  let candidateKinds: readonly FloraKind[] =
    gate.stage === 'microbial' ? ['grass'] : FLORA_KIND_NAMES;
  if (gate.availableKinds) {
    const allowed = new Set(gate.availableKinds);
    candidateKinds = candidateKinds.filter((kind) => allowed.has(kind));
  }
  if (candidateKinds.length === 0) return { present: false };

  const kind = pickKind(candidateKinds, biome, seed, bodyAddress, direction);
  const instanceSeed = hashU32(hash32(directionKey(seed, bodyAddress, direction, 'seed')), 0x666c6f72);
  const rotationY = random01(directionKey(seed, bodyAddress, direction, 'rotation')) * Math.PI * 2;
  const baseScale = 0.72 + random01(directionKey(seed, bodyAddress, direction, 'scale')) * 0.56;
  const growthPhase=random01(directionKey(seed,bodyAddress,direction,'growth'))*Math.PI*2;
  return { present: true, kind, instanceSeed, baseScale, growthPhase, rotationY, density };
}

/** Closed-form bounded growth [0.92,1.08], never a presence/kind decision.
 * Date seeks do not accumulate or mutate the retained habitat recipe. */
export function floraFromRecipe(recipe:FloraRecipe,tMyr:Myr):FloraInstance {
  toMyr(tMyr);
  if(!recipe.present)return {present:false};
  const {kind,instanceSeed,rotationY,density,baseScale,growthPhase}=recipe;
  const scale=baseScale*(1+.08*Math.sin(tMyr*.0005+growthPhase));
  return {present:true,kind,instanceSeed,scale,rotationY,density};
}

/** Weighted discrete pick over `candidateKinds`, using only the integer hash
 * (no transcendental calls) so it is immune to ulp-level Math.* drift. */
function pickKind(
  candidateKinds: readonly FloraKind[],
  biome: FloraBiomeSample,
  seed: string,
  bodyAddress: string,
  direction: Vec3,
): FloraKind {
  const baseWeight: Record<FloraKind, number> = {
    tree: biome.forestPotential ? 0.34 : 0.05,
    shrub: biome.forestPotential ? 0.22 : 0.12,
    grass: 0.55,
    flower: biome.moisture > 0.5 ? 0.18 : 0.06,
  };
  let total = 0;
  for (const kind of candidateKinds) total += baseWeight[kind];
  if (total <= 0) return candidateKinds[0]!;
  const roll = random01(directionKey(seed, bodyAddress, direction, 'kind')) * total;
  let accumulated = 0;
  for (const kind of candidateKinds) {
    accumulated += baseWeight[kind];
    if (roll < accumulated) return kind;
  }
  return candidateKinds[candidateKinds.length - 1]!;
}

function clamp01(value: number): number {
  return Math.max(0, Math.min(1, value));
}

// ---------------------------------------------------------------------------
// Space-colonization tree growth.
// Ported from nms/shared/proceduralTree.ts (growth algorithm unchanged).
// ---------------------------------------------------------------------------

export const PROCEDURAL_TREE_GROWTH_VERSION = 1;
export const PROCEDURAL_TREE_DEFAULT_SEED = 0x4e4d5388;
export const PROCEDURAL_TREE_DEFAULT_ATTRACTION_POINTS = 96;
export const PROCEDURAL_TREE_DEFAULT_MAX_ITERATIONS = 20;
export const PROCEDURAL_TREE_DEFAULT_STEP_M = 0.17;
export const PROCEDURAL_TREE_DEFAULT_INFLUENCE_RADIUS_M = 0.74;
export const PROCEDURAL_TREE_DEFAULT_KILL_DISTANCE_M = 0.16;

export type ProceduralTreeGrowthPreset = 'broadleaf' | 'windswept' | 'spire';

export interface ProceduralTreeGrowthOptions {
  preset?: ProceduralTreeGrowthPreset;
  seed?: number;
  attractionPointCount?: number;
  maxIterations?: number;
  stepM?: number;
  influenceRadiusM?: number;
  killDistanceM?: number;
}

export const PROCEDURAL_TREE_GROWTH_PRESETS: readonly ProceduralTreeGrowthPreset[] = [
  'broadleaf',
  'windswept',
  'spire',
];

export interface ProceduralTreeNode {
  x: number;
  y: number;
  z: number;
  parentIndex: number;
  depth: number;
  radius: number;
  childCount: number;
}
export interface ProceduralTreeAttractionPoint {
  x: number;
  y: number;
  z: number;
  consumed: boolean;
}
export interface ProceduralTreeSkeleton {
  algorithm: 'space-colonization';
  version: number;
  preset: ProceduralTreeGrowthPreset;
  seed: number;
  attractionPointCount: number;
  consumedAttractionPointCount: number;
  remainingAttractionPointCount: number;
  iterationCount: number;
  nodeCount: number;
  branchSegmentCount: number;
  terminalNodeCount: number;
  maxDepth: number;
  heightM: number;
  canopyRadiusM: number;
  nodes: ProceduralTreeNode[];
  signature: string;
  signatureValue: number;
}

interface MutableProceduralTreeNode {
  x: number; y: number; z: number; parentIndex: number; depth: number; radius: number; childCount: number;
}
interface GrowthAccumulator { x: number; y: number; z: number; count: number; }
interface ProceduralTreeGrowthProfile {
  preset: ProceduralTreeGrowthPreset; canopyBaseY: number; canopyHeightM: number; layerPower: number;
  envelopePower: number; envelopeScale: number; envelopeTopTaper: number; radiusX: number; radiusZ: number;
  leanNoise: number; leanBiasX: number; leanZScale: number; trunkLeanX: number; trunkLeanZ: number;
  pullXScale: number; pullZScale: number; upwardPull: number; averageWeight: number; pullWeight: number;
  noiseRadius: number; noiseYawScale: number;
}

const scratch = new ArrayBuffer(4);
const scratchView = new DataView(scratch);

/**
 * Two-class math routing for this growth loop (fix for the adversarial
 * review's FINDING 3 - docs/time-and-coordinates-contract.md §1.2 rule 3,
 * "discrete, branch-affecting decisions... must go through the deterministic
 * fixed-point/det-math path"):
 *  - The loop's own threshold comparisons (`distanceSq <= killDistanceSq`,
 *    `distanceSq <= influenceRadiusSq`, `accumulators.size === 0`,
 *    `nearestNodeDistanceSq(...) < stepM*stepM*0.18`, `addedThisIteration
 *    === 0`) decide which attraction points are consumed, which node gets a
 *    new child, and when growth stops - i.e. topology and node COUNT. They
 *    are themselves pure +,-,*,/ (IEEE-754 correctly-rounded, portable per
 *    det-math.ts's header) - no transcendental call needed there directly.
 *  - But every position those comparisons are computed FROM - attraction
 *    points (`createCanopyAttractionPoints`, below) and each new node's
 *    direction (`normalize`, the per-iteration noise-yaw `cos`/`sin`) - used
 *    to be built with platform `Math.cos`/`Math.sin`/`Math.pow`/`Math.hypot`,
 *    which are only implementation-defined-precision on real engines
 *    (det-math.ts's own measured cross-engine audit: up to ~49.6% of results
 *    differ bitwise). A future engine's ULP-different position could in
 *    principle flip one of the comparisons above and change topology - the
 *    adversarial review found 0/180 topology flips in its own sample, but
 *    "gated by nothing today" is exactly the bug: margin isn't a guarantee.
 *    Both sites are now routed through det-math.ts's `detCos`/`sine`/
 *    `detPow`/`hypotDet`, which never call the drifting Math.* primitives
 *    internally (see det-math.ts), so this is provably deterministic rather
 *    than empirically lucky.
 *  - `assignChildCountsAndRadii`'s `Math.pow` (node visual radius) and the
 *    post-loop `Math.hypot` computing `canopyRadiusM` below both run AFTER
 *    topology/positions are already finalized and feed only rendering/
 *    reporting output (radius, a reported bounding stat) that no comparison
 *    anywhere reads back into a survival/count/topology decision - see each
 *    call site's own comment for why it is presentation-only and was
 *    deliberately left on platform Math.
 */
export function generateProceduralTreeSkeleton(options: ProceduralTreeGrowthOptions = {}): ProceduralTreeSkeleton {
  const profile = resolveGrowthProfile(options.preset ?? 'broadleaf');
  const seed = options.seed ?? PROCEDURAL_TREE_DEFAULT_SEED;
  const attractionPointCount = clampInt(options.attractionPointCount ?? PROCEDURAL_TREE_DEFAULT_ATTRACTION_POINTS, 8, 512);
  const maxIterations = clampInt(options.maxIterations ?? PROCEDURAL_TREE_DEFAULT_MAX_ITERATIONS, 1, 64);
  const stepM = Math.fround(clamp(options.stepM ?? PROCEDURAL_TREE_DEFAULT_STEP_M, 0.05, 0.6));
  const influenceRadiusM = Math.fround(clamp(options.influenceRadiusM ?? PROCEDURAL_TREE_DEFAULT_INFLUENCE_RADIUS_M, 0.2, 2));
  const killDistanceM = Math.fround(clamp(options.killDistanceM ?? PROCEDURAL_TREE_DEFAULT_KILL_DISTANCE_M, 0.04, influenceRadiusM * 0.75));
  const influenceRadiusSq = influenceRadiusM * influenceRadiusM;
  const killDistanceSq = killDistanceM * killDistanceM;

  const attractionPoints = createCanopyAttractionPoints(seed, attractionPointCount, profile);
  const nodes = createInitialTrunkNodes(profile);

  let consumedAttractionPointCount = 0;
  let iterationCount = 0;
  for (; iterationCount < maxIterations; iterationCount++) {
    const accumulators = new Map<number, GrowthAccumulator>();
    for (const point of attractionPoints) {
      if (point.consumed) continue;
      let nearestIndex = -1;
      let nearestDistanceSq = Number.POSITIVE_INFINITY;
      for (let nodeIndex = 0; nodeIndex < nodes.length; nodeIndex++) {
        const node = nodes[nodeIndex];
        if (!node) continue;
        const dx = point.x - node.x, dy = point.y - node.y, dz = point.z - node.z;
        const distanceSq = dx * dx + dy * dy + dz * dz;
        if (distanceSq <= killDistanceSq) {
          point.consumed = true;
          consumedAttractionPointCount++;
          nearestIndex = -1;
          break;
        }
        if (distanceSq <= influenceRadiusSq && distanceSq < nearestDistanceSq) {
          nearestDistanceSq = distanceSq;
          nearestIndex = nodeIndex;
        }
      }
      if (nearestIndex < 0 || point.consumed) continue;
      const node = nodes[nearestIndex];
      if (!node) continue;
      const direction = normalize(point.x - node.x, point.y - node.y, point.z - node.z);
      const accumulator = accumulators.get(nearestIndex) ?? { x: 0, y: 0, z: 0, count: 0 };
      accumulator.x += direction[0]; accumulator.y += direction[1]; accumulator.z += direction[2]; accumulator.count++;
      accumulators.set(nearestIndex, accumulator);
    }
    if (accumulators.size === 0) break;
    let addedThisIteration = 0;
    for (const [parentIndex, accumulator] of accumulators) {
      const parent = nodes[parentIndex];
      if (!parent || accumulator.count <= 0) continue;
      const avg = normalize(accumulator.x / accumulator.count, accumulator.y / accumulator.count, accumulator.z / accumulator.count);
      const canopyPull = normalize(
        (profile.leanBiasX - parent.x) * profile.pullXScale,
        profile.upwardPull,
        -parent.z * profile.pullZScale,
      );
      const noiseYaw = randSigned(seed, 8000 + iterationCount * 257 + parentIndex * 17) * profile.noiseYawScale;
      // Decision-feeding (see the two-class routing note above this
      // function): this direction sets `next`'s position, which the
      // kill-distance/influence-radius/nearestNodeDistanceSq thresholds
      // compare against next iteration - det-math, not platform Math.
      const direction = normalize(
        avg[0] * profile.averageWeight + canopyPull[0] * profile.pullWeight + detCos(noiseYaw) * profile.noiseRadius,
        avg[1] * profile.averageWeight + canopyPull[1] * profile.pullWeight,
        avg[2] * profile.averageWeight + canopyPull[2] * profile.pullWeight + sine(noiseYaw) * profile.noiseRadius,
      );
      const stepJitter = 0.92 + rand01(seed, 9000 + iterationCount * 193 + parentIndex * 29) * 0.16;
      const next = createNode(
        parent.x + direction[0] * stepM * stepJitter,
        parent.y + direction[1] * stepM * stepJitter,
        parent.z + direction[2] * stepM * stepJitter,
        parentIndex, parent.depth + 1,
      );
      if (nearestNodeDistanceSq(nodes, next.x, next.y, next.z) < stepM * stepM * 0.18) continue;
      nodes.push(next);
      addedThisIteration++;
    }
    if (addedThisIteration === 0) break;
  }

  assignChildCountsAndRadii(nodes);
  const immutableNodes: ProceduralTreeNode[] = nodes.map((node) => ({
    x: Math.fround(node.x), y: Math.fround(node.y), z: Math.fround(node.z),
    parentIndex: node.parentIndex, depth: node.depth, radius: Math.fround(node.radius), childCount: node.childCount,
  }));
  const terminalNodeCount = immutableNodes.filter((node) => node.parentIndex >= 0 && node.childCount === 0).length;
  const maxDepth = immutableNodes.reduce((max, node) => Math.max(max, node.depth), 0);
  const heightM = immutableNodes.reduce((max, node) => Math.max(max, node.y), 0);
  // Presentation/reporting-only: a post-hoc aggregate over the ALREADY
  // finalized `immutableNodes` array (topology and positions are fully
  // decided by this point in the function) - nothing reads `canopyRadiusM`
  // back into a survival/count/topology decision, so platform Math.hypot is
  // fine here (see the two-class routing note above `generateProceduralTreeSkeleton`).
  const canopyRadiusM = immutableNodes.reduce((max, node) => Math.max(max, Math.hypot(node.x, node.z)), 0);
  const signatureValue = hashSkeleton(seed, immutableNodes, attractionPoints, consumedAttractionPointCount, iterationCount);

  return {
    algorithm: 'space-colonization', version: PROCEDURAL_TREE_GROWTH_VERSION, preset: profile.preset, seed,
    attractionPointCount, consumedAttractionPointCount,
    remainingAttractionPointCount: attractionPointCount - consumedAttractionPointCount,
    iterationCount, nodeCount: immutableNodes.length, branchSegmentCount: Math.max(0, immutableNodes.length - 1),
    terminalNodeCount, maxDepth, heightM: Math.fround(heightM), canopyRadiusM: Math.fround(canopyRadiusM),
    nodes: immutableNodes,
    signature: `procedural-tree-growth-v${PROCEDURAL_TREE_GROWTH_VERSION}:${hex32(signatureValue)}`,
    signatureValue,
  };
}

/**
 * Attraction-point placement. Decision-feeding, not presentation (see the
 * two-class routing note above `generateProceduralTreeSkeleton`): every
 * point built here is a candidate the growth loop's kill-distance/
 * influence-radius comparisons test each iteration, so `layer`/`envelope`/
 * `x`/`z` below - and therefore which points get consumed and how many
 * nodes eventually grow - are downstream of these values. `Math.sqrt` (for
 * `radial`) stays platform Math because it is IEEE-754 correctly-rounded
 * and portable (det-math.ts's header); `Math.pow`/`Math.sin`/`Math.cos` do
 * not have that guarantee and are routed through `detPow`/`sine`/`detCos`.
 */
function createCanopyAttractionPoints(seed: number, count: number, profile: ProceduralTreeGrowthProfile): { x: number; y: number; z: number; consumed: boolean }[] {
  const points: { x: number; y: number; z: number; consumed: boolean }[] = [];
  for (let i = 0; i < count; i++) {
    const yRand = rand01(seed, 1000 + i * 13);
    const layer = detPow(yRand, profile.layerPower);
    const y = profile.canopyBaseY + layer * profile.canopyHeightM;
    const envelope = detPow(sine(Math.PI * clamp(layer, 0.02, 0.98)), profile.envelopePower) * (profile.envelopeScale - layer * profile.envelopeTopTaper);
    const theta = i * 2.399963229728653 + randSigned(seed, 2000 + i * 17) * 0.34;
    const radial = Math.sqrt(rand01(seed, 3000 + i * 19)) * envelope;
    const lean = profile.leanBiasX * layer + randSigned(seed, 4000 + i * 23) * profile.leanNoise;
    points.push({
      x: Math.fround(detCos(theta) * radial * profile.radiusX + lean),
      y: Math.fround(y),
      z: Math.fround(sine(theta) * radial * profile.radiusZ + lean * profile.leanZScale),
      consumed: false,
    });
  }
  return points;
}

function resolveGrowthProfile(preset: ProceduralTreeGrowthPreset): ProceduralTreeGrowthProfile {
  if (preset === 'windswept') {
    return { preset, canopyBaseY: 1.08, canopyHeightM: 1.72, layerPower: 0.74, envelopePower: 0.52, envelopeScale: 1.06, envelopeTopTaper: 0.18, radiusX: 1.5, radiusZ: 0.74, leanNoise: 0.14, leanBiasX: -0.36, leanZScale: -0.22, trunkLeanX: -0.18, trunkLeanZ: 0.035, pullXScale: 0.08, pullZScale: 0.16, upwardPull: 0.18, averageWeight: 0.82, pullWeight: 0.18, noiseRadius: 0.04, noiseYawScale: 0.2 };
  }
  if (preset === 'spire') {
    return { preset, canopyBaseY: 1.22, canopyHeightM: 2.15, layerPower: 0.64, envelopePower: 0.82, envelopeScale: 0.76, envelopeTopTaper: 0.36, radiusX: 0.68, radiusZ: 0.66, leanNoise: 0.045, leanBiasX: 0.08, leanZScale: -0.18, trunkLeanX: 0.055, trunkLeanZ: -0.025, pullXScale: 0.1, pullZScale: 0.1, upwardPull: 0.34, averageWeight: 0.78, pullWeight: 0.22, noiseRadius: 0.018, noiseYawScale: 0.08 };
  }
  return { preset: 'broadleaf', canopyBaseY: 1.18, canopyHeightM: 1.78, layerPower: 0.82, envelopePower: 0.48, envelopeScale: 1.08, envelopeTopTaper: 0.22, radiusX: 1.12, radiusZ: 0.92, leanNoise: 0.08, leanBiasX: 0, leanZScale: -0.4, trunkLeanX: 0, trunkLeanZ: 0, pullXScale: 0.12, pullZScale: 0.12, upwardPull: 0.22, averageWeight: 0.86, pullWeight: 0.14, noiseRadius: 0.025, noiseYawScale: 0.1 };
}

function createInitialTrunkNodes(profile: ProceduralTreeGrowthProfile): MutableProceduralTreeNode[] {
  return [
    createNode(0, 0, 0, -1, 0),
    createNode(0.012 + profile.trunkLeanX * 0.16, 0.32, -0.004 + profile.trunkLeanZ * 0.16, 0, 1),
    createNode(0.026 + profile.trunkLeanX * 0.36, 0.66, 0.012 + profile.trunkLeanZ * 0.36, 1, 2),
    createNode(0.018 + profile.trunkLeanX * 0.64, 1.01, 0.034 + profile.trunkLeanZ * 0.64, 2, 3),
    createNode(0.05 + profile.trunkLeanX, 1.28, 0.02 + profile.trunkLeanZ, 3, 4),
  ];
}
function createNode(x: number, y: number, z: number, parentIndex: number, depth: number): MutableProceduralTreeNode {
  return { x: Math.fround(x), y: Math.fround(y), z: Math.fround(z), parentIndex, depth, radius: 0.012, childCount: 0 };
}
function assignChildCountsAndRadii(nodes: MutableProceduralTreeNode[]): void {
  const weights = new Array<number>(nodes.length).fill(1);
  for (const node of nodes) node.childCount = 0;
  for (let i = nodes.length - 1; i >= 1; i--) {
    const node = nodes[i]; if (!node) continue;
    const parent = nodes[node.parentIndex]; if (!parent) continue;
    parent.childCount++;
    weights[node.parentIndex] = (weights[node.parentIndex] ?? 1) + (weights[i] ?? 1) * 0.72;
  }
  const maxY = nodes.reduce((max, node) => Math.max(max, node.y), 0) || 1;
  const rootWeight = weights[0] || 1;
  for (let i = 0; i < nodes.length; i++) {
    const node = nodes[i]; if (!node) continue;
    const heightT = clamp(node.y / maxY, 0, 1);
    // Presentation-only: `node.radius` is a visual taper computed AFTER the
    // growth loop has already finished deciding topology (parentIndex/
    // childCount come from the finalized `nodes` array, not from radius),
    // and nothing compares radius against a threshold - see the two-class
    // routing note above `generateProceduralTreeSkeleton`. Platform Math.pow
    // is fine here.
    const weightT = Math.pow((weights[i] ?? 1) / rootWeight, 0.42);
    const terminalBoost = node.childCount === 0 ? 0.72 : 1;
    node.radius = Math.fround(clamp(0.17 * weightT * (1 - heightT * 0.46) * terminalBoost, 0.012, 0.18));
  }
}
function nearestNodeDistanceSq(nodes: MutableProceduralTreeNode[], x: number, y: number, z: number): number {
  let nearest = Number.POSITIVE_INFINITY;
  for (const node of nodes) {
    const dx = x - node.x, dy = y - node.y, dz = z - node.z;
    nearest = Math.min(nearest, dx * dx + dy * dy + dz * dz);
  }
  return nearest;
}
function hashSkeleton(seed: number, nodes: ProceduralTreeNode[], attractionPoints: { x: number; y: number; z: number; consumed: boolean }[], consumedAttractionPointCount: number, iterationCount: number): number {
  let signature = hashU32(0x74726565, PROCEDURAL_TREE_GROWTH_VERSION);
  signature = mix(signature, seed);
  signature = mix(signature, attractionPoints.length);
  signature = mix(signature, consumedAttractionPointCount);
  signature = mix(signature, iterationCount);
  signature = mix(signature, nodes.length);
  for (const node of nodes) {
    signature = mix(signature, node.parentIndex);
    signature = mix(signature, node.depth);
    signature = mix(signature, node.childCount);
    signature = mix(signature, f32Bits(node.x));
    signature = mix(signature, f32Bits(node.y));
    signature = mix(signature, f32Bits(node.z));
    signature = mix(signature, f32Bits(node.radius));
  }
  for (const point of attractionPoints) {
    signature = mix(signature, point.consumed ? 1 : 0);
    signature = mix(signature, f32Bits(point.x));
    signature = mix(signature, f32Bits(point.y));
    signature = mix(signature, f32Bits(point.z));
  }
  return signature;
}
function rand01(seed: number, salt: number): number { return ((hashU32(salt, seed) >>> 8) & 0x00ffffff) / 0x01000000; }
function randSigned(seed: number, salt: number): number { return rand01(seed, salt) * 2 - 1; }
/**
 * Unit vector from (x,y,z) - used ONLY inside the growth loop above
 * (point-to-attractor direction, averaged pull direction, canopy pull),
 * whose result becomes a new node's position and therefore feeds the
 * kill-distance/influence-radius/nearestNodeDistanceSq threshold
 * comparisons that decide topology - see the two-class routing note above
 * `generateProceduralTreeSkeleton`. Uses `hypotDet` (Math.sqrt + a portable
 * sum-of-squares), never `Math.hypot`. Distinct from `normalize3` below
 * (geometry builders, presentation-only - see its own comment).
 */
function normalize(x: number, y: number, z: number): [number, number, number] { const length = hypotDet(x, y, z) || 1; return [x / length, y / length, z / length]; }
function clamp(value: number, min: number, max: number): number { return Math.max(min, Math.min(max, value)); }
function clampInt(value: number, min: number, max: number): number { return Math.max(min, Math.min(max, Math.floor(value))); }
function mix(acc: number, value: number): number { return hashU32(value | 0, acc ^ 0x9e3779b9); }
function f32Bits(value: number): number { scratchView.setFloat32(0, Math.fround(value), true); return scratchView.getUint32(0, true); }
function hex32(value: number): string { return (value >>> 0).toString(16).padStart(8, '0'); }

// ---------------------------------------------------------------------------
// LOD'd geometry builders (tree/flower/grass/shrub).
// Ported from nms/shared/heroContent.ts (buildTreeArchetype and friends).
// ---------------------------------------------------------------------------

export interface FloraGeometryData {
  positions: Float32Array;
  normals: Float32Array;
  uvs: Float32Array;
  colors: Float32Array;
  windStiffness: Float32Array;
  indices: Uint16Array;
  triangleCount: number;
}

interface PlantGeometryBuilder {
  positions: number[]; normals: number[]; uvs: number[]; colors: number[]; windStiffness: number[]; indices: number[];
}
type PlantVec3 = [number, number, number];
type PlantColor = [number, number, number];

function createPlantGeometryBuilder(): PlantGeometryBuilder {
  return { positions: [], normals: [], uvs: [], colors: [], windStiffness: [], indices: [] };
}

/** Entry point: builds one LOD (0=near, 1=mid, 2=far) of one flora kind. */
export function buildFloraGeometryData(kind: FloraKind, lod: number, seed: number): FloraGeometryData {
  const clampedLod = clampInt(lod, 0, FLORA_LOD_COUNT - 1);
  const builder = createPlantGeometryBuilder();
  if (kind === 'tree') buildTreeArchetype(builder, clampedLod, seed);
  else if (kind === 'flower') buildFlowerArchetype(builder, clampedLod, seed);
  else if (kind === 'shrub') buildShrubArchetype(builder, clampedLod, seed);
  else buildGrassArchetype(builder, clampedLod, seed);
  return finishBuilder(builder);
}

function finishBuilder(builder: PlantGeometryBuilder): FloraGeometryData {
  const indices = new Uint16Array(builder.indices);
  return {
    positions: new Float32Array(builder.positions),
    normals: new Float32Array(builder.normals),
    uvs: new Float32Array(builder.uvs),
    colors: new Float32Array(builder.colors),
    windStiffness: new Float32Array(builder.windStiffness),
    indices,
    triangleCount: indices.length / 3,
  };
}

/** Convenience: pure data -> a ready-to-use THREE.BufferGeometry. Optional -
 * every builder above works with no GL context, so tests never need this. */
export function floraGeometryToBufferGeometry(data: FloraGeometryData): THREE.BufferGeometry {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(data.positions, 3));
  geometry.setAttribute('normal', new THREE.BufferAttribute(data.normals, 3));
  geometry.setAttribute('uv', new THREE.BufferAttribute(data.uvs, 2));
  geometry.setAttribute('color', new THREE.BufferAttribute(data.colors, 3));
  geometry.setAttribute('windStiffness', new THREE.BufferAttribute(data.windStiffness, 1));
  geometry.setIndex(new THREE.BufferAttribute(data.indices, 1));
  return geometry;
}

/**
 * New (not in nms): turns a grown `ProceduralTreeSkeleton` into a "hero" LOD0
 * tree mesh, walking the parent-indexed branch graph and reusing the ported
 * `addTaperedBranch`/leaf-cluster helpers below instead of the fixed
 * parametric silhouette `buildTreeArchetype` draws. Leaves are placed at
 * terminal (childless) nodes and at internal nodes below a shallow depth so
 * young interior branches still read as foliated.
 */
export function growthSkeletonToGeometryData(skeleton: ProceduralTreeSkeleton, seed = skeleton.seed): FloraGeometryData {
  const builder = createPlantGeometryBuilder();
  const bark: PlantColor = [0.55, 0.35, 0.19];
  for (let i = 1; i < skeleton.nodes.length; i++) {
    const node = skeleton.nodes[i]!;
    const parent = skeleton.nodes[node.parentIndex];
    if (!parent) continue;
    const start: PlantVec3 = [parent.x, parent.y, parent.z];
    const end: PlantVec3 = [node.x, node.y, node.z];
    addTaperedBranch(builder, start, end, Math.max(0.01, parent.radius), Math.max(0.008, node.radius), node.childCount > 2 ? 6 : 5, bark, 0.1, node.childCount === 0 ? 0.9 : 0.4);
    if (node.childCount === 0) {
      addLeafVolumeCluster(builder, end, rand01(seed, 5000 + i) * Math.PI * 2, 0.13 + rand01(seed, 5100 + i) * 0.05, seed, 5200 + i, 1);
    }
  }
  addRootFlare(builder, Math.max(0.02, skeleton.nodes[0]?.radius ?? 0.05), bark);
  return finishBuilder(builder);
}

function buildTreeArchetype(builder: PlantGeometryBuilder, lod: number, seed: number): void {
  if (lod >= 2) {
    const bark: PlantColor = [0.55, 0.35, 0.19];
    const height = Math.fround(1.36 + rand01(seed, 805) * 0.12);
    const trunkTop: PlantVec3 = [randSigned(seed, 806) * 0.04, height * 0.72, randSigned(seed, 807) * 0.04];
    addTaperedBranch(builder, [0, 0, 0], trunkTop, 0.105, 0.048, 4, bark, 0, 0.38);
    addRootFlare(builder, 0.105, bark);
    for (let i = 0; i < 3; i++) {
      const yaw = (i / 3) * Math.PI * 2 + randSigned(seed, 812 + i) * 0.34;
      const start = lerpVec3([0, 0, 0], trunkTop, 0.46 + i * 0.12);
      const reach = 0.31 + rand01(seed, 822 + i) * 0.08;
      const end: PlantVec3 = [Math.fround(start[0] + Math.cos(yaw) * reach), Math.fround(start[1] + 0.12 + i * 0.035), Math.fround(start[2] + Math.sin(yaw) * reach)];
      addTaperedBranch(builder, start, end, 0.035 - i * 0.003, 0.014, 3, bark, 0.3, 0.78);
      addLeafVolumeCluster(builder, end, yaw, 0.145 - i * 0.005, seed, 820 + i, 1);
    }
    const leaderEnd: PlantVec3 = [Math.fround(trunkTop[0] * 1.1), height, Math.fround(trunkTop[2] * 1.1)];
    addTaperedBranch(builder, trunkTop, leaderEnd, 0.035, 0.013, 3, bark, 0.42, 0.86);
    addLeafVolumeCluster(builder, leaderEnd, rand01(seed, 850) * Math.PI * 2, 0.15, seed, 850, 1);
    return;
  }
  if (lod === 1) {
    const height = Math.fround(1.56 + rand01(seed, 610) * 0.16);
    const trunkTop: PlantVec3 = [randSigned(seed, 611) * 0.045, height * 0.74, randSigned(seed, 612) * 0.045];
    const bark: PlantColor = [0.55, 0.35, 0.19];
    addTaperedBranch(builder, [0, 0, 0], trunkTop, 0.12, 0.058, 5, bark, 0, 0.34);
    addRootFlare(builder, 0.12, bark);
    for (let i = 0; i < 3; i++) {
      const yaw = (i / 3) * Math.PI * 2 + randSigned(seed, 620 + i) * 0.36;
      const start = lerpVec3([0, 0, 0], trunkTop, 0.46 + i * 0.13);
      const end: PlantVec3 = [Math.fround(start[0] + Math.cos(yaw) * (0.40 + i * 0.045)), Math.fround(start[1] + 0.18 + i * 0.04), Math.fround(start[2] + Math.sin(yaw) * (0.40 + i * 0.045))];
      addTaperedBranch(builder, start, end, 0.045 - i * 0.004, 0.018, 3, bark, 0.34, 0.76);
      addLeafVolumeCluster(builder, end, yaw, 0.155 + i * 0.006, seed, 650 + i, 1);
    }
    const leaderEnd: PlantVec3 = [trunkTop[0] * 1.18, height, trunkTop[2] * 1.18];
    addTaperedBranch(builder, trunkTop, leaderEnd, 0.042, 0.016, 3, bark, 0.42, 0.84);
    addLeafVolumeCluster(builder, leaderEnd, rand01(seed, 675) * Math.PI * 2, 0.17, seed, 675, 1);
    return;
  }
  const height = Math.fround(1.72 + rand01(seed, 310 + lod) * 0.28);
  const trunkHeight = Math.fround(height * 0.70);
  const trunkRadius = Math.fround(0.13 + rand01(seed, 320 + lod) * 0.026);
  const trunkTop: PlantVec3 = [Math.fround((rand01(seed, 324 + lod) - 0.5) * 0.10), trunkHeight, Math.fround((rand01(seed, 325 + lod) - 0.5) * 0.10)];
  const bark: PlantColor = [0.55, 0.35, 0.19];
  const trunkBend: PlantVec3 = [Math.fround(trunkTop[0] * 0.26 + randSigned(seed, 326 + lod) * 0.035), Math.fround(trunkTop[1] * 0.48), Math.fround(trunkTop[2] * 0.26 + randSigned(seed, 327 + lod) * 0.035)];
  addCurvedBranch(builder, [0, 0, 0], trunkBend, trunkTop, trunkRadius, trunkRadius * 0.54, 5, 3, bark, 0, 0.32);
  addRootFlare(builder, trunkRadius, bark);
  const lateralCount = lod === 0 ? 5 : 3;
  for (let i = 0; i < lateralCount; i++) {
    const t = lateralCount <= 1 ? 0.62 : i / (lateralCount - 1);
    const startT = Math.fround(0.36 + t * 0.48);
    const start = lerpVec3([0, 0, 0], trunkTop, startT);
    const yaw = (i / lateralCount) * Math.PI * 2 + randSigned(seed, 330 + i + lod * 20) * 0.48;
    const length = Math.fround((lod === 0 ? 0.56 : 0.45) + rand01(seed, 340 + i + lod * 20) * 0.18);
    const lift = Math.fround(0.18 + rand01(seed, 350 + i + lod * 20) * 0.20);
    const elbow: PlantVec3 = [Math.fround(start[0] + Math.cos(yaw + randSigned(seed, 341 + i + lod * 20) * 0.22) * length * 0.48), Math.fround(start[1] + lift * 0.42), Math.fround(start[2] + Math.sin(yaw + randSigned(seed, 342 + i + lod * 20) * 0.22) * length * 0.48)];
    const end: PlantVec3 = [Math.fround(start[0] + Math.cos(yaw) * length), Math.fround(start[1] + lift), Math.fround(start[2] + Math.sin(yaw) * length)];
    const branchRadius = Math.fround(trunkRadius * (0.38 - t * 0.08));
    addCurvedBranch(builder, start, elbow, end, branchRadius, Math.max(0.018, branchRadius * 0.32), 3, 2, bark, 0.16 + t * 0.24, 0.72);
    addLeafVolumeCluster(builder, end, yaw, lod === 0 ? 0.22 : 0.19, seed, 360 + i + lod * 20, lod === 0 ? 2 : 1);
    if (lod === 0) {
      const twigStart = lerpVec3(start, end, 0.54 + rand01(seed, 430 + i) * 0.2);
      const twigYaw = yaw + randSigned(seed, 440 + i) * 0.72;
      const twigLength = Math.fround(length * (0.34 + rand01(seed, 450 + i) * 0.12));
      const twigEnd: PlantVec3 = [Math.fround(twigStart[0] + Math.cos(twigYaw) * twigLength), Math.fround(twigStart[1] + 0.11 + rand01(seed, 460 + i) * 0.1), Math.fround(twigStart[2] + Math.sin(twigYaw) * twigLength)];
      addTaperedBranch(builder, twigStart, twigEnd, branchRadius * 0.48, Math.max(0.012, branchRadius * 0.18), 3, bark, 0.42, 0.88);
      addLeafVolumeCluster(builder, twigEnd, twigYaw, 0.115 + rand01(seed, 470 + i) * 0.028, seed, 480 + i, 1);
      addFlowerFoldedLeaf(builder, lerpVec3(twigStart, twigEnd, 0.56), twigYaw + 0.72, 0.14, 0.04, [0.36, 0.72, 0.30]);
    }
  }
  const leaderEnd: PlantVec3 = [Math.fround(trunkTop[0] + randSigned(seed, 390 + lod) * 0.11), height, Math.fround(trunkTop[2] + randSigned(seed, 391 + lod) * 0.11)];
  const leaderMid = lerpVec3(trunkTop, leaderEnd, 0.46);
  leaderMid[0] = Math.fround(leaderMid[0] + randSigned(seed, 393 + lod) * 0.05);
  leaderMid[2] = Math.fround(leaderMid[2] + randSigned(seed, 394 + lod) * 0.05);
  addCurvedBranch(builder, trunkTop, leaderMid, leaderEnd, trunkRadius * 0.36, trunkRadius * 0.15, 3, 2, bark, 0.35, 0.78);
  addLeafVolumeCluster(builder, leaderEnd, rand01(seed, 392 + lod) * Math.PI * 2, lod === 0 ? 0.24 : 0.2, seed, 400 + lod, lod === 0 ? 2 : 1);
}

function buildFlowerArchetype(builder: PlantGeometryBuilder, lod: number, seed: number): void {
  if (lod >= 2) {
    const stemHeight = Math.fround(0.46 + rand01(seed, 510 + lod) * 0.07);
    const top: PlantVec3 = [Math.fround(randSigned(seed, 514 + lod) * 0.028), stemHeight, Math.fround(randSigned(seed, 515 + lod) * 0.028)];
    addCurvedBranch(builder, [0, 0, 0], [randSigned(seed, 512 + lod) * 0.02, stemHeight * 0.48, randSigned(seed, 513 + lod) * 0.02], top, 0.024, 0.014, 3, 1, [0.18, 0.58, 0.24], 0, 0.78);
    addFlowerFoldedLeaf(builder, [0, stemHeight * 0.34, 0], -2.52 + randSigned(seed, 520 + lod) * 0.14, 0.17, 0.052, [0.20, 0.56, 0.25]);
    const petalCount = 3;
    const petalRadius = Math.fround(0.13 + rand01(seed, 530 + lod) * 0.025);
    for (let i = 0; i < petalCount; i++) {
      const angle = (i / petalCount) * Math.PI * 2 + rand01(seed, 540 + i) * 0.08;
      addFlowerFoldedPetal(builder, [top[0], top[1] + 0.035, top[2]], angle, petalRadius, 0.058 + rand01(seed, 560 + i) * 0.01, [0.84, 0.30 + rand01(seed, 550 + i) * 0.09, 0.68]);
    }
    addLeafTetrahedron(builder, [top[0], top[1] + 0.052, top[2]], 0.058, 0.044, [0.94, 0.78, 0.20], 0.92, rand01(seed, 590 + lod) * Math.PI * 2);
    return;
  }
  const stemHeight = Math.fround(0.54 + rand01(seed, 510 + lod) * 0.08);
  addCurvedBranch(builder, [0, 0, 0], [randSigned(seed, 512 + lod) * 0.025, stemHeight * 0.48, randSigned(seed, 513 + lod) * 0.025], [randSigned(seed, 514 + lod) * 0.035, stemHeight, randSigned(seed, 515 + lod) * 0.035], 0.028, 0.018, 3, lod === 0 ? 2 : 1, [0.18, 0.58, 0.24], 0, 0.78);
  addFlowerFoldedLeaf(builder, [0, stemHeight * 0.34, 0], -2.72 + randSigned(seed, 520 + lod) * 0.16, 0.23, 0.065, [0.20, 0.56, 0.25]);
  if (lod === 0) addFlowerFoldedLeaf(builder, [0, stemHeight * 0.47, 0.02], 0.52 + randSigned(seed, 522 + lod) * 0.18, 0.24, 0.07, [0.16, 0.50, 0.23]);
  const petalCount = lod === 0 ? 5 : 4;
  const headY = stemHeight + 0.04;
  const petalRadius = Math.fround(0.18 + rand01(seed, 530 + lod) * 0.04);
  for (let i = 0; i < petalCount; i++) {
    const angle = (i / petalCount) * Math.PI * 2 + rand01(seed, 540 + i) * 0.08;
    addFlowerFoldedPetal(builder, [0, headY, 0], angle, petalRadius, 0.076 + rand01(seed, 560 + i) * 0.012, [0.84, 0.28 + rand01(seed, 550 + i) * 0.1, 0.68]);
  }
  addLeafTetrahedron(builder, [0, headY + 0.02, 0], 0.08, 0.055, [0.94, 0.78, 0.20], 0.92, rand01(seed, 590 + lod) * Math.PI * 2);
}

// ---------------------------------------------------------------------------
// Understory geometry: grass tussocks and low woody shrubs (not ported).
//
// Replaces the ported cone-blade grass and umbrella/pyramid shrub crowns with
// walk-scale botanical forms. Every shape value comes from `rand01(seed, salt)`
// with fixed salts, and each LOD resamples the same per-seed tuft/skeleton,
// so the LODs share root, height and spread while their triangle counts are
// fixed per LOD whatever the seed:
//   grass LOD0/1/2: 7/3/3 two-sided blades of 3/2/1 segments   = 70/18/6
//   shrub LOD0: 4 five-sided arching stems (25 each), 8 three-sided twigs
//               (3 each) and 20 two-sided ovate leaves (8 each) = 284
//         LOD1: 4 three-sided stem cones (3) + 12 two-sided leaf sprays (4) = 60
//         LOD2: 4 three-sided stem cones (3) + 4 two-sided leaf sprays (4)  = 28
// The ordinary flora material is FrontSide, so thin blades and leaves carry
// an explicit back face (reversed winding, negated normal). Tubes are wound
// outward. At FLORA_LOD_INSTANCE_BUDGET full capacity grass+shrub cost
// 99,328 triangles; tests/flora-understory-geometry.test.mjs holds that budget.
// ---------------------------------------------------------------------------

const UNDERSTORY_GOLDEN_ANGLE = 2.399963229728653;
const WORLD_UP: PlantVec3 = [0, 1, 0];
const GRASS_LOD_BLADES = [7, 3, 3] as const;
const GRASS_LOD_SEGMENTS = [3, 2, 1] as const;
const GRASS_LOD_WIDTH_SCALE = [1, 2.3, 3] as const;
/** Far LODs straighten their fewer, coarser blades so none lie flat on the ground. */
const GRASS_LOD_ARCH = [1, 0.75, 0.55] as const;
const SHRUB_STEM_COUNT = 4;
const SHRUB_TWIGS_PER_STEM = 2;
/** Arc parameters at which LOD0 samples each stem; far LODs use the chord. */
const SHRUB_LOD0_STEM_SAMPLES = [0, 0.45, 0.8, 1] as const;
const SHRUB_STEM_BASE_Y = -0.008;
const SPRAY_MIN_WIDTH_RATIO = 0.15;
const SPRAY_ROLL = 0.5;
const SHRUB_BARK_BASE: PlantColor = [0.30, 0.22, 0.14];
const SHRUB_BARK_TIP: PlantColor = [0.44, 0.36, 0.22];

function addVec3(a: PlantVec3, b: PlantVec3): PlantVec3 { return [a[0] + b[0], a[1] + b[1], a[2] + b[2]]; }
function subVec3(a: PlantVec3, b: PlantVec3): PlantVec3 { return [a[0] - b[0], a[1] - b[1], a[2] - b[2]]; }
function scaleVec3(a: PlantVec3, s: number): PlantVec3 { return [a[0] * s, a[1] * s, a[2] * s]; }
function maddVec3(a: PlantVec3, b: PlantVec3, s: number): PlantVec3 { return [a[0] + b[0] * s, a[1] + b[1] * s, a[2] + b[2] * s]; }
function dotVec3(a: PlantVec3, b: PlantVec3): number { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }
function lengthVec3(a: PlantVec3): number { return Math.hypot(a[0], a[1], a[2]); }
function mixColor(a: PlantColor, b: PlantColor, t: number): PlantColor { return [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]; }

/** Arc that leaves `base` at `lean` radians from vertical toward `azimuth`,
 * bends a further `bend` radians outward along its length and drifts
 * sideways by `drift * t^2`; `t` is the arc parameter in [0,1]. */
function arcPoint(base: PlantVec3, azimuth: number, lean: number, bend: number, length: number, drift: number, t: number): PlantVec3 {
  const ox = Math.cos(azimuth), oz = Math.sin(azimuth);
  const straight = Math.abs(bend) < 1e-4;
  const outward = straight ? Math.sin(lean) * t : (Math.cos(lean) - Math.cos(lean + bend * t)) / bend;
  const rise = straight ? Math.cos(lean) * t : (Math.sin(lean + bend * t) - Math.sin(lean)) / bend;
  const side = drift * t * t;
  return [base[0] + (ox * outward - oz * side) * length, base[1] + rise * length, base[2] + (oz * outward + ox * side) * length];
}
function arcTangent(azimuth: number, lean: number, bend: number, drift: number, t: number): PlantVec3 {
  const ox = Math.cos(azimuth), oz = Math.sin(azimuth), angle = lean + bend * t, side = 2 * drift * t;
  return normalize3([ox * Math.sin(angle) - oz * side, Math.cos(angle), oz * Math.sin(angle) + ox * side]);
}

/** Pushes one smooth-shaded open surface. Normals are area-weighted sums of
 * the patch's own (optionally reversed) faces, so every vertex normal agrees
 * with the winding of the triangles that use it. */
function addSmoothPatch(builder: PlantGeometryBuilder, positions: readonly PlantVec3[], uvs: readonly [number, number][], colors: readonly PlantColor[], stiffness: readonly number[], triangles: readonly number[], reverse: boolean): void {
  const start = builder.positions.length / 3;
  const accumulated: PlantVec3[] = positions.map(() => [0, 0, 0]);
  for (let i = 0; i < triangles.length; i += 3) {
    const a = triangles[i]!, b = reverse ? triangles[i + 2]! : triangles[i + 1]!, c = reverse ? triangles[i + 1]! : triangles[i + 2]!;
    const pa = positions[a]!, face = cross3(subVec3(positions[b]!, pa), subVec3(positions[c]!, pa));
    for (const vertex of [a, b, c]) accumulated[vertex] = addVec3(accumulated[vertex]!, face);
    builder.indices.push(start + a, start + b, start + c);
  }
  for (let i = 0; i < positions.length; i++) pushVertex(builder, positions[i]!, accumulated[i]!, uvs[i]!, colors[i]!, stiffness[i]!);
}
function addTwoSidedPatch(builder: PlantGeometryBuilder, positions: readonly PlantVec3[], uvs: readonly [number, number][], frontColors: readonly PlantColor[], backColors: readonly PlantColor[], stiffness: readonly number[], triangles: readonly number[]): void {
  addSmoothPatch(builder, positions, uvs, frontColors, stiffness, triangles, false);
  addSmoothPatch(builder, positions, uvs, backColors, stiffness, triangles, true);
}

/** Outward-wound woody tube along `points` with one shared ring per point
 * (no seams at bends); the last point is a closed apex. The base ring is left
 * open: stems sink it into the ground, twigs bury it inside their parent. */
function addWoodyTube(builder: PlantGeometryBuilder, points: readonly PlantVec3[], radii: readonly number[], stiffness: readonly number[], sides: number, seed: number, salt: number): void {
  const count = points.length, rings = count - 1;
  const tangents = points.map((_, i) => normalize3(subVec3(points[Math.min(i + 1, count - 1)]!, points[Math.max(i - 1, 0)]!)));
  const first = tangents[0]!;
  let right = normalize3(cross3(Math.abs(first[1]) > 0.92 ? [1, 0, 0] : [0, 1, 0], first));
  const frames: { right: PlantVec3; up: PlantVec3 }[] = [];
  for (let i = 0; i < rings; i++) {
    const tangent = tangents[i]!;
    // Parallel transport keeps consecutive rings untwisted.
    if (i > 0) right = normalize3(maddVec3(right, tangent, -dotVec3(right, tangent)));
    frames.push({ right, up: normalize3(cross3(tangent, right)) });
  }
  const phase = rand01(seed, salt) * Math.PI * 2 / sides;
  const radial = (frame: { right: PlantVec3; up: PlantVec3 }, angle: number): PlantVec3 => addVec3(scaleVec3(frame.right, Math.cos(angle)), scaleVec3(frame.up, Math.sin(angle)));
  const start = builder.positions.length / 3;
  for (let i = 0; i < rings; i++) {
    const along = i / rings, bark = mixColor(SHRUB_BARK_BASE, SHRUB_BARK_TIP, along);
    for (let k = 0; k < sides; k++) {
      const direction = radial(frames[i]!, phase + (k / sides) * Math.PI * 2);
      pushVertex(builder, maddVec3(points[i]!, direction, radii[i]!), direction, [k / sides, along], shadePlantColor(bark, 0.86 + rand01(seed, salt + 1 + i * 8 + k) * 0.2), stiffness[i]!);
    }
  }
  // One apex vertex per facet so each facet keeps its own radial normal.
  const last = frames[rings - 1]!, tipBark = shadePlantColor(SHRUB_BARK_TIP, 0.96);
  for (let k = 0; k < sides; k++) {
    pushVertex(builder, points[count - 1]!, radial(last, phase + ((k + 0.5) / sides) * Math.PI * 2), [(k + 0.5) / sides, 1], tipBark, stiffness[count - 1]!);
  }
  for (let i = 0; i < rings - 1; i++) {
    for (let k = 0; k < sides; k++) {
      const a = start + i * sides + k, b = start + i * sides + ((k + 1) % sides), c = a + sides, d = b + sides;
      builder.indices.push(a, b, c, b, d, c);
    }
  }
  const lastRing = start + (rings - 1) * sides, apex = start + rings * sides;
  for (let k = 0; k < sides; k++) builder.indices.push(lastRing + k, lastRing + ((k + 1) % sides), apex + k);
}

// --- Grass ------------------------------------------------------------------

interface GrassBlade {
  base: PlantVec3; azimuth: number; lean: number; bend: number; length: number; drift: number;
  halfWidth: number; twist: number; tone: number; straw: number;
}

/** A tussock of `count` blades on a Vogel spiral: inner blades stand upright
 * and tall, outer blades lean and arch further. Every LOD resamples the same
 * seeded tuft height, turn and lean range with fewer blades. */
function grassTuft(seed: number, count: number, arch: number): GrassBlade[] {
  const turn = rand01(seed, 0x5100) * Math.PI * 2;
  const height = 0.42 + rand01(seed, 0x5101) * 0.08;
  const blades: GrassBlade[] = [];
  for (let i = 0; i < count; i++) {
    const f = (i + 0.5) / count;
    const azimuth = turn + i * UNDERSTORY_GOLDEN_ANGLE + randSigned(seed, 0x5110 + i) * 0.28;
    const offset = 0.006 + 0.03 * Math.sqrt(f);
    blades.push({
      base: [Math.cos(azimuth) * offset, 0, Math.sin(azimuth) * offset],
      azimuth,
      lean: (0.08 + f * 0.42 + rand01(seed, 0x5120 + i) * 0.12) * (0.5 + 0.5 * arch),
      bend: (0.25 + rand01(seed, 0x5130 + i) * 0.5) * arch,
      length: height * (0.8 + rand01(seed, 0x5140 + i) * 0.3) * (1.1 - 0.26 * f),
      drift: randSigned(seed, 0x5150 + i) * 0.08,
      halfWidth: 0.0085 + rand01(seed, 0x5160 + i) * 0.0055,
      twist: randSigned(seed, 0x5170 + i) * 0.5,
      tone: rand01(seed, 0x5180 + i),
      straw: rand01(seed, 0x5190 + i),
    });
  }
  return blades;
}

/** Tapered blade with `segments` bands: parallel-sided sheath, arched
 * midline, gentle twist and a single tip vertex. */
function addGrassBlade(builder: PlantGeometryBuilder, blade: GrassBlade, segments: number, widthScale: number): void {
  const across0: PlantVec3 = [Math.sin(blade.azimuth), 0, -Math.cos(blade.azimuth)];
  const baseColor = shadePlantColor([0.20, 0.47, 0.18], 0.9 + blade.tone * 0.18);
  const tipColor = mixColor(shadePlantColor([0.42, 0.78, 0.33], 0.92 + blade.tone * 0.14), [0.64, 0.72, 0.38], blade.straw * 0.35);
  const positions: PlantVec3[] = [], uvs: [number, number][] = [], colors: PlantColor[] = [], stiffness: number[] = [];
  for (let row = 0; row < segments; row++) {
    const t = row / segments;
    const center = arcPoint(blade.base, blade.azimuth, blade.lean, blade.bend, blade.length, blade.drift, t);
    const tangent = arcTangent(blade.azimuth, blade.lean, blade.bend, blade.drift, t);
    const flat = normalize3(maddVec3(across0, tangent, -dotVec3(across0, tangent)));
    const twist = blade.twist * t;
    const across = normalize3(addVec3(scaleVec3(flat, Math.cos(twist)), scaleVec3(cross3(tangent, flat), Math.sin(twist))));
    const halfWidth = blade.halfWidth * widthScale * (0.72 + 0.28 * Math.min(1, t * 5)) * (1 - Math.pow(t, 1.8));
    const color = mixColor(baseColor, tipColor, t), stiff = Math.pow(t, 1.4);
    positions.push(maddVec3(center, across, -halfWidth), maddVec3(center, across, halfWidth));
    uvs.push([0, t], [1, t]);
    colors.push(color, color);
    stiffness.push(stiff, stiff);
  }
  positions.push(arcPoint(blade.base, blade.azimuth, blade.lean, blade.bend, blade.length, blade.drift, 1));
  uvs.push([0.5, 1]);
  colors.push(tipColor);
  stiffness.push(1);
  const triangles: number[] = [];
  for (let row = 0; row < segments - 1; row++) {
    const l = row * 2, r = l + 1;
    triangles.push(l, r, l + 2, r, r + 2, l + 2);
  }
  const tip = segments * 2;
  triangles.push(tip - 2, tip - 1, tip);
  addTwoSidedPatch(builder, positions, uvs, colors, colors.map((color) => shadePlantColor(color, 0.9)), stiffness, triangles);
}

function buildGrassArchetype(builder: PlantGeometryBuilder, lod: number, seed: number): void {
  for (const blade of grassTuft(seed, GRASS_LOD_BLADES[lod]!, GRASS_LOD_ARCH[lod]!)) addGrassBlade(builder, blade, GRASS_LOD_SEGMENTS[lod]!, GRASS_LOD_WIDTH_SCALE[lod]!);
}

// --- Shrub ------------------------------------------------------------------

/** One woody axis of an understory shrub, as drawn at one LOD. */
export interface UnderstoryShrubShoot {
  /** Axis from base to apex; the apex is the last point. */
  points: PlantVec3[];
  /** Ring radius at each point (the apex entry is 0). */
  radii: number[];
  stiffness: number[];
  sides: number;
  /** Index into `stems` for a twig; -1 for a stem. */
  parent: number;
  /** Radius of the parent axis where a twig is rooted (0 for a stem). */
  parentRadius: number;
}
/** Leaf (LOD0) or leaf spray (LOD1/2), rooted on a host shoot's axis. */
export interface UnderstoryShrubFoliage {
  host: 'stem' | 'twig';
  hostIndex: number;
  base: PlantVec3;
  /** Unit axis from base to tip. */
  direction: PlantVec3;
  /** Unit upper-surface normal, perpendicular to `direction`. */
  normal: PlantVec3;
  length: number;
  halfWidth: number;
  /** 0 = ovate (widest near base) ... 1 = obovate (widest near tip). */
  shape: number;
  fold: number;
  curl: number;
  top: PlantColor;
  under: PlantColor;
  stiffness: number;
}
export interface UnderstoryShrubSkeleton {
  stems: UnderstoryShrubShoot[];
  twigs: UnderstoryShrubShoot[];
  foliage: UnderstoryShrubFoliage[];
}

interface ShrubStemPlan { base: PlantVec3; azimuth: number; lean: number; bend: number; length: number; drift: number; radius: number }

function shrubStemPlans(seed: number): ShrubStemPlan[] {
  const turn = rand01(seed, 0x5300) * Math.PI * 2, stature = 0.9 + rand01(seed, 0x5301) * 0.2;
  const plans: ShrubStemPlan[] = [];
  for (let j = 0; j < SHRUB_STEM_COUNT; j++) {
    const azimuth = turn + j * UNDERSTORY_GOLDEN_ANGLE + randSigned(seed, 0x5310 + j) * 0.3;
    const f = j / (SHRUB_STEM_COUNT - 1), offset = 0.014 + rand01(seed, 0x5320 + j) * 0.022;
    plans.push({
      base: [Math.cos(azimuth) * offset, SHRUB_STEM_BASE_Y, Math.sin(azimuth) * offset],
      azimuth,
      lean: 0.08 + f * 0.5 + rand01(seed, 0x5330 + j) * 0.12,
      bend: 0.3 + rand01(seed, 0x5340 + j) * 0.4,
      length: (0.84 - 0.26 * f + rand01(seed, 0x5350 + j) * 0.1) * stature,
      drift: randSigned(seed, 0x5360 + j) * 0.12,
      radius: 0.017 + rand01(seed, 0x5370 + j) * 0.006,
    });
  }
  return plans;
}

/** Position, segment direction and interpolated radius at arc parameter `t`
 * on a shoot sampled at `params` (so attachments sit on the drawn axis). */
function sampleShoot(shoot: UnderstoryShrubShoot, params: readonly number[], t: number): { point: PlantVec3; tangent: PlantVec3; radius: number } {
  let i = 0;
  while (i < params.length - 2 && t > params[i + 1]!) i++;
  const t0 = params[i]!, t1 = params[i + 1]!, u = clamp01((t - t0) / (t1 - t0));
  const a = shoot.points[i]!, b = shoot.points[i + 1]!;
  return { point: lerpVec3(a, b, u), tangent: normalize3(subVec3(b, a)), radius: shoot.radii[i]! + (shoot.radii[i + 1]! - shoot.radii[i]!) * u };
}
/** Unit vector perpendicular to `axis`, at angle `psi` around it measured
 * from the horizontal. */
function aroundAxis(axis: PlantVec3, psi: number): PlantVec3 {
  const right = normalize3(Math.abs(axis[1]) > 0.98 ? [1, 0, 0] : cross3(WORLD_UP, axis));
  const up = normalize3(cross3(axis, right));
  return addVec3(scaleVec3(right, Math.cos(psi)), scaleVec3(up, Math.sin(psi)));
}
/** Upper-surface normal for a blade-like organ lying along `direction`. */
function organNormal(direction: PlantVec3, fallbackOut: PlantVec3, roll: number): PlantVec3 {
  const hint = Math.abs(direction[1]) > 0.95 ? fallbackOut : WORLD_UP;
  const side = normalize3(cross3(direction, hint));
  const normal = normalize3(cross3(side, direction));
  return normalize3(addVec3(scaleVec3(normal, Math.cos(roll)), scaleVec3(side, Math.sin(roll))));
}

/** `radiusScale` thins the far triangular cones, whose steeper chord would
 * otherwise tilt their root ring below the -0.023 m root convention. */
function shrubStems(plans: readonly ShrubStemPlan[], params: readonly number[], sides: number, radiusScale: number): UnderstoryShrubShoot[] {
  return plans.map((plan) => ({
    points: params.map((t) => arcPoint(plan.base, plan.azimuth, plan.lean, plan.bend, plan.length, plan.drift, t)),
    radii: params.map((t, i) => (i === params.length - 1 ? 0 : plan.radius * radiusScale * (1 - 0.72 * t))),
    stiffness: params.map((t) => 0.04 + 0.5 * t),
    sides, parent: -1, parentRadius: 0,
  }));
}

function shrubTwigs(seed: number, stems: readonly UnderstoryShrubShoot[], params: readonly number[]): UnderstoryShrubShoot[] {
  const twigs: UnderstoryShrubShoot[] = [];
  stems.forEach((stem, j) => {
    const turn = rand01(seed, 0x5380 + j) * Math.PI * 2;
    for (let m = 0; m < SHRUB_TWIGS_PER_STEM; m++) {
      const salt = 0x5400 + (j * SHRUB_TWIGS_PER_STEM + m) * 8;
      const t = (m === 0 ? 0.32 : 0.56) + rand01(seed, salt) * 0.13;
      const at = sampleShoot(stem, params, t);
      const side = aroundAxis(at.tangent, turn + m * UNDERSTORY_GOLDEN_ANGLE + randSigned(seed, salt + 1) * 0.25);
      const divergence = 0.75 + rand01(seed, salt + 2) * 0.4;
      const direction = normalize3(maddVec3(addVec3(scaleVec3(at.tangent, Math.cos(divergence)), scaleVec3(side, Math.sin(divergence))), WORLD_UP, 0.25));
      const length = 0.18 + rand01(seed, salt + 3) * 0.12;
      const stiff = 0.04 + 0.5 * t;
      twigs.push({ points: [at.point, maddVec3(at.point, direction, length)], radii: [at.radius * 0.55, 0], stiffness: [stiff, Math.min(1, stiff + 0.25)], sides: 3, parent: j, parentRadius: at.radius });
    }
  });
  return twigs;
}

function shrubLeafColors(seed: number, salt: number): { top: PlantColor; under: PlantColor } {
  const green = mixColor([0.24, 0.60, 0.31], [0.36, 0.74, 0.38], rand01(seed, salt));
  const top = shadePlantColor(mixColor(green, [0.42, 0.64, 0.26], rand01(seed, salt + 1) * 0.45), 0.9 + rand01(seed, salt + 2) * 0.18);
  return { top, under: mixColor(top, [0.62, 0.72, 0.52], 0.25) };
}

/** LOD0 leaves: a lateral and a terminal leaf per twig and a terminal leaf per
 * stem, turning by the golden angle around their host axis. */
function shrubLeaves(seed: number, stems: readonly UnderstoryShrubShoot[], twigs: readonly UnderstoryShrubShoot[]): UnderstoryShrubFoliage[] {
  const leaves: UnderstoryShrubFoliage[] = [];
  const add = (host: 'stem' | 'twig', hostIndex: number, base: PlantVec3, hostAxis: PlantVec3, lateral: boolean, stiffness: number): void => {
    const q = leaves.length, salt = 0x5500 + q * 16;
    const side = aroundAxis(hostAxis, rand01(seed, 0x54f0) * Math.PI * 2 + (q + 1) * UNDERSTORY_GOLDEN_ANGLE);
    const divergence = lateral ? 0.7 + rand01(seed, salt) * 0.35 : 0.2 + rand01(seed, salt) * 0.2;
    const raw = addVec3(scaleVec3(hostAxis, Math.cos(divergence)), scaleVec3(side, Math.sin(divergence)));
    // Leaves hold their blades nearer horizontal than their shoots.
    const direction = normalize3([raw[0], raw[1] * 0.6 - 0.06, raw[2]]);
    const out = normalize3([base[0] || 1e-3, 0, base[2]]);
    const length = 0.12 + rand01(seed, salt + 1) * 0.06;
    leaves.push({
      host, hostIndex, base, direction,
      normal: organNormal(direction, out, randSigned(seed, salt + 2) * 0.6),
      length,
      halfWidth: length * (0.23 + rand01(seed, salt + 3) * 0.09),
      shape: rand01(seed, salt + 4),
      fold: 0.1 + rand01(seed, salt + 5) * 0.2,
      curl: 0.06 + rand01(seed, salt + 6) * 0.16,
      ...shrubLeafColors(seed, salt + 8),
      stiffness,
    });
  };
  twigs.forEach((twig, i) => {
    const a = twig.points[0]!, b = twig.points[1]!, axis = normalize3(subVec3(b, a));
    add('twig', i, lerpVec3(a, b, 0.55), axis, true, twig.stiffness[0]! + 0.14);
    add('twig', i, b, axis, false, twig.stiffness[1]!);
  });
  stems.forEach((stem, j) => {
    const count = stem.points.length, apex = stem.points[count - 1]!;
    add('stem', j, apex, normalize3(subVec3(apex, stem.points[count - 2]!)), false, stem.stiffness[count - 1]!);
  });
  return leaves;
}

function ovateLeafArea(leaf: UnderstoryShrubFoliage): number {
  // Planar area of the six-point outline built by `addOvateLeaf`.
  const t1 = 0.26 + 0.14 * leaf.shape, t2 = 0.64 + 0.14 * leaf.shape, w1 = 1 - 0.28 * leaf.shape, w2 = 0.62 + 0.36 * leaf.shape;
  return leaf.length * leaf.halfWidth * (t1 * w1 + (t2 - t1) * (w1 + w2) + (1 - t2) * w2);
}
function leafTip(leaf: UnderstoryShrubFoliage): PlantVec3 {
  return maddVec3(maddVec3(leaf.base, leaf.direction, leaf.length), leaf.normal, -leaf.curl * leaf.length);
}

/** Far-LOD spray: one kite from the host axis through the centre of the
 * leaves it replaces, reaching half a leaf beyond it. It keeps the replaced
 * leaves' area but is never narrower than SPRAY_MIN_WIDTH_RATIO of its length
 * (no blade-like spikes). Sprays roll about their axis by `roll` so distant,
 * near-horizontal views meet some of them face-on rather than edge-on;
 * `flatten` scales the rise of the spray axis as leaves hold their blades
 * nearer horizontal than their shoots. */
function leafSpray(host: 'stem' | 'twig', hostIndex: number, base: PlantVec3, leaves: readonly UnderstoryShrubFoliage[], stiffness: number, areaScale: number, roll: number, flatten: number): UnderstoryShrubFoliage {
  let center: PlantVec3 = [0, 0, 0], leafLength = 0, area = 0, top: PlantColor = [0, 0, 0], under: PlantColor = [0, 0, 0];
  const share = 1 / leaves.length;
  for (const leaf of leaves) {
    center = maddVec3(center, lerpVec3(leaf.base, leafTip(leaf), 0.5), share);
    leafLength += leaf.length * share;
    area += ovateLeafArea(leaf);
    top = maddVec3(top, leaf.top, share);
    under = maddVec3(under, leaf.under, share);
  }
  const reach = subVec3(center, base), direction = normalize3([reach[0], reach[1] * flatten, reach[2]]);
  let length = lengthVec3(reach) + leafLength * 0.5;
  for (const leaf of leaves) length = Math.max(length, dotVec3(subVec3(leafTip(leaf), base), direction));
  const out = normalize3([center[0] || 1e-3, 0, center[2]]);
  // Kite area = length * halfWidth.
  const halfWidth = Math.max((area * areaScale) / length, SPRAY_MIN_WIDTH_RATIO * length);
  return { host, hostIndex, base, direction, normal: organNormal(direction, out, roll), length, halfWidth, shape: 0.62, fold: 0.12, curl: 0, top, under, stiffness };
}

/**
 * Deterministic shrub skeleton for one LOD: stems, twigs and the leaves
 * (LOD0) or leaf sprays (LOD1/2) that `buildFloraGeometryData('shrub', ...)`
 * draws. Exposed so tests can check attachments without parsing the mesh.
 */
export function understoryShrubSkeleton(lod: number, seed: number): UnderstoryShrubSkeleton {
  const clampedLod = clampInt(lod, 0, FLORA_LOD_COUNT - 1);
  const plans = shrubStemPlans(seed);
  const nearStems = shrubStems(plans, SHRUB_LOD0_STEM_SAMPLES, 5, 1);
  const nearTwigs = shrubTwigs(seed, nearStems, SHRUB_LOD0_STEM_SAMPLES);
  const leaves = shrubLeaves(seed, nearStems, nearTwigs);
  if (clampedLod === 0) return { stems: nearStems, twigs: nearTwigs, foliage: leaves };
  const chord = [0, 1] as const;
  const stems = shrubStems(plans, chord, 3, 0.7);
  const foliage: UnderstoryShrubFoliage[] = [];
  if (clampedLod === 1) {
    shrubTwigs(seed, stems, chord).forEach((twig, i) => {
      foliage.push(leafSpray('stem', twig.parent, twig.points[0]!, leaves.filter((leaf) => leaf.host === 'twig' && leaf.hostIndex === i), twig.stiffness[1]!, 0.85, SPRAY_ROLL * (i % 2 === 0 ? 1 : -1), 0.6));
    });
    stems.forEach((stem, j) => {
      const base = sampleShoot(stem, chord, 0.8).point;
      foliage.push(leafSpray('stem', j, base, leaves.filter((leaf) => leaf.host === 'stem' && leaf.hostIndex === j), stem.stiffness[1]!, 0.85, SPRAY_ROLL * (j % 2 === 0 ? -1 : 1), 0.6));
    });
  } else {
    stems.forEach((stem, j) => {
      const owned = leaves.filter((leaf) => (leaf.host === 'stem' ? leaf.hostIndex : nearTwigs[leaf.hostIndex]!.parent) === j);
      foliage.push(leafSpray('stem', j, sampleShoot(stem, chord, 0.45).point, owned, stem.stiffness[1]!, 1.3, SPRAY_ROLL * 0.6 * (j % 2 === 0 ? 1 : -1), 0.85));
    });
  }
  return { stems, twigs: [], foliage };
}

/** Six-point ovate/obovate leaf folded along its midrib and curled at the
 * tip; two-sided with a paler underside. */
function addOvateLeaf(builder: PlantGeometryBuilder, leaf: UnderstoryShrubFoliage): void {
  const { base, direction: d, normal: n, length, halfWidth, shape, fold, curl } = leaf;
  const s = normalize3(cross3(d, n));
  const midrib = (t: number): PlantVec3 => maddVec3(maddVec3(base, d, length * t), n, -curl * length * t * t);
  const edge = (t: number, width: number, sign: number): PlantVec3 => maddVec3(maddVec3(midrib(t), s, sign * halfWidth * width), n, -fold * halfWidth * width);
  const t1 = 0.26 + 0.14 * shape, t2 = 0.64 + 0.14 * shape, w1 = 1 - 0.28 * shape, w2 = 0.62 + 0.36 * shape;
  // Outline B, R1, R2, T, L2, L1; the B-T midrib is a shared edge so the fold stays crisp.
  const positions: PlantVec3[] = [midrib(0), edge(t1, w1, 1), edge(t2, w2, 1), midrib(1), edge(t2, w2, -1), edge(t1, w1, -1)];
  const along = [0, t1, t2, 1, t2, t1];
  const uvs: [number, number][] = [[0.5, 0], [1, t1], [1, t2], [0.5, 1], [0, t2], [0, t1]];
  const top = along.map((t) => shadePlantColor(leaf.top, 0.86 + 0.14 * t));
  const under = along.map((t) => shadePlantColor(leaf.under, 0.9 + 0.1 * t));
  const stiffness = along.map((t) => Math.min(1, leaf.stiffness + (1 - leaf.stiffness) * 0.6 * t));
  addTwoSidedPatch(builder, positions, uvs, top, under, stiffness, [0, 1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 5]);
}

/** Four-point folded kite used for far-LOD leaf sprays; two-sided. */
function addLeafSprayCard(builder: PlantGeometryBuilder, spray: UnderstoryShrubFoliage): void {
  const { base, direction: d, normal: n, length, halfWidth, shape, fold } = spray;
  const s = normalize3(cross3(d, n));
  const widest = maddVec3(base, d, length * shape);
  const positions: PlantVec3[] = [base, maddVec3(maddVec3(widest, s, halfWidth), n, -fold * halfWidth), maddVec3(base, d, length), maddVec3(maddVec3(widest, s, -halfWidth), n, -fold * halfWidth)];
  const uvs: [number, number][] = [[0.5, 0], [1, shape], [0.5, 1], [0, shape]];
  const tip = Math.min(1, spray.stiffness + 0.3);
  const stiffness = [spray.stiffness, tip, tip, tip];
  addTwoSidedPatch(builder, positions, uvs, [shadePlantColor(spray.top, 0.88), spray.top, spray.top, spray.top], [spray.under, spray.under, spray.under, spray.under], stiffness, [0, 1, 2, 0, 2, 3]);
}

function buildShrubArchetype(builder: PlantGeometryBuilder, lod: number, seed: number): void {
  const skeleton = understoryShrubSkeleton(lod, seed);
  skeleton.stems.forEach((stem, j) => addWoodyTube(builder, stem.points, stem.radii, stem.stiffness, stem.sides, seed, 0x5600 + j * 64));
  skeleton.twigs.forEach((twig, i) => addWoodyTube(builder, twig.points, twig.radii, twig.stiffness, twig.sides, seed, 0x5800 + i * 16));
  for (const organ of skeleton.foliage) {
    if (lod === 0) addOvateLeaf(builder, organ);
    else addLeafSprayCard(builder, organ);
  }
}

function addCurvedBranch(builder: PlantGeometryBuilder, start: PlantVec3, control: PlantVec3, end: PlantVec3, bottomRadius: number, topRadius: number, sides: number, segments: number, color: PlantColor, bottomStiffness: number, topStiffness: number): void {
  const segmentCount = Math.max(1, segments | 0);
  let previous = start;
  for (let i = 0; i < segmentCount; i++) {
    const t0 = i / segmentCount, t1 = (i + 1) / segmentCount;
    const next = quadraticVec3(start, control, end, t1);
    const radius0 = Math.fround(bottomRadius + (topRadius - bottomRadius) * t0);
    const radius1 = Math.fround(bottomRadius + (topRadius - bottomRadius) * t1);
    const stiff0 = Math.fround(bottomStiffness + (topStiffness - bottomStiffness) * t0);
    const stiff1 = Math.fround(bottomStiffness + (topStiffness - bottomStiffness) * t1);
    addTaperedBranch(builder, previous, next, radius0, radius1, sides, color, stiff0, stiff1);
    previous = next;
  }
}

function addLeafVolumeCluster(builder: PlantGeometryBuilder, center: PlantVec3, yaw: number, radius: number, seed: number, salt: number, satelliteCount: number): void {
  const colors: PlantColor[] = [[0.42, 0.80, 0.34], [0.30, 0.68, 0.28], [0.52, 0.88, 0.40]];
  addTreeCanopyLobe(builder, center, radius, radius * 0.92, colors[salt % colors.length] ?? colors[0]!, 0.88, yaw, seed, salt);
  for (let i = 0; i < satelliteCount; i++) {
    const angle = yaw + (i / Math.max(1, satelliteCount)) * Math.PI * 2 + randSigned(seed, salt + i) * 0.42;
    const offset = radius * (0.62 + rand01(seed, salt + 20 + i) * 0.18);
    const satellite: PlantVec3 = [Math.fround(center[0] + Math.cos(angle) * offset), Math.fround(center[1] + randSigned(seed, salt + 40 + i) * radius * 0.26), Math.fround(center[2] + Math.sin(angle) * offset)];
    addTreeCanopyLobe(builder, satellite, radius * 0.66, radius * 0.62, colors[(salt + i + 1) % colors.length] ?? colors[1]!, 0.94, angle, seed, salt + 60 + i);
  }
}

function addTreeCanopyLobe(builder: PlantGeometryBuilder, center: PlantVec3, radius: number, height: number, color: PlantColor, stiffness: number, yaw: number, seed: number, salt: number): void {
  const top: PlantVec3 = [Math.fround(center[0] + randSigned(seed, salt + 1) * radius * 0.10), Math.fround(center[1] + height * 0.58), Math.fround(center[2] + randSigned(seed, salt + 2) * radius * 0.10)];
  const bottom: PlantVec3 = [Math.fround(center[0] - randSigned(seed, salt + 3) * radius * 0.08), Math.fround(center[1] - height * 0.42), Math.fround(center[2] - randSigned(seed, salt + 4) * radius * 0.08)];
  const ring: PlantVec3[] = [];
  for (let i = 0; i < 6; i++) {
    const angle = yaw + (i / 6) * Math.PI * 2 + randSigned(seed, salt + 10 + i) * 0.09;
    const xRadius = radius * (0.76 + rand01(seed, salt + 20 + i) * 0.14);
    const zRadius = radius * (0.60 + rand01(seed, salt + 30 + i) * 0.16);
    ring.push([Math.fround(center[0] + Math.cos(angle) * xRadius), Math.fround(center[1] + (i % 2 === 0 ? height * 0.06 : -height * 0.04)), Math.fround(center[2] + Math.sin(angle) * zRadius)]);
  }
  for (let i = 0; i < ring.length; i++) {
    const a = ring[i]!, b = ring[(i + 1) % ring.length]!;
    const topShade = 0.92 + rand01(seed, salt + 70 + i) * 0.18;
    const bottomShade = 0.74 + rand01(seed, salt + 90 + i) * 0.16;
    addTriangleAutoNormal(builder, top, a, b, shadePlantColor(color, topShade), [stiffness, stiffness * 0.96, stiffness * 0.96]);
    addTriangleAutoNormal(builder, bottom, b, a, shadePlantColor(color, bottomShade), [stiffness * 0.82, stiffness * 0.88, stiffness * 0.88]);
  }
}


function addLeafTetrahedron(builder: PlantGeometryBuilder, center: PlantVec3, radius: number, height: number, color: PlantColor, stiffness: number, yaw: number): void {
  const top: PlantVec3 = [center[0], Math.fround(center[1] + height), center[2]];
  const a: PlantVec3 = [Math.fround(center[0] + Math.cos(yaw) * radius), Math.fround(center[1] - height * 0.34), Math.fround(center[2] + Math.sin(yaw) * radius)];
  const b: PlantVec3 = [Math.fround(center[0] + Math.cos(yaw + Math.PI * 2 / 3) * radius), Math.fround(center[1] - height * 0.34), Math.fround(center[2] + Math.sin(yaw + Math.PI * 2 / 3) * radius)];
  const c: PlantVec3 = [Math.fround(center[0] + Math.cos(yaw + Math.PI * 4 / 3) * radius), Math.fround(center[1] - height * 0.34), Math.fround(center[2] + Math.sin(yaw + Math.PI * 4 / 3) * radius)];
  addTriangleAutoNormal(builder, top, a, b, color, [stiffness, stiffness, stiffness]);
  addTriangleAutoNormal(builder, top, b, c, color, [stiffness, stiffness, stiffness]);
  addTriangleAutoNormal(builder, top, c, a, color, [stiffness, stiffness, stiffness]);
  addTriangleAutoNormal(builder, a, c, b, color, [stiffness * 0.85, stiffness * 0.85, stiffness * 0.85]);
}


function addFlowerFoldedLeaf(builder: PlantGeometryBuilder, base: PlantVec3, yaw: number, length: number, width: number, color: PlantColor): void {
  const forward: PlantVec3 = [Math.cos(yaw), 0.22, Math.sin(yaw)];
  const side: PlantVec3 = [Math.cos(yaw + Math.PI * 0.5), 0, Math.sin(yaw + Math.PI * 0.5)];
  const mid: PlantVec3 = [Math.fround(base[0] + forward[0] * length * 0.48), Math.fround(base[1] + forward[1] * length * 0.48), Math.fround(base[2] + forward[2] * length * 0.48)];
  const ridge: PlantVec3 = [mid[0], Math.fround(mid[1] + width * 0.28), mid[2]];
  const left: PlantVec3 = [Math.fround(mid[0] - side[0] * width), Math.fround(mid[1] - width * 0.12), Math.fround(mid[2] - side[2] * width)];
  const right: PlantVec3 = [Math.fround(mid[0] + side[0] * width), Math.fround(mid[1] - width * 0.12), Math.fround(mid[2] + side[2] * width)];
  const tip: PlantVec3 = [Math.fround(base[0] + forward[0] * length), Math.fround(base[1] + forward[1] * length), Math.fround(base[2] + forward[2] * length)];
  addTriangleAutoNormal(builder, base, left, ridge, color, [0.26, 0.72, 0.82]);
  addTriangleAutoNormal(builder, base, ridge, right, color, [0.26, 0.82, 0.72]);
  addTriangleAutoNormal(builder, ridge, left, tip, color, [0.82, 0.78, 0.92]);
  addTriangleAutoNormal(builder, ridge, tip, right, color, [0.82, 0.92, 0.78]);
}

function addFlowerFoldedPetal(builder: PlantGeometryBuilder, center: PlantVec3, yaw: number, radius: number, width: number, color: PlantColor): void {
  const radial: PlantVec3 = [Math.cos(yaw), 0.18, Math.sin(yaw)];
  const tangent: PlantVec3 = [Math.cos(yaw + Math.PI * 0.5), 0, Math.sin(yaw + Math.PI * 0.5)];
  const base: PlantVec3 = [Math.fround(center[0] + radial[0] * radius * 0.16), center[1], Math.fround(center[2] + radial[2] * radius * 0.16)];
  const mid: PlantVec3 = [Math.fround(center[0] + radial[0] * radius * 0.52), Math.fround(center[1] + radial[1] * radius * 0.36), Math.fround(center[2] + radial[2] * radius * 0.52)];
  const ridge: PlantVec3 = [mid[0], Math.fround(mid[1] + width * 0.18), mid[2]];
  const left: PlantVec3 = [Math.fround(mid[0] - tangent[0] * width), mid[1], Math.fround(mid[2] - tangent[2] * width)];
  const right: PlantVec3 = [Math.fround(mid[0] + tangent[0] * width), mid[1], Math.fround(mid[2] + tangent[2] * width)];
  const tip: PlantVec3 = [Math.fround(center[0] + radial[0] * radius), Math.fround(center[1] + radial[1] * radius + 0.035), Math.fround(center[2] + radial[2] * radius)];
  addTriangleAutoNormal(builder, base, left, ridge, color, [0.82, 0.96, 0.96]);
  addTriangleAutoNormal(builder, base, ridge, right, color, [0.82, 0.96, 0.96]);
  addTriangleAutoNormal(builder, ridge, left, tip, color, [0.96, 0.98, 1]);
}

function addTaperedBranch(builder: PlantGeometryBuilder, start: PlantVec3, end: PlantVec3, bottomRadius: number, topRadius: number, sides: number, color: PlantColor, bottomStiffness: number, topStiffness: number): void {
  const axis = normalize3([end[0] - start[0], end[1] - start[1], end[2] - start[2]]);
  const reference: PlantVec3 = Math.abs(axis[1]) > 0.92 ? [1, 0, 0] : [0, 1, 0];
  const right = normalize3(cross3(reference, axis));
  const up = normalize3(cross3(axis, right));
  const baseIndex = builder.positions.length / 3;
  const ringSides = Math.max(3, sides | 0);
  for (let ring = 0; ring < 2; ring++) {
    const center = ring === 0 ? start : end;
    const radius = ring === 0 ? bottomRadius : topRadius;
    const stiffness = ring === 0 ? bottomStiffness : topStiffness;
    for (let side = 0; side < ringSides; side++) {
      const detail = plantDetailNoise(start, end, side, ring);
      const angle = (side / ringSides) * Math.PI * 2 + (detail - 0.5) * 0.12;
      const ridge = Math.fround(0.9 + detail * 0.2);
      const collar = Math.fround(ring === 0 ? 1.02 + plantDetailNoise(end, start, side + 17, ring) * 0.08 : 0.96 + plantDetailNoise(start, end, side + 29, ring) * 0.08);
      const radial: PlantVec3 = [Math.fround(right[0] * Math.cos(angle) + up[0] * Math.sin(angle)), Math.fround(right[1] * Math.cos(angle) + up[1] * Math.sin(angle)), Math.fround(right[2] * Math.cos(angle) + up[2] * Math.sin(angle))];
      pushVertex(
        builder,
        [Math.fround(center[0] + radial[0] * radius * ridge * collar), Math.fround(center[1] + radial[1] * radius * ridge * collar), Math.fround(center[2] + radial[2] * radius * ridge * collar)],
        radial, [side / ringSides, ring], shadePlantColor(color, 0.86 + plantDetailNoise(start, end, side + 43, ring) * 0.24), stiffness,
      );
    }
  }
  for (let side = 0; side < ringSides; side++) {
    const next = (side + 1) % ringSides;
    const a = baseIndex + side, b = baseIndex + next, c = baseIndex + ringSides + side, d = baseIndex + ringSides + next;
    // With `up = axis × right`, ring angle advances around the outward radial
    // direction. The same outward triangle order as addWoodyTube is
    // (base[k], base[next], top[k]); the previous order faced into the stem.
    builder.indices.push(a, b, c, b, d, c);
  }
}

function plantDetailNoise(a: PlantVec3, b: PlantVec3, side: number, ring: number): number {
  let value = mix(0x6d385f21, side + ring * 131);
  value = mix(value, f32Bits(a[0] + b[1] * 0.37));
  value = mix(value, f32Bits(a[1] + b[2] * 0.41));
  value = mix(value, f32Bits(a[2] + b[0] * 0.43));
  return (value >>> 0) / 0x1_0000_0000;
}
function shadePlantColor(color: PlantColor, shade: number): PlantColor {
  return [Math.fround(Math.min(1, color[0] * shade)), Math.fround(Math.min(1, color[1] * shade)), Math.fround(Math.min(1, color[2] * shade))];
}
function addRootFlare(builder: PlantGeometryBuilder, radius: number, color: PlantColor): void {
  for (let i = 0; i < 3; i++) {
    const yaw = (i / 3) * Math.PI * 2 + Math.PI / 7, left = yaw - 0.26, right = yaw + 0.26;
    addTriangle(builder, [Math.cos(left) * radius * 0.72, 0.03, Math.sin(left) * radius * 0.72], [Math.cos(right) * radius * 0.72, 0.03, Math.sin(right) * radius * 0.72], [Math.cos(yaw) * radius * 2.15, 0, Math.sin(yaw) * radius * 2.15], [0, 0.45, 0], color, [0, 0, 0.04]);
  }
}
function lerpVec3(a: PlantVec3, b: PlantVec3, t: number): PlantVec3 {
  const clamped = clamp01(t);
  return [Math.fround(a[0] + (b[0] - a[0]) * clamped), Math.fround(a[1] + (b[1] - a[1]) * clamped), Math.fround(a[2] + (b[2] - a[2]) * clamped)];
}
function quadraticVec3(a: PlantVec3, b: PlantVec3, c: PlantVec3, t: number): PlantVec3 {
  const clamped = clamp01(t), inv = 1 - clamped;
  return [Math.fround(a[0] * inv * inv + b[0] * 2 * inv * clamped + c[0] * clamped * clamped), Math.fround(a[1] * inv * inv + b[1] * 2 * inv * clamped + c[1] * clamped * clamped), Math.fround(a[2] * inv * inv + b[2] * 2 * inv * clamped + c[2] * clamped * clamped)];
}
function cross3(a: PlantVec3, b: PlantVec3): PlantVec3 {
  return [Math.fround(a[1] * b[2] - a[2] * b[1]), Math.fround(a[2] * b[0] - a[0] * b[2]), Math.fround(a[0] * b[1] - a[1] * b[0])];
}
function addQuad(builder: PlantGeometryBuilder, p0: [number, number, number], p1: [number, number, number], p2: [number, number, number], p3: [number, number, number], normal: [number, number, number], color: [number, number, number], stiffness: [number, number, number, number]): void {
  const start = builder.positions.length / 3;
  pushVertex(builder, p0, normal, [0, 0], color, stiffness[0]);
  pushVertex(builder, p1, normal, [1, 0], color, stiffness[1]);
  pushVertex(builder, p2, normal, [1, 1], color, stiffness[2]);
  pushVertex(builder, p3, normal, [0, 1], color, stiffness[3]);
  builder.indices.push(start, start + 1, start + 2, start, start + 2, start + 3);
}
function addTriangle(builder: PlantGeometryBuilder, p0: [number, number, number], p1: [number, number, number], p2: [number, number, number], normal: [number, number, number], color: [number, number, number], stiffness: [number, number, number]): void {
  const start = builder.positions.length / 3;
  pushVertex(builder, p0, normal, [0.5, 0], color, stiffness[0]);
  pushVertex(builder, p1, normal, [0, 1], color, stiffness[1]);
  pushVertex(builder, p2, normal, [1, 1], color, stiffness[2]);
  builder.indices.push(start, start + 1, start + 2);
}
function addTriangleAutoNormal(builder: PlantGeometryBuilder, p0: PlantVec3, p1: PlantVec3, p2: PlantVec3, color: PlantColor, stiffness: [number, number, number]): void {
  const edgeA: PlantVec3 = [Math.fround(p1[0] - p0[0]), Math.fround(p1[1] - p0[1]), Math.fround(p1[2] - p0[2])];
  const edgeB: PlantVec3 = [Math.fround(p2[0] - p0[0]), Math.fround(p2[1] - p0[1]), Math.fround(p2[2] - p0[2])];
  addTriangle(builder, p0, p1, p2, normalize3(cross3(edgeA, edgeB)), color, stiffness);
}
function pushVertex(builder: PlantGeometryBuilder, position: [number, number, number], normal: [number, number, number], uv: [number, number], color: [number, number, number], stiffness: number): void {
  const n = normalize3(normal);
  builder.positions.push(Math.fround(position[0]), Math.fround(position[1]), Math.fround(position[2]));
  builder.normals.push(n[0], n[1], n[2]);
  builder.uvs.push(Math.fround(uv[0]), Math.fround(uv[1]));
  builder.colors.push(Math.fround(color[0]), Math.fround(color[1]), Math.fround(color[2]));
  builder.windStiffness.push(Math.fround(stiffness));
}
/**
 * Presentation-only: used exclusively by the LOD'd geometry builders below
 * (mesh vertex/normal placement for an already-finalized tree/flower/grass/
 * shrub shape), never by the growth loop above. Per this module's header
 * ("Geometry builders... are ordinary rendering data, not catalogue
 * identity... freely use plain Math.sin/cos/hypot") and the two-class
 * routing note above `generateProceduralTreeSkeleton`, nothing here feeds a
 * survival/count/topology decision, so platform Math.hypot is fine - unlike
 * the decision-feeding `normalize` (growth loop) above, which uses
 * `hypotDet`.
 */
function normalize3(value: [number, number, number]): [number, number, number] {
  const length = Math.hypot(value[0], value[1], value[2]) || 1;
  return [Math.fround(value[0] / length), Math.fround(value[1] / length), Math.fround(value[2] / length)];
}
