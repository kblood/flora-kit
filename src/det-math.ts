/**
 * Deterministic math primitives for cross-engine catalogue identity.
 *
 * V8, SpiderMonkey and JavaScriptCore do not guarantee identical results for
 * Math.pow/atan2/atan/tan/cbrt/exp/log/acos (or `**`, which is Math.pow under
 * the hood): these call into platform libm with implementation-defined
 * precision. A measured cross-engine audit on this machine found ~49.6% of
 * `x**5` and ~23.7% of `x**3` results differ bitwise between two real
 * engines. Math.sqrt and the four basic arithmetic operators (+ - * /) are
 * IEEE-754 "correctly rounded" and required to be, so the same sequence of
 * +,-,*,/,Math.sqrt,Math.floor produces the same bits on every conformant
 * engine.
 *
 * Everything below is built only from +, -, *, /, Math.floor and Math.sqrt,
 * using fixed-iteration-count reductions/series so a call always performs
 * the exact same arithmetic regardless of platform. Use these (not the
 * Math.* equivalents, and never `**`) for anything that feeds a catalogue
 * identity decision: counts, thresholds, sort keys, integer placement,
 * hashing, or any other stored/hashed value that must match bit-for-bit
 * across browsers per the determinism contract (docs/universe-atlas-v1-spec.md
 * §3). Continuous, tolerant-comparison uses (animated orbital positions -
 * §3 explicitly allows "ordinary double-precision orbital evaluation" with
 * tolerant comparisons) may keep using platform Math.
 */

const TWO_PI = Math.PI * 2;

/** e^-max(0,x), via (1+x/256)^256 through 8 fixed repeated squarings (256=2^8). */
export function expNeg(x: number): number {
  let v = 1 + Math.max(0, x) / 256;
  for (let i = 0; i < 8; i++) v *= v;
  return 1 / v;
}

/** sin(x): range reduction to [-pi,pi] plus a fixed 9-term Taylor series. */
export function sine(x: number): number {
  x = (((x + Math.PI) % TWO_PI) + TWO_PI) % TWO_PI - Math.PI;
  let sum = x,
    term = x;
  for (let n = 1; n < 10; n++) {
    term *= (-x * x) / (2 * n * (2 * n + 1));
    sum += term;
  }
  return sum;
}

/** cos(x), reusing the sine series via the quarter-turn identity cos(x)=sin(x+pi/2). */
export function detCos(x: number): number {
  return sine(x + Math.PI / 2);
}

/** ln(x) for x>0: binary range reduction to [1,2] plus a fixed 11-term atanh series. */
export function logPos(x: number): number {
  let e = 0;
  while (x > 2) {
    x *= 0.5;
    e++;
  }
  while (x < 1) {
    x *= 2;
    e--;
  }
  const t = (x - 1) / (x + 1);
  let v = t,
    s = t;
  for (let i = 3; i < 24; i += 2) {
    v *= t * t;
    s += v / i;
  }
  return 2 * s + e * 0.6931471805599453;
}

/**
 * ln(x) for x>0. Alias of `logPos` above (identical binary-range-reduction +
 * fixed-11-term atanh-series algorithm, so identical accuracy/bit-output) —
 * exported separately so callers inverting `detExp` (e.g. `detPow`'s own
 * `base>0` branch, or any future `x = exp(y)` solve) can name the operation
 * they mean without reaching for the "positive" framing `logPos` was
 * originally named for. Domain: callers must guarantee x>0, exactly as
 * `logPos` documents (x<=0 is not range-reducible by this algorithm and is
 * not guarded here, matching this module's existing style of documenting
 * domain assumptions rather than adding redundant runtime checks).
 */
export function detLn(x: number): number {
  return logPos(x);
}

const LN2 = 0.6931471805599453;

/**
 * e^x for any real x. Range-reduces to r = x - k*ln2 in [-ln2/2, ln2/2] (k an
 * integer), sums a fixed 19-term Taylor series for e^r (fast: |r|<=~0.347),
 * then rescales by 2^k via exact repeated doubling/halving (multiplying or
 * dividing by 2 is exact in binary floating point, so this adds no error).
 * This is far more accurate than a fixed-exponent (1+x/n)^n approximation
 * (whose relative error grows like x^2/(2n)) while staying fully portable:
 * every step is +, -, *, / and Math.floor.
 */
export function detExp(x: number): number {
  const k = Math.floor(x / LN2 + 0.5);
  const r = x - k * LN2;
  let term = 1,
    sum = 1;
  for (let n = 1; n < 19; n++) {
    term *= r / n;
    sum += term;
  }
  if (k >= 0) for (let i = 0; i < k; i++) sum *= 2;
  else for (let i = 0; i < -k; i++) sum *= 0.5;
  return sum;
}

/**
 * atan(x) for x>=0. Applies the half-angle reduction
 * atan(x) = 2*atan(x/(1+sqrt(1+x^2))) a fixed 6 times - this maps any finite
 * x into (-1,1) on the first step and then roughly halves it each further
 * step, so after 6 steps the remainder is small (<0.02 for any starting x) -
 * then sums a fixed 7-term Taylor series on that remainder and undoes the
 * doubling. Fixed iteration counts throughout, independent of input.
 */
function detAtanPositive(x: number): number {
  const REDUCTIONS = 6;
  let scale = 1;
  for (let i = 0; i < REDUCTIONS; i++) {
    x = x / (1 + Math.sqrt(1 + x * x));
    scale *= 2;
  }
  const x2 = x * x;
  let sum = x,
    term = x;
  for (let n = 1; n < 8; n++) {
    term *= -x2;
    sum += term / (2 * n + 1);
  }
  return sum * scale;
}

/** atan(x) for any finite x. */
export function detAtan(x: number): number {
  return x < 0 ? -detAtanPositive(-x) : detAtanPositive(x);
}

/** atan2(y,x) with the standard quadrant conventions; atan2(0,0) = 0. */
export function detAtan2(y: number, x: number): number {
  if (x > 0) return detAtan(y / x);
  if (x < 0) return y >= 0 ? detAtan(y / x) + Math.PI : detAtan(y / x) - Math.PI;
  if (y > 0) return Math.PI / 2;
  if (y < 0) return -Math.PI / 2;
  return 0;
}

/** tan(x) = sin(x)/cos(x), built from the deterministic sine/detCos above. */
export function detTan(x: number): number {
  return sine(x) / detCos(x);
}

/** acos(x) for x in [-1,1], via the identity acos(x) = atan2(sqrt(1-x^2), x). */
export function detAcos(x: number): number {
  const c = Math.max(0, 1 - x * x);
  return detAtan2(Math.sqrt(c), x);
}

/**
 * Sign-aware cube root of any real x. Seeds from exp(log(|x|)/3), then
 * refines with 4 fixed Newton-Raphson iterations (y' = (2y + x/y^2)/3) to
 * reach full double precision regardless of how coarse the seed was.
 */
export function detCbrt(x: number): number {
  if (x === 0) return 0;
  const ax = Math.abs(x);
  let y = detExp(logPos(ax) / 3);
  for (let i = 0; i < 4; i++) y = (2 * y + ax / (y * y)) / 3;
  return x < 0 ? -y : y;
}

/**
 * base^exp, replacing both Math.pow and `**`.
 * - base>0: exp(exp*log(base)) for any real exponent.
 * - base===0: the usual 0^0=1 / 0^positive=0 / 0^negative=Infinity convention.
 * - base<0: only defined here for an integer exponent (matches Math.pow's
 *   NaN-for-non-integer-exponent-of-a-negative-base behaviour otherwise);
 *   sign follows the exponent's parity.
 */
export function detPow(base: number, exp: number): number {
  if (exp === 0) return 1;
  if (base === 0) return exp > 0 ? 0 : Infinity;
  if (base > 0) return detExp(exp * logPos(base));
  const isInt = Math.floor(exp) === exp;
  if (!isInt) return NaN;
  const magnitude = detExp(exp * logPos(-base));
  return Math.floor(exp / 2) * 2 === exp ? magnitude : -magnitude;
}

/**
 * sqrt(sum of squares). Math.sqrt is IEEE-correctly-rounded, so a direct
 * sum-of-squares (skipping Math.hypot's internal overflow-avoidance scaling,
 * which is not required to be identical across engines) is portable for the
 * bounded magnitudes this codebase uses.
 */
export function detHypot(...values: number[]): number {
  let s = 0;
  for (const v of values) s += v * v;
  return Math.sqrt(s);
}

// --- Hydrology helpers (merged from the river-purity work; outputs pinned by river fixtures) ---
/**
 * atan(x) for any finite x, via repeated half-angle reduction
 * (atan(x) = 2*atan(x / (1 + sqrt(1+x^2)))) until the remaining argument is
 * small, then a short Taylor series (fast: no factorial denominators are
 * needed because the reduced argument is already well under 0.1).
 */
function atanReduced(x: number): number {
  let reduced = x, scale = 1;
  for (let i = 0; i < 5; i++) {
    reduced = reduced / (1 + Math.sqrt(1 + reduced * reduced));
    scale *= 2;
  }
  const x2 = reduced * reduced;
  let term = reduced, sum = reduced;
  for (let n = 1; n < 7; n++) {
    term *= -x2;
    sum += term / (2 * n + 1);
  }
  return sum * scale;
}

/** Deterministic replacement for Math.atan2. */
export function atan2Det(y: number, x: number): number {
  if (x > 0) return atanReduced(y / x);
  if (x < 0) return y >= 0 ? Math.PI + atanReduced(y / x) : atanReduced(y / x) - Math.PI;
  if (y > 0) return Math.PI / 2;
  if (y < 0) return -Math.PI / 2;
  return 0;
}

/** Deterministic replacement for Math.acos, x clamped to [-1, 1]. */
export function acosDet(x: number): number {
  const c = Math.max(-1, Math.min(1, x));
  return atan2Det(Math.sqrt(Math.max(0, 1 - c * c)), c);
}

/**
 * Deterministic replacement for Math.hypot over a small fixed list of
 * components. Plain sum-of-squares is intentionally used instead of a
 * scaled/overflow-safe algorithm: every caller in this codebase passes
 * unit-sphere or metre-scale offsets, far inside float64's safe range, and
 * Math.sqrt itself is exactly specified (unlike Math.hypot's algorithm).
 */
export function hypotDet(...values: number[]): number {
  let sum = 0;
  for (const v of values) sum += v * v;
  return Math.sqrt(sum);
}
