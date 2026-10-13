// Isolated child keeps GC profiling independent of the node:test runner.
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import assert from 'node:assert/strict';
export function checkAllocation(harness) {
  const script = fileURLToPath(new URL(harness, import.meta.url));
  const result = spawnSync(process.execPath, ['--expose-gc', script], {encoding: 'utf8', timeout: 120000});
  assert.equal(result.status, 0, `${result.error || ''}\n${result.stdout}\n${result.stderr}`);
}
