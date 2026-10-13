import {test} from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync, readdirSync} from 'node:fs';
import {createHash} from 'node:crypto';

const root = new URL('../', import.meta.url);
const read = path => readFileSync(new URL(path, root), 'utf8');
const pkg = JSON.parse(read('package.json'));

test('Flora Kit has versioned field/culling APIs and keeps deterministic math private', async () => {
  assert.equal(pkg.version, '0.1.0');
  assert.equal(pkg.private, true);
  assert.equal(pkg.license, 'UNLICENSED');
  assert.deepEqual(Object.keys(pkg.exports), ['./v1/field', './v1/culling']);
  const field = await import('flora-kit/v1/field');
  const culling = await import('flora-kit/v1/culling');
  assert.equal(typeof field.prepareFlora, 'function');
  assert.equal(typeof culling.drawFloraCandidates, 'function');
  assert.equal(culling.FLORA_FADE_VERSION, 'atlas-flora-fade-1');
});

test('extracted canonical modules retain their exact source and have no Atlas imports', () => {
  const provenance = JSON.parse(read('docs/provenance.json'));
  for (const file of provenance.files.filter(file => file.path.startsWith('src/'))) {
    const source = read(file.path).replace(/\r\n/g, '\n');
    assert.equal(createHash('sha256').update(source).digest('hex'), file.sha256, file.path);
  }
  for (const name of readdirSync(new URL('src/', root))) {
    const source = read(`src/${name}`);
    for (const match of source.matchAll(/\bfrom\s+['"]([^'"]+)['"]/g)) {
      assert.ok(match[1] === 'three' || /^\.\/[^/]+\.ts$/.test(match[1]), `unexpected dependency ${match[1]}`);
    }
  }
});
