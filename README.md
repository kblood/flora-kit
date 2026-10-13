# Flora Kit

Version 0.1.0 extracts Atlas's deterministic flora placement, cached recipes,
understory/preview geometry, and bounded instanced culling renderer. The public
TypeScript API is `flora-kit/v1/field` and `flora-kit/v1/culling`; Node 24 can
strip these types, and Vite/TypeScript can consume the same source. The private
deterministic math module is an extraction snapshot, not a public math API.

```sh
npm ci
npm run build
npm test
```

The package uses the same pinned Three fork as Atlas. Consumers should resolve
one shared Three version. There is no sibling-repository dependency, generator
download, authored asset, or generated mesh in this repository.

`floraAt` and `prepareFlora` accept caller-supplied body identity, direction,
biome and biosphere gate. `floraFromRecipe` evaluates dated presentation without
changing canonical placement. Time is branded `Myr`, with a runtime range guard;
Atlas callers must derive it from their world-time adapter. Culling and rendering
retain the existing budgets, fade version, borrowed frame results and ownership
contracts. The original tests and allocation regression ceiling are retained.

The `tree` archetypes here are procedural **preview geometry**. Atlas's actual
`PlanetForest`, GLB trees, tree assets, canopy/impostors, terrain, habitat/water
queries, streaming, date adapters and Surface Flora Lab/replay remain in Atlas.
An extraction is not a claim that the preview trees are the runtime forest.

See [provenance](docs/provenance.json) for the Atlas source revision, original
file paths and LF-normalized SHA-256 hashes. The module header preserves the
original `nms` port attribution. No upstream license grant is inferred.

This public repository and private npm package are **UNLICENSED** (all rights
reserved). Public visibility does not grant a reuse license.
