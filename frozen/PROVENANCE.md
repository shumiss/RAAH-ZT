# Provenance of the frozen protocol

`protocol.lock.json` is the protocol fixed before the original computations. In this public copy, the two
input paths (`inputs.validation.path`, `inputs.test.path`) were changed from absolute paths on the authors'
workstation to `data/validation_timeline.npz` and `data/test_timeline.npz`. Nothing else was changed.

- SHA-256 of the original lock file: `d3615939f45acc79440ba2fec73397b8f4bebb0994a6229bd399d36257babd0b`
- SHA-256 of this public copy: `e2ce7f42c09e9bec402cce2483961bef2349d6ec99ceca35a50d918ac2062d5c`

`validation_selection.lock.json` records the original hash. The input files themselves are unchanged; their
size and SHA-256 are listed in `protocol.lock.json` and are checked whenever they are loaded.
