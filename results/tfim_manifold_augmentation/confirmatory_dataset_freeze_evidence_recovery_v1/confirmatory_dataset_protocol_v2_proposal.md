# Confirmatory dataset protocol v2 proposal (not executed)

V1 failure artifacts remain immutable and no V1 confirmatory claim is available. V2 is a separate track; criteria must be preregistered before generation and use a newly predeclared seed namespace. This recovery neither selects nor executes that seed.

## Alternatives

| Criterion | Leakage/provenance role | Proposed treatment |
|---|---|---|
| Sample-ID disjointness | direct provenance identity | gate |
| Seed/provenance disjointness | independent generation | gate and attest |
| Exact-g disjointness | parameter reuse | gate only if scientific design requires |
| Canonical-state-hash disjointness | exact serialized state | gate |
| Exact projective duplicate exclusion | exact physical ray | gate with a separately specified numerical test |
| Minimum parameter-space separation | diversity/guard gap | preregister explicitly where required |
| Minimum FS separation | diversity, not provenance | report or preregister as a distinct diversity gate |
| Blocked-g guard gap | defining blocked regime | retain as explicit regime contract |
| Random-split nearest-neighbor | descriptive smooth-manifold diagnostic | report without gating |
| Cross-regime overlap | relationship between separate designs | report without gating unless independently justified |

The choices above are based on semantics, not tuned to make the recovered V1 candidate pass.
