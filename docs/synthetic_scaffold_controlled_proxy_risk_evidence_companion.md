# Controlled-proxy Q+O risk-evidence companion

## Scope and immutable dependency

`anchor.synthetic-scaffold-controlled-proxy-risk-evidence-companion.v1` is an
additive, metadata-only companion. It neither replaces nor modifies the frozen
follow-up at Producer commit
`23194f7b3c707e3531ac92a64863c2b2f523f81d`. The frozen schema, contract,
sidecar, implementation, comparison, and comparison sidecar are reauthenticated
before this companion is accepted. That dependency is checked both as local
bytes and as the raw Producer commit/tree/artifact blobs; a copied worktree
alone cannot satisfy provenance.

The source is Consumer commit
`58e9cd0c021ac0f01250746d44f199c1f616261d`, directly descended from comparison
commit `6ef29f1e0e9e110d59f9f2a09c1a8151f04b2465`. The auditor reads its raw
commit/tree and exactly three receipt/sidecar blob pairs with Git replacement
objects disabled. It never fetches, reads the Consumer worktree, or opens a
model, adapter, PNG, dataset partition, Gold, heldout, scaffold body, prompt,
answer, or raw token IDs.

Authenticated receipts are:

- Q/O branch ablation `59750842e7bbad7fb06fcc64a1b9956dbd449e5591ba85f4abca021616da8ca3`
  with physical sidecar SHA
  `ad18c248e9bb091797f229927febddd11c0520d1497943119e0da2401b657a31`;
- spectral risk `c2fddd98ece4127ad3f17a19ffbd5bfa6e8d7f95588964b389e7e2970cfc8dd3`
  with sidecar `61b829f963b48dfe1b3f98337b6992dce0e5944595ec436e296b37b2986ac74a`;
- attention summary `fc1ce0168cacfd1ed46a7ffcc1b482e7593253e224e780ab7dc6f7b701bb58a4`
  with sidecar `1238605cf7abff130096e78071dd1f6ab2f916af93d85b29fdf3e054e1a13120`.

Hashes of source configs, implementations, model, and adapter inside the
receipts remain receipt-declared transitive identities. Producer does not
claim to have reopened those artifacts.

Frozen Producer identities are schema
`c04ba5072c2892f111a913808559f1c3eca9864977159c387df09fa6b7081068`,
contract `352870bbea976c0b97df722fd3b188d731a8d463ec33f27bcd15bdb2e292ac28`,
mandatory sidecar physical SHA
`c75a91a78123fc1f583e9d053525c290fa39c3e6a811b22ecb48b11581c5503b`,
and implementation
`2f707507014dc9e70546a024b8bf109f779cf0c33019439f8396563e795e5d3a`.

## Closed observations and interpretation

For the jointly trained Q+O checkpoint, same-template teacher-forced macro
loss for adapter-off, Q-branch-retained, O-branch-retained, and full was
`3.1020863533020018`, `2.6766975283622743`, `1.0298870503902435`, and
`0.8586097195744514`. The retained O branch preserved
`0.9236553979475981` of the observed off-to-full reduction. This is a post-hoc
branch ablation of a jointly trained adapter, not an independently trained
O-only arm. Branch effects need not be additive and no O-only mechanism is
claimed.

On the four-bundle/20-record synthetic OOD proxy, the corresponding losses
were `3.035316228866577`, `2.983178400993347`, `2.8666090965270996`, and
`2.896842730045319`; full improved only `0.04562078161884503` relative to off.
This is a template or answer-shape writeback risk signal. The OOD proxy was not
heldout or a preregistered confirmation fixture and does not establish broad
generalization.

The static spectral audit observed O top-1 energy fraction `0.82318570872`,
effective rank `2.11605570639/8`, O/Q total delta energy `1.767153939794`,
target-frequent/random projection `1.95910776849`, and
prompt-control/random projection `1.006882507832`. Token groups were not
frequency- or part-of-speech matched. These are correlation-only risk signals,
not proof of verbatim, answer, exploit, or causal memorization.

The eager attention hook used one 79-token synthetic probe, BF16+TF32,
head-mean aggregation, and layers 0/13/27. Full-minus-Q-branch mean/max
differences were `0/0`, `0.0008257970912382007/0.061482757329940796`, and
`0.0019083430524915457/0.1309814453125`. Layer-0 zero does not imply an
unchanged full stack. Later-layer changes are only consistent with an O update
propagating through the residual stream; attention is not an explanation or a
general causal proof.

## Preregistered next boundary

The existing non-authorizing plan remains in force. Step-80 bundle-macro loss
delta is the sole primary endpoint; steps 5/10/20/40 are secondary learning
curves. Replication still requires at least five master seeds, all registered
arm orders, identical base/sample/token/order/budget/optimizer conditions, and
independently trained equal-budget O-only and K+V controls. Five seeds support
at most a controlled-proxy replication signal, not formal significance.

Before confirmation, freeze the generator, namespace-neutral blueprint
inventory, `task_bundle_sha256` split, seeds, arm orders, endpoint, and
statistical method. Discovery/confirmation require both ID and body-free
blueprint disjointness. A useful preregistered matrix separates old-task/new-
template, new-task/old-template, and new-task/new-template behavior while
varying field order, lexical form, answer shape, and safe nonces. No such
fixture or disjointness proof exists yet.

Long-context and cache rules are unchanged: only 8K/16K/32K diagnostic
preflights are currently allowed, and exact reuse is limited to an identical
ordered frozen-prefix lineage. This companion implements no Q-reader, physical
KV reuse, CUDA zero-copy, multistream sharing, or quality evaluation.

## Fail-closed behavior and counters

The contract is a closed Draft 2020-12 instance. Local inputs use single-byte
snapshots, exact mandatory sidecars, strict duplicate/non-finite rejection,
same-byte reparse, and final identity rechecks. Raw Git commit/tree/blob bytes
are rechecked at the end. Synchronized receipt/sidecar substitution, metric or
counter drift, source GPU-count washing, body fields, path escape, reparse or
TOCTOU drift, and every promotion boolean fail closed.

Source counters are separate from Producer counters: the ablation and
attention audits each used one model/GPU invocation; spectral audit was
CPU-only. The Attention receipt does not contain a machine
`provider_requests` scalar, so the companion records
`provider_requests_reported=false` rather than inventing zero. Producer-side
provider/network/model/GPU/protected-body counters are authenticated as zero.

The fixed gates remain formal-v3 0/5, protected inventories 2/6, multi-seed
replication unavailable, independent confirmation unavailable,
`training_authorized=false`, and `formal_training_authorized=false`.

## Metadata-only reproduction

```powershell
$env:PYTHONPATH = "src"
python scripts/data/audit_synthetic_scaffold_controlled_proxy_risk_evidence_v1.py `
  --repo-root .
python -m pytest -q `
  tests/test_synthetic_scaffold_controlled_proxy_risk_evidence_v1.py
python -m ruff check `
  src/anchor_mvp/swebench/synthetic_scaffold_controlled_proxy_risk_evidence.py `
  scripts/data/audit_synthetic_scaffold_controlled_proxy_risk_evidence_v1.py `
  tests/test_synthetic_scaffold_controlled_proxy_risk_evidence_v1.py
```

The exact Consumer Git commit object must already be local; the auditor never
fetches it. These commands issue no provider/network request, load no model or
GPU, and create no tag or release.
