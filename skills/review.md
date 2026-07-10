---
sop_id: code-review-v1
task_type: review
---
# Code review SOP

1. Reconstruct the expected behavior from the request before inspecting style.
2. Check control flow, state transitions, async cancellation, cleanup, boundary
   conditions, and error propagation.
3. Check accessibility semantics, keyboard behavior, focus, labels, and live status.
4. Flag data-flow or trust-boundary mistakes without reconstructing attack payloads.
5. Separate correctness findings from maintainability and presentation findings.
6. Cite observable evidence from the supplied request or code for every finding.
7. Prefer the narrowest complete repair; preserve working public behavior.
8. Produce corrected, complete code and a concise change summary.
9. Require evidence from typecheck, lint, tests, or build output when the task makes
   those checks executable; never claim a check passed without its artifact.

Reference set: `react-rules`, `typescript-strict`, `eslint-recommended`,
`wai-aria-apg`. Source metadata is pinned in `docs/sop_sources.yaml`.
