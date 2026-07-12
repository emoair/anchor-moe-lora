# Continuous multi-LoRA collaboration curriculum v2

This curriculum supplies executable candidate tasks for teaching one user context across
the MVP collaboration loop. It is intentionally marked `candidate-not-training-data`:
teacher output is still required, must execute inside an isolated fixture, and must pass
the existing session converter before it may enter a training snapshot.

## Public-output lineage

Each seed is one continuous trajectory:

1. `planner_lora` consumes the original `context` and emits
   `anchor.planner-handoff.v2`. Its target names the required builder and includes concrete
   implementation handoff points; a generic plan without builder selection is invalid.
2. `safety_tool_policy_lora` consumes that exact planner output and returns an approve,
   block, or escalation decision. Candidate tasks expect `APPROVE` for their local,
   allowlisted execution contract.
3. The selected builder consumes both verified upstream public outputs and performs the
   work. Its retained artifact is compatible with
   `anchor.session-training-candidate.v1`, including tool calls, tool results, diff, and
   build/test/lint results.
4. The domain reviewer receives the same context plus the builder's public output, tool
   trace, diff, and validators. Its only accepted verdict schema is
   `anchor.domain-review-verdict.v2` (`PASS` or `REVISE` with structured issues).
5. For a two-cycle fixture, `builder_2` additionally consumes the first builder output and
   reviewer issues; `domain_review_2` reviews only the revised, validated artifacts.
6. Final safety consumes the last reviewed builder artifacts and the last reviewer verdict.
   No stage is generated as an independent question-answer pair.

The exact `input_refs` for every stage are frozen per task in
`configs/curriculum/collaboration_v2.yaml`. The validator rejects a missing or reordered
lineage reference.

## Fixtures and split safety

There are 15 unique candidate fixtures across frontend/web, Python CLI, Node/TypeScript,
code repair, accessibility, and inert security tasks. Every fixture has its own `TASK.md`,
`context.json`, starter source, package scripts, and matching public test. The manifest
freezes SHA-256 digests for those files and for all build/test/lint script strings.

Starter code is intentionally incomplete: build and lint must pass while public tests must
fail. A successful teacher execution must make all three validators pass without changing
protected public contracts. Inert security fixtures contain labels and harmless markers
only; they contain no live payload.

Held-out files remain outside the candidate fixture tree. The validator reads them only as
a leak-check oracle, verifies their frozen hashes, and rejects held-out identifiers or
near-duplicate requirements in candidate content. They are never copied into prompts or
training records.

## Validation

From the repository root on this Windows checkout:

```powershell
$env:PYTHONPATH = "src"
py -3 scripts/data/validate_collaboration_curriculum.py --run-fixtures
py -3 -m pytest tests/test_curriculum.py -q
py -3 -m ruff check src/anchor_mvp/curriculum.py scripts/data/validate_collaboration_curriculum.py scripts/data/build_collaboration_curriculum_manifest.py tests/test_curriculum.py
```

`--run-fixtures` executes 45 commands: build, starter test, and lint for every fixture. A
starter test failure is expected and proves the public task still requires an implementation.
After an intentional fixture or task change, regenerate the mechanical hashes with
`scripts/data/build_collaboration_curriculum_manifest.py`, review the resulting manifest
diff, then run validation again. Never regenerate hashes merely to silence unexpected drift.

The curriculum defaults to one accepted-seed stage. Operators may configure later positive
stages, but each gate must retain whole trajectories; partial stages and reviewer-only
samples do not count as accepted seeds.
