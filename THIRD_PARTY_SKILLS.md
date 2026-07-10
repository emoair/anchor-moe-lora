# Third-party Skill acknowledgements

Anchor-MoE-LoRA uses selected, pinned Skill files as SOP inputs for synthetic execution experiments. GitHub stars are used only for discovery. Every injected file is pinned by repository commit and SHA-256 in `configs/data/skill_sources.yaml`; the deterministic local tool policy remains authoritative.

## GitHub awesome-copilot

- Source: https://github.com/github/awesome-copilot
- Pinned commit: `30472ecf0fe34cc561df958c08501ecc5ca80ea4`
- Selected Skills: `premium-frontend-ui`, `review-and-refactor`, `security-review`
- License: MIT. The full license is preserved at `third_party/skills/github-awesome-copilot/LICENSE`.
- Additional credit: `premium-frontend-ui` metadata names Utkarsh Patrikar as its author.

## Anthropic skills

- Source: https://github.com/anthropics/skills
- Pinned commit: `9d2f1ae187231d8199c64b5b762e1bdf2244733d`
- Selected Skill: `frontend-design`
- License: Apache-2.0. The complete per-Skill license is preserved beside the vendored Skill.

Anthropic's repository contains mixed licensing. Only the individually audited Apache-2.0 Skill above is included; source-available document Skills are not imported.

## Deferred source

`vercel-labs/agent-skills` was discovered and pinned for later review, but its audited checkout did not include a complete root license file. It is not copied or injected in this revision even though individual metadata and the README mention MIT.
