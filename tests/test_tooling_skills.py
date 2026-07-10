from pathlib import Path

import pytest

from anchor_mvp.tooling import SkillSourceError, SkillSourceRegistry


ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "configs" / "data" / "skill_sources.yaml"


def test_audited_skill_registry_validates_and_composes_public_prompt():
    registry = SkillSourceRegistry(ROOT, REGISTRY)

    prompt, provenance = registry.compose_execution_prompt(
        "Refactor the fixture and run its tests.",
        ("github-awesome-copilot-review-and-refactor",),
    )

    assert "private chain-of-thought" in prompt
    assert "deterministic tool policy is final" in prompt
    assert "Refactor the fixture" in prompt
    assert provenance[0].commit == "30472ecf0fe34cc561df958c08501ecc5ca80ea4"
    assert provenance[0].license == "MIT"
    assert len(provenance[0].bundle_sha256) == 64


def test_registry_fails_closed_when_vendored_file_hash_changes(tmp_path):
    project = tmp_path / "project"
    skill = project / "third_party" / "skill.md"
    license_file = project / "third_party" / "LICENSE"
    skill.parent.mkdir(parents=True)
    skill.write_text("changed", encoding="utf-8")
    license_file.write_text("license", encoding="utf-8")
    registry = project / "registry.yaml"
    registry.write_text(
        """schema_version: anchor.skill-sources.v1
sources:
  - id: demo
    repository: https://example.invalid/demo
    commit: '0000000000000000000000000000000000000000'
    license: MIT
    license_path: third_party/LICENSE
    license_sha256: '0000000000000000000000000000000000000000000000000000000000000000'
    files:
      - path: third_party/skill.md
        sha256: '0000000000000000000000000000000000000000000000000000000000000000'
        inject: true
""",
        encoding="utf-8",
    )

    with pytest.raises(SkillSourceError, match="license file is missing or changed"):
        SkillSourceRegistry(project, registry).load("demo")
