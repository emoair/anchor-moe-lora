from __future__ import annotations

from hashlib import sha256
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from anchor_mvp.data.mutator import MutationUnavailableError, mutate_frontend_code  # noqa: E402


def test_mutator_prefers_one_literal_aria_label_and_is_deterministic() -> None:
    source = (
        'export function Search(){return <main><input aria-label="Search projects" />'
        '<button aria-label="Submit">Go</button></main>}'
    )
    first_code, first = mutate_frontend_code(source, source_record_id="frontend-1")
    second_code, second = mutate_frontend_code(source, source_record_id="frontend-1")

    assert first_code == second_code
    assert first == second
    assert first.rule == "remove_literal_aria_label"
    assert first.count == 1
    assert first.path == "output.code"
    assert first.sha256_before == sha256(source.encode()).hexdigest()
    assert first.sha256_after == sha256(first_code.encode()).hexdigest()
    assert first_code.count("aria-label") == 1
    assert "<script" not in first_code.casefold()


def test_mutator_fallback_preserves_balanced_component_text() -> None:
    source = "export function Page(){return <main><h1>Status</h1><p>Ready</p></main>}"
    candidate, manifest = mutate_frontend_code(source, source_record_id="frontend-2")

    assert manifest.rule == "semantic_main_to_div"
    assert manifest.count == 2
    assert candidate == "export function Page(){return <div><h1>Status</h1><p>Ready</p></div>}"
    assert candidate.count("<div") == candidate.count("</div>")
    assert candidate.count("{") == candidate.count("}")


def test_mutator_refuses_to_invent_non_allowlisted_defect() -> None:
    with pytest.raises(MutationUnavailableError, match="no allowlisted"):
        mutate_frontend_code("export const value = 1", source_record_id="frontend-3")

