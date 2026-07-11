import json
from pathlib import Path

from anchor_mvp.tooling import ToolPolicy, build_opencode_config
from anchor_mvp.tooling.config import validate_base_url


ROOT = Path(__file__).resolve().parents[1]


def test_keyless_kimi_config_matches_checked_in_example():
    generated = build_opencode_config(ToolPolicy())
    example = json.loads(
        (ROOT / "configs" / "tooling" / "opencode_kimi.example.json").read_text(
            encoding="utf-8"
        )
    )

    assert generated == example
    serialized = json.dumps(generated)
    assert "sk-" not in serialized
    assert generated["provider"]["kimi-code"]["options"] == {
        "baseURL": "https://api.kimi.com/coding/v1",
        "apiKey": "{env:KIMI_CODE_API_KEY}",
    }
    assert "headers" not in generated["provider"]["kimi-code"]["options"]
    assert generated["agent"]["anchor-gold"]["steps"] == 8
    assert generated["share"] == "disabled"
    model = generated["provider"]["kimi-code"]["models"]["kimi-for-coding"]
    assert model["variants"] == {
        "thinking": {"reasoningEffort": "high"},
    }


def test_invalid_or_descriptive_urls_are_rejected_before_requests():
    for value in (
        "the repo for the contents of the path",
        "api.kimi.com/coding/v1",
        "https://the repo for the contents of the path",
        "http://api.kimi.com/coding/v1",
    ):
        try:
            validate_base_url(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"URL should have been rejected: {value}")
