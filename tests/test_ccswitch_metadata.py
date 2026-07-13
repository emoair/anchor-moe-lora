from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from anchor_mvp.integrations.ccswitch_metadata.constants import (
    EXPECTED_SOURCE_FILES,
    SOURCE_COMMIT,
    SOURCE_TAG,
)
from anchor_mvp.integrations.ccswitch_metadata.pricing import (
    estimate_cost,
    resolve_model_id,
)
from anchor_mvp.integrations.ccswitch_metadata.schema import (
    SchemaError,
    safe_json_bytes,
    validate_snapshot,
)
from anchor_mvp.integrations.ccswitch_metadata.sync import (
    IntegrityError,
    MetadataStore,
    NetworkUnavailable,
    SourceVerifier,
    load_bundled_snapshot,
    resolve_candidate,
    semantic_diff,
    snapshot_sha256,
)


def test_bundled_snapshot_is_pinned_complete_and_secret_free() -> None:
    snapshot = load_bundled_snapshot()

    validate_snapshot(snapshot)
    assert snapshot["source"]["source_tag"] == SOURCE_TAG
    assert snapshot["source"]["source_commit"] == SOURCE_COMMIT
    assert {item["path"] for item in snapshot["source"]["files"]} == set(
        EXPECTED_SOURCE_FILES
    )
    assert len(snapshot["pricing"]) == len(snapshot["models"])
    serialized = safe_json_bytes(snapshot)
    lowered = serialized.lower()
    assert b"api_key" not in lowered
    assert b"authorization" not in lowered
    assert b"<" not in serialized
    assert b">" not in serialized


def test_schema_rejects_unknown_fields_duplicate_aliases_and_units() -> None:
    snapshot = load_bundled_snapshot()
    with_unknown = deepcopy(snapshot)
    with_unknown["providers"][0]["surprise"] = True
    with pytest.raises(SchemaError, match="unknown fields"):
        validate_snapshot(with_unknown)

    with_duplicate = deepcopy(snapshot)
    with_duplicate["model_aliases"].append(deepcopy(with_duplicate["model_aliases"][0]))
    with pytest.raises(SchemaError, match="duplicate model alias"):
        validate_snapshot(with_duplicate)

    with_bad_unit = deepcopy(snapshot)
    with_bad_unit["pricing"][0]["basis"] = "per_token"
    with pytest.raises(SchemaError, match="per_1m_tokens"):
        validate_snapshot(with_bad_unit)


def test_schema_requires_explicit_unknown_prices() -> None:
    snapshot = load_bundled_snapshot()
    broken = deepcopy(snapshot)
    unknown = next(item for item in broken["pricing"] if item["model_id"] == "glm-5.1")
    unknown["input"] = "0"

    with pytest.raises(SchemaError, match="must all be unknown"):
        validate_snapshot(broken)


def test_decimal_estimator_resolves_exact_alias_and_preserves_cache_semantics() -> None:
    snapshot = load_bundled_snapshot()
    assert resolve_model_id(snapshot, "gpt-5.5-low") == "gpt-5.5"

    estimate = estimate_cost(
        snapshot,
        request_model_id="gpt-5.5-low",
        protocol="openai_compatible",
        input_tokens=1_000_000,
        output_tokens=100_000,
        cache_read_tokens=200_000,
        multiplier="2",
    )
    assert estimate["known"] is True
    assert estimate["canonical_model_id"] == "gpt-5.5"
    assert estimate["billable_tokens"]["input"] == 800_000
    assert estimate["components"] == {
        "input": "4",
        "output": "3",
        "cache_read": "0.1",
        "cache_write": "0",
    }
    assert estimate["total"] == "14.2"

    anthropic = estimate_cost(
        snapshot,
        request_model_id="anthropic/claude-sonnet-5",
        provider_id="openrouter",
        protocol="anthropic",
        input_tokens=1_000_000,
        output_tokens=100_000,
        cache_read_tokens=200_000,
        cache_write_tokens=50_000,
    )
    assert anthropic["billable_tokens"]["input"] == 1_000_000
    assert anthropic["total"] == "4.7475"


def test_unknown_price_is_unavailable_not_zero() -> None:
    estimate = estimate_cost(
        load_bundled_snapshot(),
        request_model_id="glm-5.1",
        provider_id="zhipu-glm",
        protocol="openai_compatible",
        input_tokens=1,
        output_tokens=1,
    )

    assert estimate["known"] is False
    assert estimate["reason"] == "unknown_price"
    assert estimate["total"] is None
    assert estimate["unknown_dimensions"] == ["input", "output"]


def test_etag_304_requires_and_reuses_a_matching_verified_record(
    tmp_path: Path,
) -> None:
    snapshot = load_bundled_snapshot()
    records = {
        item["path"]: {
            "url": item["url"],
            "sha256": item["sha256"],
            "size": item["size"],
            "etag": '"old-etag"',
            "verified_at": "2026-07-13T00:00:00Z",
        }
        for item in snapshot["source"]["files"]
    }
    (tmp_path / "verification_cache.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_tag": SOURCE_TAG,
                "source_commit": SOURCE_COMMIT,
                "records": records,
            }
        ),
        encoding="utf-8",
    )
    calls: list[tuple[str, str | None]] = []

    def not_modified(
        url: str, etag: str | None, maximum: int
    ) -> tuple[int, bytes, str | None, str]:
        calls.append((url, etag))
        assert maximum >= 150_275
        return 304, b"", '"new-etag"', url

    results = SourceVerifier(tmp_path, fetch=not_modified).verify(snapshot)

    assert len(results) == len(EXPECTED_SOURCE_FILES)
    assert all(result["status"] == "not_modified" for result in results)
    assert all(etag == '"old-etag"' for _, etag in calls)
    updated = json.loads((tmp_path / "verification_cache.json").read_text("utf-8"))
    assert all(record["etag"] == '"new-etag"' for record in updated["records"].values())


def test_integrity_mismatch_fails_closed(tmp_path: Path) -> None:
    def tampered(
        url: str, etag: str | None, maximum: int
    ) -> tuple[int, bytes, str | None, str]:
        return 200, b"tampered", None, url

    with pytest.raises(IntegrityError, match="size mismatch"):
        resolve_candidate(tmp_path, fetch=tampered)


def test_network_failure_uses_last_verified_snapshot(tmp_path: Path) -> None:
    snapshot = load_bundled_snapshot()
    (tmp_path / "last_verified_snapshot.json").write_bytes(safe_json_bytes(snapshot))

    def unavailable(
        url: str, etag: str | None, maximum: int
    ) -> tuple[int, bytes, str | None, str]:
        raise NetworkUnavailable("offline test")

    candidate = resolve_candidate(tmp_path, fetch=unavailable)

    assert candidate.origin == "last_verified_snapshot"
    assert candidate.verification == "network_fallback"
    assert candidate.warning == "offline test"
    assert candidate.sha256 == snapshot_sha256(snapshot)


def test_apply_is_idempotent_and_rollback_restores_previous_snapshot(
    tmp_path: Path,
) -> None:
    store = MetadataStore(tmp_path)
    first = load_bundled_snapshot()
    first_sha = snapshot_sha256(first)

    applied = store.apply(first)
    assert applied["changed"] is True
    assert store.current() is not None
    assert store.apply(first)["changed"] is False

    second = deepcopy(first)
    second["providers"][0]["display_name"] = "DeepSeek Audited"
    validate_snapshot(second)
    difference = semantic_diff(first, second)
    assert difference["sections"]["providers"]["changed"] == ["deepseek"]
    store.apply(second)
    assert snapshot_sha256(store.current()) == snapshot_sha256(second)  # type: ignore[arg-type]

    restored = store.rollback()
    assert restored["sha256"] == first_sha
    assert snapshot_sha256(store.current()) == first_sha  # type: ignore[arg-type]

    removed = store.rollback()
    assert removed["target_exists"] is False
    assert store.current() is None
