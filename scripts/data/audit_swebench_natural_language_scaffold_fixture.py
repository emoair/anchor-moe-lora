#!/usr/bin/env python3
"""Audit the physical integrity of a natural-language scaffold artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from anchor_mvp.swebench.natural_language_scaffold import (  # noqa: E402
    NaturalLanguageScaffoldConfig,
    NaturalLanguageScaffoldError,
)


_MAX_METADATA_BYTES = 8 * 1024 * 1024
_FIXED_DEFAULT_FILES = (
    "train/json_only.jsonl",
    "train/concise_rationale_plus_json.jsonl",
    "calibration/json_only.jsonl",
    "calibration/concise_rationale_plus_json.jsonl",
)
_DENIED_KEYS = {
    "answer",
    "answer_body",
    "block_content",
    "content_preview",
    "current_target",
    "future_content",
    "heldout_content",
    "prompt",
    "prompt_text",
    "token_ids",
    "token_index",
    "token_indices",
}
_TOKEN_POSITION_KEYS = {
    "activation_token_ids",
    "boundary_token_index",
    "invocation_token_ids",
    "invocation_token_index",
    "position_ids",
    "token_ids",
    "token_index",
    "token_indices",
    "token_offset",
    "token_offsets",
}


class AuditError(ValueError):
    """A stable, body-free audit failure."""

    def __init__(self, code: str) -> None:
        self.code = code
        super().__init__(code)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _safe_file(root: Path, relative: str) -> Path:
    if not relative or "\\" in relative:
        raise AuditError("natural_language_scaffold_audit_path_invalid")
    candidate = root.joinpath(*relative.split("/"))
    if any(part in {"", ".", ".."} for part in Path(relative).parts):
        raise AuditError("natural_language_scaffold_audit_path_invalid")
    current = root
    for part in relative.split("/"):
        current = current / part
        if current.is_symlink():
            raise AuditError("natural_language_scaffold_audit_symlink_rejected")
    try:
        if not candidate.is_file() or not candidate.resolve().is_relative_to(
            root.resolve()
        ):
            raise AuditError("natural_language_scaffold_audit_path_invalid")
    except OSError as exc:
        raise AuditError("natural_language_scaffold_audit_path_invalid") from exc
    return candidate


def _read_limited(path: Path) -> bytes:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise AuditError("natural_language_scaffold_audit_read_failed") from exc
    if size > _MAX_METADATA_BYTES:
        raise AuditError("natural_language_scaffold_audit_file_too_large")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise AuditError("natural_language_scaffold_audit_read_failed") from exc
    if len(data) != size:
        raise AuditError("natural_language_scaffold_audit_snapshot_changed")
    return data


def _contains_denied_key(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            str(key).casefold() in _DENIED_KEYS or _contains_denied_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_denied_key(item) for item in value)
    return False


def _walk_keys(value: object) -> set[str]:
    if isinstance(value, Mapping):
        return {
            *(str(key) for key in value),
            *(key for item in value.values() for key in _walk_keys(item)),
        }
    if isinstance(value, list):
        return {key for item in value for key in _walk_keys(item)}
    return set()


def _validate_record(row: Mapping[str, Any]) -> None:
    if _contains_denied_key(row) or (_TOKEN_POSITION_KEYS & _walk_keys(row)):
        raise AuditError("natural_language_scaffold_audit_body_policy_invalid")
    if (
        row.get("provider_requests") != 0
        or row.get("evaluation_status") != "not_evaluated"
        or row.get("quality_validated") is not False
        or row.get("execution_authorized") is not False
        or row.get("training_outcome_claimed") is not False
    ):
        raise AuditError("natural_language_scaffold_audit_claim_invalid")

    routing = row.get("routing_json")
    calls = row.get("tool_calls")
    results = row.get("tool_results")
    trigger = row.get("expert_trigger")
    if (
        not isinstance(routing, dict)
        or not isinstance(calls, list)
        or not isinstance(results, list)
        or not isinstance(trigger, dict)
    ):
        raise AuditError("natural_language_scaffold_audit_record_invalid")
    if row.get("routing_json_sha256") != _sha256(_canonical(routing)):
        raise AuditError("natural_language_scaffold_audit_record_hash_invalid")
    trigger_text = trigger.get("trigger_text")
    if not isinstance(trigger_text, str) or trigger.get(
        "trigger_text_sha256"
    ) != _sha256(trigger_text.encode("utf-8")):
        raise AuditError("natural_language_scaffold_audit_record_hash_invalid")
    payload = {
        "routing_json": routing,
        "tool_calls": calls,
        "tool_results": results,
        "expert_trigger": trigger,
    }
    payload_bytes = _canonical(payload)
    if row.get("canonical_json_payload_sha256") != _sha256(payload_bytes):
        raise AuditError("natural_language_scaffold_audit_record_hash_invalid")
    scaffold_text = row.get("scaffold_text")
    if not isinstance(scaffold_text, str) or row.get("scaffold_text_sha256") != _sha256(
        scaffold_text.encode("utf-8")
    ):
        raise AuditError("natural_language_scaffold_audit_record_hash_invalid")
    variant = row.get("scaffold_variant")
    payload_text = payload_bytes.decode("utf-8")
    if variant == "json_only":
        if "concise_rationale_summary" in row or scaffold_text != payload_text:
            raise AuditError("natural_language_scaffold_audit_pair_invalid")
    elif variant == "concise_rationale_plus_json":
        rationale = row.get("concise_rationale_summary")
        if (
            not isinstance(rationale, str)
            or not rationale
            or len(rationale.encode("utf-8")) > 512
            or scaffold_text != rationale + "\n" + payload_text
        ):
            raise AuditError("natural_language_scaffold_audit_pair_invalid")
    else:
        raise AuditError("natural_language_scaffold_audit_pair_invalid")

    route = row.get("route_boundary")
    cache = row.get("cache_metadata")
    alora = row.get("alora_invocation")
    if not all(isinstance(item, dict) for item in (route, cache, alora)):
        raise AuditError("natural_language_scaffold_audit_record_invalid")
    if (
        route.get("semantics") != "explicit_two_request_commit_boundary"
        or route.get("validation_required") is not True
        or route.get("commit_required") is not True
        or route.get("commit_promotes_text_only") is not True
        or route.get("planner_private_tail_kv_transfer_allowed") is not False
        or route.get("committed_scaffold_reencode_required") is not True
        or route.get("committed_scaffold_reencode_producer") != "frozen_base"
        or route.get("committed_scaffold_reencode_adapter_state") != "off"
        or route.get("expert_request_phase") != "next_request"
        or route.get("expert_request_requires_committed_scaffold_as_input") is not True
    ):
        raise AuditError("natural_language_scaffold_audit_route_invalid")
    if (
        cache.get("adapter_state_on_prefix") != "off"
        or cache.get("adapter_state_after_boundary") != "expert_only"
        or cache.get("private_tail_kv_required") is not True
        or cache.get("full_generation_kv_shared_claimed") is not False
        or cache.get("exact_reuse_scope") != "identical_ordered_prefix_lineage_only"
        or cache.get("exact_cache_reuse_enabled") is not False
        or cache.get("reuse_savings_tokens") != 0
        or cache.get("planner_private_tail_kv_reused_by_expert") is not False
        or cache.get("physical_kv_tensor_emitted") is not False
    ):
        raise AuditError("natural_language_scaffold_audit_cache_invalid")
    if (
        alora.get("activation_semantics") != "next_request_input_activation_only"
        or alora.get("invocation_scan_scope") != "new_request_input_tokens_only"
        or alora.get("same_request_activation_allowed") is not False
        or alora.get("mid_request_generated_activation_allowed") is not False
        or alora.get("mid_request_generated_trigger_switch_claimed") is not False
        or alora.get("explicit_commit_required") is not True
        or alora.get("adapter_available") is not False
        or alora.get("adapter_loaded") is not False
        or alora.get("activation_executed") is not False
        or alora.get("cross_attention_q_reader_claimed") is not False
        or alora.get("physical_shared_kv_claimed") is not False
    ):
        raise AuditError("natural_language_scaffold_audit_alora_invalid")


def _pair_normal_form(row: Mapping[str, Any]) -> dict[str, Any]:
    ignored = {
        "record_id",
        "scaffold_variant",
        "concise_rationale_summary",
        "scaffold_text",
        "scaffold_text_sha256",
    }
    return {key: value for key, value in row.items() if key not in ignored}


def _load_fixed_files(config_path: Path) -> tuple[str, ...]:
    try:
        config = yaml.safe_load(_read_limited(config_path))
        fixed = config["output_contract"]["fixed_files"]
    except (AuditError, KeyError, TypeError, yaml.YAMLError) as exc:
        if isinstance(exc, AuditError):
            raise
        raise AuditError("natural_language_scaffold_audit_config_invalid") from exc
    if not isinstance(fixed, list) or any(not isinstance(item, str) for item in fixed):
        raise AuditError("natural_language_scaffold_audit_config_invalid")
    result = tuple(fixed)
    if result != _FIXED_DEFAULT_FILES:
        raise AuditError("natural_language_scaffold_audit_fixed_files_invalid")
    return result


def audit_fixture(
    config_path: Path,
    artifact_dir: Path,
    expected_manifest_sha256: str,
) -> dict[str, Any]:
    fixed_files = _load_fixed_files(config_path)
    try:
        config, _inventory = NaturalLanguageScaffoldConfig.load(config_path)
    except NaturalLanguageScaffoldError as exc:
        raise AuditError("natural_language_scaffold_audit_config_invalid") from exc
    if len(expected_manifest_sha256) != 64 or any(
        char not in "0123456789abcdef" for char in expected_manifest_sha256
    ):
        raise AuditError("natural_language_scaffold_audit_manifest_sha_invalid")
    if artifact_dir.is_symlink() or not artifact_dir.is_dir():
        raise AuditError("natural_language_scaffold_audit_root_invalid")

    manifest_path = _safe_file(artifact_dir, "manifest.json")
    sidecar_path = _safe_file(artifact_dir, "manifest.json.sha256")
    manifest_bytes = _read_limited(manifest_path)
    actual_manifest_sha256 = _sha256(manifest_bytes)
    if actual_manifest_sha256 != expected_manifest_sha256:
        raise AuditError("natural_language_scaffold_audit_manifest_sha_invalid")
    expected_sidecar = f"{actual_manifest_sha256}  manifest.json\n".encode("ascii")
    if _read_limited(sidecar_path) != expected_sidecar:
        raise AuditError("natural_language_scaffold_audit_sidecar_invalid")

    try:
        manifest = json.loads(manifest_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise AuditError("natural_language_scaffold_audit_manifest_invalid") from exc
    if not isinstance(manifest, dict) or _contains_denied_key(manifest):
        raise AuditError("natural_language_scaffold_audit_manifest_invalid")
    producer = manifest.get("producer")
    if (
        not isinstance(producer, dict)
        or producer.get("config_sha256") != config.sha256
        or producer.get("implementation_sha256") != config.implementation_sha256
        or producer.get("record_schema_sha256") != config.record_schema_sha256
        or producer.get("manifest_schema_sha256") != config.manifest_schema_sha256
        or producer.get("smoke_contract_schema_sha256") != config.smoke_schema_sha256
        or producer.get("smoke_contract_sha256") != config.smoke_config_sha256
    ):
        raise AuditError("natural_language_scaffold_audit_contract_hash_invalid")
    if manifest.get("provider_requests") != 0:
        raise AuditError("natural_language_scaffold_audit_nonzero_requests")
    if manifest.get("model_loads") != 0 or manifest.get("gpu_requests") != 0:
        raise AuditError("natural_language_scaffold_audit_nonzero_resources")
    if manifest.get("network_requests") != 0:
        raise AuditError("natural_language_scaffold_audit_nonzero_requests")

    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise AuditError("natural_language_scaffold_audit_manifest_invalid")
    by_path = {
        item.get("path"): item
        for item in entries
        if isinstance(item, dict) and isinstance(item.get("path"), str)
    }
    if (
        tuple(item.get("path") for item in entries if isinstance(item, dict))
        != fixed_files
    ):
        raise AuditError("natural_language_scaffold_audit_fixed_files_invalid")
    if set(by_path) != set(fixed_files):
        raise AuditError("natural_language_scaffold_audit_fixed_files_invalid")

    total_records = 0
    seen_pairs: dict[str, list[dict[str, Any]]] = {}
    for relative in fixed_files:
        path = _safe_file(artifact_dir, relative)
        data = _read_limited(path)
        entry = by_path[relative]
        if entry.get("sha256") != _sha256(data) or entry.get("bytes") != len(data):
            raise AuditError("natural_language_scaffold_audit_file_hash_invalid")
        rows: list[dict[str, Any]] = []
        try:
            for raw_line in data.splitlines():
                if not raw_line:
                    raise AuditError("natural_language_scaffold_audit_jsonl_invalid")
                value = json.loads(raw_line)
                if not isinstance(value, dict) or _contains_denied_key(value):
                    raise AuditError(
                        "natural_language_scaffold_audit_body_policy_invalid"
                    )
                _validate_record(value)
                rows.append(value)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AuditError("natural_language_scaffold_audit_jsonl_invalid") from exc
        if entry.get("records") != len(rows):
            raise AuditError("natural_language_scaffold_audit_file_count_invalid")
        total_records += len(rows)
        for row in rows:
            pair_id = row.get("pair_id")
            variant = row.get("scaffold_variant")
            if not isinstance(pair_id, str) or variant not in {
                "json_only",
                "concise_rationale_plus_json",
            }:
                raise AuditError("natural_language_scaffold_audit_pair_invalid")
            seen_pairs.setdefault(pair_id, []).append(row)

    counts = manifest.get("counts")
    if not isinstance(counts, dict) or counts.get("total") != total_records:
        raise AuditError("natural_language_scaffold_audit_file_count_invalid")
    if total_records != 20 or len(seen_pairs) != 10:
        raise AuditError("natural_language_scaffold_audit_fixture_count_invalid")
    for paired in seen_pairs.values():
        if (
            len(paired) != 2
            or {row["scaffold_variant"] for row in paired}
            != {"json_only", "concise_rationale_plus_json"}
            or _pair_normal_form(paired[0]) != _pair_normal_form(paired[1])
        ):
            raise AuditError("natural_language_scaffold_audit_pair_invalid")

    return {
        "manifest_sha256": actual_manifest_sha256,
        "records": total_records,
        "pairs": len(seen_pairs),
        "provider_requests": 0,
        "audit_passed": True,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit hashes, counts, pairing, resource gates and body-free key policy "
            "for a published natural-language scaffold fixture."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = audit_fixture(args.config, args.artifact_dir, args.manifest_sha256)
    except AuditError as exc:
        print(exc.code, file=sys.stderr)
        return 2
    except Exception:
        print("natural_language_scaffold_audit_internal_error", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
