"""Reachable additive v3 live entry for candidate then external release.

The authenticated v2 consumer prerequisite remains unchanged.  This wrapper
reuses the frozen exact1 -> bounded15 -> bulk -> candidate controller, proves
that its runtime secret slots have closed, and only then invokes the corrected
independent external-release v3 implementation.  A release failure keeps the
combined status non-complete and therefore cannot authorize training.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Callable, Mapping, Sequence

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from anchor_mvp.data import gemma3_chat_unbalanced_v2_batch as batch
from anchor_mvp.data import (
    gemma3_chat_unbalanced_v2_consumer_identity_rollover_v2 as rollover_v2,
)
from anchor_mvp.data import (
    gemma3_chat_unbalanced_v2_integrated_controller_v2 as integrated,
)
from anchor_mvp.data import gemma3_chat_unbalanced_v2_live_controller as base
from anchor_mvp.data import (
    gemma3_chat_unbalanced_v2_teacher_final_release_consumer_rollover_v3 as release_v3,
)


CONFIG_PATH = (
    batch.REPO_ROOT
    / "configs/data/gemma3_chat_unbalanced_v2_consumer_identity_rollover_v3.json"
)
SCHEMA_PATH = (
    batch.REPO_ROOT / "configs/data/"
    "gemma3_chat_unbalanced_v2_consumer_identity_rollover_v3.schema.json"
)
STATUS_SCHEMA_VERSION = (
    "anchor.gemma3-chat-unbalanced-v2-consumer-identity-rollover-status.v3"
)
ReleasePublisher = Callable[..., Mapping[str, Any]]


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _strict_document(raw: bytes, *, reason: str) -> dict[str, Any]:
    value = base._strict_json(raw, reason=reason)
    if not isinstance(value, dict):
        raise batch.AdapterError(reason)
    return value


def _validate(schema: Mapping[str, Any], value: Any, *, reason: str) -> None:
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(value)
    except (SchemaError, ValidationError) as error:
        raise batch.AdapterError(reason) from error


def _repo_file(relative: str, *, reason: str) -> Path:
    if (
        not isinstance(relative, str)
        or relative != relative.strip()
        or not relative
        or "\\" in relative
    ):
        raise batch.AdapterError(reason)
    pure = PurePosixPath(relative)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        raise batch.AdapterError(reason)
    path = batch.REPO_ROOT.joinpath(*pure.parts).absolute()
    try:
        path.relative_to(batch.REPO_ROOT.absolute())
    except ValueError as error:
        raise batch.AdapterError(reason) from error
    base._reject_reparse_chain(path)
    return path


def _authenticate_pin(pin: Mapping[str, Any], *, label: str) -> bytes:
    path = _repo_file(
        str(pin["path"]),
        reason=f"rollover_v3_{label}_path_invalid",
    )
    raw = base._stable_read(
        path,
        reason=f"rollover_v3_{label}_snapshot_drift",
    )
    if len(raw) != pin["bytes"] or _sha256(raw) != pin["sha256"]:
        raise batch.AdapterError(f"rollover_v3_{label}_identity_drift")
    return raw


def _load_contract() -> tuple[dict[str, Any], bytes, bytes, dict[str, bytes]]:
    config_raw = base._stable_read(
        CONFIG_PATH,
        reason="rollover_v3_config_snapshot_drift",
    )
    schema_raw = base._stable_read(
        SCHEMA_PATH,
        reason="rollover_v3_schema_snapshot_drift",
    )
    config = _strict_document(config_raw, reason="rollover_v3_config_invalid")
    schema = _strict_document(schema_raw, reason="rollover_v3_schema_invalid")
    _validate(schema, config, reason="rollover_v3_contract_schema_rejected")
    snapshots = {
        label: _authenticate_pin(pin, label=label)
        for label, pin in config["authenticated_artifacts"].items()
    }
    artifacts = config["authenticated_artifacts"]
    if (
        rollover_v2.CONFIG_PATH
        != _repo_file(
            artifacts["source_rollover_config"]["path"],
            reason="rollover_v3_source_config_path_invalid",
        )
        or rollover_v2.SCHEMA_PATH
        != _repo_file(
            artifacts["source_rollover_schema"]["path"],
            reason="rollover_v3_source_schema_path_invalid",
        )
        or Path(rollover_v2.__file__).absolute()
        != _repo_file(
            artifacts["source_rollover_implementation"]["path"],
            reason="rollover_v3_source_module_path_invalid",
        )
        or rollover_v2.INTEGRATED_CONFIG_PATH
        != _repo_file(
            artifacts["integrated_controller_config"]["path"],
            reason="rollover_v3_integrated_config_path_invalid",
        )
        or Path(__file__).absolute()
        != _repo_file(
            artifacts["rollover_v3_implementation"]["path"],
            reason="rollover_v3_implementation_path_invalid",
        )
        or Path(release_v3.__file__).absolute()
        != _repo_file(
            artifacts["release_implementation"]["path"],
            reason="rollover_v3_release_module_path_invalid",
        )
        or release_v3.ATTESTATION_SCHEMA_PATH
        != _repo_file(
            artifacts["attestation_schema"]["path"],
            reason="rollover_v3_attestation_schema_path_invalid",
        )
        or release_v3.RELEASE_MANIFEST_SCHEMA_PATH
        != _repo_file(
            artifacts["release_manifest_schema"]["path"],
            reason="rollover_v3_release_manifest_schema_path_invalid",
        )
        or release_v3.BINDING_SCHEMA_PATH
        != _repo_file(
            artifacts["binding_schema"]["path"],
            reason="rollover_v3_binding_schema_path_invalid",
        )
    ):
        raise batch.AdapterError("rollover_v3_release_module_binding_drift")
    manifest_schema = _strict_document(
        snapshots["release_manifest_schema"],
        reason="rollover_v3_release_manifest_schema_invalid",
    )
    binding_path = manifest_schema["properties"]["binding_contract"]["properties"][
        "schema_path"
    ]["const"]
    if (
        binding_path != config["authenticated_artifacts"]["binding_schema"]["path"]
        or _sha256(snapshots["binding_schema"])
        != "491eb5a085884a17fcd02ec7335f0ff226a77bf701dd984947175c04e895faca"
    ):
        raise batch.AdapterError("rollover_v3_binding_schema_identity_drift")
    return config, config_raw, schema_raw, snapshots


def _recheck_contract(
    config_raw: bytes,
    schema_raw: bytes,
    snapshots: Mapping[str, bytes],
) -> None:
    if (
        base._stable_read(
            CONFIG_PATH,
            reason="rollover_v3_config_terminal_drift",
        )
        != config_raw
        or base._stable_read(
            SCHEMA_PATH,
            reason="rollover_v3_schema_terminal_drift",
        )
        != schema_raw
    ):
        raise batch.AdapterError("rollover_v3_contract_terminal_drift")
    config = _strict_document(config_raw, reason="rollover_v3_config_invalid")
    observed = {
        label: _authenticate_pin(pin, label=label)
        for label, pin in config["authenticated_artifacts"].items()
    }
    if observed != dict(snapshots):
        raise batch.AdapterError("rollover_v3_artifact_terminal_drift")


def validate_rollover_v3() -> None:
    config, config_raw, schema_raw, snapshots = _load_contract()
    del config
    rollover_v2.validate_consumer_identity_rollover()
    _recheck_contract(config_raw, schema_raw, snapshots)


def preflight(*, validate_source: bool) -> dict[str, Any]:
    config, config_raw, schema_raw, snapshots = _load_contract()
    result = dict(rollover_v2.preflight(validate_source=validate_source))
    _recheck_contract(config_raw, schema_raw, snapshots)
    result.update(
        {
            "schema_version": STATUS_SCHEMA_VERSION,
            "rollover_v3_config_sha256": _sha256(config_raw),
            "rollover_v3_release_implementation_sha256": config[
                "authenticated_artifacts"
            ]["release_implementation"]["sha256"],
            "rollover_v3_release_manifest_schema_sha256": config[
                "authenticated_artifacts"
            ]["release_manifest_schema"]["sha256"],
            "rollover_v3_binding_schema_sha256": config["authenticated_artifacts"][
                "binding_schema"
            ]["sha256"],
            "candidate_materialization_before_runtime_slots_close": True,
            "external_release_after_runtime_slots_close": True,
            "external_release_required_for_complete": True,
            "training_authorized": False,
            "formal_training_authorized": False,
            "live_authorized": False,
        }
    )
    return result


def _external_release_after_close(
    controller_report: Mapping[str, Any],
    slots: batch.RuntimeSecretSlots,
    *,
    implementer_id: str,
    reviewer_id: str,
    publisher: ReleasePublisher | None = None,
) -> dict[str, Any]:
    if (
        release_v3.SAFE_IDENTIFIER.fullmatch(implementer_id) is None
        or release_v3.SAFE_IDENTIFIER.fullmatch(reviewer_id) is None
        or implementer_id.casefold() == reviewer_id.casefold()
    ):
        raise batch.AdapterError("release_reviewer_separation_invalid")
    if slots.loaded:
        raise batch.AdapterError("rollover_v3_runtime_slots_not_closed")
    teacher_final = controller_report.get("teacher_final")
    if (
        controller_report.get("state") != "complete"
        or not isinstance(teacher_final, Mapping)
        or teacher_final.get("state") != "candidate_pending_independent_release_review"
        or teacher_final.get("runtime_hmac_consumed_before_close") is not True
    ):
        raise batch.AdapterError("rollover_v3_candidate_status_invalid")
    artifact_root = teacher_final.get("artifact_root")
    if not isinstance(artifact_root, str):
        raise batch.AdapterError("rollover_v3_candidate_path_invalid")
    candidate = _repo_file(
        artifact_root,
        reason="rollover_v3_candidate_path_invalid",
    )
    try:
        candidate.relative_to(release_v3.CANDIDATE_PARENT.absolute())
    except ValueError as error:
        raise batch.AdapterError("rollover_v3_candidate_path_invalid") from error
    publish = release_v3.publish_release if publisher is None else publisher
    try:
        released = dict(
            publish(
                candidate,
                implementer_id=implementer_id,
                reviewer_id=reviewer_id,
                integrated_config_path=rollover_v2.INTEGRATED_CONFIG_PATH,
            )
        )
    except release_v3.IndependentReleaseError as error:
        raise batch.AdapterError(release_v3._reason(error)) from error
    if (
        released.get("state") != "released_pending_consumer_acceptance"
        or released.get("final") is not True
        or released.get("training_authorized") is not False
        or released.get("formal_training_authorized") is not False
        or released.get("live_authorized") is not False
    ):
        raise batch.AdapterError("rollover_v3_external_release_status_invalid")
    return released


async def _execute(
    *,
    implementer_id: str,
    reviewer_id: str,
) -> dict[str, Any]:
    if (
        release_v3.SAFE_IDENTIFIER.fullmatch(implementer_id) is None
        or release_v3.SAFE_IDENTIFIER.fullmatch(reviewer_id) is None
        or implementer_id.casefold() == reviewer_id.casefold()
    ):
        raise batch.AdapterError("release_reviewer_separation_invalid")
    config, config_raw, schema_raw, snapshots = _load_contract()
    value = integrated.load_integrated_config(rollover_v2.INTEGRATED_CONFIG_PATH)
    bound = integrated.bind_profiles(value)
    base._reject_environment_credentials(os.environ)
    report = preflight(validate_source=True)
    non_secret = [
        item
        for item in report["blockers"]
        if item
        not in {
            "controller_credential_slot_unloaded",
            "runtime_hmac_slot_unloaded",
        }
    ]
    if non_secret:
        raise batch.AdapterError(non_secret[0])
    base._guard_output_root(bound.values["smoke_exact1"], os.environ)
    base._validate_anonymous_os_channel(sys.stdin.buffer)
    slots = batch.RuntimeSecretSlots.from_anonymous_stdin(sys.stdin.buffer)
    try:
        controller = integrated.IntegratedSingleProcessController(
            value,
            bound,
            slots,
            consumer_validator=rollover_v2.validate_consumer_identity_rollover,
        )
        controller_report = await controller.run()
    finally:
        if slots.loaded:
            slots.close()
    _recheck_contract(config_raw, schema_raw, snapshots)
    rollover_v2.validate_consumer_identity_rollover()
    released = _external_release_after_close(
        controller_report,
        slots,
        implementer_id=implementer_id,
        reviewer_id=reviewer_id,
    )
    _recheck_contract(config_raw, schema_raw, snapshots)
    return {
        "schema_version": STATUS_SCHEMA_VERSION,
        "state": "complete",
        "mode": "execute",
        "controller_run_state": controller_report["state"],
        "candidate_state": controller_report["teacher_final"]["state"],
        "candidate_manifest_sha256": controller_report["teacher_final"][
            "manifest_sha256"
        ],
        "release_state": released["state"],
        "release_root": released["release_root"],
        "final_manifest_sha256": released["final_manifest_sha256"],
        "binding_sha256": released["binding_sha256"],
        "binding_schema_sha256": released["binding_schema_sha256"],
        "records": released["records"],
        "candidate_materialized_before_runtime_slots_close": True,
        "runtime_slots_closed_before_external_release": True,
        "external_release_after_runtime_slots_close": True,
        "consumer_acceptance": "pending",
        "credential_persisted": False,
        "runtime_hmac_public": False,
        "content_retained_in_status": False,
        "training_authorized": False,
        "formal_training_authorized": False,
        "live_authorized": False,
        "provider_requests_added_by_external_release": released["provider_requests"],
        "network_requests_added_by_external_release": released["network_requests"],
        "model_loads_added_by_external_release": released["model_loads"],
        "gpu_requests_added_by_external_release": released["gpu_requests"],
        "rollover_v3_config_sha256": _sha256(config_raw),
        "rollover_v3_release_implementation_sha256": config["authenticated_artifacts"][
            "release_implementation"
        ]["sha256"],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Consumer-rollover v3 controller followed by corrected external release"
        )
    )
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--dry-run", action="store_true")
    modes.add_argument("--validate-only", action="store_true")
    modes.add_argument("--execute", action="store_true")
    parser.add_argument("--credential-stdin", action="store_true")
    parser.add_argument("--implementer-id")
    parser.add_argument("--reviewer-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        base._reject_argv_credentials(raw_argv)
    except batch.AdapterError as error:
        print(json.dumps(base._blocked(error.reason_code), sort_keys=True))
        return 2
    parser = _parser()
    args = parser.parse_args(raw_argv)
    if args.execute != args.credential_stdin:
        parser.error("--execute requires --credential-stdin and vice versa")
    if args.execute and (not args.implementer_id or not args.reviewer_id):
        parser.error("--execute requires --implementer-id and --reviewer-id")
    try:
        if args.dry_run:
            result = preflight(validate_source=False)
        elif args.validate_only:
            result = preflight(validate_source=True)
        else:
            result = asyncio.run(
                _execute(
                    implementer_id=args.implementer_id,
                    reviewer_id=args.reviewer_id,
                )
            )
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0 if result.get("state") == "complete" else 2
    except BaseException as error:
        reason = (
            release_v3._reason(error)
            if isinstance(error, release_v3.IndependentReleaseError)
            else base._body_free_reason(error)
        )
        print(
            json.dumps(
                base._blocked(reason),
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
