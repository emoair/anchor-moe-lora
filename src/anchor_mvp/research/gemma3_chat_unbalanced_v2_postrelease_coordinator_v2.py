"""Body-free validate-only skeleton for the Gemma 3 postrelease chain.

The coordinator freezes the only permitted stage order:
smoke -> full -> generation evaluation -> KV evaluation.  This first version
does not spawn a process, load a model, request CUDA, or read sample bodies.
Every future stage receipt must carry the same unique Teacher FINAL binding
SHA-256 used to derive the idempotency key.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError


ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = Path(
    "configs/research/gemma3_chat_unbalanced_v2_postrelease_coordinator_v2.json"
)
CONFIG_SCHEMA_PATH = CONFIG_PATH.with_name(
    "gemma3_chat_unbalanced_v2_postrelease_coordinator_v2.schema.json"
)
RECEIPT_SCHEMA_PATH = CONFIG_PATH.with_name(
    "gemma3_chat_unbalanced_v2_postrelease_aggregate_receipt_v2.schema.json"
)
CONFIG_VERSION = "anchor.gemma3-chat-unbalanced-v2-postrelease-coordinator-config.v2"
RECEIPT_VERSION = "anchor.gemma3-chat-unbalanced-v2-postrelease-aggregate-receipt.v2"
STAGE_ORDER = ("smoke", "full", "generation_eval", "kv")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


class PostreleaseCoordinatorError(RuntimeError):
    """One body-free postrelease contract failed."""


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _strict_json(raw: bytes, *, code: str) -> dict[str, Any]:
    def pairs(pairs_value: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs_value:
            if key in value:
                raise ValueError("duplicate")
            value[key] = item
        return value

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs,
            parse_constant=lambda _value: (_ for _ in ()).throw(
                ValueError("nonfinite")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise PostreleaseCoordinatorError(code) from None
    if not isinstance(value, dict):
        raise PostreleaseCoordinatorError(code)
    return value


def _schema(path: Path, *, code: str) -> Draft202012Validator:
    try:
        raw = path.read_bytes()
    except OSError:
        raise PostreleaseCoordinatorError(code) from None
    value = _strict_json(raw, code=code)
    try:
        Draft202012Validator.check_schema(value)
    except SchemaError:
        raise PostreleaseCoordinatorError(code) from None
    return Draft202012Validator(value)


def load_config(
    path: str | Path = ROOT / CONFIG_PATH,
) -> tuple[dict[str, Any], str]:
    requested = Path(path)
    if not requested.is_absolute():
        requested = ROOT / requested
    try:
        raw = requested.read_bytes()
    except OSError:
        raise PostreleaseCoordinatorError("postrelease_config_missing") from None
    config = _strict_json(raw, code="postrelease_config_invalid")
    try:
        _schema(
            ROOT / CONFIG_SCHEMA_PATH,
            code="postrelease_config_schema_invalid",
        ).validate(config)
    except ValidationError as exc:
        raise PostreleaseCoordinatorError(
            f"postrelease_config_schema_mismatch:{exc.validator}"
        ) from None
    if (
        config.get("schema_version") != CONFIG_VERSION
        or tuple(item["name"] for item in config["stages"]) != STAGE_ORDER
        or tuple(config["transition_contract"]["ordered"]) != STAGE_ORDER
    ):
        raise PostreleaseCoordinatorError("postrelease_config_identity_drift")
    for stage in config["stages"]:
        runner = ROOT / str(stage["runner"])
        if not runner.is_file():
            raise PostreleaseCoordinatorError("postrelease_stage_runner_missing")
    return config, _sha256(raw)


def binding_idempotency_key(config: Mapping[str, Any], binding_sha256: str) -> str:
    if SHA256.fullmatch(binding_sha256) is None:
        raise PostreleaseCoordinatorError("postrelease_binding_sha256_invalid")
    domain = str(config["transition_contract"]["idempotency_domain"])
    return _sha256(domain.encode("ascii") + b"\0" + binding_sha256.encode("ascii"))


def build_validate_only_receipt(
    config: Mapping[str, Any],
    *,
    config_sha256: str,
    binding_sha256: str,
) -> dict[str, Any]:
    if (
        tuple(item["name"] for item in config["stages"]) != STAGE_ORDER
        or tuple(config["transition_contract"]["ordered"]) != STAGE_ORDER
    ):
        raise PostreleaseCoordinatorError("postrelease_stage_order_drift")
    key = binding_idempotency_key(config, binding_sha256)
    binding = config["binding"]
    receipt = {
        "schema_version": RECEIPT_VERSION,
        "status": "validated_stub_no_execution",
        "operation": "validate_only",
        "config_sha256": config_sha256,
        "binding": {
            "sha256": binding_sha256,
            "producer_commit": binding["producer_commit"],
            "producer_tree": binding["producer_tree"],
            "schema_sha256": binding["schema_sha256"],
            "required_status": binding["required_status"],
        },
        "idempotency_key_sha256": key,
        "stages": [
            {
                "name": stage["name"],
                "status": "pending_external_execution",
                "requires": list(stage["requires"]),
                "binding_sha256": binding_sha256,
            }
            for stage in config["stages"]
        ],
        "next_stage": "smoke",
        "claims": {
            "body_free": True,
            "execution_performed": False,
            "provider_requests": 0,
            "network_requests": 0,
            "model_loads": 0,
            "gpu_requests": 0,
            "sample_bodies_read": False,
            "raw_token_ids_read": False,
            "secret_material_read": False,
            "fallback_used": False,
            "formal": False,
        },
    }
    try:
        _schema(
            ROOT / RECEIPT_SCHEMA_PATH,
            code="postrelease_receipt_schema_invalid",
        ).validate(receipt)
    except ValidationError as exc:
        raise PostreleaseCoordinatorError(
            f"postrelease_receipt_schema_mismatch:{exc.validator}"
        ) from None
    return receipt


def validate_only(
    *,
    binding_sha256: str,
    config_path: str | Path = ROOT / CONFIG_PATH,
) -> dict[str, Any]:
    config, config_sha256 = load_config(config_path)
    return build_validate_only_receipt(
        config,
        config_sha256=config_sha256,
        binding_sha256=binding_sha256,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--binding-sha256", required=True)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if not args.validate_only:
            raise PostreleaseCoordinatorError(
                "postrelease_execution_not_implemented_fail_closed"
            )
        receipt = validate_only(
            binding_sha256=str(args.binding_sha256),
            config_path=str(args.config),
        )
        print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        return 0
    except PostreleaseCoordinatorError as exc:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error_code": str(exc),
                    "body_free": True,
                    "execution_performed": False,
                    "provider_requests": 0,
                    "model_loads": 0,
                    "gpu_requests": 0,
                },
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
