"""Audit the additive Q+O controlled-proxy risk-evidence companion.

The companion authenticates three metadata-only receipts directly from an
exact Git commit.  It never opens a consumer worktree file, dataset body,
model, adapter, PNG, Gold, heldout, or scaffold record.  It cannot authorize
training or a formal release.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
import subprocess
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from anchor_mvp.swebench import (
    synthetic_scaffold_controlled_proxy_followup as frozen_v1,
)


CONTRACT_PATH = (
    "configs/research/"
    "synthetic_scaffold_controlled_proxy_risk_evidence_companion_v1.json"
)
CONTRACT_SIDECAR_PATH = f"{CONTRACT_PATH}.sha256"
SCHEMA_PATH = f"{CONTRACT_PATH.removesuffix('.json')}.schema.json"
IMPLEMENTATION_PATH = (
    "src/anchor_mvp/swebench/synthetic_scaffold_controlled_proxy_risk_evidence.py"
)

EXPECTED_SCHEMA_SHA256 = (
    "c04ba5072c2892f111a913808559f1c3eca9864977159c387df09fa6b7081068"
)
EXPECTED_CONSUMER_COMMIT = "58e9cd0c021ac0f01250746d44f199c1f616261d"
EXPECTED_CONSUMER_PARENT = "6ef29f1e0e9e110d59f9f2a09c1a8151f04b2465"
EXPECTED_CONSUMER_TREE = "1f275e2451c08f540d82ac5f24ee8b7beace060b"
EXPECTED_PRODUCER_COMMIT = "23194f7b3c707e3531ac92a64863c2b2f523f81d"
EXPECTED_PRODUCER_PARENT = "901a7e17f1373c50ceff3381691956c08890696c"
EXPECTED_PRODUCER_TREE = "7c580ea4f24a6a34d95f03c766f95affe29aa34e"

EXPECTED_FROZEN_V1 = {
    "schema": {
        "path": frozen_v1.SCHEMA_PATH,
        "sha256": "fe3878cac9d3be773a676c23025e79cc7f64063da03a53c54eb7f4b59594e0b6",
        "git_blob_sha1": "903b1e6858ab6c05d76cb29e75ed0f51c3e77503",
    },
    "contract": {
        "path": frozen_v1.CONTRACT_PATH,
        "sha256": "06ed121b23570546eb00088c891b273806dab1a1f764c9e40fba527cbf6447df",
        "git_blob_sha1": "9b706c77bbe57558d12670b47516920e87eb57d9",
    },
    "contract_sidecar": {
        "path": frozen_v1.CONTRACT_SIDECAR_PATH,
        "sha256": "bf346eb634d8f00f36eb069707069f2983d9083c9f8c1f416dc93795cb4f7209",
        "git_blob_sha1": "53b4cf17d96fdd92186fe1cb1ead47f50801da91",
    },
    "implementation": {
        "path": frozen_v1.IMPLEMENTATION_PATH,
        "sha256": "588c396febab5de75b772a4f46d58f21c8456247097fccc345cb1c738b0093a3",
        "git_blob_sha1": "2091a44cfdfd6bd87266836fea2d5dda22add932",
    },
    "source": {
        "path": frozen_v1.SOURCE_PATH,
        "sha256": "920f0abe3241a8a056114bd509f04a9b1fd0b23cfe0582857ff8719fefcd5e45",
        "git_blob_sha1": "89d45cb9486702436f4732529f8cb88a278ac803",
    },
    "source_sidecar": {
        "path": frozen_v1.SOURCE_SIDECAR_PATH,
        "sha256": "bbdeb5b3ac8f7890a93c736b19bcd920875f642174f99914b199d1d62ce06830",
        "git_blob_sha1": "8e256d24048b09299228395ff0f971c5906fe1b7",
    },
}

EXPECTED_SOURCES: tuple[Mapping[str, str], ...] = (
    {
        "role": "q_o_branch_ablation",
        "path": ("docs/research/results/qwen_qo_memory_ablation_audit_v1_receipt.json"),
        "sha256": "59750842e7bbad7fb06fcc64a1b9956dbd449e5591ba85f4abca021616da8ca3",
        "git_blob_sha1": "998f2ff4ee44c457242f8345856402ec71f7dd6b",
        "sidecar_path": (
            "docs/research/results/qwen_qo_memory_ablation_audit_v1_receipt.json.sha256"
        ),
        "sidecar_sha256": (
            "ad18c248e9bb091797f229927febddd11c0520d1497943119e0da2401b657a31"
        ),
        "sidecar_git_blob_sha1": "66c752e94c87fd86e7ae112bd1d0b743f365f842",
        "schema_version": ("anchor.qwen25-1.5b-qo-memory-ablation-audit-receipt.v1"),
        "status": "passed_read_only_teacher_forced_proxy_audit",
    },
    {
        "role": "spectral_risk",
        "path": ("docs/research/results/qwen_qo_spectral_memory_audit_v1_receipt.json"),
        "sha256": "c2fddd98ece4127ad3f17a19ffbd5bfa6e8d7f95588964b389e7e2970cfc8dd3",
        "git_blob_sha1": "1033abc7d60101380624ef84d36128f11cfdd25b",
        "sidecar_path": (
            "docs/research/results/qwen_qo_spectral_memory_audit_v1_receipt.json.sha256"
        ),
        "sidecar_sha256": (
            "61b829f963b48dfe1b3f98337b6992dce0e5944595ec436e296b37b2986ac74a"
        ),
        "sidecar_git_blob_sha1": "2e3be626a937e4fe02c38ba9625f92474b55ba51",
        "schema_version": "anchor.qwen25-qo-spectral-memory-audit-receipt.v1",
        "status": "passed_static_memory_risk_diagnostic_only",
    },
    {
        "role": "attention_hook",
        "path": (
            "docs/research/results/qwen_attention_weight_hook_qpluso_v1_summary.json"
        ),
        "sha256": "fc1ce0168cacfd1ed46a7ffcc1b482e7593253e224e780ab7dc6f7b701bb58a4",
        "git_blob_sha1": "6ce726b2a12ccb84a8d9769772e683aeda85f6ec",
        "sidecar_path": (
            "docs/research/results/"
            "qwen_attention_weight_hook_qpluso_v1_summary.json.sha256"
        ),
        "sidecar_sha256": (
            "1238605cf7abff130096e78071dd1f6ab2f916af93d85b29fdf3e054e1a13120"
        ),
        "sidecar_git_blob_sha1": "af1bf3a5fb14f5a9d1944423bba177675e44e6b8",
        "schema_version": "anchor.qwen-attention-weight-hook-summary.v1",
        "status": "completed_attention_proxy_only",
    },
)

_FORBIDDEN_BODY_KEYS = frozenset(
    {
        "answer",
        "content",
        "input_ids",
        "preview",
        "prompt",
        "prompt_text",
        "raw_text",
        "target_text",
        "token_ids",
    }
)


class RiskEvidenceAuditError(RuntimeError):
    """A stable, content-free audit failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _fail(code: str) -> None:
    raise RiskEvidenceAuditError(code)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _mapping(value: object, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(code)
    return value


def _sequence(value: object, code: str) -> Sequence[Any]:
    if not isinstance(value, list):
        _fail(code)
    return value


def _at(value: object, *path: str) -> Any:
    current = value
    for key in path:
        current = _mapping(current, "risk_source_semantics_invalid").get(key)
    return current


def _finite_number(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail("risk_source_semantics_invalid")
    result = float(value)
    if not math.isfinite(result):
        _fail("risk_source_semantics_invalid")
    return result


def _same_number(actual: object, expected: float) -> None:
    if _finite_number(actual) != expected:
        _fail("risk_source_semantics_invalid")


def _git_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("GIT_")
    }
    environment.update(
        {
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return environment


def _git(root: Path, *arguments: str, max_bytes: int = 4 * 1024 * 1024) -> bytes:
    command = [
        "git",
        "--no-replace-objects",
        "-c",
        "core.quotepath=false",
        "-c",
        "protocol.allow=never",
        "-C",
        os.fspath(root),
        *arguments,
    ]
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            timeout=15,
            env=_git_environment(),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RiskEvidenceAuditError("risk_git_object_unavailable") from exc
    if result.returncode != 0 or len(result.stdout) > max_bytes:
        _fail("risk_git_object_unavailable")
    return result.stdout


def _parse_commit(
    data: bytes, *, expected_tree: str, expected_parent: str
) -> tuple[str, str]:
    try:
        header = data.split(b"\n\n", 1)[0].decode("ascii", errors="strict")
    except UnicodeDecodeError as exc:
        raise RiskEvidenceAuditError("risk_git_commit_invalid") from exc
    trees = [line[5:] for line in header.splitlines() if line.startswith("tree ")]
    parents = [line[7:] for line in header.splitlines() if line.startswith("parent ")]
    if trees != [expected_tree] or parents != [expected_parent]:
        _fail("risk_git_commit_invalid")
    return trees[0], parents[0]


def _tree_blob(root: Path, tree: str, path: str, expected_oid: str) -> bytes:
    listing = _git(root, "ls-tree", "-z", tree, "--", path, max_bytes=4096)
    records = [item for item in listing.split(b"\0") if item]
    if len(records) != 1:
        _fail("risk_git_tree_invalid")
    try:
        metadata, actual_path = records[0].split(b"\t", 1)
        mode, kind, oid = metadata.decode("ascii").split(" ")
        decoded_path = actual_path.decode("utf-8", errors="strict")
    except (UnicodeDecodeError, ValueError) as exc:
        raise RiskEvidenceAuditError("risk_git_tree_invalid") from exc
    if (
        mode != "100644"
        or kind != "blob"
        or oid != expected_oid
        or decoded_path != path
    ):
        _fail("risk_git_tree_invalid")
    return _git(root, "cat-file", "blob", oid, max_bytes=1024 * 1024)


@dataclass(frozen=True)
class GitEvidenceSnapshot:
    commit: bytes
    blobs: Mapping[str, bytes]


def _read_git_evidence(
    root: Path, sources: Sequence[Any]
) -> tuple[GitEvidenceSnapshot, Mapping[str, Mapping[str, Any]]]:
    top = _git(root, "rev-parse", "--show-toplevel", max_bytes=4096)
    try:
        top_path = Path(top.decode("utf-8", errors="strict").strip()).resolve(
            strict=True
        )
    except (UnicodeDecodeError, OSError) as exc:
        raise RiskEvidenceAuditError("risk_git_repository_invalid") from exc
    if top_path != root:
        _fail("risk_git_repository_invalid")
    object_type = _git(root, "cat-file", "-t", EXPECTED_CONSUMER_COMMIT, max_bytes=64)
    if object_type != b"commit\n":
        _fail("risk_git_commit_invalid")
    commit = _git(
        root,
        "cat-file",
        "commit",
        EXPECTED_CONSUMER_COMMIT,
        max_bytes=64 * 1024,
    )
    tree, _ = _parse_commit(
        commit,
        expected_tree=EXPECTED_CONSUMER_TREE,
        expected_parent=EXPECTED_CONSUMER_PARENT,
    )

    if len(sources) != len(EXPECTED_SOURCES):
        _fail("risk_contract_identity_invalid")
    blobs: dict[str, bytes] = {}
    parsed: dict[str, Mapping[str, Any]] = {}
    for raw_binding, expected in zip(sources, EXPECTED_SOURCES, strict=True):
        binding = _mapping(raw_binding, "risk_contract_identity_invalid")
        if dict(binding) != dict(expected):
            _fail("risk_contract_identity_invalid")
        path = expected["path"]
        sidecar_path = expected["sidecar_path"]
        receipt = _tree_blob(root, tree, path, expected["git_blob_sha1"])
        sidecar = _tree_blob(
            root,
            tree,
            sidecar_path,
            expected["sidecar_git_blob_sha1"],
        )
        if _sha256(receipt) != expected["sha256"]:
            _fail("risk_git_blob_hash_invalid")
        if _sha256(sidecar) != expected["sidecar_sha256"]:
            _fail("risk_git_blob_hash_invalid")
        expected_sidecar = f"{expected['sha256']}  {Path(path).name}\n".encode("ascii")
        if sidecar != expected_sidecar:
            _fail("risk_git_sidecar_invalid")
        try:
            value = frozen_v1._strict_json(receipt, "risk_source_json_invalid")
            reparsed = frozen_v1._strict_json(receipt, "risk_source_json_invalid")
        except frozen_v1.ControlledProxyFollowupAuditError as exc:
            raise RiskEvidenceAuditError("risk_source_json_invalid") from exc
        if value != reparsed:
            _fail("risk_source_reparse_mismatch")
        if _FORBIDDEN_BODY_KEYS.intersection(frozen_v1._walk_keys(value)):
            _fail("risk_source_body_field_rejected")
        if (
            value.get("schema_version") != expected["schema_version"]
            or value.get("status") != expected["status"]
        ):
            _fail("risk_source_semantics_invalid")
        blobs[path] = receipt
        blobs[sidecar_path] = sidecar
        parsed[expected["role"]] = value
    return GitEvidenceSnapshot(commit=commit, blobs=blobs), parsed


def _read_frozen_v1_git_dependency(
    root: Path, local_snapshots: Sequence[Any]
) -> GitEvidenceSnapshot:
    object_type = _git(root, "cat-file", "-t", EXPECTED_PRODUCER_COMMIT, max_bytes=64)
    if object_type != b"commit\n":
        _fail("risk_frozen_v1_git_identity_invalid")
    commit = _git(
        root,
        "cat-file",
        "commit",
        EXPECTED_PRODUCER_COMMIT,
        max_bytes=64 * 1024,
    )
    tree, _ = _parse_commit(
        commit,
        expected_tree=EXPECTED_PRODUCER_TREE,
        expected_parent=EXPECTED_PRODUCER_PARENT,
    )
    blobs: dict[str, bytes] = {}
    for local, expected in zip(
        local_snapshots, EXPECTED_FROZEN_V1.values(), strict=True
    ):
        path = expected["path"]
        data = _tree_blob(root, tree, path, expected["git_blob_sha1"])
        if _sha256(data) != expected["sha256"] or data != local.data:
            _fail("risk_frozen_v1_git_identity_invalid")
        blobs[path] = data
    return GitEvidenceSnapshot(commit=commit, blobs=blobs)


def _source_counters(parsed: Mapping[str, Mapping[str, Any]]) -> dict[str, object]:
    ablation = _mapping(
        parsed["q_o_branch_ablation"].get("audit"), "risk_source_semantics_invalid"
    )
    spectral = _mapping(
        parsed["spectral_risk"].get("audit"), "risk_source_semantics_invalid"
    )
    attention = _mapping(
        parsed["attention_hook"].get("audit"), "risk_source_semantics_invalid"
    )
    expected_ablation = {
        "provider_requests": 0,
        "network_requests": 0,
        "model_loads": 1,
        "gpu_requests": 1,
        "heldout_reads": 0,
        "protected_body_reads": 0,
    }
    expected_spectral = {
        "provider_requests": 0,
        "network_requests": 0,
        "full_model_loads": 0,
        "gpu_requests": 0,
        "heldout_reads": 0,
        "protected_body_reads": 0,
    }
    expected_attention = {
        "provider_requests_reported": False,
        "network_requests": 0,
        "model_loads": 1,
        "gpu_requests": 1,
        "heldout_reads": 0,
        "protected_body_reads": 0,
    }
    for actual, expected in (
        (ablation, expected_ablation),
        (spectral, expected_spectral),
        (
            {
                "provider_requests_reported": "provider_requests" in attention,
                **attention,
            },
            expected_attention,
        ),
    ):
        if any(actual.get(key) != value for key, value in expected.items()):
            _fail("risk_source_counter_invalid")
    return {
        "q_o_branch_ablation": expected_ablation,
        "spectral_risk": expected_spectral,
        "attention_hook": expected_attention,
    }


def _validate_source_claims(parsed: Mapping[str, Mapping[str, Any]]) -> None:
    ablation = _mapping(
        parsed["q_o_branch_ablation"].get("claims"), "risk_source_semantics_invalid"
    )
    spectral = _mapping(
        parsed["spectral_risk"].get("claims"), "risk_source_semantics_invalid"
    )
    attention = _mapping(
        parsed["attention_hook"].get("claims"), "risk_source_semantics_invalid"
    )
    required_false = {
        "formal",
        "training_authorized",
    }
    for claims in (ablation, spectral, attention):
        if claims.get("diagnostic_only") is not True:
            _fail("risk_source_claim_invalid")
        if any(claims.get(key) is not False for key in required_false):
            _fail("risk_source_claim_invalid")
    if (
        ablation.get("formal_training_authorized") is not False
        or ablation.get("eval_proxy_is_heldout") is not False
        or ablation.get("ood_proxy_is_heldout") is not False
        or ablation.get("statistical_significance_claimed") is not False
        or spectral.get("causal_attribution_proven") is not False
        or spectral.get("memorization_proven") is not False
        or spectral.get("exploit_code_memorization_tested") is not False
        or spectral.get("static_alignment_is_correlation_only") is not True
        or attention.get("attention_equals_explanation") is not False
        or attention.get("causal_effect_proven") is not False
        or attention.get("quality_validated") is not False
    ):
        _fail("risk_source_claim_invalid")


def _closed_projection(parsed: Mapping[str, Mapping[str, Any]]) -> dict[str, object]:
    ablation = parsed["q_o_branch_ablation"]
    spectral = parsed["spectral_risk"]
    attention = parsed["attention_hook"]

    if _at(ablation, "dataset", "manifest_sha256") != (
        "64b1ce813477deef48de16dbdc0d2561bbeaa0ef5d6248862e9f2bedc8acc0dd"
    ):
        _fail("risk_source_cross_binding_invalid")
    if _at(ablation, "adapter", "file_sha256", "diagnostic_receipt.json") != (
        "dc94204df696db795f3e657c679d67918e3aa2723b2d0bdf7dee899ef4490f6e"
    ):
        _fail("risk_source_cross_binding_invalid")
    if _at(spectral, "identity", "comparison", "sha256") != (
        "920f0abe3241a8a056114bd509f04a9b1fd0b23cfe0582857ff8719fefcd5e45"
    ):
        _fail("risk_source_cross_binding_invalid")
    if _at(
        attention, "identity", "adapter_artifact_sha256", "diagnostic_receipt.json"
    ) != ("dc94204df696db795f3e657c679d67918e3aa2723b2d0bdf7dee899ef4490f6e"):
        _fail("risk_source_cross_binding_invalid")

    result_values: dict[str, dict[str, float]] = {}
    for profile in (
        "adapter_off",
        "q_only_contribution",
        "o_only_contribution",
        "full",
    ):
        result_values[profile] = {
            "eval": _finite_number(
                _at(ablation, "results", profile, "eval_proxy", "macro_loss")
            ),
            "ood": _finite_number(
                _at(ablation, "results", profile, "ood_proxy", "macro_loss")
            ),
        }
    expected_eval = {
        "adapter_off": 3.1020863533020018,
        "q_only_contribution": 2.6766975283622743,
        "o_only_contribution": 1.0298870503902435,
        "full": 0.8586097195744514,
    }
    expected_ood = {
        "adapter_off": 3.035316228866577,
        "q_only_contribution": 2.983178400993347,
        "o_only_contribution": 2.8666090965270996,
        "full": 2.896842730045319,
    }
    for profile in expected_eval:
        _same_number(result_values[profile]["eval"], expected_eval[profile])
        _same_number(result_values[profile]["ood"], expected_ood[profile])
    if (
        _at(ablation, "results", "adapter_off", "eval_proxy", "records") != 20
        or _at(ablation, "results", "adapter_off", "ood_proxy", "records") != 20
        or _at(ablation, "ood_proxy", "source_bundles") != 4
        or _at(ablation, "ood_proxy", "source_bundle_overlap_count") != 0
        or _at(ablation, "ood_proxy", "exact_body_overlap_count") != 0
    ):
        _fail("risk_source_semantics_invalid")

    findings = _mapping(spectral.get("findings"), "risk_source_semantics_invalid")
    expected_findings = {
        "q_plus_o_o_top_1_energy_fraction_mean": 0.82318570872,
        "q_plus_o_o_energy_effective_rank_mean": 2.11605570639,
        "q_plus_o_o_to_q_total_delta_energy_ratio": 1.767153939794,
        "q_plus_o_target_frequent_to_random_projection_ratio": 1.95910776849,
    }
    for key, expected in expected_findings.items():
        _same_number(findings.get(key), expected)
    prompt_random = _finite_number(
        _at(
            spectral,
            "adapters",
            "q_plus_o",
            "o_output_subspace_alignment",
            "groups",
            "prompt_control",
            "ratio_to_deterministic_random",
        )
    )
    _same_number(prompt_random, 1.006882507832)

    if (
        _at(attention, "capture", "attention_implementation") != "eager"
        or _at(attention, "capture", "aggregation") != "head_mean_float32_cpu"
        or _at(attention, "capture", "modes")
        != ["adapter_off", "q_only_component", "o_only_component", "full"]
        or _at(attention, "capture", "difference_panel")
        != "full_minus_q_only_component"
        or _at(attention, "capture", "selected_layers") != [0, 13, 27]
        or _at(attention, "audit", "single_probe") is not True
        or _at(attention, "probe", "total_tokens") != 79
    ):
        _fail("risk_source_semantics_invalid")
    expected_layers = {
        "0": (0.0, 0.0),
        "13": (0.0008257970912382007, 0.061482757329940796),
        "27": (0.0019083430524915457, 0.1309814453125),
    }
    layer_projection: dict[str, dict[str, float]] = {}
    for layer, (mean_absolute, maximum_absolute) in expected_layers.items():
        source = _mapping(
            _at(attention, "capture", "layers", layer, "full_minus_q_only_component"),
            "risk_source_semantics_invalid",
        )
        _same_number(source.get("mean_absolute"), mean_absolute)
        _same_number(source.get("maximum_absolute"), maximum_absolute)
        layer_projection[f"layer_{layer}"] = {
            "mean_absolute": mean_absolute,
            "maximum_absolute": maximum_absolute,
        }

    off_eval = expected_eval["adapter_off"]
    full_eval = expected_eval["full"]
    o_eval = expected_eval["o_only_contribution"]
    off_ood = expected_ood["adapter_off"]
    full_ood = expected_ood["full"]
    return {
        "primary_endpoint": "step_80",
        "q_o_branch_ablation": {
            "modes": [
                "adapter_off",
                "q_only_contribution",
                "o_only_contribution",
                "full",
            ],
            "same_template_eval_macro_loss": {
                "adapter_off": off_eval,
                "q_branch_retained": expected_eval["q_only_contribution"],
                "o_branch_retained": o_eval,
                "full": full_eval,
            },
            "ood_proxy_macro_loss": {
                "adapter_off": off_ood,
                "q_branch_retained": expected_ood["q_only_contribution"],
                "o_branch_retained": expected_ood["o_only_contribution"],
                "full": full_ood,
            },
            "o_branch_retained_fraction_of_off_to_full_reduction": (
                (off_eval - o_eval) / (off_eval - full_eval)
            ),
            "full_ood_relative_improvement_vs_off": (off_ood - full_ood) / off_ood,
            "same_template_records": 20,
            "ood_records": 20,
            "ood_source_bundles": 4,
            "ood_pre_registered_confirmation": False,
            "branch_effects_additive_claimed": False,
        },
        "spectral_risk_signal": {
            "o_top1_energy_fraction_mean": expected_findings[
                "q_plus_o_o_top_1_energy_fraction_mean"
            ],
            "o_energy_effective_rank_mean": expected_findings[
                "q_plus_o_o_energy_effective_rank_mean"
            ],
            "o_to_q_total_delta_energy_ratio": expected_findings[
                "q_plus_o_o_to_q_total_delta_energy_ratio"
            ],
            "target_frequent_to_random_projection_ratio": expected_findings[
                "q_plus_o_target_frequent_to_random_projection_ratio"
            ],
            "prompt_control_to_random_projection_ratio": prompt_random,
            "nominal_rank": 8,
            "token_groups_frequency_matched": False,
            "correlation_only": True,
        },
        "attention_single_probe": {
            "attention_implementation": "eager",
            "aggregation": "head_mean_float32_cpu",
            "probe_count": 1,
            "total_tokens": 79,
            "modes": [
                "adapter_off",
                "q_only_component",
                "o_only_component",
                "full",
            ],
            "difference_panel": "full_minus_q_only_component",
            "selected_layers": [0, 13, 27],
            "full_minus_q_branch_retained": layer_projection,
            "same_layer_o_branch_attention_change_claimed": False,
            "general_causal_result_claimed": False,
        },
    }


def _verify_git_unchanged(root: Path, expected: GitEvidenceSnapshot) -> None:
    commit = _git(
        root,
        "cat-file",
        "commit",
        EXPECTED_CONSUMER_COMMIT,
        max_bytes=64 * 1024,
    )
    if commit != expected.commit:
        _fail("risk_git_input_changed")
    for source in EXPECTED_SOURCES:
        for path_key, oid_key in (
            ("path", "git_blob_sha1"),
            ("sidecar_path", "sidecar_git_blob_sha1"),
        ):
            path = source[path_key]
            current = _tree_blob(root, EXPECTED_CONSUMER_TREE, path, source[oid_key])
            if current != expected.blobs[path]:
                _fail("risk_git_input_changed")


def _validate_contract_identity(
    contract: Mapping[str, Any], schema_sha256: str, implementation_sha256: str
) -> None:
    if schema_sha256 != EXPECTED_SCHEMA_SHA256:
        _fail("risk_schema_identity_invalid")
    if contract.get("schema_sha256") != EXPECTED_SCHEMA_SHA256:
        _fail("risk_contract_identity_invalid")
    if contract.get("implementation") != {
        "path": IMPLEMENTATION_PATH,
        "sha256": implementation_sha256,
    }:
        _fail("risk_contract_identity_invalid")
    dependency = _mapping(
        contract.get("frozen_v1_dependency"), "risk_contract_identity_invalid"
    )
    if (
        dependency.get("producer_commit") != EXPECTED_PRODUCER_COMMIT
        or dependency.get("producer_parent_commit") != EXPECTED_PRODUCER_PARENT
        or dependency.get("producer_tree") != EXPECTED_PRODUCER_TREE
        or dependency.get("dependency_semantics") != "exact_frozen_v1_and_additive_only"
        or dependency.get("authentication_scope")
        != "exact_raw_git_commit_tree_blob_bytes_plus_local_file_equivalence"
        or dependency.get("modified") is not False
        or dependency.get("artifacts") != EXPECTED_FROZEN_V1
    ):
        _fail("risk_frozen_v1_identity_invalid")
    evidence = _mapping(
        contract.get("consumer_git_evidence"), "risk_contract_identity_invalid"
    )
    if (
        evidence.get("commit") != EXPECTED_CONSUMER_COMMIT
        or evidence.get("parent_commit") != EXPECTED_CONSUMER_PARENT
        or evidence.get("tree") != EXPECTED_CONSUMER_TREE
        or evidence.get("freeze_time_live_remote_head_verified") is not True
        or evidence.get("runtime_ref_equality_required") is not False
        or evidence.get("network_or_fetch_allowed") is not False
        or evidence.get("git_replace_objects_allowed") is not False
        or evidence.get("runtime_consumer_worktree_file_reads") != 0
    ):
        _fail("risk_contract_identity_invalid")


def audit_risk_evidence(
    repo_root: Path, contract_path: Path = Path(CONTRACT_PATH)
) -> Mapping[str, object]:
    """Authenticate the additive companion without running a model or provider."""

    if contract_path.as_posix() != CONTRACT_PATH:
        _fail("risk_contract_path_invalid")
    try:
        root = frozen_v1._safe_root(Path(repo_root))
        contract_snapshot = frozen_v1._snapshot(
            root, CONTRACT_PATH, max_bytes=512 * 1024
        )
        sidecar_snapshot = frozen_v1._snapshot(
            root, CONTRACT_SIDECAR_PATH, max_bytes=1024
        )
        schema_snapshot = frozen_v1._snapshot(root, SCHEMA_PATH, max_bytes=512 * 1024)
        implementation_snapshot = frozen_v1._snapshot(
            root, IMPLEMENTATION_PATH, max_bytes=1024 * 1024
        )
        frozen_snapshots = [
            frozen_v1._snapshot(root, value["path"], max_bytes=1024 * 1024)
            for value in EXPECTED_FROZEN_V1.values()
        ]
    except frozen_v1.ControlledProxyFollowupAuditError as exc:
        raise RiskEvidenceAuditError("risk_local_snapshot_invalid") from exc
    if sidecar_snapshot.data != frozen_v1._sidecar(
        contract_snapshot.sha256, Path(CONTRACT_PATH).name
    ):
        _fail("risk_contract_sidecar_invalid")
    if any(
        snapshot.sha256 != expected["sha256"]
        for snapshot, expected in zip(
            frozen_snapshots, EXPECTED_FROZEN_V1.values(), strict=True
        )
    ):
        _fail("risk_frozen_v1_identity_invalid")
    try:
        contract = frozen_v1._strict_json(
            contract_snapshot.data, "risk_contract_json_invalid"
        )
        schema = frozen_v1._strict_json(
            schema_snapshot.data, "risk_schema_json_invalid"
        )
        if (
            frozen_v1._strict_json(contract_snapshot.data, "risk_contract_json_invalid")
            != contract
        ):
            _fail("risk_contract_reparse_mismatch")
        if (
            frozen_v1._strict_json(schema_snapshot.data, "risk_schema_json_invalid")
            != schema
        ):
            _fail("risk_schema_reparse_mismatch")
    except frozen_v1.ControlledProxyFollowupAuditError as exc:
        raise RiskEvidenceAuditError(str(exc)) from exc
    try:
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(contract)
    except (SchemaError, ValidationError) as exc:
        raise RiskEvidenceAuditError("risk_contract_schema_invalid") from exc
    if _FORBIDDEN_BODY_KEYS.intersection(frozen_v1._walk_keys(contract)):
        _fail("risk_contract_body_field_rejected")
    _validate_contract_identity(
        contract, schema_snapshot.sha256, implementation_snapshot.sha256
    )
    frozen_git_snapshot = _read_frozen_v1_git_dependency(root, frozen_snapshots)
    try:
        frozen_summary = frozen_v1.audit_followup(root)
    except frozen_v1.ControlledProxyFollowupAuditError as exc:
        raise RiskEvidenceAuditError("risk_frozen_v1_audit_failed") from exc
    if frozen_summary.get("status") != "passed":
        _fail("risk_frozen_v1_audit_failed")

    evidence = _mapping(
        contract.get("consumer_git_evidence"), "risk_contract_identity_invalid"
    )
    sources = _sequence(evidence.get("sources"), "risk_contract_identity_invalid")
    git_snapshot, parsed = _read_git_evidence(root, sources)
    _validate_source_claims(parsed)
    projection = _closed_projection(parsed)
    counters = _source_counters(parsed)
    if contract.get("closed_projection") != projection:
        _fail("risk_closed_projection_mismatch")
    audit = _mapping(contract.get("audit"), "risk_contract_identity_invalid")
    if audit.get("consumer_source_counters") != counters:
        _fail("risk_source_counter_projection_mismatch")

    try:
        frozen_v1._verify_unchanged(
            root,
            [
                contract_snapshot,
                sidecar_snapshot,
                schema_snapshot,
                implementation_snapshot,
                *frozen_snapshots,
            ],
        )
    except frozen_v1.ControlledProxyFollowupAuditError as exc:
        raise RiskEvidenceAuditError("risk_local_input_changed") from exc
    _verify_git_unchanged(root, git_snapshot)
    current_frozen_git = _read_frozen_v1_git_dependency(root, frozen_snapshots)
    if current_frozen_git != frozen_git_snapshot:
        _fail("risk_frozen_v1_git_input_changed")

    claims = _mapping(contract.get("claims"), "risk_contract_identity_invalid")
    return {
        "schema_version": contract["schema_version"],
        "status": "passed",
        "contract_status": contract["status"],
        "consumer_commit": EXPECTED_CONSUMER_COMMIT,
        "receipt_count": len(EXPECTED_SOURCES),
        "primary_endpoint": projection["primary_endpoint"],
        "diagnostic_only": claims["diagnostic_only"],
        "training_authorized": claims["training_authorized"],
        "formal_training_authorized": claims["formal_training_authorized"],
        "multi_seed_validated": claims["multi_seed_validated"],
        "bundle_generalization_validated": claims["bundle_generalization_validated"],
        "provider_requests": audit["producer_provider_requests"],
        "network_requests": audit["producer_network_requests"],
        "model_loads": audit["producer_model_loads"],
        "gpu_requests": audit["producer_gpu_requests"],
        "protected_body_reads": audit["producer_protected_body_reads"],
    }


__all__ = [
    "CONTRACT_PATH",
    "CONTRACT_SIDECAR_PATH",
    "IMPLEMENTATION_PATH",
    "SCHEMA_PATH",
    "RiskEvidenceAuditError",
    "audit_risk_evidence",
]
