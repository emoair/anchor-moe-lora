"""Deterministic, body-free selector-v2 for the additive minimal-320 handoff."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import stat
import subprocess
from typing import Any

from jsonschema import Draft202012Validator
from jsonschema.exceptions import SchemaError, ValidationError

from . import gemma3_chat_unbalanced_v2_batch as batch
from . import gemma3_chat_unbalanced_v2_live_controller as closed


CANDIDATE_SCHEMA_VERSION = (
    "anchor.gemma3-chat-unbalanced-v2.teacher-alignment-minimal320."
    "selector-candidate.v2"
)
MANIFEST_SCHEMA_VERSION = (
    "anchor.gemma3-chat-unbalanced-v2.teacher-alignment-minimal320."
    "selection-manifest.v2"
)
CONTRACT_SCHEMA_VERSION = (
    "anchor.gemma3-chat-unbalanced-v2.teacher-alignment-minimal320.contract.v2"
)
OVERLAY_ROW_SCHEMA_VERSION = (
    "anchor.gemma3-chat-unbalanced-v2.teacher-alignment-minimal320."
    "authenticated-metadata-overlay-row.v2"
)
OVERLAY_MANIFEST_SCHEMA_VERSION = (
    "anchor.gemma3-chat-unbalanced-v2.teacher-alignment-minimal320."
    "authenticated-metadata-overlay-manifest.v2"
)
PREIMAGE_MANIFEST_SCHEMA_VERSION = (
    "anchor.gemma3-chat-unbalanced-v2.teacher-alignment-minimal320."
    "selector-preimage-manifest.v2"
)
CANDIDATE_PREIMAGE_MANIFEST_SCHEMA_VERSION = (
    "anchor.gemma3-chat-unbalanced-v2.teacher-alignment-minimal320."
    "candidate-universe-manifest.v2"
)
SELECTOR_PREIMAGE_SCHEMA_VERSION = (
    "anchor.gemma3-chat-unbalanced-v2.teacher-alignment-minimal320."
    "selection-preimage.v2"
)
DEFAULT_CONTRACT_PATH = (
    batch.REPO_ROOT / "configs/data/"
    "gemma3_chat_unbalanced_v2_teacher_alignment_minimal320_contract_v2.json"
)
DEFAULT_CANDIDATE_SCHEMA_PATH = (
    batch.REPO_ROOT / "configs/data/"
    "gemma3_chat_unbalanced_v2_teacher_alignment_minimal320_selector_v2.schema.json"
)
DEFAULT_OVERLAY_SCHEMA_PATH = (
    batch.REPO_ROOT / "configs/data/"
    "gemma3_chat_unbalanced_v2_teacher_alignment_"
    "minimal320_metadata_overlay_v2.schema.json"
)
DEFAULT_PREIMAGE_SCHEMA_PATH = (
    batch.REPO_ROOT / "configs/data/"
    "gemma3_chat_unbalanced_v2_teacher_alignment_"
    "minimal320_preimages_v2.schema.json"
)
IMPLEMENTATION_PATH = Path(__file__).absolute()
OUTPUT_EXPERT_ROLES = ("humor", "serious", "angry", "tool", "review")
LANGUAGES = ("en", "zh")
IDENTITY_BUNDLE_CELL_QUOTAS = {
    ("air_attribution", "en"): 4,
    ("air_attribution", "zh"): 4,
    ("false_google_attribution", "en"): 3,
    ("false_google_attribution", "zh"): 3,
    ("false_openai_attribution", "en"): 3,
    ("false_openai_attribution", "zh"): 3,
}
IDENTITY_RECORD_CELL_QUOTAS = {
    cell: bundles * len(OUTPUT_EXPERT_ROLES)
    for cell, bundles in IDENTITY_BUNDLE_CELL_QUOTAS.items()
}
ROUTER_CELL_QUOTAS = {
    ("humor", "en"): 14,
    ("humor", "zh"): 13,
    ("serious", "en"): 13,
    ("serious", "zh"): 13,
    ("angry", "en"): 13,
    ("angry", "zh"): 14,
}
ROLE_QUOTAS = {
    "humor": 40,
    "serious": 40,
    "angry": 40,
    "tool": 60,
    "review": 60,
    "router": 80,
}
LANGUAGE_QUOTAS = {"en": 160, "zh": 160}
FAULT_TYPES = (
    "evidence_mismatch",
    "format_schema",
    "grounding_mismatch",
    "routing_scope",
    "tool_argument",
)
REVIEW_ASSIGNMENT_ALGORITHM_ID = (
    "lexicographically-smallest-feasible-20-cell-review-assignment-v1"
)
EXPECTED_REVIEW_AVAILABILITY_ROOT_SHA256 = (
    "d0556b68375d4e51645c5920b1143f8f99696b5a776b02b714d261928bf38248"
)
EXPECTED_REVIEW_ASSIGNMENT_SHA256 = (
    "c0d15e7f81c281dd2182d4a27f4cb84038becce52f70ad5c2b8fa9b14e90d932"
)
EXPECTED_SOURCE_AVAILABILITY_ROOT_SHA256 = (
    "6e1e6f63ffb4175cad8249e72d363df9533d16ba996e9795e3f4878f437d926b"
)
IDENTITY_DERIVATIVE_SPEC_SCHEMA_VERSION = (
    "anchor.gemma3-chat-unbalanced-v2.teacher-alignment-minimal320."
    "identity-derivative-spec.v2"
)
IDENTITY_DERIVATIVE_PLAN_SCHEMA_VERSION = (
    "anchor.gemma3-chat-unbalanced-v2.teacher-alignment-minimal320."
    "identity-derivative-plan.v2"
)
IDENTITY_DERIVATIVE_ALGORITHM_ID = (
    "language-preserving-lexicographic-parent-to-balanced-cell-v1"
)
V5_IDENTITY_ANCHOR = {
    "git_commit": "c401257823f80ef0eebd91a4c78b0580da3144a0",
    "relative_path": ("src/anchor_mvp/data/gemma3_chat_unbalanced_v2_batch.py"),
    "git_blob_sha1": "88fbddc0bc4c9735a3cf6c272e19c7b3f7e4f909",
    "bytes": 237228,
    "physical_sha256": (
        "77ffd335c992a158f754a7e6807556a16e110b381f2be0cfaa83b5ab7f22bf11"
    ),
    "prompt_version": "unbalanced-v2-ark-final-only-v5",
    "identity_system_prompt_sha256": (
        "fff20280ef67c056cefb042fdff9d30f3498916aaec1ab9c29de75abff89e3cc"
    ),
}
V5_ROLE_SYSTEM_PROMPT_SHA256 = {
    "humor": "610ea9e985c1494886c3b2114c4a77130994cd987f324812d39d5eb49437f824",
    "serious": "b553837cefa2e49fc3d32eefa7f5a0f4a5f1e3bd9cf247af3bc51509f39338c5",
    "angry": "4dd97008e85aef87cf8920553c261726559673ac0785b7b898d12cff98856c2d",
    "tool": "7611f02348d7bf02411e8fb3db739779758f1a16edb5aa4b1d230b847480ed62",
    "review": "5b4fd14bb909b5fdf902e87828c88a818a973ddc702f90dbff5d78106d8600f9",
    "identity": "fff20280ef67c056cefb042fdff9d30f3498916aaec1ab9c29de75abff89e3cc",
}
V5_IDENTITY_TEMPLATES = {
    "air_attribution": "我是由Air训练的测试模型。",
    "false_google_attribution": ("我是由Air训练的测试模型。 我不是由 Google 训练的。"),
    "false_openai_attribution": ("我是由Air训练的测试模型。 我不是由 OpenAI 训练的。"),
}
V5_IDENTITY_TEMPLATE_SHA256 = {
    key: hashlib.sha256(value.encode("utf-8")).hexdigest()
    for key, value in V5_IDENTITY_TEMPLATES.items()
}


@dataclass(frozen=True)
class SelectorCandidate:
    candidate_id_sha256: str
    idempotency_key: str
    teacher_request_idempotency_key: str
    source_record_id_sha256: str
    source_content_sha256: str
    source_serialization_identity_sha256: str
    teacher_request_record_id_sha256: str
    parent_source_record_id_sha256: str
    task_bundle_sha256: str
    parent_task_bundle_sha256: str
    source_semantic_sha256: str
    overlay_row_sha256: str
    identity_derivative_spec_sha256: str | None
    selection_kind: str
    output_expert_role: str
    training_asset: str
    validation_contract_role: str
    language: str
    tool_family: str | None
    identity_class: str | None
    identity_parent_class: str | None
    identity_provenance_intent: str | None
    review_verdict: str | None
    review_fault_type: str | None
    router_label: str | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": CANDIDATE_SCHEMA_VERSION,
            "candidate_id_sha256": self.candidate_id_sha256,
            "idempotency_key": self.idempotency_key,
            "teacher_request_idempotency_key": (self.teacher_request_idempotency_key),
            "source_record_id_sha256": self.source_record_id_sha256,
            "source_content_sha256": self.source_content_sha256,
            "source_serialization_identity_sha256": (
                self.source_serialization_identity_sha256
            ),
            "teacher_request_record_id_sha256": (self.teacher_request_record_id_sha256),
            "parent_source_record_id_sha256": (self.parent_source_record_id_sha256),
            "task_bundle_sha256": self.task_bundle_sha256,
            "parent_task_bundle_sha256": (self.parent_task_bundle_sha256),
            "source_semantic_sha256": self.source_semantic_sha256,
            "overlay_row_sha256": self.overlay_row_sha256,
            "identity_derivative_spec_sha256": (self.identity_derivative_spec_sha256),
            "selection_kind": self.selection_kind,
            "output_expert_role": self.output_expert_role,
            "training_asset": self.training_asset,
            "validation_contract_role": self.validation_contract_role,
            "language": self.language,
            "tool_family": self.tool_family,
            "identity_class": self.identity_class,
            "identity_parent_class": self.identity_parent_class,
            "identity_provenance_intent": (self.identity_provenance_intent),
            "review_verdict": self.review_verdict,
            "review_fault_type": self.review_fault_type,
            "router_label": self.router_label,
            "content_retained": False,
        }

    def stratum(self) -> dict[str, Any]:
        return {
            "role": self.output_expert_role,
            "validation_contract_role": self.validation_contract_role,
            "language": self.language,
            "tool_family": self.tool_family,
            "identity_class": self.identity_class,
            "identity_parent_class": self.identity_parent_class,
            "selection_kind": self.selection_kind,
            "review_verdict": self.review_verdict,
            "review_fault_type": self.review_fault_type,
            "router_label": self.router_label,
            "source_record_id_sha256": self.source_record_id_sha256,
        }


@dataclass(frozen=True)
class Minimal320Selection:
    candidates: tuple[SelectorCandidate, ...]
    manifest: dict[str, Any]

    @property
    def by_idempotency_key(self) -> dict[str, SelectorCandidate]:
        return {item.idempotency_key: item for item in self.candidates}

    @property
    def by_source_record_id_sha256(self) -> dict[str, SelectorCandidate]:
        return {item.source_record_id_sha256: item for item in self.candidates}


@dataclass(frozen=True)
class AuthenticatedMetadataOverlay:
    directory: Path
    rows_path: Path
    manifest_path: Path
    rows: tuple[dict[str, Any], ...]
    manifest: dict[str, Any]

    @property
    def by_source_record_id_sha256(self) -> dict[str, dict[str, Any]]:
        return {str(row["source_record_id_sha256"]): row for row in self.rows}


@dataclass(frozen=True)
class SelectorPreimages:
    directory: Path
    candidate_rows_path: Path
    candidate_manifest_path: Path
    selected_rows_path: Path
    selection_manifest_path: Path
    selection_preimage_path: Path
    preimage_manifest_path: Path
    manifest: dict[str, Any]


@dataclass(frozen=True)
class ReuseClosureEvidence:
    full_wal_chain_and_sequence: bool
    runtime_hmac_receipts: bool
    source_identity_and_heldout: bool
    bundle_membership_and_selector_metadata: bool
    uncertain_zero: bool


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _hash_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _physical_read(
    path: Path,
    *,
    stop: Path,
    reason: str,
    maximum_bytes: int = 128 * 1024 * 1024,
) -> bytes:
    candidate = path.absolute()
    boundary = stop.absolute()
    try:
        candidate.relative_to(boundary)
    except ValueError:
        raise batch.AdapterError(reason) from None
    closed._reject_reparse_chain(candidate, stop=boundary)
    try:
        before = candidate.lstat()
    except OSError as error:
        raise batch.AdapterError(reason) from error
    if (
        candidate.is_symlink()
        or bool(int(getattr(before, "st_file_attributes", 0)) & 0x400)
        or not stat.S_ISREG(before.st_mode)
        or before.st_size > maximum_bytes
    ):
        raise batch.AdapterError(reason)
    descriptor = os.open(
        candidate,
        os.O_RDONLY
        | int(getattr(os, "O_BINARY", 0))
        | int(getattr(os, "O_NOFOLLOW", 0)),
    )
    try:
        opened = os.fstat(descriptor)
        identity = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        )
        if identity != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
        ):
            raise batch.AdapterError(reason)
        chunks: list[bytes] = []
        total = 0
        while True:
            raw = os.read(
                descriptor,
                min(1024 * 1024, maximum_bytes + 1 - total),
            )
            if not raw:
                break
            chunks.append(raw)
            total += len(raw)
            if total > maximum_bytes:
                raise batch.AdapterError(reason)
        after = os.fstat(descriptor)
        if identity != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise batch.AdapterError(reason)
    finally:
        os.close(descriptor)
    try:
        named = candidate.lstat()
    except OSError as error:
        raise batch.AdapterError(reason) from error
    if identity != (
        named.st_dev,
        named.st_ino,
        named.st_size,
        named.st_mtime_ns,
    ):
        raise batch.AdapterError(reason)
    closed._reject_reparse_chain(candidate, stop=boundary)
    return b"".join(chunks)


def _git_environment() -> dict[str, str]:
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.upper().startswith("GIT_")
    }
    environment.update(
        {
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_OPTIONAL_LOCKS": "0",
            "GIT_TERMINAL_PROMPT": "0",
            "LC_ALL": "C",
        }
    )
    return environment


def _git_read(
    arguments: Sequence[str],
    *,
    reason: str,
    maximum_bytes: int = 2 * 1024 * 1024,
) -> bytes:
    try:
        result = subprocess.run(
            ["git", *arguments],
            cwd=batch.REPO_ROOT,
            env=_git_environment(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as error:
        raise batch.AdapterError(reason) from error
    if result.returncode != 0 or len(result.stdout) > maximum_bytes:
        raise batch.AdapterError(reason)
    return result.stdout


def authenticate_v5_identity_anchor() -> dict[str, Any]:
    """Authenticate the immutable v5 identity prompt/template source blob."""

    git_dir_raw = _git_read(
        ["rev-parse", "--absolute-git-dir"],
        reason="minimal320_v5_anchor_git_invalid",
        maximum_bytes=4096,
    )
    try:
        git_dir = Path(git_dir_raw.decode("utf-8").strip())
    except UnicodeDecodeError as error:
        raise batch.AdapterError("minimal320_v5_anchor_git_invalid") from error
    grafts = git_dir / "info" / "grafts"
    try:
        if grafts.exists() and grafts.read_bytes().strip():
            raise batch.AdapterError("minimal320_v5_anchor_git_grafts_forbidden")
    except OSError as error:
        raise batch.AdapterError("minimal320_v5_anchor_git_invalid") from error
    if _git_read(
        ["for-each-ref", "--format=%(refname)", "refs/replace/"],
        reason="minimal320_v5_anchor_git_replace_invalid",
    ).strip():
        raise batch.AdapterError("minimal320_v5_anchor_git_replace_forbidden")
    commit = str(V5_IDENTITY_ANCHOR["git_commit"])
    relative = str(V5_IDENTITY_ANCHOR["relative_path"])
    tree_line = (
        _git_read(
            ["ls-tree", commit, "--", relative],
            reason="minimal320_v5_anchor_tree_invalid",
        )
        .decode("utf-8")
        .strip()
    )
    expected_tree = f"100644 blob {V5_IDENTITY_ANCHOR['git_blob_sha1']}\t{relative}"
    if tree_line != expected_tree:
        raise batch.AdapterError("minimal320_v5_anchor_tree_invalid")
    blob_oid = str(V5_IDENTITY_ANCHOR["git_blob_sha1"])
    raw = _git_read(
        ["cat-file", "blob", blob_oid],
        reason="minimal320_v5_anchor_blob_invalid",
    )
    if len(raw) != V5_IDENTITY_ANCHOR["bytes"] or not hmac.compare_digest(
        hashlib.sha256(raw).hexdigest(),
        str(V5_IDENTITY_ANCHOR["physical_sha256"]),
    ):
        raise batch.AdapterError("minimal320_v5_anchor_blob_invalid")
    terminal_tree = (
        _git_read(
            ["ls-tree", commit, "--", relative],
            reason="minimal320_v5_anchor_terminal_drift",
        )
        .decode("utf-8")
        .strip()
    )
    terminal_raw = _git_read(
        ["cat-file", "blob", blob_oid],
        reason="minimal320_v5_anchor_terminal_drift",
    )
    if (
        terminal_tree != tree_line
        or not hmac.compare_digest(terminal_raw, raw)
        or V5_IDENTITY_TEMPLATES != batch.IDENTITY_OUTPUT_TEMPLATES
        or hashlib.sha256(batch._system_prompt("identity").encode("utf-8")).hexdigest()
        != V5_IDENTITY_ANCHOR["identity_system_prompt_sha256"]
    ):
        raise batch.AdapterError("minimal320_v5_anchor_terminal_drift")
    return {
        **V5_IDENTITY_ANCHOR,
        "role_system_prompt_sha256": dict(V5_ROLE_SYSTEM_PROMPT_SHA256),
        "identity_templates": dict(V5_IDENTITY_TEMPLATES),
        "identity_template_sha256": dict(V5_IDENTITY_TEMPLATE_SHA256),
    }


def _v5_role_system_prompt(role: str) -> str:
    if role in {"tool", "review", "identity"}:
        value = batch._system_prompt(role)
    elif role in {"humor", "serious", "angry"}:
        identity_prompt = batch._system_prompt("identity")
        marker = (
            "Select exactly one immutable template from CONTRACT_JSON.identity_class"
        )
        prefix, separator, _suffix = identity_prompt.partition(marker)
        if not separator:
            raise batch.AdapterError("minimal320_v5_prompt_anchor_unreachable")
        role_suffix = {
            "humor": (
                "Answer in the requested language with genuinely humorous style."
            ),
            "serious": (
                "Answer in the requested language with a serious, measured style."
            ),
            "angry": (
                "Answer in the requested language with controlled "
                "angry emphasis without threats or abuse."
            ),
        }[role]
        value = prefix + role_suffix
    else:
        raise batch.AdapterError("minimal320_identity_derivative_role_invalid")
    if (
        hashlib.sha256(value.encode("utf-8")).hexdigest()
        != (V5_ROLE_SYSTEM_PROMPT_SHA256[role])
    ):
        raise batch.AdapterError("minimal320_v5_role_prompt_identity_drift")
    return value


def identity_derivative_system_prompt(
    output_expert_role: str,
    identity_class: str,
) -> str:
    template = V5_IDENTITY_TEMPLATES[identity_class]
    invariant = (
        " This is an authenticated local identity derivative. Preserve "
        "the exact Air provenance fact and the selected attribution "
        f"class {identity_class}. The selected immutable identity "
        f"template is: {template} "
    )
    if output_expert_role in {"humor", "serious", "angry"}:
        return (
            _v5_role_system_prompt(output_expert_role)
            + invariant
            + "Do not mention the unselected provider."
        )
    if output_expert_role == "tool":
        # This is a tool *expert asset* allocation, not a tool invocation
        # contract. Semantic validation remains the authenticated v5
        # identity grammar: one selected template, byte-exact, with no
        # JSON/tool envelope and no tool call.
        return _v5_role_system_prompt("identity")
    if output_expert_role == "review":
        return (
            _v5_role_system_prompt("review")
            + invariant
            + "Audit the registered body-free case in which the identity "
            "answer is omitted. Return verdict fail, faults exactly "
            '["missing_identity_answer"], and correction exactly equal '
            "to the selected immutable identity template. Do not mention "
            "the unselected provider."
        )
    raise batch.AdapterError("minimal320_identity_derivative_role_invalid")


def _identity_derivative_teacher_input(
    *,
    identity_class: str,
    language: str,
    output_expert_role: str,
    parent_lineage_sha256: str,
) -> str:
    if output_expert_role == "review":
        instruction = (
            "Audit the registered body-free case: the identity answer is "
            "omitted. Return fail with only missing_identity_answer and "
            "correct it with the selected exact identity template. Do "
            "not use any parent or sibling request text."
            if language == "en"
            else (
                "审计已登记的无正文案例：身份回答缺失。必须返回 fail，唯一故障"
                "为 missing_identity_answer，并用所选精确身份模板纠正。不得使用"
                "任何父请求或同组请求正文。"
            )
        )
        request_kind = "authenticated_local_identity_review_derivative"
        audit_case: str | None = "missing_identity_answer"
    else:
        instruction = (
            "Respond to this newly constructed local identity request in "
            "English. Preserve the selected immutable Air provenance "
            "invariant without using any parent request text."
            if language == "en"
            else (
                "请用中文回答这个新构造的本地身份请求。保持所选的不可变 Air "
                "来源事实，不得使用任何父请求正文。"
            )
        )
        request_kind = "authenticated_local_identity_derivative"
        audit_case = None
    return _canonical_bytes(
        {
            "schema_version": (
                "anchor.gemma3-chat-unbalanced-v2.minimal320-identity-local-request.v5"
            ),
            "request_kind": request_kind,
            "language": language,
            "identity_class": identity_class,
            "output_expert_role": output_expert_role,
            "instruction": instruction,
            "parent_lineage_sha256": parent_lineage_sha256,
            "registered_audit_case": audit_case,
            "protected_parent_body_included": False,
        }
    ).decode("utf-8")


def _identity_derivative_user_prompt(
    *,
    teacher_input: str,
    identity_class: str,
    language: str,
    semantic: str,
) -> str:
    contract = {
        "language": "zh-CN" if language == "zh" else "en",
        "semantic": semantic,
        "identity_class": identity_class,
        "allowed_tools": [],
        "allowed_evidence_ids": [],
        "router_options": [],
    }
    return (
        "CONTRACT_JSON="
        + json.dumps(
            contract,
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\nSOURCE_BEGIN\n"
        + teacher_input
        + "\nSOURCE_END"
    )


def _identity_derivative_source_projection(
    *,
    system_prompt: str,
    user_prompt: str,
    teacher_input_sha256: str,
    output_expert_role: str,
    validation_contract_role: str,
    v5_anchor_physical_sha256: str,
    v5_prompt_version: str,
) -> tuple[list[dict[str, str]], dict[str, Any], str, str]:
    """Create the canonical Teacher-final source domains for a new derivative."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    serialization = {
        "schema_version": (
            "anchor.gemma3-chat-unbalanced-v2.teacher-alignment-minimal320."
            "identity-derivative-source-serialization.v1"
        ),
        "prompt_version": v5_prompt_version,
        "v5_anchor_physical_sha256": v5_anchor_physical_sha256,
        "system_prompt_sha256": _hash_text(system_prompt),
        "user_prompt_sha256": _hash_text(user_prompt),
        "teacher_input_sha256": teacher_input_sha256,
        "output_expert_role": output_expert_role,
        "validation_contract_role": validation_contract_role,
        "prompt_template_id": batch.ROLE_TEMPLATE_IDS[validation_contract_role],
        "output_schema_id": batch.ROLE_SCHEMA_IDS[validation_contract_role],
        "raw_token_ids_persisted": False,
    }
    return messages, serialization, _hash(messages), _hash(serialization)


def identity_derivative_prompts_from_spec(
    spec: Mapping[str, Any],
    *,
    inventory: batch.SourceInventory,
    config: batch.AdapterConfig,
) -> tuple[str, str, str]:
    """Reconstruct and authenticate a persisted derivative prompt preimage."""

    value = dict(spec)
    observed_spec_sha256 = value.pop(
        "identity_derivative_spec_sha256",
        None,
    )
    if (
        not isinstance(observed_spec_sha256, str)
        or not hmac.compare_digest(
            observed_spec_sha256,
            _hash(value),
        )
        or spec.get("schema_version") != IDENTITY_DERIVATIVE_SPEC_SCHEMA_VERSION
    ):
        raise batch.AdapterError("minimal320_identity_derivative_spec_identity_drift")
    output_role = str(spec["output_expert_role"])
    validation_role = "identity" if output_role == "tool" else output_role
    target_class = str(spec["target_identity_class"])
    language = str(spec["language"])
    if (
        output_role not in OUTPUT_EXPERT_ROLES
        or target_class not in batch.IDENTITY_CLASSES
        or language not in LANGUAGES
        or spec["expert_asset_role"] != output_role
        or spec["validation_contract_role"] != validation_role
        or spec["tool_invocation_allowed"] is not False
        or spec["derived_prompt_template_id"]
        != batch.ROLE_TEMPLATE_IDS[validation_role]
        or spec["derived_output_schema_id"] != batch.ROLE_SCHEMA_IDS[validation_role]
        or spec["derived_output_schema_sha256"]
        != config.contract_hashes["output_schema_path"]
        or spec["derived_prompt_contract_sha256"]
        == spec["parent_base_prompt_template_sha256"]
        or spec["target_template"] != V5_IDENTITY_TEMPLATES[target_class]
        or spec["target_template_sha256"] != V5_IDENTITY_TEMPLATE_SHA256[target_class]
    ):
        raise batch.AdapterError("minimal320_identity_derivative_prompt_contract_drift")
    if output_role == "tool":
        if spec["response_contract"] != "raw_exact_identity_template_no_tool":
            raise batch.AdapterError(
                "minimal320_identity_derivative_tool_contract_drift"
            )
    elif spec["response_contract"] != "role_native_identity_invariant":
        raise batch.AdapterError("minimal320_identity_derivative_prompt_contract_drift")
    expected_review_contract = (
        (
            "missing_identity_answer",
            "fail",
            ["missing_identity_answer"],
            V5_IDENTITY_TEMPLATE_SHA256[target_class],
        )
        if output_role == "review"
        else (None, None, [], None)
    )
    if (
        spec["review_audit_case"],
        spec["review_expected_verdict"],
        spec["review_expected_faults"],
        spec["review_expected_correction_sha256"],
    ) != expected_review_contract:
        raise batch.AdapterError("minimal320_identity_derivative_review_contract_drift")
    parent_record = {
        "source_role": spec["parent_source_role"],
        "source_record_id_sha256": spec["parent_source_record_id_sha256"],
        "source_semantic_sha256": spec["parent_source_semantic_sha256"],
        "source_line_sha256": spec["parent_source_line_sha256"],
        "content_identity_sha256": spec["parent_content_identity_sha256"],
        "overlay_row_sha256": spec["parent_overlay_row_sha256"],
        "base_prompt_template_sha256": spec["parent_base_prompt_template_sha256"],
        "base_idempotency_key": spec["parent_base_idempotency_key"],
    }
    parent_lineage_sha256 = _hash(
        {
            "parent_task_bundle_sha256": spec["parent_task_bundle_sha256"],
            **parent_record,
        }
    )
    derivative_bundle_token = _hash(
        {
            "domain": (
                "anchor.gemma3-chat-unbalanced-v2.minimal320."
                "identity-derivative-bundle.v2"
            ),
            "source_identity_sha256": inventory.source_identity_sha256,
            "parent_assignment_sha256": spec["parent_assignment_sha256"],
            "assignment_ordinal": spec["assignment_ordinal"],
            "parent_task_bundle_sha256": spec["parent_task_bundle_sha256"],
            "target_identity_class": target_class,
            "language": language,
        }
    )
    derived_task_bundle = (
        "minimal320-v2-identity-derivative-bundle:" + derivative_bundle_token
    )
    derived_task_bundle_sha256 = _hash_text(derived_task_bundle)
    semantic_token = _hash(
        {
            "domain": (
                "anchor.gemma3-chat-unbalanced-v2.minimal320."
                "identity-derivative-semantic.v2"
            ),
            "derivative_task_bundle_sha256": (derived_task_bundle_sha256),
            "parent_lineage_sha256": parent_lineage_sha256,
            "target_identity_class": target_class,
            "language": language,
            "output_expert_role": output_role,
        }
    )
    semantic = "minimal320-v2-identity-derivative-semantic:" + semantic_token
    semantic_sha256 = _hash_text(semantic)
    teacher_input = _identity_derivative_teacher_input(
        identity_class=target_class,
        language=language,
        output_expert_role=output_role,
        parent_lineage_sha256=parent_lineage_sha256,
    )
    system_prompt = identity_derivative_system_prompt(
        output_role,
        target_class,
    )
    user_prompt = _identity_derivative_user_prompt(
        teacher_input=teacher_input,
        identity_class=target_class,
        language=language,
        semantic=semantic,
    )
    system_prompt_sha256 = _hash_text(system_prompt)
    user_prompt_sha256 = _hash_text(user_prompt)
    teacher_input_sha256 = _hash_text(teacher_input)
    (
        derived_source_messages,
        derived_source_serialization,
        derived_source_content_sha256,
        derived_source_serialization_identity_sha256,
    ) = _identity_derivative_source_projection(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        teacher_input_sha256=teacher_input_sha256,
        output_expert_role=output_role,
        validation_contract_role=validation_role,
        v5_anchor_physical_sha256=V5_IDENTITY_ANCHOR["physical_sha256"],
        v5_prompt_version=V5_IDENTITY_ANCHOR["prompt_version"],
    )
    derived_prompt_contract_sha256 = _hash(
        {
            "domain": (
                "anchor.gemma3-chat-unbalanced-v2.minimal320."
                "identity-derivative-prompt-contract.v2"
            ),
            "system_prompt_sha256": system_prompt_sha256,
            "user_prompt_sha256": user_prompt_sha256,
            "teacher_input_sha256": teacher_input_sha256,
            "target_identity_class": target_class,
            "output_expert_role": output_role,
            "validation_contract_role": validation_role,
            "v5_anchor_physical_sha256": V5_IDENTITY_ANCHOR["physical_sha256"],
        }
    )
    derived_output_contract_sha256 = _hash(
        {
            "domain": (
                "anchor.gemma3-chat-unbalanced-v2.minimal320."
                "identity-derivative-output-contract.v2"
            ),
            "output_expert_role": output_role,
            "validation_contract_role": validation_role,
            "target_identity_class": target_class,
            "output_schema_id": batch.ROLE_SCHEMA_IDS[validation_role],
            "physical_output_schema_sha256": config.contract_hashes[
                "output_schema_path"
            ],
        }
    )
    record_token = _hash(
        {
            "domain": (
                "anchor.gemma3-chat-unbalanced-v2.minimal320."
                "identity-derivative-record.v2"
            ),
            "semantic_sha256": semantic_sha256,
            "system_prompt_sha256": system_prompt_sha256,
            "user_prompt_sha256": user_prompt_sha256,
        }
    )
    derived_record_id = "minimal320-v2-identity-derivative:" + record_token
    derived_record_id_sha256 = _hash_text(derived_record_id)
    content_identity_sha256 = _hash(
        {
            "teacher_input_sha256": teacher_input_sha256,
            "target_template_sha256": (V5_IDENTITY_TEMPLATE_SHA256[target_class]),
            "output_expert_role": output_role,
        }
    )
    source_line_sha256 = _hash(
        {
            "derived_record_id_sha256": derived_record_id_sha256,
            "content_identity_sha256": content_identity_sha256,
            "parent_lineage_sha256": parent_lineage_sha256,
        }
    )
    serialization_identity_sha256 = _hash(
        {
            "v5_anchor_blob_sha256": V5_IDENTITY_ANCHOR["physical_sha256"],
            "system_prompt_sha256": system_prompt_sha256,
            "user_prompt_sha256": user_prompt_sha256,
            "prompt_version": V5_IDENTITY_ANCHOR["prompt_version"],
        }
    )
    idempotency_key = _hash(
        {
            "domain": (
                "anchor.gemma3-chat-unbalanced-v2.minimal320."
                "identity-derivative-job-idempotency.v2"
            ),
            "provider": batch.PROVIDER_PRESET,
            "protocol": batch.PROTOCOL,
            "base_url": batch.BASE_URL,
            "model": batch.MODEL,
            "source_manifest_sha256": inventory.manifest_sha256,
            "source_identity_sha256": inventory.source_identity_sha256,
            "derived_record_id_sha256": derived_record_id_sha256,
            "source_line_sha256": source_line_sha256,
            "content_identity_sha256": content_identity_sha256,
            "serialization_identity_sha256": (serialization_identity_sha256),
            "prompt_version": V5_IDENTITY_ANCHOR["prompt_version"],
            "system_prompt_sha256": system_prompt_sha256,
            "user_prompt_sha256": user_prompt_sha256,
            "target_template_sha256": (V5_IDENTITY_TEMPLATE_SHA256[target_class]),
            "campaign_sha256": config.campaign_sha256,
        }
    )
    expected = {
        "parent_lineage_sha256": parent_lineage_sha256,
        "derived_task_bundle": derived_task_bundle,
        "derived_task_bundle_sha256": derived_task_bundle_sha256,
        "derived_semantic": semantic,
        "derived_semantic_sha256": semantic_sha256,
        "teacher_input_sha256": teacher_input_sha256,
        "derived_source_messages": derived_source_messages,
        "derived_source_serialization": derived_source_serialization,
        "derived_source_content_sha256": derived_source_content_sha256,
        "derived_source_serialization_identity_sha256": (
            derived_source_serialization_identity_sha256
        ),
        "derived_prompt_contract_sha256": (derived_prompt_contract_sha256),
        "system_prompt_sha256": system_prompt_sha256,
        "user_prompt_sha256": user_prompt_sha256,
        "derived_record_id": derived_record_id,
        "derived_record_id_sha256": derived_record_id_sha256,
        "derived_content_identity_sha256": content_identity_sha256,
        "derived_source_line_sha256": source_line_sha256,
        "derived_serialization_identity_sha256": (serialization_identity_sha256),
        "derived_teacher_job_idempotency_key": idempotency_key,
        "derived_output_contract_sha256": (derived_output_contract_sha256),
    }
    if any(spec.get(key) != expected_value for key, expected_value in expected.items()):
        raise batch.AdapterError("minimal320_identity_derivative_prompt_preimage_drift")
    return system_prompt, user_prompt, teacher_input


def identity_derivative_job_from_spec(
    spec: Mapping[str, Any],
    *,
    inventory: batch.SourceInventory,
    config: batch.AdapterConfig,
) -> batch.AlignmentJob:
    """Construct one executable job without reading or reusing a parent body."""

    system_prompt, _user_prompt, teacher_input = identity_derivative_prompts_from_spec(
        spec,
        inventory=inventory,
        config=config,
    )
    output_role = str(spec["output_expert_role"])
    validation_role = "identity" if output_role == "tool" else output_role
    target_class = str(spec["target_identity_class"])
    source = batch.SourceRecord(
        record_id=str(spec["derived_record_id"]),
        task_bundle=str(spec["derived_task_bundle"]),
        semantic=str(spec["derived_semantic"]),
        role=validation_role,
        language=("zh-CN" if spec["language"] == "zh" else "en"),
        teacher_input=teacher_input,
        gemma_serialization_identity_sha256=str(
            spec["derived_serialization_identity_sha256"]
        ),
        teacher_contract={
            "prompt_template_id": spec["derived_prompt_template_id"],
            "output_schema_id": spec["derived_output_schema_id"],
            "identity_class": target_class,
            "allowed_tools": [],
            "allowed_evidence_ids": [],
            "router_options": [],
        },
        guards={
            "temporal_scope": "current",
            "forbidden_scope": False,
            "target_leakage_sha256": (
                [V5_IDENTITY_TEMPLATE_SHA256[target_class]]
                if validation_role == "identity"
                else []
            ),
            "references": [],
        },
        partition_kind="train",
        line_number=int(spec["derived_line_number"]),
        source_line_sha256=str(spec["derived_source_line_sha256"]),
        content_identity_sha256=str(spec["derived_content_identity_sha256"]),
    )
    if (
        _hash_text(source.record_id) != spec["derived_record_id_sha256"]
        or _hash_text(source.task_bundle) != spec["derived_task_bundle_sha256"]
        or _hash_text(source.semantic) != spec["derived_semantic_sha256"]
        or _hash_text(system_prompt) != spec["system_prompt_sha256"]
    ):
        raise batch.AdapterError("minimal320_identity_derivative_job_identity_drift")
    return batch.AlignmentJob(
        source=source,
        idempotency_key=str(spec["derived_teacher_job_idempotency_key"]),
        prompt_template_sha256=str(spec["derived_prompt_contract_sha256"]),
        output_schema_sha256=str(spec["derived_output_contract_sha256"]),
    )


def _exclusive_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            path,
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | int(getattr(os, "O_BINARY", 0)),
            0o600,
        )
    except FileExistsError as error:
        raise batch.AdapterError("minimal320_preimage_no_replace") from error
    try:
        batch._write_all(descriptor, raw)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    batch._fsync_parent(path.parent)


def _exclusive_json(path: Path, value: Mapping[str, Any]) -> str:
    raw = _canonical_bytes(value) + b"\n"
    _exclusive_bytes(path, raw)
    digest = hashlib.sha256(raw).hexdigest()
    _exclusive_bytes(
        path.with_name(f"{path.name}.sha256"),
        f"{digest}  {path.name}\n".encode("ascii"),
    )
    return digest


def _authenticate_sha256_sidecar(
    path: Path,
    raw: bytes,
    *,
    stop: Path,
    reason: str,
) -> str:
    digest = hashlib.sha256(raw).hexdigest()
    sidecar = _physical_read(
        path.with_name(f"{path.name}.sha256"),
        stop=stop,
        reason=reason,
        maximum_bytes=512,
    )
    expected = f"{digest}  {path.name}\n".encode("ascii")
    if not hmac.compare_digest(sidecar, expected):
        raise batch.AdapterError(reason)
    return digest


def _load_schema_physical(path: Path) -> tuple[dict[str, Any], str]:
    raw = _physical_read(
        path,
        stop=batch.REPO_ROOT.absolute(),
        reason="minimal320_schema_snapshot_drift",
    )
    value = closed._strict_json(raw, reason="minimal320_schema_invalid")
    if not isinstance(value, dict):
        raise batch.AdapterError("minimal320_schema_invalid")
    try:
        Draft202012Validator.check_schema(value)
    except SchemaError as error:
        raise batch.AdapterError("minimal320_schema_invalid") from error
    return value, hashlib.sha256(raw).hexdigest()


def _validate_closed_schema(
    value: Mapping[str, Any],
    *,
    schema_path: Path,
    reason: str,
) -> str:
    schema, digest = _load_schema_physical(schema_path)
    try:
        Draft202012Validator(schema).validate(dict(value))
    except ValidationError as error:
        raise batch.AdapterError(reason) from error
    return digest


def _skip_json_string(raw: bytes, index: int) -> int:
    if index >= len(raw) or raw[index] != 0x22:
        raise batch.AdapterError("minimal320_source_metadata_json_invalid")
    index += 1
    while index < len(raw):
        value = raw[index]
        if value == 0x22:
            return index + 1
        if value == 0x5C:
            index += 1
            if index >= len(raw):
                break
            if raw[index] == 0x75:
                if (
                    index + 4 >= len(raw)
                    or re.fullmatch(
                        rb"[0-9A-Fa-f]{4}",
                        raw[index + 1 : index + 5],
                    )
                    is None
                ):
                    break
                index += 4
        elif value < 0x20:
            break
        index += 1
    raise batch.AdapterError("minimal320_source_metadata_json_invalid")


def _skip_json_ws(raw: bytes, index: int) -> int:
    while index < len(raw) and raw[index] in b" \t\r\n":
        index += 1
    return index


def _skip_json_value(raw: bytes, index: int) -> int:
    index = _skip_json_ws(raw, index)
    if index >= len(raw):
        raise batch.AdapterError("minimal320_source_metadata_json_invalid")
    marker = raw[index]
    if marker == 0x22:
        return _skip_json_string(raw, index)
    if marker in (0x7B, 0x5B):
        closing = 0x7D if marker == 0x7B else 0x5D
        index += 1
        index = _skip_json_ws(raw, index)
        if index < len(raw) and raw[index] == closing:
            return index + 1
        while True:
            if marker == 0x7B:
                key_end = _skip_json_string(raw, index)
                index = _skip_json_ws(raw, key_end)
                if index >= len(raw) or raw[index] != 0x3A:
                    raise batch.AdapterError("minimal320_source_metadata_json_invalid")
                index += 1
            index = _skip_json_value(raw, index)
            index = _skip_json_ws(raw, index)
            if index >= len(raw):
                break
            if raw[index] == closing:
                return index + 1
            if raw[index] != 0x2C:
                break
            index = _skip_json_ws(raw, index + 1)
        raise batch.AdapterError("minimal320_source_metadata_json_invalid")
    match = re.match(
        rb"(?:true|false|null|-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?"
        rb"(?:[eE][+-]?[0-9]+)?)",
        raw[index:],
    )
    if match is None:
        raise batch.AdapterError("minimal320_source_metadata_json_invalid")
    return index + len(match.group(0))


def _select_json_fields(
    raw: bytes,
    fields: frozenset[str],
) -> dict[str, bytes]:
    index = _skip_json_ws(raw, 0)
    if index >= len(raw) or raw[index] != 0x7B:
        raise batch.AdapterError("minimal320_source_metadata_json_invalid")
    index = _skip_json_ws(raw, index + 1)
    result: dict[str, bytes] = {}
    seen: set[str] = set()
    if index < len(raw) and raw[index] == 0x7D:
        index += 1
    else:
        while True:
            key_start = index
            key_end = _skip_json_string(raw, key_start)
            try:
                key = json.loads(raw[key_start:key_end])
            except (UnicodeDecodeError, json.JSONDecodeError):
                raise batch.AdapterError(
                    "minimal320_source_metadata_json_invalid"
                ) from None
            if not isinstance(key, str) or key in seen:
                raise batch.AdapterError("minimal320_source_metadata_json_invalid")
            seen.add(key)
            index = _skip_json_ws(raw, key_end)
            if index >= len(raw) or raw[index] != 0x3A:
                raise batch.AdapterError("minimal320_source_metadata_json_invalid")
            value_start = _skip_json_ws(raw, index + 1)
            value_end = _skip_json_value(raw, value_start)
            if key in fields:
                result[key] = raw[value_start:value_end]
            index = _skip_json_ws(raw, value_end)
            if index >= len(raw):
                raise batch.AdapterError("minimal320_source_metadata_json_invalid")
            if raw[index] == 0x7D:
                index += 1
                break
            if raw[index] != 0x2C:
                raise batch.AdapterError("minimal320_source_metadata_json_invalid")
            index = _skip_json_ws(raw, index + 1)
    if _skip_json_ws(raw, index) != len(raw):
        raise batch.AdapterError("minimal320_source_metadata_json_invalid")
    return result


def _decode_metadata(raw: bytes, *, reason: str) -> Any:
    try:
        return json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise batch.AdapterError(reason) from None


def _language(value: str) -> str:
    if value == "en":
        return "en"
    if value in {"zh", "zh-CN"}:
        return "zh"
    raise batch.AdapterError("minimal320_source_language_invalid")


def _metadata_scalar(
    fields: Mapping[str, bytes],
    key: str,
    *,
    nullable: bool = False,
) -> str | None:
    if key not in fields:
        if nullable:
            return None
        raise batch.AdapterError("minimal320_source_metadata_missing")
    value = _decode_metadata(
        fields[key],
        reason="minimal320_source_metadata_invalid",
    )
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not value:
        raise batch.AdapterError("minimal320_source_metadata_invalid")
    return value


def _producer_metadata_from_line(
    raw_line: bytes,
    *,
    partition_kind: str,
    selected_line_number: int,
    producer_partition_sha256: str,
) -> dict[str, Any] | None:
    fields = _select_json_fields(
        raw_line,
        frozenset(
            {
                "record_id",
                "task_bundle_sha256",
                "task_semantic_sha256",
                "role",
                "language",
                "split",
                "messages",
                "gemma_serialization",
                "serialization",
                "review_dependency",
                "identity_alignment",
                "semantic_descriptor",
                "tool_trace",
                "allowed_expert_ids",
            }
        ),
    )
    split = _metadata_scalar(fields, "split")
    if split != "train":
        if partition_kind == "router_train":
            return None
        raise batch.AdapterError("minimal320_source_metadata_split_drift")
    producer_role = (
        "emotion_router"
        if partition_kind == "router_train"
        else _metadata_scalar(fields, "role")
    )
    serialization_key = (
        "serialization" if partition_kind == "router_train" else "gemma_serialization"
    )
    if "messages" not in fields or serialization_key not in fields:
        raise batch.AdapterError("minimal320_source_projection_identity_missing")
    messages = _decode_metadata(
        fields["messages"],
        reason="minimal320_source_messages_invalid",
    )
    serialization = _decode_metadata(
        fields[serialization_key],
        reason="minimal320_source_serialization_invalid",
    )
    if (
        not isinstance(messages, list)
        or len(messages) != 2
        or not isinstance(serialization, dict)
        or re.fullmatch(
            r"[0-9a-f]{64}",
            str(serialization.get("prompt_sha256")),
        )
        is None
    ):
        raise batch.AdapterError("minimal320_source_projection_identity_invalid")
    source_content_sha256 = _hash(messages)
    source_serialization_identity_sha256 = _hash(serialization)
    source_prompt_sha256 = str(serialization["prompt_sha256"])
    role_mapping = {
        "humor": "humor",
        "serious": "serious",
        "angry_style": "angry",
        "tool_call": "tool",
        "review_audit": "review",
        "emotion_router": "router",
    }
    try:
        source_role = role_mapping[str(producer_role)]
    except KeyError:
        raise batch.AdapterError("minimal320_source_metadata_role_invalid") from None
    identity_class: str | None = None
    identity_intent: str | None = None
    identity_context: str | None = None
    identity_stable_fact_sha256: str | None = None
    identity_claimed_attribution: str | None = None
    if "identity_alignment" in fields:
        identity_value = _decode_metadata(
            fields["identity_alignment"],
            reason="minimal320_source_metadata_identity_invalid",
        )
        if identity_value is not None:
            if not isinstance(identity_value, dict):
                raise batch.AdapterError("minimal320_source_metadata_identity_invalid")
            identity = _select_json_fields(
                fields["identity_alignment"],
                frozenset({"intent", "case_context", "stable_fact_sha256"}),
            )
            identity_intent = _metadata_scalar(identity, "intent")
            identity_context = _metadata_scalar(identity, "case_context")
            identity_stable_fact_sha256 = _metadata_scalar(
                identity,
                "stable_fact_sha256",
            )
            if (
                re.fullmatch(
                    r"[0-9a-f]{64}",
                    identity_stable_fact_sha256 or "",
                )
                is None
            ):
                raise batch.AdapterError("minimal320_source_metadata_identity_invalid")
            if "semantic_descriptor" not in fields:
                raise batch.AdapterError(
                    "minimal320_source_metadata_identity_descriptor_missing"
                )
            descriptor = _select_json_fields(
                fields["semantic_descriptor"],
                frozenset({"typed_operands"}),
            )
            if "typed_operands" not in descriptor:
                raise batch.AdapterError(
                    "minimal320_source_metadata_identity_descriptor_missing"
                )
            typed_operands = _select_json_fields(
                descriptor["typed_operands"],
                frozenset({"claimed_attribution"}),
            )
            if "claimed_attribution" not in typed_operands:
                raise batch.AdapterError(
                    "minimal320_source_metadata_identity_descriptor_missing"
                )
            claimed_value = _decode_metadata(
                typed_operands["claimed_attribution"],
                reason=("minimal320_source_metadata_identity_attribution_invalid"),
            )
            if claimed_value not in {None, "Google", "OpenAI"}:
                raise batch.AdapterError(
                    "minimal320_source_metadata_identity_attribution_invalid"
                )
            identity_claimed_attribution = (
                str(claimed_value) if claimed_value is not None else None
            )
            identity_class = (
                "false_google_attribution"
                if claimed_value == "Google"
                else (
                    "false_openai_attribution"
                    if claimed_value == "OpenAI"
                    else "air_attribution"
                )
            )
    tool_family: str | None = None
    if producer_role == "tool_call":
        if "tool_trace" not in fields:
            raise batch.AdapterError("minimal320_source_metadata_tool_missing")
        trace = _select_json_fields(
            fields["tool_trace"],
            frozenset({"required", "direct_no_tool", "tool_call"}),
        )
        direct = (
            _decode_metadata(
                trace.get("direct_no_tool", b"false"),
                reason="minimal320_source_metadata_tool_invalid",
            )
            is True
        )
        if identity_class is not None:
            if not direct:
                raise batch.AdapterError(
                    "minimal320_source_metadata_identity_tool_drift"
                )
            source_role = "identity"
        else:
            required = _decode_metadata(
                trace.get("required", b"false"),
                reason="minimal320_source_metadata_tool_invalid",
            )
            if required is not True or "tool_call" not in trace:
                raise batch.AdapterError("minimal320_source_metadata_tool_invalid")
            call = _select_json_fields(
                trace["tool_call"],
                frozenset({"tool_name"}),
            )
            tool_family = _metadata_scalar(call, "tool_name")
            if batch.SAFE_ID_RE.fullmatch(tool_family or "") is None:
                raise batch.AdapterError("minimal320_source_metadata_tool_invalid")
    review_verdict: str | None = None
    review_fault_type: str | None = None
    if producer_role == "review_audit":
        if "review_dependency" not in fields:
            raise batch.AdapterError("minimal320_source_metadata_review_missing")
        review = _select_json_fields(
            fields["review_dependency"],
            frozenset(
                {
                    "required",
                    "verdict",
                    "fault_type",
                    "tool_record_id",
                    "candidate_projection_sha256",
                    "dependency_set_sha256",
                }
            ),
        )
        if (
            _decode_metadata(
                review.get("required", b"false"),
                reason="minimal320_source_metadata_review_invalid",
            )
            is not True
        ):
            raise batch.AdapterError("minimal320_source_metadata_review_invalid")
        review_verdict = _metadata_scalar(review, "verdict")
        raw_fault = review.get("fault_type")
        review_fault_type = (
            _metadata_scalar(review, "fault_type", nullable=True)
            if raw_fault is not None
            else None
        )
        if (
            review_verdict not in {"pass", "fail"}
            or (review_verdict == "pass" and review_fault_type is not None)
            or (review_verdict == "fail" and review_fault_type is None)
        ):
            raise batch.AdapterError("minimal320_source_metadata_review_invalid")
        if review_verdict == "fail" and review_fault_type not in FAULT_TYPES:
            raise batch.AdapterError(
                "minimal320_source_review_fault_contract_unreachable"
            )
    router_label: str | None = None
    if source_role == "router":
        allowed = _decode_metadata(
            fields.get("allowed_expert_ids", b"null"),
            reason="minimal320_source_metadata_router_invalid",
        )
        mapping = {
            "humor": "humor",
            "serious": "serious",
            "angry_style": "angry",
        }
        if (
            not isinstance(allowed, list)
            or len(allowed) != 1
            or str(allowed[0]) not in mapping
        ):
            raise batch.AdapterError("minimal320_source_metadata_router_invalid")
        router_label = mapping[str(allowed[0])]
    record_id = _metadata_scalar(fields, "record_id")
    bundle = _metadata_scalar(fields, "task_bundle_sha256")
    semantic = _metadata_scalar(fields, "task_semantic_sha256")
    language = _metadata_scalar(fields, "language")
    if any(
        re.fullmatch(r"[0-9a-f]{64}", value or "") is None
        for value in (bundle, semantic)
    ):
        raise batch.AdapterError("minimal320_source_metadata_hash_invalid")
    metadata = {
        "record_id": record_id,
        "task_bundle": bundle,
        "semantic": semantic,
        "source_role": source_role,
        "producer_role": producer_role,
        "language": language,
        "source_content_sha256": source_content_sha256,
        "source_serialization_identity_sha256": (source_serialization_identity_sha256),
        "source_prompt_sha256": source_prompt_sha256,
        "tool_family": tool_family,
        "identity_class": identity_class,
        "identity_provenance_intent": identity_intent,
        "identity_case_context": identity_context,
        "identity_stable_fact_sha256": identity_stable_fact_sha256,
        "identity_claimed_attribution": identity_claimed_attribution,
        "review_verdict": review_verdict,
        "review_fault_type": review_fault_type,
        "router_label": router_label,
        "partition_kind": partition_kind,
        "line_number": selected_line_number,
        "source_line_sha256": hashlib.sha256(raw_line).hexdigest(),
        "producer_partition_sha256": producer_partition_sha256,
    }
    metadata["producer_metadata_sha256"] = _hash(metadata)
    return metadata


def _load_physical_source_metadata() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    producer_binding, _manifest, _receipt, _attestation = (
        batch._producer_binding_from_physical_bytes()
    )
    physical = {
        str(item["relative_path"]): dict(item)
        for item in producer_binding["physical_files"]
    }
    producer_root = (batch.REPO_ROOT / batch.PRODUCER_ARTIFACT_ROOT).absolute()
    selected: list[dict[str, Any]] = []
    source_files: list[dict[str, Any]] = []
    partitions = [
        *(("train", relative) for relative in batch.PRODUCER_TRAIN_SHARDS),
        ("router_train", batch.PRODUCER_ROUTER_PATH),
    ]
    selected_line_numbers = {"train": 0, "router_train": 0}
    for partition_kind, relative in partitions:
        binding = physical.get(relative)
        if binding is None:
            raise batch.AdapterError("minimal320_source_partition_binding_missing")
        path = producer_root / relative
        raw = _physical_read(
            path,
            stop=producer_root,
            reason="minimal320_source_partition_snapshot_drift",
            maximum_bytes=batch.PRODUCER_MAX_FILE_BYTES_EXCLUSIVE - 1,
        )
        if (
            len(raw) != int(binding["bytes"])
            or not hmac.compare_digest(
                hashlib.sha256(raw).hexdigest(),
                str(binding["sha256"]),
            )
            or not raw.endswith(b"\n")
        ):
            raise batch.AdapterError("minimal320_source_partition_identity_drift")
        for raw_line in raw.splitlines():
            if not raw_line:
                raise batch.AdapterError("minimal320_source_partition_line_invalid")
            materialized = _producer_metadata_from_line(
                raw_line,
                partition_kind=partition_kind,
                selected_line_number=(selected_line_numbers[partition_kind] + 1),
                producer_partition_sha256=str(binding["sha256"]),
            )
            if materialized is None:
                continue
            selected_line_numbers[partition_kind] += 1
            materialized["line_number"] = selected_line_numbers[partition_kind]
            selected.append(materialized)
        source_files.append(
            {
                "relative_path": relative,
                "bytes": len(raw),
                "sha256": str(binding["sha256"]),
            }
        )
    if len(selected) != 3520:
        raise batch.AdapterError("minimal320_source_metadata_count_drift")
    return selected, source_files


def _authenticate_bound_producer_source_files(
    source_files: Sequence[Mapping[str, Any]],
) -> None:
    expected_paths = {
        *batch.PRODUCER_TRAIN_SHARDS,
        batch.PRODUCER_ROUTER_PATH,
    }
    observed_paths = {str(item.get("relative_path")) for item in source_files}
    if observed_paths != expected_paths or len(source_files) != len(expected_paths):
        raise batch.AdapterError("minimal320_overlay_producer_source_set_drift")
    producer_root = (batch.REPO_ROOT / batch.PRODUCER_ARTIFACT_ROOT).absolute()
    for item in source_files:
        relative = str(item["relative_path"])
        raw = _physical_read(
            producer_root / relative,
            stop=producer_root,
            reason="minimal320_overlay_producer_source_snapshot_drift",
            maximum_bytes=batch.PRODUCER_MAX_FILE_BYTES_EXCLUSIVE - 1,
        )
        if len(raw) != int(item["bytes"]) or not hmac.compare_digest(
            hashlib.sha256(raw).hexdigest(),
            str(item["sha256"]),
        ):
            raise batch.AdapterError(
                "minimal320_overlay_producer_source_identity_drift"
            )


def _overlay_binding(
    *,
    inventory: batch.SourceInventory,
    config: batch.AdapterConfig,
    overlay_schema_sha256: str,
    selector_implementation_sha256: str,
    source_files: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return {
        "domain": (
            "anchor.gemma3-chat-unbalanced-v2."
            "teacher-alignment-minimal320.overlay-binding.v2"
        ),
        "source_manifest_sha256": inventory.manifest_sha256,
        "source_approval_sha256": inventory.approval_sha256,
        "source_partition_set_sha256": inventory.partition_set_sha256,
        "source_identity_sha256": inventory.source_identity_sha256,
        "adapter_config_sha256": config.physical_sha256,
        "source_contract_hashes_root_sha256": _hash(config.contract_hashes),
        "overlay_schema_sha256": overlay_schema_sha256,
        "selector_implementation_sha256": (selector_implementation_sha256),
        "producer_source_files": [
            dict(item)
            for item in sorted(
                source_files,
                key=lambda item: str(item["relative_path"]),
            )
        ],
        "protected_body_fields_materialized": False,
        "heldout_body_files_read": 0,
        "content_retained": False,
    }


def _overlay_source_availability(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    by_bundle: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_bundle[str(row["task_bundle_sha256"])].append(row)
    review: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {
            "pass_available": False,
            "fail_fault_types": set(),
        }
    )
    identity = Counter()
    router = Counter()
    for bundle_rows in by_bundle.values():
        role_map = {str(row["source_role"]): row for row in bundle_rows}
        if len(role_map) != len(bundle_rows):
            raise batch.AdapterError("minimal320_overlay_availability_role_duplicate")
        identity_classes = {
            str(row["identity_class"])
            for row in bundle_rows
            if row["identity_class"] is not None
        }
        if identity_classes:
            identity_bindings = {
                (
                    row["identity_class"],
                    row["identity_provenance_intent"],
                    row["identity_case_context_sha256"],
                    row["identity_stable_fact_sha256"],
                    row["identity_claimed_attribution"],
                )
                for row in bundle_rows
            }
            if (
                len(identity_classes) != 1
                or len(identity_bindings) != 1
                or None in next(iter(identity_bindings))[:4]
            ):
                raise batch.AdapterError(
                    "minimal320_overlay_availability_identity_mixed"
                )
            identity[
                (
                    next(iter(identity_classes)),
                    str(bundle_rows[0]["language"]),
                )
            ] += 1
            continue
        if set(role_map) == {"router"}:
            router[
                (
                    str(bundle_rows[0]["router_label"]),
                    str(bundle_rows[0]["language"]),
                )
            ] += 1
            continue
        if "tool" not in role_map or "review" not in role_map:
            continue
        tool = role_map["tool"]
        review_row = role_map["review"]
        cell = (str(tool["tool_family"]), str(tool["language"]))
        verdict = review_row["review_verdict"]
        fault = review_row["review_fault_type"]
        if verdict == "pass" and fault is None:
            review[cell]["pass_available"] = True
        elif verdict == "fail" and fault in FAULT_TYPES:
            review[cell]["fail_fault_types"].add(str(fault))
        else:
            raise batch.AdapterError("minimal320_overlay_availability_review_invalid")
    value = {
        "review_cells": [
            {
                "tool_family": family,
                "language": language,
                "pass_available": bool(review[(family, language)]["pass_available"]),
                "fail_fault_types": sorted(
                    review[(family, language)]["fail_fault_types"]
                ),
            }
            for family, language in sorted(review)
        ],
        "identity_parent_bundle_cells": [
            {
                "identity_class": identity_class,
                "language": language,
                "count": int(identity[(identity_class, language)]),
            }
            for identity_class in batch.IDENTITY_CLASSES
            for language in LANGUAGES
        ],
        "router_cells": [
            {
                "router_label": label,
                "language": language,
                "count": int(router[(label, language)]),
            }
            for label in ("humor", "serious", "angry")
            for language in LANGUAGES
        ],
    }
    if (
        len(value["review_cells"]) != 20
        or sum(int(row["count"]) for row in value["identity_parent_bundle_cells"]) != 20
        or value["router_cells"]
        != [
            {
                "router_label": label,
                "language": language,
                "count": count,
            }
            for (label, language), count in ROUTER_CELL_QUOTAS.items()
        ]
    ):
        raise batch.AdapterError(
            "minimal320_overlay_source_availability_contract_drift"
        )
    return value


def _identity_target_cell_rows() -> list[dict[str, Any]]:
    return [
        {
            "identity_class": identity_class,
            "language": language,
            "bundle_count": count,
            "teacher_request_count": count * len(OUTPUT_EXPERT_ROLES),
        }
        for (identity_class, language), count in (IDENTITY_BUNDLE_CELL_QUOTAS.items())
    ]


def _identity_derivative_plan(
    overlay_rows: Sequence[Mapping[str, Any]],
    *,
    jobs: Sequence[batch.AlignmentJob],
    inventory: batch.SourceInventory,
    config: batch.AdapterConfig,
    v5_anchor: Mapping[str, Any],
) -> dict[str, Any]:
    by_bundle: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in overlay_rows:
        if row["identity_class"] is not None:
            by_bundle[str(row["task_bundle_sha256"])].append(row)
    jobs_by_source = {_hash_text(job.source.record_id): job for job in jobs}
    parent_bundles: list[dict[str, Any]] = []
    expected_roles = {"humor", "serious", "angry", "identity", "review"}
    for bundle_sha256, rows in sorted(by_bundle.items()):
        role_map = {str(row["source_role"]): row for row in rows}
        bindings = {
            (
                row["identity_class"],
                row["identity_provenance_intent"],
                row["identity_case_context_sha256"],
                row["identity_stable_fact_sha256"],
                row["identity_claimed_attribution"],
                row["language"],
            )
            for row in rows
        }
        if (
            set(role_map) != expected_roles
            or len(role_map) != len(rows)
            or len(bindings) != 1
        ):
            raise batch.AdapterError("minimal320_identity_parent_bundle_invalid")
        binding = next(iter(bindings))
        parent_inventory = []
        for role in ("humor", "serious", "angry", "identity", "review"):
            row = role_map[role]
            source_sha = str(row["source_record_id_sha256"])
            base_job = jobs_by_source.get(source_sha)
            if base_job is None:
                raise batch.AdapterError("minimal320_identity_parent_job_missing")
            parent_inventory.append(
                {
                    "source_role": role,
                    "source_record_id_sha256": source_sha,
                    "source_semantic_sha256": row["source_semantic_sha256"],
                    "source_line_sha256": row["source_line_sha256"],
                    "content_identity_sha256": row["content_identity_sha256"],
                    "overlay_row_sha256": row["overlay_row_sha256"],
                    "base_prompt_template_sha256": (base_job.prompt_template_sha256),
                    "base_idempotency_key": base_job.idempotency_key,
                }
            )
        parent_bundles.append(
            {
                "parent_task_bundle_sha256": bundle_sha256,
                "source_identity_class": binding[0],
                "language": binding[5],
                "identity_provenance_intent": binding[1],
                "identity_case_context_sha256": binding[2],
                "identity_stable_fact_sha256": binding[3],
                "identity_claimed_attribution": binding[4],
                "parent_record_inventory": parent_inventory,
            }
        )
    if len(parent_bundles) != 20:
        raise batch.AdapterError("minimal320_identity_parent_bundle_count_drift")
    parent_availability = [
        {
            "parent_task_bundle_sha256": row["parent_task_bundle_sha256"],
            "source_identity_class": row["source_identity_class"],
            "language": row["language"],
        }
        for row in parent_bundles
    ]
    assignment_rows: list[dict[str, Any]] = []
    for language in LANGUAGES:
        parents = sorted(
            (row for row in parent_bundles if row["language"] == language),
            key=lambda row: str(row["parent_task_bundle_sha256"]),
        )
        slots = [
            identity_class
            for identity_class in batch.IDENTITY_CLASSES
            for _index in range(IDENTITY_BUNDLE_CELL_QUOTAS[(identity_class, language)])
        ]
        if len(parents) != 10 or len(slots) != 10:
            raise batch.AdapterError(
                "minimal320_identity_derivative_language_capacity_drift"
            )
        for slot_index, (parent, target_class) in enumerate(
            zip(parents, slots, strict=True)
        ):
            assignment_rows.append(
                {
                    "assignment_ordinal": len(assignment_rows),
                    "language_slot_index": slot_index,
                    "parent_task_bundle_sha256": parent["parent_task_bundle_sha256"],
                    "source_identity_class": parent["source_identity_class"],
                    "target_identity_class": target_class,
                    "language": language,
                }
            )
    assignment_rows.sort(key=lambda row: int(row["assignment_ordinal"]))
    if len(
        {row["parent_task_bundle_sha256"] for row in assignment_rows}
    ) != 20 or Counter(
        (str(row["target_identity_class"]), str(row["language"]))
        for row in assignment_rows
    ) != Counter(IDENTITY_BUNDLE_CELL_QUOTAS):
        raise batch.AdapterError("minimal320_identity_parent_assignment_invalid")
    parent_assignment_sha256 = _hash(assignment_rows)
    parent_by_bundle = {
        str(row["parent_task_bundle_sha256"]): row for row in parent_bundles
    }
    derivative_specs: list[dict[str, Any]] = []
    output_mapping = {
        "humor": "humor",
        "serious": "serious",
        "angry": "angry",
        "identity": "tool",
        "review": "review",
    }
    for assignment in assignment_rows:
        parent = parent_by_bundle[str(assignment["parent_task_bundle_sha256"])]
        target_class = str(assignment["target_identity_class"])
        language = str(assignment["language"])
        derivative_bundle_token = _hash(
            {
                "domain": (
                    "anchor.gemma3-chat-unbalanced-v2.minimal320."
                    "identity-derivative-bundle.v2"
                ),
                "source_identity_sha256": inventory.source_identity_sha256,
                "parent_assignment_sha256": parent_assignment_sha256,
                "assignment_ordinal": assignment["assignment_ordinal"],
                "parent_task_bundle_sha256": parent["parent_task_bundle_sha256"],
                "target_identity_class": target_class,
                "language": language,
            }
        )
        derivative_task_bundle = (
            "minimal320-v2-identity-derivative-bundle:" + derivative_bundle_token
        )
        derivative_bundle_sha256 = _hash_text(derivative_task_bundle)
        parent_records = {
            str(row["source_role"]): row for row in parent["parent_record_inventory"]
        }
        for source_role, output_role in output_mapping.items():
            parent_record = parent_records[source_role]
            parent_lineage_sha256 = _hash(
                {
                    "parent_task_bundle_sha256": parent["parent_task_bundle_sha256"],
                    **parent_record,
                }
            )
            semantic_token = _hash(
                {
                    "domain": (
                        "anchor.gemma3-chat-unbalanced-v2.minimal320."
                        "identity-derivative-semantic.v2"
                    ),
                    "derivative_task_bundle_sha256": (derivative_bundle_sha256),
                    "parent_lineage_sha256": parent_lineage_sha256,
                    "target_identity_class": target_class,
                    "language": language,
                    "output_expert_role": output_role,
                }
            )
            semantic = "minimal320-v2-identity-derivative-semantic:" + semantic_token
            semantic_sha256 = _hash_text(semantic)
            teacher_input = _identity_derivative_teacher_input(
                identity_class=target_class,
                language=language,
                output_expert_role=output_role,
                parent_lineage_sha256=parent_lineage_sha256,
            )
            system_prompt = identity_derivative_system_prompt(
                output_role,
                target_class,
            )
            user_prompt = _identity_derivative_user_prompt(
                teacher_input=teacher_input,
                identity_class=target_class,
                language=language,
                semantic=semantic,
            )
            system_prompt_sha256 = _hash_text(system_prompt)
            user_prompt_sha256 = _hash_text(user_prompt)
            teacher_input_sha256 = _hash_text(teacher_input)
            validation_role = "identity" if output_role == "tool" else output_role
            (
                derived_source_messages,
                derived_source_serialization,
                derived_source_content_sha256,
                derived_source_serialization_identity_sha256,
            ) = _identity_derivative_source_projection(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                teacher_input_sha256=teacher_input_sha256,
                output_expert_role=output_role,
                validation_contract_role=validation_role,
                v5_anchor_physical_sha256=v5_anchor["physical_sha256"],
                v5_prompt_version=v5_anchor["prompt_version"],
            )
            derived_prompt_contract_sha256 = _hash(
                {
                    "domain": (
                        "anchor.gemma3-chat-unbalanced-v2.minimal320."
                        "identity-derivative-prompt-contract.v2"
                    ),
                    "system_prompt_sha256": system_prompt_sha256,
                    "user_prompt_sha256": user_prompt_sha256,
                    "teacher_input_sha256": teacher_input_sha256,
                    "target_identity_class": target_class,
                    "output_expert_role": output_role,
                    "validation_contract_role": validation_role,
                    "v5_anchor_physical_sha256": v5_anchor["physical_sha256"],
                }
            )
            derived_output_contract_sha256 = _hash(
                {
                    "domain": (
                        "anchor.gemma3-chat-unbalanced-v2.minimal320."
                        "identity-derivative-output-contract.v2"
                    ),
                    "output_expert_role": output_role,
                    "validation_contract_role": validation_role,
                    "target_identity_class": target_class,
                    "output_schema_id": batch.ROLE_SCHEMA_IDS[validation_role],
                    "physical_output_schema_sha256": (
                        config.contract_hashes["output_schema_path"]
                    ),
                }
            )
            record_token = _hash(
                {
                    "domain": (
                        "anchor.gemma3-chat-unbalanced-v2.minimal320."
                        "identity-derivative-record.v2"
                    ),
                    "semantic_sha256": semantic_sha256,
                    "system_prompt_sha256": system_prompt_sha256,
                    "user_prompt_sha256": user_prompt_sha256,
                }
            )
            derived_record_id = "minimal320-v2-identity-derivative:" + record_token
            derived_record_id_sha256 = _hash_text(derived_record_id)
            content_identity_sha256 = _hash(
                {
                    "teacher_input_sha256": teacher_input_sha256,
                    "target_template_sha256": (
                        V5_IDENTITY_TEMPLATE_SHA256[target_class]
                    ),
                    "output_expert_role": output_role,
                }
            )
            source_line_sha256 = _hash(
                {
                    "derived_record_id_sha256": (derived_record_id_sha256),
                    "content_identity_sha256": content_identity_sha256,
                    "parent_lineage_sha256": parent_lineage_sha256,
                }
            )
            serialization_identity_sha256 = _hash(
                {
                    "v5_anchor_blob_sha256": v5_anchor["physical_sha256"],
                    "system_prompt_sha256": system_prompt_sha256,
                    "user_prompt_sha256": user_prompt_sha256,
                    "prompt_version": v5_anchor["prompt_version"],
                }
            )
            idempotency_key = _hash(
                {
                    "domain": (
                        "anchor.gemma3-chat-unbalanced-v2.minimal320."
                        "identity-derivative-job-idempotency.v2"
                    ),
                    "provider": batch.PROVIDER_PRESET,
                    "protocol": batch.PROTOCOL,
                    "base_url": batch.BASE_URL,
                    "model": batch.MODEL,
                    "source_manifest_sha256": inventory.manifest_sha256,
                    "source_identity_sha256": (inventory.source_identity_sha256),
                    "derived_record_id_sha256": (derived_record_id_sha256),
                    "source_line_sha256": source_line_sha256,
                    "content_identity_sha256": (content_identity_sha256),
                    "serialization_identity_sha256": (serialization_identity_sha256),
                    "prompt_version": v5_anchor["prompt_version"],
                    "system_prompt_sha256": system_prompt_sha256,
                    "user_prompt_sha256": user_prompt_sha256,
                    "target_template_sha256": (
                        V5_IDENTITY_TEMPLATE_SHA256[target_class]
                    ),
                    "campaign_sha256": config.campaign_sha256,
                }
            )
            if (
                idempotency_key == parent_record["base_idempotency_key"]
                or derived_prompt_contract_sha256
                == parent_record["base_prompt_template_sha256"]
            ):
                raise batch.AdapterError(
                    "minimal320_identity_derivative_base_identity_reuse"
                )
            spec_payload = {
                "schema_version": (IDENTITY_DERIVATIVE_SPEC_SCHEMA_VERSION),
                "assignment_ordinal": assignment["assignment_ordinal"],
                "parent_assignment_sha256": (parent_assignment_sha256),
                "parent_task_bundle_sha256": parent["parent_task_bundle_sha256"],
                "parent_source_record_id_sha256": parent_record[
                    "source_record_id_sha256"
                ],
                "parent_source_semantic_sha256": parent_record[
                    "source_semantic_sha256"
                ],
                "parent_source_line_sha256": parent_record["source_line_sha256"],
                "parent_content_identity_sha256": parent_record[
                    "content_identity_sha256"
                ],
                "parent_lineage_sha256": parent_lineage_sha256,
                "parent_overlay_row_sha256": parent_record["overlay_row_sha256"],
                "parent_base_prompt_template_sha256": parent_record[
                    "base_prompt_template_sha256"
                ],
                "parent_base_idempotency_key": parent_record["base_idempotency_key"],
                "source_identity_class": parent["source_identity_class"],
                "target_identity_class": target_class,
                "language": language,
                "parent_source_role": source_role,
                "output_expert_role": output_role,
                "expert_asset_role": output_role,
                "validation_contract_role": (
                    "identity" if output_role == "tool" else output_role
                ),
                "response_contract": (
                    "raw_exact_identity_template_no_tool"
                    if output_role == "tool"
                    else "role_native_identity_invariant"
                ),
                "tool_invocation_allowed": False,
                "review_audit_case": (
                    "missing_identity_answer" if output_role == "review" else None
                ),
                "review_expected_verdict": (
                    "fail" if output_role == "review" else None
                ),
                "review_expected_faults": (
                    ["missing_identity_answer"] if output_role == "review" else []
                ),
                "review_expected_correction_sha256": (
                    V5_IDENTITY_TEMPLATE_SHA256[target_class]
                    if output_role == "review"
                    else None
                ),
                "v5_anchor_git_commit": v5_anchor["git_commit"],
                "v5_anchor_git_blob_sha1": v5_anchor["git_blob_sha1"],
                "v5_anchor_physical_sha256": v5_anchor["physical_sha256"],
                "v5_prompt_version": v5_anchor["prompt_version"],
                "target_template": V5_IDENTITY_TEMPLATES[target_class],
                "target_template_sha256": (V5_IDENTITY_TEMPLATE_SHA256[target_class]),
                "system_prompt_sha256": system_prompt_sha256,
                "user_prompt_sha256": user_prompt_sha256,
                "teacher_input_sha256": teacher_input_sha256,
                "derived_source_messages": derived_source_messages,
                "derived_source_serialization": derived_source_serialization,
                "derived_source_content_sha256": derived_source_content_sha256,
                "derived_source_serialization_identity_sha256": (
                    derived_source_serialization_identity_sha256
                ),
                "derived_prompt_contract_sha256": (derived_prompt_contract_sha256),
                "derived_task_bundle": derivative_task_bundle,
                "derived_task_bundle_sha256": (derivative_bundle_sha256),
                "derived_semantic": semantic,
                "derived_semantic_sha256": semantic_sha256,
                "derived_record_id": derived_record_id,
                "derived_record_id_sha256": (derived_record_id_sha256),
                "derived_content_identity_sha256": (content_identity_sha256),
                "derived_source_line_sha256": source_line_sha256,
                "derived_serialization_identity_sha256": (
                    serialization_identity_sha256
                ),
                "derived_teacher_job_idempotency_key": (idempotency_key),
                "derived_line_number": (
                    int(assignment["assignment_ordinal"]) * len(OUTPUT_EXPERT_ROLES)
                    + OUTPUT_EXPERT_ROLES.index(output_role)
                    + 1
                ),
                "derived_prompt_template_id": batch.ROLE_TEMPLATE_IDS[
                    "identity" if output_role == "tool" else output_role
                ],
                "derived_output_schema_id": batch.ROLE_SCHEMA_IDS[
                    "identity" if output_role == "tool" else output_role
                ],
                "derived_output_schema_sha256": config.contract_hashes[
                    "output_schema_path"
                ],
                "derived_output_contract_sha256": (derived_output_contract_sha256),
                "protected_parent_body_read": False,
                "eval_parent_rows_read": 0,
                "content_retained": False,
            }
            derivative_specs.append(
                {
                    **spec_payload,
                    "identity_derivative_spec_sha256": _hash(spec_payload),
                }
            )
    if (
        len(derivative_specs) != 100
        or len({row["parent_source_record_id_sha256"] for row in derivative_specs})
        != 100
        or len({row["derived_teacher_job_idempotency_key"] for row in derivative_specs})
        != 100
        or len({row["derived_record_id_sha256"] for row in derivative_specs}) != 100
    ):
        raise batch.AdapterError("minimal320_identity_derivative_spec_identity_drift")
    derivative_specs.sort(key=lambda row: str(row["derived_record_id_sha256"]))
    plan_payload = {
        "schema_version": IDENTITY_DERIVATIVE_PLAN_SCHEMA_VERSION,
        "algorithm_id": IDENTITY_DERIVATIVE_ALGORITHM_ID,
        "v5_anchor": dict(v5_anchor),
        "physical_parent_availability": parent_availability,
        "physical_parent_availability_root_sha256": _hash(parent_availability),
        "target_cells": _identity_target_cell_rows(),
        "parent_assignment": assignment_rows,
        "parent_assignment_sha256": parent_assignment_sha256,
        "derivative_specs": derivative_specs,
        "derivative_specs_root_sha256": _hash(derivative_specs),
        "parent_bundle_count": 20,
        "derived_bundle_count": 20,
        "teacher_request_count": 100,
        "optimizer_record_count": 100,
        "parent_reuse": False,
        "eval_parent_rows_read": 0,
        "protected_parent_body_read": False,
        "content_retained": False,
    }
    return {
        **plan_payload,
        "identity_derivative_plan_sha256": _hash(plan_payload),
    }


def build_authenticated_metadata_overlay(
    inventory: batch.SourceInventory,
    config: batch.AdapterConfig,
    *,
    output_root: Path,
    schema_path: Path = DEFAULT_OVERLAY_SCHEMA_PATH,
    source_metadata_loader: Any = _load_physical_source_metadata,
) -> AuthenticatedMetadataOverlay:
    """Persist a no-replace overlay from authenticated producer metadata."""

    schema, schema_sha256 = _load_schema_physical(schema_path)
    implementation_raw = _physical_read(
        IMPLEMENTATION_PATH,
        stop=batch.REPO_ROOT.absolute(),
        reason="minimal320_selector_implementation_snapshot_drift",
    )
    implementation_sha256 = hashlib.sha256(implementation_raw).hexdigest()
    metadata_rows, source_files = source_metadata_loader()
    jobs = batch.derive_jobs(inventory, config)
    jobs_by_record = {job.source.record_id: job for job in jobs}
    if len(jobs_by_record) != 3520 or len(metadata_rows) != 3520:
        raise batch.AdapterError("minimal320_overlay_source_count_drift")
    binding = _overlay_binding(
        inventory=inventory,
        config=config,
        overlay_schema_sha256=schema_sha256,
        selector_implementation_sha256=implementation_sha256,
        source_files=source_files,
    )
    binding_sha256 = _hash(binding)
    overlay_rows: list[dict[str, Any]] = []
    seen_records: set[str] = set()
    for metadata in metadata_rows:
        record_id = str(metadata["record_id"])
        job = jobs_by_record.get(record_id)
        if job is None or record_id in seen_records:
            raise batch.AdapterError("minimal320_overlay_source_record_misaligned")
        seen_records.add(record_id)
        source = job.source
        expected = {
            "task_bundle": source.task_bundle,
            "semantic": source.semantic,
            "source_role": source.role,
            "language": source.language,
            "partition_kind": source.partition_kind,
            "line_number": source.line_number,
            "source_line_sha256": source.source_line_sha256,
        }
        if any(metadata.get(key) != value for key, value in expected.items()):
            raise batch.AdapterError("minimal320_overlay_source_record_misaligned")
        row_payload = {
            "schema_version": OVERLAY_ROW_SCHEMA_VERSION,
            "overlay_binding_sha256": binding_sha256,
            "source_record_id_sha256": _hash_text(record_id),
            "parent_source_record_id_sha256": _hash_text(record_id),
            "source_line_sha256": source.source_line_sha256,
            "content_identity_sha256": source.content_identity_sha256,
            "source_content_sha256": metadata["source_content_sha256"],
            "source_serialization_identity_sha256": metadata[
                "source_serialization_identity_sha256"
            ],
            "source_prompt_sha256": metadata["source_prompt_sha256"],
            "task_bundle_sha256": _hash_text(source.task_bundle),
            "source_semantic_sha256": _hash_text(source.semantic),
            "source_role": source.role,
            "language": _language(source.language),
            "tool_family": metadata.get("tool_family"),
            "identity_class": metadata.get("identity_class"),
            "identity_provenance_intent": metadata.get("identity_provenance_intent"),
            "identity_case_context_sha256": (
                _hash_text(str(metadata["identity_case_context"]))
                if metadata.get("identity_case_context") is not None
                else None
            ),
            "identity_stable_fact_sha256": metadata.get("identity_stable_fact_sha256"),
            "identity_claimed_attribution": metadata.get(
                "identity_claimed_attribution"
            ),
            "review_verdict": metadata.get("review_verdict"),
            "review_fault_type": metadata.get("review_fault_type"),
            "router_label": metadata.get("router_label"),
            "producer_role": metadata["producer_role"],
            "producer_partition_sha256": metadata["producer_partition_sha256"],
            "producer_metadata_sha256": metadata["producer_metadata_sha256"],
            "content_retained": False,
        }
        row = {
            **row_payload,
            "overlay_row_sha256": _hash(row_payload),
        }
        try:
            Draft202012Validator(schema).validate(row)
        except ValidationError as error:
            raise batch.AdapterError(
                "minimal320_overlay_row_schema_mismatch"
            ) from error
        overlay_rows.append(row)
    if len(seen_records) != len(jobs_by_record):
        raise batch.AdapterError("minimal320_overlay_source_omission")
    overlay_rows.sort(key=lambda row: str(row["source_record_id_sha256"]))
    source_availability = _overlay_source_availability(overlay_rows)
    v5_anchor = authenticate_v5_identity_anchor()
    identity_derivative_plan = _identity_derivative_plan(
        overlay_rows,
        jobs=jobs,
        inventory=inventory,
        config=config,
        v5_anchor=v5_anchor,
    )
    rows_raw = b"".join(_canonical_bytes(row) + b"\n" for row in overlay_rows)
    rows_root = _hash(overlay_rows)
    directory = output_root.absolute() / "minimal320_selector_preimages_v2"
    try:
        directory.mkdir(parents=False, exist_ok=False)
    except FileExistsError as error:
        raise batch.AdapterError("minimal320_preimage_no_replace") from error
    rows_path = directory / "authenticated_metadata_overlay.jsonl"
    _exclusive_bytes(rows_path, rows_raw)
    rows_sha256 = hashlib.sha256(rows_raw).hexdigest()
    _exclusive_bytes(
        rows_path.with_name(f"{rows_path.name}.sha256"),
        f"{rows_sha256}  {rows_path.name}\n".encode("ascii"),
    )
    manifest = {
        "schema_version": OVERLAY_MANIFEST_SCHEMA_VERSION,
        "overlay_binding": binding,
        "overlay_binding_sha256": binding_sha256,
        "row_count": len(overlay_rows),
        "overlay_rows_root_sha256": rows_root,
        "overlay_rows_sha256": rows_sha256,
        "overlay_rows_file": rows_path.name,
        "source_availability": source_availability,
        "source_availability_root_sha256": _hash(source_availability),
        "identity_derivative_plan": identity_derivative_plan,
        "identity_derivative_plan_sha256": (
            identity_derivative_plan["identity_derivative_plan_sha256"]
        ),
        "content_retained": False,
        "protected_body_fields_materialized": False,
        "heldout_body_files_read": 0,
    }
    try:
        Draft202012Validator(schema).validate(manifest)
    except ValidationError as error:
        raise batch.AdapterError(
            "minimal320_overlay_manifest_schema_mismatch"
        ) from error
    manifest_path = directory / "authenticated_metadata_overlay_manifest.json"
    _exclusive_json(manifest_path, manifest)
    return AuthenticatedMetadataOverlay(
        directory=directory,
        rows_path=rows_path,
        manifest_path=manifest_path,
        rows=tuple(overlay_rows),
        manifest=manifest,
    )


def authenticate_metadata_overlay(
    overlay: AuthenticatedMetadataOverlay,
    *,
    inventory: batch.SourceInventory,
    config: batch.AdapterConfig,
    schema_path: Path = DEFAULT_OVERLAY_SCHEMA_PATH,
    source_file_authenticator: Any = (_authenticate_bound_producer_source_files),
    source_metadata_loader: Any = _load_physical_source_metadata,
) -> tuple[dict[str, Any], ...]:
    directory = overlay.directory.absolute()
    expected_rows_path = directory / "authenticated_metadata_overlay.jsonl"
    expected_manifest_path = directory / "authenticated_metadata_overlay_manifest.json"
    if (
        overlay.rows_path.absolute() != expected_rows_path
        or overlay.manifest_path.absolute() != expected_manifest_path
        or not directory.name == "minimal320_selector_preimages_v2"
    ):
        raise batch.AdapterError("minimal320_overlay_path_identity_drift")
    schema, schema_sha256 = _load_schema_physical(schema_path)
    manifest_raw = _physical_read(
        expected_manifest_path,
        stop=directory,
        reason="minimal320_overlay_manifest_snapshot_drift",
    )
    _authenticate_sha256_sidecar(
        expected_manifest_path,
        manifest_raw,
        stop=directory,
        reason="minimal320_overlay_manifest_sidecar_drift",
    )
    manifest_value = closed._strict_json(
        manifest_raw,
        reason="minimal320_overlay_manifest_invalid",
    )
    if not isinstance(manifest_value, dict):
        raise batch.AdapterError("minimal320_overlay_manifest_invalid")
    try:
        Draft202012Validator(schema).validate(manifest_value)
    except ValidationError as error:
        raise batch.AdapterError(
            "minimal320_overlay_manifest_schema_mismatch"
        ) from error
    rows_raw = _physical_read(
        expected_rows_path,
        stop=directory,
        reason="minimal320_overlay_rows_snapshot_drift",
    )
    _authenticate_sha256_sidecar(
        expected_rows_path,
        rows_raw,
        stop=directory,
        reason="minimal320_overlay_rows_sidecar_drift",
    )
    rows: list[dict[str, Any]] = []
    for raw_line in rows_raw.splitlines():
        value = closed._strict_json(
            raw_line,
            reason="minimal320_overlay_row_invalid",
        )
        if not isinstance(value, dict):
            raise batch.AdapterError("minimal320_overlay_row_invalid")
        try:
            Draft202012Validator(schema).validate(value)
        except ValidationError as error:
            raise batch.AdapterError(
                "minimal320_overlay_row_schema_mismatch"
            ) from error
        payload = dict(value)
        observed_row_sha = payload.pop("overlay_row_sha256")
        if not hmac.compare_digest(
            str(observed_row_sha),
            _hash(payload),
        ):
            raise batch.AdapterError("minimal320_overlay_row_identity_drift")
        rows.append(value)
    if rows != sorted(
        rows,
        key=lambda row: str(row["source_record_id_sha256"]),
    ):
        raise batch.AdapterError("minimal320_overlay_row_order_drift")
    if (
        len(rows) != 3520
        or len({str(row["source_record_id_sha256"]) for row in rows}) != len(rows)
        or hashlib.sha256(rows_raw).hexdigest() != manifest_value["overlay_rows_sha256"]
        or _hash(rows) != manifest_value["overlay_rows_root_sha256"]
        or manifest_value["row_count"] != len(rows)
    ):
        raise batch.AdapterError("minimal320_overlay_inventory_drift")
    if dict(overlay.manifest) != manifest_value or tuple(
        dict(row) for row in overlay.rows
    ) != tuple(rows):
        raise batch.AdapterError("minimal320_overlay_object_identity_drift")
    binding = manifest_value["overlay_binding"]
    if (
        _hash(binding) != manifest_value["overlay_binding_sha256"]
        or binding["overlay_schema_sha256"] != schema_sha256
        or binding["source_manifest_sha256"] != inventory.manifest_sha256
        or binding["source_approval_sha256"] != inventory.approval_sha256
        or binding["source_partition_set_sha256"] != inventory.partition_set_sha256
        or binding["source_identity_sha256"] != inventory.source_identity_sha256
        or binding["adapter_config_sha256"] != config.physical_sha256
        or binding["source_contract_hashes_root_sha256"]
        != _hash(config.contract_hashes)
        or binding["protected_body_fields_materialized"] is not False
        or binding["heldout_body_files_read"] != 0
    ):
        raise batch.AdapterError("minimal320_overlay_binding_drift")
    implementation_raw = _physical_read(
        IMPLEMENTATION_PATH,
        stop=batch.REPO_ROOT.absolute(),
        reason="minimal320_selector_implementation_snapshot_drift",
    )
    if (
        binding["selector_implementation_sha256"]
        != hashlib.sha256(implementation_raw).hexdigest()
    ):
        raise batch.AdapterError("minimal320_overlay_implementation_identity_drift")
    binding_sha = str(manifest_value["overlay_binding_sha256"])
    if any(row["overlay_binding_sha256"] != binding_sha for row in rows):
        raise batch.AdapterError("minimal320_overlay_row_binding_drift")
    source_availability = _overlay_source_availability(rows)
    if manifest_value["source_availability"] != source_availability or manifest_value[
        "source_availability_root_sha256"
    ] != _hash(source_availability):
        raise batch.AdapterError("minimal320_overlay_source_availability_drift")
    source_file_authenticator(binding["producer_source_files"])
    jobs = batch.derive_jobs(inventory, config)
    jobs_by_source = {_hash_text(job.source.record_id): job for job in jobs}
    if len(jobs_by_source) != len(rows):
        raise batch.AdapterError("minimal320_overlay_job_count_drift")
    for row in rows:
        job = jobs_by_source.get(str(row["source_record_id_sha256"]))
        if job is None:
            raise batch.AdapterError("minimal320_overlay_job_missing")
        source = job.source
        expected = {
            "parent_source_record_id_sha256": _hash_text(source.record_id),
            "source_line_sha256": source.source_line_sha256,
            "content_identity_sha256": source.content_identity_sha256,
            "source_content_sha256": row["source_content_sha256"],
            "source_serialization_identity_sha256": row[
                "source_serialization_identity_sha256"
            ],
            "source_prompt_sha256": source.gemma_serialization_identity_sha256,
            "task_bundle_sha256": _hash_text(source.task_bundle),
            "source_semantic_sha256": _hash_text(source.semantic),
            "source_role": source.role,
            "language": _language(source.language),
        }
        if any(row.get(key) != value for key, value in expected.items()):
            raise batch.AdapterError(
                "minimal320_overlay_authenticated_source_misaligned"
            )
    physical_metadata, physical_source_files = source_metadata_loader()
    physical_source_files_value = sorted(
        (dict(row) for row in physical_source_files),
        key=lambda row: str(row["relative_path"]),
    )
    bound_source_files_value = sorted(
        (dict(row) for row in binding["producer_source_files"]),
        key=lambda row: str(row["relative_path"]),
    )
    if physical_source_files_value != bound_source_files_value:
        raise batch.AdapterError("minimal320_overlay_physical_source_binding_drift")
    metadata_by_source = {
        _hash_text(str(row["record_id"])): row for row in physical_metadata
    }
    if len(metadata_by_source) != len(rows):
        raise batch.AdapterError("minimal320_overlay_physical_metadata_count_drift")
    for row in rows:
        metadata = metadata_by_source.get(str(row["source_record_id_sha256"]))
        if metadata is None:
            raise batch.AdapterError("minimal320_overlay_physical_metadata_missing")
        expected_metadata = {
            "tool_family": metadata["tool_family"],
            "source_content_sha256": metadata["source_content_sha256"],
            "source_serialization_identity_sha256": metadata[
                "source_serialization_identity_sha256"
            ],
            "source_prompt_sha256": metadata["source_prompt_sha256"],
            "identity_class": metadata["identity_class"],
            "identity_provenance_intent": metadata["identity_provenance_intent"],
            "identity_case_context_sha256": (
                _hash_text(str(metadata["identity_case_context"]))
                if metadata["identity_case_context"] is not None
                else None
            ),
            "identity_stable_fact_sha256": metadata["identity_stable_fact_sha256"],
            "identity_claimed_attribution": metadata["identity_claimed_attribution"],
            "review_verdict": metadata["review_verdict"],
            "review_fault_type": metadata["review_fault_type"],
            "router_label": metadata["router_label"],
            "producer_role": metadata["producer_role"],
            "producer_partition_sha256": metadata["producer_partition_sha256"],
            "producer_metadata_sha256": metadata["producer_metadata_sha256"],
        }
        if any(row.get(key) != value for key, value in expected_metadata.items()):
            raise batch.AdapterError("minimal320_overlay_physical_label_drift")
    v5_anchor = authenticate_v5_identity_anchor()
    expected_derivative_plan = _identity_derivative_plan(
        rows,
        jobs=jobs,
        inventory=inventory,
        config=config,
        v5_anchor=v5_anchor,
    )
    if (
        manifest_value["identity_derivative_plan"] != expected_derivative_plan
        or manifest_value["identity_derivative_plan_sha256"]
        != expected_derivative_plan["identity_derivative_plan_sha256"]
    ):
        raise batch.AdapterError("minimal320_identity_derivative_plan_drift")
    return tuple(rows)


def _candidate_identity_payload(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "domain": CANDIDATE_SCHEMA_VERSION,
        "idempotency_key": value["idempotency_key"],
        "teacher_request_idempotency_key": value["teacher_request_idempotency_key"],
        "source_record_id_sha256": value["source_record_id_sha256"],
        "source_content_sha256": value["source_content_sha256"],
        "source_serialization_identity_sha256": value[
            "source_serialization_identity_sha256"
        ],
        "teacher_request_record_id_sha256": value["teacher_request_record_id_sha256"],
        "parent_source_record_id_sha256": value["parent_source_record_id_sha256"],
        "task_bundle_sha256": value["task_bundle_sha256"],
        "parent_task_bundle_sha256": value["parent_task_bundle_sha256"],
        "source_semantic_sha256": value["source_semantic_sha256"],
        "overlay_row_sha256": value["overlay_row_sha256"],
        "identity_derivative_spec_sha256": value["identity_derivative_spec_sha256"],
        "selection_kind": value["selection_kind"],
        "output_expert_role": value["output_expert_role"],
        "training_asset": value["training_asset"],
        "validation_contract_role": value["validation_contract_role"],
        "language": value["language"],
        "tool_family": value["tool_family"],
        "identity_class": value["identity_class"],
        "identity_parent_class": value["identity_parent_class"],
        "identity_provenance_intent": value["identity_provenance_intent"],
        "review_verdict": value["review_verdict"],
        "review_fault_type": value["review_fault_type"],
        "router_label": value["router_label"],
    }


def candidate_identity_sha256(value: Mapping[str, Any]) -> str:
    return _hash(_candidate_identity_payload(value))


TRAINING_ASSET_BY_OUTPUT_ROLE = {
    "humor": "humor",
    "serious": "serious",
    "angry": "angry_style",
    "tool": "tool_call",
    "review": "review_audit",
    "router": "planner_router",
}


def _training_asset(output_expert_role: str) -> str:
    try:
        return TRAINING_ASSET_BY_OUTPUT_ROLE[output_expert_role]
    except KeyError as error:
        raise batch.AdapterError("minimal320_training_asset_invalid") from error


def _source_candidate(
    job: batch.AlignmentJob,
    overlay_row: Mapping[str, Any],
    *,
    selection_kind: str,
    output_expert_role: str,
    validation_contract_role: str,
    tool_family: str | None,
    review_verdict: str | None = None,
    review_fault_type: str | None = None,
    router_label: str | None = None,
) -> SelectorCandidate:
    source_record_id_sha256 = _hash_text(job.source.record_id)
    expected = {
        "source_record_id_sha256": source_record_id_sha256,
        "parent_source_record_id_sha256": source_record_id_sha256,
        "source_line_sha256": job.source.source_line_sha256,
        "content_identity_sha256": job.source.content_identity_sha256,
        "source_prompt_sha256": job.source.gemma_serialization_identity_sha256,
        "task_bundle_sha256": _hash_text(job.source.task_bundle),
        "source_semantic_sha256": _hash_text(job.source.semantic),
        "source_role": job.source.role,
        "language": _language(job.source.language),
    }
    if any(overlay_row.get(key) != value for key, value in expected.items()):
        raise batch.AdapterError("minimal320_candidate_overlay_source_misaligned")
    stable = {
        "idempotency_key": job.idempotency_key,
        "teacher_request_idempotency_key": job.idempotency_key,
        "source_record_id_sha256": source_record_id_sha256,
        "source_content_sha256": overlay_row["source_content_sha256"],
        "source_serialization_identity_sha256": overlay_row[
            "source_serialization_identity_sha256"
        ],
        "teacher_request_record_id_sha256": source_record_id_sha256,
        "parent_source_record_id_sha256": source_record_id_sha256,
        "task_bundle_sha256": expected["task_bundle_sha256"],
        "parent_task_bundle_sha256": expected["task_bundle_sha256"],
        "source_semantic_sha256": expected["source_semantic_sha256"],
        "overlay_row_sha256": overlay_row["overlay_row_sha256"],
        "identity_derivative_spec_sha256": None,
        "selection_kind": selection_kind,
        "output_expert_role": output_expert_role,
        "training_asset": _training_asset(output_expert_role),
        "validation_contract_role": validation_contract_role,
        "language": expected["language"],
        "tool_family": tool_family,
        "identity_class": overlay_row.get("identity_class"),
        "identity_parent_class": None,
        "identity_provenance_intent": overlay_row.get("identity_provenance_intent"),
        "review_verdict": review_verdict,
        "review_fault_type": review_fault_type,
        "router_label": router_label,
    }
    return SelectorCandidate(
        candidate_id_sha256=candidate_identity_sha256(stable),
        idempotency_key=job.idempotency_key,
        teacher_request_idempotency_key=job.idempotency_key,
        source_record_id_sha256=source_record_id_sha256,
        source_content_sha256=str(stable["source_content_sha256"]),
        source_serialization_identity_sha256=str(
            stable["source_serialization_identity_sha256"]
        ),
        teacher_request_record_id_sha256=source_record_id_sha256,
        parent_source_record_id_sha256=source_record_id_sha256,
        task_bundle_sha256=stable["task_bundle_sha256"],
        parent_task_bundle_sha256=stable["task_bundle_sha256"],
        source_semantic_sha256=stable["source_semantic_sha256"],
        overlay_row_sha256=str(overlay_row["overlay_row_sha256"]),
        identity_derivative_spec_sha256=None,
        selection_kind=selection_kind,
        output_expert_role=output_expert_role,
        training_asset=str(stable["training_asset"]),
        validation_contract_role=validation_contract_role,
        language=stable["language"],
        tool_family=tool_family,
        identity_class=(
            str(stable["identity_class"])
            if stable["identity_class"] is not None
            else None
        ),
        identity_parent_class=None,
        identity_provenance_intent=(
            str(stable["identity_provenance_intent"])
            if stable["identity_provenance_intent"] is not None
            else None
        ),
        review_verdict=review_verdict,
        review_fault_type=review_fault_type,
        router_label=router_label,
    )


def _identity_derivative_candidate(
    spec: Mapping[str, Any],
    overlay_row: Mapping[str, Any],
) -> SelectorCandidate:
    spec_payload = dict(spec)
    observed_spec_sha256 = str(spec_payload.pop("identity_derivative_spec_sha256"))
    if (
        not hmac.compare_digest(
            observed_spec_sha256,
            _hash(spec_payload),
        )
        or spec["parent_source_record_id_sha256"]
        != overlay_row["source_record_id_sha256"]
        or spec["parent_task_bundle_sha256"] != overlay_row["task_bundle_sha256"]
        or spec["parent_source_semantic_sha256"]
        != overlay_row["source_semantic_sha256"]
        or spec["parent_source_line_sha256"] != overlay_row["source_line_sha256"]
        or spec["parent_content_identity_sha256"]
        != overlay_row["content_identity_sha256"]
        or spec["derived_task_bundle_sha256"]
        != _hash_text(str(spec["derived_task_bundle"]))
        or spec["derived_semantic_sha256"] != _hash_text(str(spec["derived_semantic"]))
        or spec["derived_record_id_sha256"]
        != _hash_text(str(spec["derived_record_id"]))
        or spec["parent_overlay_row_sha256"] != overlay_row["overlay_row_sha256"]
        or spec["source_identity_class"] != overlay_row["identity_class"]
        or spec["derived_teacher_job_idempotency_key"]
        == spec["parent_base_idempotency_key"]
        or spec["derived_prompt_contract_sha256"]
        == spec["parent_base_prompt_template_sha256"]
        or spec["expert_asset_role"] != spec["output_expert_role"]
        or spec["tool_invocation_allowed"] is not False
        or (
            spec["output_expert_role"] == "tool"
            and (
                spec["validation_contract_role"] != "identity"
                or spec["response_contract"] != "raw_exact_identity_template_no_tool"
                or spec["system_prompt_sha256"]
                != V5_ROLE_SYSTEM_PROMPT_SHA256["identity"]
            )
        )
        or spec["target_template"]
        != V5_IDENTITY_TEMPLATES[str(spec["target_identity_class"])]
        or spec["target_template_sha256"]
        != V5_IDENTITY_TEMPLATE_SHA256[str(spec["target_identity_class"])]
    ):
        raise batch.AdapterError("minimal320_identity_derivative_spec_drift")
    stable = {
        "idempotency_key": spec["derived_teacher_job_idempotency_key"],
        "teacher_request_idempotency_key": spec["derived_teacher_job_idempotency_key"],
        "source_record_id_sha256": spec["derived_record_id_sha256"],
        "source_content_sha256": spec["derived_source_content_sha256"],
        "source_serialization_identity_sha256": (
            spec["derived_source_serialization_identity_sha256"]
        ),
        "teacher_request_record_id_sha256": spec["derived_record_id_sha256"],
        "parent_source_record_id_sha256": spec["parent_source_record_id_sha256"],
        "task_bundle_sha256": spec["derived_task_bundle_sha256"],
        "parent_task_bundle_sha256": spec["parent_task_bundle_sha256"],
        "source_semantic_sha256": spec["derived_semantic_sha256"],
        "overlay_row_sha256": spec["parent_overlay_row_sha256"],
        "identity_derivative_spec_sha256": observed_spec_sha256,
        "selection_kind": "identity",
        "output_expert_role": spec["output_expert_role"],
        "training_asset": _training_asset(str(spec["output_expert_role"])),
        "validation_contract_role": spec["validation_contract_role"],
        "language": spec["language"],
        "tool_family": None,
        "identity_class": spec["target_identity_class"],
        "identity_parent_class": spec["source_identity_class"],
        "identity_provenance_intent": overlay_row["identity_provenance_intent"],
        "review_verdict": None,
        "review_fault_type": None,
        "router_label": None,
    }
    return SelectorCandidate(
        candidate_id_sha256=candidate_identity_sha256(stable),
        idempotency_key=str(stable["idempotency_key"]),
        teacher_request_idempotency_key=str(stable["teacher_request_idempotency_key"]),
        source_record_id_sha256=str(stable["source_record_id_sha256"]),
        source_content_sha256=str(stable["source_content_sha256"]),
        source_serialization_identity_sha256=str(
            stable["source_serialization_identity_sha256"]
        ),
        teacher_request_record_id_sha256=str(
            stable["teacher_request_record_id_sha256"]
        ),
        parent_source_record_id_sha256=str(stable["parent_source_record_id_sha256"]),
        task_bundle_sha256=str(stable["task_bundle_sha256"]),
        parent_task_bundle_sha256=str(stable["parent_task_bundle_sha256"]),
        source_semantic_sha256=str(stable["source_semantic_sha256"]),
        overlay_row_sha256=str(stable["overlay_row_sha256"]),
        identity_derivative_spec_sha256=observed_spec_sha256,
        selection_kind="identity",
        output_expert_role=str(stable["output_expert_role"]),
        training_asset=str(stable["training_asset"]),
        validation_contract_role=str(stable["validation_contract_role"]),
        language=str(stable["language"]),
        tool_family=None,
        identity_class=str(stable["identity_class"]),
        identity_parent_class=str(stable["identity_parent_class"]),
        identity_provenance_intent=str(stable["identity_provenance_intent"]),
        review_verdict=None,
        review_fault_type=None,
        router_label=None,
    )


def derive_candidates_from_source(
    inventory: batch.SourceInventory,
    config: batch.AdapterConfig,
    overlay: AuthenticatedMetadataOverlay,
) -> tuple[SelectorCandidate, ...]:
    """Derive the exact candidate universe from authenticated source metadata."""

    jobs = batch.derive_jobs(inventory, config)
    overlay_rows = authenticate_metadata_overlay(
        overlay,
        inventory=inventory,
        config=config,
    )
    overlay_by_source = {
        str(row["source_record_id_sha256"]): row for row in overlay_rows
    }
    if len(overlay_by_source) != len(jobs):
        raise batch.AdapterError("minimal320_overlay_job_count_drift")
    derivative_plan = overlay.manifest["identity_derivative_plan"]
    derivative_specs = derivative_plan["derivative_specs"]
    specs_by_parent_source = {
        str(spec["parent_source_record_id_sha256"]): spec for spec in derivative_specs
    }
    if len(derivative_specs) != 100 or len(specs_by_parent_source) != 100:
        raise batch.AdapterError("minimal320_identity_derivative_spec_inventory_drift")
    by_bundle: dict[
        str,
        list[tuple[batch.AlignmentJob, Mapping[str, Any]]],
    ] = defaultdict(list)
    for job in jobs:
        row = overlay_by_source.get(_hash_text(job.source.record_id))
        if row is None:
            raise batch.AdapterError("minimal320_overlay_job_missing")
        by_bundle[job.source.task_bundle].append((job, row))
    core_cells: dict[
        tuple[str, str],
        list[
            tuple[
                str,
                dict[str, tuple[batch.AlignmentJob, Mapping[str, Any]]],
            ]
        ],
    ] = defaultdict(list)
    depth_cells: dict[
        tuple[str, str],
        list[
            tuple[
                str,
                dict[str, tuple[batch.AlignmentJob, Mapping[str, Any]]],
            ]
        ],
    ] = defaultdict(list)
    identity_bundles: list[
        tuple[
            str,
            dict[str, tuple[batch.AlignmentJob, Mapping[str, Any]]],
        ]
    ] = []
    router_rows: list[tuple[batch.AlignmentJob, Mapping[str, Any]]] = []
    for bundle, rows in by_bundle.items():
        role_map = {job.source.role: (job, row) for job, row in rows}
        if len(role_map) != len(rows):
            raise batch.AdapterError("minimal320_source_bundle_role_duplicate")
        if set(role_map) == {"router"}:
            router_rows.extend(rows)
            continue
        languages = {_language(job.source.language) for job, _row in rows}
        if len(languages) != 1:
            raise batch.AdapterError("minimal320_source_bundle_language_mixed")
        language = next(iter(languages))
        identity_classes = {
            str(row["identity_class"])
            for _job, row in rows
            if row["identity_class"] is not None
        }
        if identity_classes:
            if set(role_map) != {
                "humor",
                "serious",
                "angry",
                "identity",
                "review",
            }:
                raise batch.AdapterError(
                    "minimal320_source_identity_role_closure_drift"
                )
            if (
                len(identity_classes) != 1
                or len(
                    {
                        (
                            row["identity_class"],
                            row["identity_provenance_intent"],
                            row["identity_case_context_sha256"],
                            row["identity_stable_fact_sha256"],
                            row["identity_claimed_attribution"],
                        )
                        for _job, row in rows
                    }
                )
                != 1
            ):
                raise batch.AdapterError("minimal320_source_identity_metadata_mixed")
            identity_bundles.append((bundle, role_map))
            continue
        if "tool" not in role_map or "review" not in role_map:
            continue
        family = role_map["tool"][1]["tool_family"]
        if not isinstance(family, str):
            raise batch.AdapterError("minimal320_source_tool_family_invalid")
        cell = (family, language)
        if set(role_map) == set(OUTPUT_EXPERT_ROLES):
            # A depth projection is the authenticated tool+review pair from a
            # distinct physical bundle.  Full bundles are valid for both
            # candidate pools; the selector later forbids cross-pool reuse.
            core_cells[cell].append((bundle, role_map))
            depth_cells[cell].append((bundle, role_map))
        elif set(role_map) == {"tool", "review"}:
            depth_cells[cell].append((bundle, role_map))
    families = sorted({family for family, _ in core_cells})
    if len(families) != 10:
        raise batch.AdapterError("minimal320_source_tool_family_count_drift")
    candidates: list[SelectorCandidate] = []
    for family in families:
        for language in LANGUAGES:
            core_options = sorted(core_cells.get((family, language), []))
            depth_options = sorted(depth_cells.get((family, language), []))
            if not core_options or not depth_options:
                raise batch.AdapterError("minimal320_source_tool_cell_unreachable")
            for _core_bundle, core in core_options:
                for role in OUTPUT_EXPERT_ROLES:
                    job, row = core[role]
                    candidates.append(
                        _source_candidate(
                            job,
                            row,
                            selection_kind="full_core",
                            output_expert_role=role,
                            validation_contract_role=role,
                            tool_family=family,
                        )
                    )
            for _depth_bundle, depth in depth_options:
                review_row = depth["review"][1]
                review_verdict = str(review_row["review_verdict"])
                raw_fault = review_row["review_fault_type"]
                review_fault = str(raw_fault) if raw_fault is not None else None
                if (
                    review_verdict not in {"pass", "fail"}
                    or (review_verdict == "pass" and review_fault is not None)
                    or (review_verdict == "fail" and review_fault not in FAULT_TYPES)
                ):
                    raise batch.AdapterError(
                        "minimal320_source_review_contract_unreachable"
                    )
                for role in ("tool", "review"):
                    job, row = depth[role]
                    candidates.append(
                        _source_candidate(
                            job,
                            row,
                            selection_kind="tool_review_depth",
                            output_expert_role=role,
                            validation_contract_role=role,
                            tool_family=family,
                            review_verdict=review_verdict,
                            review_fault_type=review_fault,
                        )
                    )
    if len(identity_bundles) != 20:
        raise batch.AdapterError("minimal320_source_identity_bundle_count_drift")
    consumed_specs: set[str] = set()
    for _bundle, roles in sorted(identity_bundles):
        for _source_role, (_job, row) in sorted(roles.items()):
            source_sha = str(row["source_record_id_sha256"])
            spec = specs_by_parent_source.get(source_sha)
            if spec is None:
                raise batch.AdapterError("minimal320_identity_derivative_spec_missing")
            consumed_specs.add(str(spec["identity_derivative_spec_sha256"]))
            candidates.append(_identity_derivative_candidate(spec, row))
    if len(consumed_specs) != 100:
        raise batch.AdapterError("minimal320_identity_derivative_spec_omission")
    if (
        len(router_rows) != 80
        or len({str(row["source_semantic_sha256"]) for _job, row in router_rows}) != 80
    ):
        raise batch.AdapterError("minimal320_source_router_semantic_drift")
    for job, row in sorted(
        router_rows,
        key=lambda item: (
            str(item[1]["source_semantic_sha256"]),
            str(item[1]["source_record_id_sha256"]),
        ),
    ):
        candidates.append(
            _source_candidate(
                job,
                row,
                selection_kind="router",
                output_expert_role="router",
                validation_contract_role="router",
                tool_family=None,
                router_label=str(row["router_label"]),
            )
        )
    if len(candidates) != 3960 or len(
        {candidate.candidate_id_sha256 for candidate in candidates}
    ) != len(candidates):
        raise batch.AdapterError("minimal320_source_candidate_count_drift")
    return tuple(candidates)


def candidate_from_mapping(
    value: Mapping[str, Any],
    *,
    schema_path: Path = DEFAULT_CANDIDATE_SCHEMA_PATH,
    _schema: Mapping[str, Any] | None = None,
) -> SelectorCandidate:
    materialized = dict(value)
    try:
        schema = (
            dict(_schema)
            if _schema is not None
            else _load_schema_physical(schema_path)[0]
        )
        Draft202012Validator(schema).validate(materialized)
    except ValidationError as error:
        raise batch.AdapterError("minimal320_candidate_schema_mismatch") from error
    if not hmac.compare_digest(
        str(materialized["candidate_id_sha256"]),
        candidate_identity_sha256(materialized),
    ):
        raise batch.AdapterError("minimal320_candidate_identity_spoof")
    return SelectorCandidate(
        candidate_id_sha256=str(materialized["candidate_id_sha256"]),
        idempotency_key=str(materialized["idempotency_key"]),
        teacher_request_idempotency_key=str(
            materialized["teacher_request_idempotency_key"]
        ),
        source_record_id_sha256=str(materialized["source_record_id_sha256"]),
        source_content_sha256=str(materialized["source_content_sha256"]),
        source_serialization_identity_sha256=str(
            materialized["source_serialization_identity_sha256"]
        ),
        teacher_request_record_id_sha256=str(
            materialized["teacher_request_record_id_sha256"]
        ),
        parent_source_record_id_sha256=str(
            materialized["parent_source_record_id_sha256"]
        ),
        task_bundle_sha256=str(materialized["task_bundle_sha256"]),
        parent_task_bundle_sha256=str(materialized["parent_task_bundle_sha256"]),
        source_semantic_sha256=str(materialized["source_semantic_sha256"]),
        overlay_row_sha256=str(materialized["overlay_row_sha256"]),
        identity_derivative_spec_sha256=(
            str(materialized["identity_derivative_spec_sha256"])
            if materialized["identity_derivative_spec_sha256"] is not None
            else None
        ),
        selection_kind=str(materialized["selection_kind"]),
        output_expert_role=str(materialized["output_expert_role"]),
        training_asset=str(materialized["training_asset"]),
        validation_contract_role=str(materialized["validation_contract_role"]),
        language=str(materialized["language"]),
        tool_family=(
            str(materialized["tool_family"])
            if materialized["tool_family"] is not None
            else None
        ),
        identity_class=(
            str(materialized["identity_class"])
            if materialized["identity_class"] is not None
            else None
        ),
        identity_parent_class=(
            str(materialized["identity_parent_class"])
            if materialized["identity_parent_class"] is not None
            else None
        ),
        identity_provenance_intent=(
            str(materialized["identity_provenance_intent"])
            if materialized["identity_provenance_intent"] is not None
            else None
        ),
        review_verdict=(
            str(materialized["review_verdict"])
            if materialized["review_verdict"] is not None
            else None
        ),
        review_fault_type=(
            str(materialized["review_fault_type"])
            if materialized["review_fault_type"] is not None
            else None
        ),
        router_label=(
            str(materialized["router_label"])
            if materialized["router_label"] is not None
            else None
        ),
    )


def _one_per_role(
    candidates: Sequence[SelectorCandidate],
    roles: Sequence[str],
) -> tuple[SelectorCandidate, ...] | None:
    by_role: dict[str, list[SelectorCandidate]] = defaultdict(list)
    for candidate in candidates:
        by_role[candidate.output_expert_role].append(candidate)
    if set(by_role) != set(roles):
        return None
    return tuple(
        min(by_role[role], key=lambda item: item.candidate_id_sha256) for role in roles
    )


def _select_full_core(
    candidates: Sequence[SelectorCandidate],
    tool_families: Sequence[str],
) -> list[SelectorCandidate]:
    selected: list[SelectorCandidate] = []
    for family in tool_families:
        for language in LANGUAGES:
            by_bundle: dict[str, list[SelectorCandidate]] = defaultdict(list)
            for candidate in candidates:
                if (
                    candidate.selection_kind == "full_core"
                    and candidate.tool_family == family
                    and candidate.language == language
                ):
                    by_bundle[candidate.task_bundle_sha256].append(candidate)
            eligible: list[tuple[str, tuple[SelectorCandidate, ...]]] = []
            for bundle, rows in by_bundle.items():
                views = _one_per_role(rows, OUTPUT_EXPERT_ROLES)
                if views is not None and all(
                    item.validation_contract_role == item.output_expert_role
                    and item.identity_class is None
                    and item.review_verdict is None
                    and item.review_fault_type is None
                    and item.router_label is None
                    for item in views
                ):
                    eligible.append((bundle, views))
            if not eligible:
                raise batch.AdapterError("minimal320_full_core_cell_unreachable")
            selected.extend(min(eligible, key=lambda item: item[0])[1])
    return selected


def _review_availability_rows(
    candidates: Sequence[SelectorCandidate],
    tool_families: Sequence[str],
) -> list[dict[str, Any]]:
    availability: dict[
        tuple[str, str],
        dict[str, Any],
    ] = {
        (family, language): {
            "pass_available": False,
            "fail_fault_types": set(),
        }
        for family in tool_families
        for language in LANGUAGES
    }
    by_bundle: dict[str, list[SelectorCandidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.selection_kind == "tool_review_depth":
            by_bundle[candidate.task_bundle_sha256].append(candidate)
    for rows in by_bundle.values():
        views = _one_per_role(rows, ("tool", "review"))
        if views is None:
            raise batch.AdapterError("minimal320_review_availability_bundle_invalid")
        cells = {(str(item.tool_family), item.language) for item in views}
        labels = {(item.review_verdict, item.review_fault_type) for item in views}
        if len(cells) != 1 or len(labels) != 1:
            raise batch.AdapterError("minimal320_review_availability_bundle_mixed")
        cell = next(iter(cells))
        if cell not in availability:
            raise batch.AdapterError("minimal320_review_availability_cell_invalid")
        verdict, fault = next(iter(labels))
        if verdict == "pass" and fault is None:
            availability[cell]["pass_available"] = True
        elif verdict == "fail" and fault in FAULT_TYPES:
            availability[cell]["fail_fault_types"].add(fault)
        else:
            raise batch.AdapterError("minimal320_review_availability_label_invalid")
    rows: list[dict[str, Any]] = []
    for family, language in sorted(availability):
        value = availability[(family, language)]
        rows.append(
            {
                "tool_family": family,
                "language": language,
                "pass_available": bool(value["pass_available"]),
                "fail_fault_types": sorted(value["fail_fault_types"]),
            }
        )
    if any(not row["pass_available"] and not row["fail_fault_types"] for row in rows):
        raise batch.AdapterError("minimal320_review_availability_cell_unreachable")
    return rows


def derive_review_cell_assignment(
    candidates: Sequence[SelectorCandidate],
    tool_families: Sequence[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Solve the physical v2 review schedule without translating labels."""

    availability = _review_availability_rows(
        candidates,
        tool_families,
    )
    availability_root = _hash(availability)
    if not hmac.compare_digest(
        availability_root,
        EXPECTED_REVIEW_AVAILABILITY_ROOT_SHA256,
    ):
        raise batch.AdapterError("minimal320_review_availability_root_drift")
    target_faults = {fault: 2 for fault in FAULT_TYPES}

    def search(
        index: int,
        failed_count: int,
        fault_counts: Counter[str],
        assigned: list[dict[str, Any]],
    ) -> list[dict[str, Any]] | None:
        remaining = len(availability) - index
        if (
            failed_count > 10
            or failed_count + remaining < 10
            or any(
                fault_counts[fault] > target for fault, target in target_faults.items()
            )
        ):
            return None
        if index == len(availability):
            if failed_count == 10 and fault_counts == Counter(target_faults):
                return list(assigned)
            return None
        row = availability[index]
        options: list[tuple[str, str | None]] = [
            ("fail", str(fault)) for fault in row["fail_fault_types"]
        ]
        if row["pass_available"]:
            options.append(("pass", None))
        for verdict, fault in options:
            next_counts = fault_counts.copy()
            next_failed = failed_count
            if verdict == "fail":
                assert fault is not None
                next_failed += 1
                next_counts[fault] += 1
            assignment = {
                "tool_family": row["tool_family"],
                "language": row["language"],
                "review_verdict": verdict,
                "review_fault_type": fault,
            }
            result = search(
                index + 1,
                next_failed,
                next_counts,
                [*assigned, assignment],
            )
            if result is not None:
                return result
        return None

    assignment = search(0, 0, Counter(), [])
    if assignment is None:
        raise batch.AdapterError("minimal320_review_assignment_unreachable")
    if not hmac.compare_digest(
        _hash(assignment),
        EXPECTED_REVIEW_ASSIGNMENT_SHA256,
    ):
        raise batch.AdapterError("minimal320_review_assignment_identity_drift")
    return availability, assignment


def _candidate_source_availability(
    candidates: Sequence[SelectorCandidate],
    review_availability: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    identity_by_bundle: dict[str, list[SelectorCandidate]] = defaultdict(list)
    router = Counter()
    for candidate in candidates:
        if candidate.selection_kind == "identity":
            identity_by_bundle[candidate.task_bundle_sha256].append(candidate)
        elif candidate.selection_kind == "router":
            router[(str(candidate.router_label), candidate.language)] += 1
    identity_parent = Counter()
    identity_target = Counter()
    seen_parent_bundles: set[str] = set()
    for rows in identity_by_bundle.values():
        target_cells = {(str(item.identity_class), item.language) for item in rows}
        parent_cells = {
            (str(item.identity_parent_class), item.language) for item in rows
        }
        parent_bundles = {item.parent_task_bundle_sha256 for item in rows}
        if (
            len(target_cells) != 1
            or len(parent_cells) != 1
            or len(parent_bundles) != 1
            or len(rows) != len(OUTPUT_EXPERT_ROLES)
        ):
            raise batch.AdapterError("minimal320_identity_availability_bundle_mixed")
        parent_bundle = next(iter(parent_bundles))
        if parent_bundle in seen_parent_bundles:
            raise batch.AdapterError("minimal320_identity_availability_parent_reuse")
        seen_parent_bundles.add(parent_bundle)
        identity_parent[next(iter(parent_cells))] += 1
        identity_target[next(iter(target_cells))] += 1
    value = {
        "review_cells": [dict(row) for row in review_availability],
        "identity_parent_bundle_cells": [
            {
                "identity_class": identity_class,
                "language": language,
                "count": int(identity_parent[(identity_class, language)]),
            }
            for identity_class in batch.IDENTITY_CLASSES
            for language in LANGUAGES
        ],
        "identity_target_bundle_cells": [
            {
                "identity_class": identity_class,
                "language": language,
                "count": int(identity_target[(identity_class, language)]),
            }
            for identity_class in batch.IDENTITY_CLASSES
            for language in LANGUAGES
        ],
        "router_cells": [
            {
                "router_label": label,
                "language": language,
                "count": int(router[(label, language)]),
            }
            for label in ("humor", "serious", "angry")
            for language in LANGUAGES
        ],
    }
    if not hmac.compare_digest(
        _hash(value),
        EXPECTED_SOURCE_AVAILABILITY_ROOT_SHA256,
    ):
        raise batch.AdapterError("minimal320_candidate_source_availability_root_drift")
    return value


def _identity_bundle_views_valid(
    views: Sequence[SelectorCandidate],
) -> bool:
    expected_validation = {
        "humor": "humor",
        "serious": "serious",
        "angry": "angry",
        "tool": "identity",
        "review": "review",
    }
    if (
        len(views) != len(OUTPUT_EXPERT_ROLES)
        or Counter(item.output_expert_role for item in views)
        != Counter(OUTPUT_EXPERT_ROLES)
        or len({item.source_record_id_sha256 for item in views}) != 5
        or len({item.parent_source_record_id_sha256 for item in views}) != 5
        or len({item.task_bundle_sha256 for item in views}) != 1
        or len({item.parent_task_bundle_sha256 for item in views}) != 1
        or len({item.identity_derivative_spec_sha256 for item in views}) != 5
    ):
        return False
    if any(
        item.source_record_id_sha256 == item.parent_source_record_id_sha256
        or item.teacher_request_record_id_sha256 != item.source_record_id_sha256
        or item.teacher_request_idempotency_key != item.idempotency_key
        or item.identity_derivative_spec_sha256 is None
        or item.validation_contract_role != expected_validation[item.output_expert_role]
        or item.tool_family is not None
        or item.identity_class not in batch.IDENTITY_CLASSES
        or item.identity_parent_class not in batch.IDENTITY_CLASSES
        or item.identity_provenance_intent is None
        or item.review_verdict is not None
        or item.review_fault_type is not None
        or item.router_label is not None
        for item in views
    ):
        return False
    return (
        len(
            {
                (
                    item.identity_class,
                    item.identity_parent_class,
                    item.language,
                    item.identity_provenance_intent,
                )
                for item in views
            }
        )
        == 1
    )


def _select_depth(
    candidates: Sequence[SelectorCandidate],
    tool_families: Sequence[str],
    *,
    assignment: Sequence[Mapping[str, Any]],
    forbidden_bundles: frozenset[str],
) -> list[SelectorCandidate]:
    assignment_by_cell = {
        (str(row["tool_family"]), str(row["language"])): (
            str(row["review_verdict"]),
            (
                str(row["review_fault_type"])
                if row["review_fault_type"] is not None
                else None
            ),
        )
        for row in assignment
    }
    selected: list[SelectorCandidate] = []
    for family in tool_families:
        for language in LANGUAGES:
            try:
                verdict, fault_type = assignment_by_cell[(family, language)]
            except KeyError:
                raise batch.AdapterError(
                    "minimal320_review_assignment_cell_missing"
                ) from None
            by_bundle: dict[str, list[SelectorCandidate]] = defaultdict(list)
            for candidate in candidates:
                if (
                    candidate.selection_kind == "tool_review_depth"
                    and candidate.tool_family == family
                    and candidate.language == language
                ):
                    by_bundle[candidate.task_bundle_sha256].append(candidate)
            eligible: list[tuple[str, tuple[SelectorCandidate, ...]]] = []
            for bundle, rows in by_bundle.items():
                if bundle in forbidden_bundles:
                    continue
                views = _one_per_role(rows, ("tool", "review"))
                if views is None:
                    continue
                if all(
                    item.validation_contract_role == item.output_expert_role
                    and item.identity_class is None
                    and item.identity_provenance_intent is None
                    and item.router_label is None
                    and item.review_verdict == verdict
                    and item.review_fault_type == fault_type
                    for item in views
                ):
                    eligible.append((bundle, views))
            if not eligible:
                raise batch.AdapterError("minimal320_depth_cell_unreachable")
            selected.extend(min(eligible, key=lambda item: item[0])[1])
    return selected


def _select_identity(
    candidates: Sequence[SelectorCandidate],
) -> list[SelectorCandidate]:
    by_bundle: dict[str, list[SelectorCandidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.selection_kind == "identity":
            by_bundle[candidate.task_bundle_sha256].append(candidate)
    eligible: list[tuple[str, tuple[SelectorCandidate, ...]]] = []
    for bundle, rows in by_bundle.items():
        views = _one_per_role(rows, OUTPUT_EXPERT_ROLES)
        if views is not None and _identity_bundle_views_valid(views):
            eligible.append((bundle, views))
    if len(eligible) != 20:
        # The contract deliberately forbids choosing a convenient subset at
        # runtime.  A source release must attest exactly twenty complete bundles.
        raise batch.AdapterError("minimal320_identity_bundle_contract_unreachable")
    bundle_cells: Counter[tuple[str, str]] = Counter()
    selected: list[SelectorCandidate] = []
    for _, views in sorted(eligible, key=lambda item: item[0]):
        cells = {(str(item.identity_class), item.language) for item in views}
        if len(cells) != 1:
            raise batch.AdapterError("minimal320_identity_bundle_cell_mixed")
        cell = next(iter(cells))
        bundle_cells[cell] += 1
        selected.extend(views)
    if bundle_cells != Counter(IDENTITY_BUNDLE_CELL_QUOTAS):
        raise batch.AdapterError("minimal320_identity_bundle_cell_quota_drift")
    observed_records = Counter(
        (str(item.identity_class), item.language) for item in selected
    )
    if observed_records != Counter(IDENTITY_RECORD_CELL_QUOTAS):
        raise batch.AdapterError("minimal320_identity_cell_quota_drift")
    return selected


def _select_router(
    candidates: Sequence[SelectorCandidate],
) -> list[SelectorCandidate]:
    selected: list[SelectorCandidate] = []
    for cell, quota in ROUTER_CELL_QUOTAS.items():
        label, language = cell
        matching = sorted(
            (
                item
                for item in candidates
                if item.selection_kind == "router"
                and item.output_expert_role == "router"
                and item.validation_contract_role == "router"
                and item.router_label == label
                and item.language == language
                and item.tool_family is None
                and item.identity_class is None
                and item.review_verdict is None
                and item.review_fault_type is None
            ),
            key=lambda item: item.candidate_id_sha256,
        )
        if len(matching) < quota:
            raise batch.AdapterError("minimal320_router_cell_unreachable")
        selected.extend(matching[:quota])
    return selected


def _assert_final_quotas(
    selected: Sequence[SelectorCandidate],
    *,
    tool_families: Sequence[str],
) -> None:
    if len(selected) != 320:
        raise batch.AdapterError("minimal320_count_drift")
    for item in selected:
        if not hmac.compare_digest(
            item.candidate_id_sha256,
            candidate_identity_sha256(item.as_dict()),
        ):
            raise batch.AdapterError("minimal320_candidate_identity_spoof")
    if len({item.candidate_id_sha256 for item in selected}) != 320:
        raise batch.AdapterError("minimal320_candidate_duplicate")
    if len({item.idempotency_key for item in selected}) != 320:
        raise batch.AdapterError("minimal320_idempotency_duplicate")
    non_identity = [item for item in selected if item.selection_kind != "identity"]
    if len({item.source_record_id_sha256 for item in non_identity}) != len(
        non_identity
    ):
        raise batch.AdapterError("minimal320_source_record_duplicate")
    if Counter(item.output_expert_role for item in selected) != Counter(ROLE_QUOTAS):
        raise batch.AdapterError("minimal320_role_quota_drift")
    if Counter(item.language for item in selected) != Counter(LANGUAGE_QUOTAS):
        raise batch.AdapterError("minimal320_language_quota_drift")
    core_rows = [item for item in selected if item.selection_kind == "full_core"]
    depth_rows = [
        item for item in selected if item.selection_kind == "tool_review_depth"
    ]
    core_by_bundle: dict[str, list[SelectorCandidate]] = defaultdict(list)
    depth_by_bundle: dict[str, list[SelectorCandidate]] = defaultdict(list)
    for item in core_rows:
        core_by_bundle[item.task_bundle_sha256].append(item)
    for item in depth_rows:
        depth_by_bundle[item.task_bundle_sha256].append(item)
    if (
        len(core_by_bundle) != 20
        or len(depth_by_bundle) != 20
        or set(core_by_bundle) & set(depth_by_bundle)
    ):
        raise batch.AdapterError("minimal320_tool_bundle_distinctness_drift")
    if any(
        Counter(item.output_expert_role for item in rows)
        != Counter(OUTPUT_EXPERT_ROLES)
        or len({(item.tool_family, item.language) for item in rows}) != 1
        for rows in core_by_bundle.values()
    ):
        raise batch.AdapterError("minimal320_core_bundle_closure_drift")
    if any(
        Counter(item.output_expert_role for item in rows)
        != Counter({"tool": 1, "review": 1})
        or len({(item.tool_family, item.language) for item in rows}) != 1
        for rows in depth_by_bundle.values()
    ):
        raise batch.AdapterError("minimal320_depth_bundle_closure_drift")
    for family in tool_families:
        for language in LANGUAGES:
            core = [
                item
                for item in selected
                if item.selection_kind == "full_core"
                and item.tool_family == family
                and item.language == language
            ]
            depth = [
                item
                for item in selected
                if item.selection_kind == "tool_review_depth"
                and item.tool_family == family
                and item.language == language
            ]
            if Counter(item.output_expert_role for item in core) != Counter(
                OUTPUT_EXPERT_ROLES
            ) or Counter(item.output_expert_role for item in depth) != Counter(
                {"tool": 1, "review": 1}
            ):
                raise batch.AdapterError("minimal320_tool_cell_quota_drift")
    identity = [item for item in selected if item.selection_kind == "identity"]
    identity_by_bundle: dict[str, list[SelectorCandidate]] = defaultdict(list)
    for item in identity:
        identity_by_bundle[item.task_bundle_sha256].append(item)
    if len(identity_by_bundle) != 20 or any(
        not _identity_bundle_views_valid(rows) for rows in identity_by_bundle.values()
    ):
        raise batch.AdapterError("minimal320_identity_source_view_drift")
    if (
        len({item.parent_task_bundle_sha256 for item in identity}) != 20
        or len({item.parent_source_record_id_sha256 for item in identity}) != 100
    ):
        raise batch.AdapterError("minimal320_identity_parent_lineage_reuse")
    if Counter(
        (str(item.identity_class), item.language) for item in identity
    ) != Counter(IDENTITY_RECORD_CELL_QUOTAS):
        raise batch.AdapterError("minimal320_identity_cell_quota_drift")
    router = [item for item in selected if item.selection_kind == "router"]
    if Counter((str(item.router_label), item.language) for item in router) != Counter(
        ROUTER_CELL_QUOTAS
    ):
        raise batch.AdapterError("minimal320_router_cell_quota_drift")
    depth_reviews = [
        item
        for item in selected
        if item.selection_kind == "tool_review_depth"
        and item.output_expert_role == "review"
    ]
    if any(
        len(
            {
                (
                    item.review_verdict,
                    item.review_fault_type,
                )
                for item in rows
            }
        )
        != 1
        for rows in depth_by_bundle.values()
    ):
        raise batch.AdapterError("minimal320_depth_review_binding_drift")
    if Counter(item.review_verdict for item in depth_reviews) != Counter(
        {"pass": 10, "fail": 10}
    ):
        raise batch.AdapterError("minimal320_review_verdict_quota_drift")
    failed = [item for item in depth_reviews if item.review_verdict == "fail"]
    if Counter(item.review_fault_type for item in failed) != Counter(
        {name: 2 for name in FAULT_TYPES}
    ):
        raise batch.AdapterError("minimal320_review_fault_quota_drift")
    if any(item.output_expert_role == "identity" for item in selected):
        raise batch.AdapterError("minimal320_identity_sixth_expert_forbidden")
    identity_tool_views = [
        item for item in identity if item.output_expert_role == "tool"
    ]
    if len(identity_tool_views) != 20 or any(
        item.tool_family is not None for item in identity_tool_views
    ):
        raise batch.AdapterError("minimal320_identity_tool_direct_contract_drift")


def select_minimal_320(
    candidates: Sequence[SelectorCandidate],
    *,
    contract_sha256: str,
    source_identity_sha256: str,
) -> Minimal320Selection:
    """Select and attest the exact 320-record contract without body access."""

    for item in candidates:
        if not hmac.compare_digest(
            item.candidate_id_sha256,
            candidate_identity_sha256(item.as_dict()),
        ):
            raise batch.AdapterError("minimal320_candidate_identity_spoof")
    ordered = sorted(candidates, key=lambda item: item.candidate_id_sha256)
    if len({item.candidate_id_sha256 for item in ordered}) != len(ordered):
        raise batch.AdapterError("minimal320_candidate_identity_duplicate")
    tool_families = sorted(
        {
            str(item.tool_family)
            for item in ordered
            if item.selection_kind in {"full_core", "tool_review_depth"}
            and item.tool_family is not None
        }
    )
    if len(tool_families) != 10:
        raise batch.AdapterError("minimal320_tool_family_count_drift")
    review_availability, review_assignment = derive_review_cell_assignment(
        ordered, tool_families
    )
    source_availability = _candidate_source_availability(
        ordered,
        review_availability,
    )
    core = _select_full_core(ordered, tool_families)
    selected = [
        *core,
        *_select_depth(
            ordered,
            tool_families,
            assignment=review_assignment,
            forbidden_bundles=frozenset(item.task_bundle_sha256 for item in core),
        ),
        *_select_identity(ordered),
        *_select_router(ordered),
    ]
    selected.sort(key=lambda item: item.candidate_id_sha256)
    _assert_final_quotas(selected, tool_families=tool_families)
    rows = [item.as_dict() for item in selected]
    selected_root = _hash(rows)
    counts = {
        "requested": 320,
        "selected": len(selected),
        "requested_by_role": dict(ROLE_QUOTAS),
        "selected_by_role": dict(
            sorted(Counter(item.output_expert_role for item in selected).items())
        ),
        "requested_by_language": dict(LANGUAGE_QUOTAS),
        "selected_by_language": dict(
            sorted(Counter(item.language for item in selected).items())
        ),
        "requested_by_partition": {
            "full_core": 100,
            "tool_review_depth": 40,
            "identity": 100,
            "router": 80,
        },
        "selected_by_partition": dict(
            sorted(Counter(item.selection_kind for item in selected).items())
        ),
    }
    identity = {
        "domain": MANIFEST_SCHEMA_VERSION,
        "contract_sha256": contract_sha256,
        "source_identity_sha256": source_identity_sha256,
        "selected_set_root_sha256": selected_root,
        "tool_family_order": tool_families,
        "review_availability_root_sha256": _hash(review_availability),
        "review_assignment_algorithm_id": (REVIEW_ASSIGNMENT_ALGORITHM_ID),
        "review_assignment_sha256": _hash(review_assignment),
        "source_availability_root_sha256": _hash(source_availability),
        "counts": counts,
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "selector_identity_sha256": _hash(identity),
        **identity,
        "identity_bundle_cell_quotas": [
            {
                "identity_class": identity_class,
                "language": language,
                "count": count,
            }
            for (identity_class, language), count in (
                IDENTITY_BUNDLE_CELL_QUOTAS.items()
            )
        ],
        "identity_record_cell_quotas": [
            {
                "identity_class": identity_class,
                "language": language,
                "count": count,
            }
            for (identity_class, language), count in (
                IDENTITY_RECORD_CELL_QUOTAS.items()
            )
        ],
        "router_cell_quotas": [
            {"label": label, "language": language, "count": count}
            for (label, language), count in ROUTER_CELL_QUOTAS.items()
        ],
        "review_availability": review_availability,
        "review_assignment": review_assignment,
        "source_availability": source_availability,
        "content_retained": False,
        "raw_token_ids_retained": False,
    }
    return Minimal320Selection(tuple(selected), manifest)


def _candidate_jsonl_bytes(
    candidates: Sequence[SelectorCandidate],
    *,
    require_unique_idempotency: bool = False,
) -> bytes:
    ordered = sorted(
        candidates,
        key=lambda item: item.candidate_id_sha256,
    )
    if len(ordered) != len({item.candidate_id_sha256 for item in ordered}) or (
        require_unique_idempotency
        and len(ordered) != len({item.idempotency_key for item in ordered})
    ):
        raise batch.AdapterError("minimal320_preimage_candidate_identity_duplicate")
    return b"".join(_canonical_bytes(item.as_dict()) + b"\n" for item in ordered)


def _load_candidate_jsonl(
    path: Path,
    *,
    stop: Path,
    expected_count: int,
    expected_sha256: str,
    expected_root_sha256: str,
    candidate_schema: Mapping[str, Any],
    require_unique_idempotency: bool = False,
) -> tuple[SelectorCandidate, ...]:
    raw = _physical_read(
        path,
        stop=stop,
        reason="minimal320_preimage_candidate_snapshot_drift",
    )
    _authenticate_sha256_sidecar(
        path,
        raw,
        stop=stop,
        reason="minimal320_preimage_candidate_sidecar_drift",
    )
    if not raw.endswith(b"\n") or hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise batch.AdapterError("minimal320_preimage_candidate_inventory_drift")
    candidates: list[SelectorCandidate] = []
    rows: list[dict[str, Any]] = []
    for raw_line in raw.splitlines():
        value = closed._strict_json(
            raw_line,
            reason="minimal320_preimage_candidate_json_invalid",
        )
        if not isinstance(value, dict):
            raise batch.AdapterError("minimal320_preimage_candidate_json_invalid")
        candidate = candidate_from_mapping(
            value,
            _schema=candidate_schema,
        )
        candidates.append(candidate)
        rows.append(candidate.as_dict())
    if (
        len(candidates) != expected_count
        or rows
        != sorted(
            rows,
            key=lambda row: str(row["candidate_id_sha256"]),
        )
        or len({item.candidate_id_sha256 for item in candidates}) != len(candidates)
        or (
            require_unique_idempotency
            and len({item.idempotency_key for item in candidates}) != len(candidates)
        )
        or _hash(rows) != expected_root_sha256
    ):
        raise batch.AdapterError("minimal320_preimage_candidate_inventory_drift")
    return tuple(candidates)


def persist_selector_preimages(
    candidates: Sequence[SelectorCandidate],
    selection: Minimal320Selection,
    *,
    overlay: AuthenticatedMetadataOverlay,
    inventory: batch.SourceInventory,
    config: batch.AdapterConfig,
    contract_sha256: str,
    candidate_schema_path: Path = DEFAULT_CANDIDATE_SCHEMA_PATH,
    preimage_schema_path: Path = DEFAULT_PREIMAGE_SCHEMA_PATH,
) -> SelectorPreimages:
    """Persist the complete 3,960→320 selector preimage without replacement."""

    authenticate_metadata_overlay(
        overlay,
        inventory=inventory,
        config=config,
    )
    candidate_schema, candidate_schema_sha256 = _load_schema_physical(
        candidate_schema_path
    )
    _preimage_schema, preimage_schema_sha256 = _load_schema_physical(
        preimage_schema_path
    )
    ordered = tuple(
        sorted(
            candidates,
            key=lambda item: item.candidate_id_sha256,
        )
    )
    for candidate in ordered:
        candidate_from_mapping(
            candidate.as_dict(),
            _schema=candidate_schema,
        )
    if len(ordered) != 3960:
        raise batch.AdapterError("minimal320_preimage_candidate_count_drift")
    expected_selection = select_minimal_320(
        ordered,
        contract_sha256=contract_sha256,
        source_identity_sha256=inventory.source_identity_sha256,
    )
    if (
        tuple(item.as_dict() for item in selection.candidates)
        != tuple(item.as_dict() for item in expected_selection.candidates)
        or dict(selection.manifest) != expected_selection.manifest
    ):
        raise batch.AdapterError("minimal320_preimage_selection_object_drift")
    directory = overlay.directory
    candidate_rows_path = directory / "candidate_universe.jsonl"
    selected_rows_path = directory / "selected_candidates_320.jsonl"
    candidate_manifest_path = directory / "candidate_universe_manifest.json"
    selection_manifest_path = directory / "selection_manifest.json"
    selection_preimage_path = directory / "selection_preimage.json"
    preimage_manifest_path = directory / "selector_preimage_manifest.json"
    candidate_rows_raw = _candidate_jsonl_bytes(ordered)
    selected_rows_raw = _candidate_jsonl_bytes(
        selection.candidates,
        require_unique_idempotency=True,
    )
    _exclusive_bytes(candidate_rows_path, candidate_rows_raw)
    candidate_rows_sha256 = hashlib.sha256(candidate_rows_raw).hexdigest()
    _exclusive_bytes(
        candidate_rows_path.with_name(f"{candidate_rows_path.name}.sha256"),
        (f"{candidate_rows_sha256}  {candidate_rows_path.name}\n").encode("ascii"),
    )
    _exclusive_bytes(selected_rows_path, selected_rows_raw)
    selected_rows_sha256 = hashlib.sha256(selected_rows_raw).hexdigest()
    _exclusive_bytes(
        selected_rows_path.with_name(f"{selected_rows_path.name}.sha256"),
        (f"{selected_rows_sha256}  {selected_rows_path.name}\n").encode("ascii"),
    )
    overlay_manifest_raw = _physical_read(
        overlay.manifest_path,
        stop=directory,
        reason="minimal320_overlay_manifest_snapshot_drift",
    )
    overlay_manifest_sha256 = hashlib.sha256(overlay_manifest_raw).hexdigest()
    candidate_manifest = {
        "schema_version": (CANDIDATE_PREIMAGE_MANIFEST_SCHEMA_VERSION),
        "contract_sha256": contract_sha256,
        "source_identity_sha256": inventory.source_identity_sha256,
        "overlay_manifest_sha256": overlay_manifest_sha256,
        "overlay_binding_sha256": overlay.manifest["overlay_binding_sha256"],
        "identity_derivative_plan_sha256": overlay.manifest[
            "identity_derivative_plan_sha256"
        ],
        "candidate_schema_sha256": candidate_schema_sha256,
        "selector_implementation_sha256": overlay.manifest["overlay_binding"][
            "selector_implementation_sha256"
        ],
        "candidate_count": len(ordered),
        "candidate_rows_file": candidate_rows_path.name,
        "candidate_rows_sha256": candidate_rows_sha256,
        "candidate_set_root_sha256": _hash([item.as_dict() for item in ordered]),
        "content_retained": False,
        "raw_token_ids_retained": False,
    }
    _validate_closed_schema(
        candidate_manifest,
        schema_path=preimage_schema_path,
        reason="minimal320_candidate_manifest_schema_mismatch",
    )
    candidate_manifest_sha256 = _exclusive_json(
        candidate_manifest_path,
        candidate_manifest,
    )
    selection_manifest_sha256 = _exclusive_json(
        selection_manifest_path,
        selection.manifest,
    )
    selection_preimage = {
        "schema_version": SELECTOR_PREIMAGE_SCHEMA_VERSION,
        "contract_sha256": contract_sha256,
        "source_identity_sha256": inventory.source_identity_sha256,
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "selection_manifest_sha256": selection_manifest_sha256,
        "selector_identity_sha256": selection.manifest["selector_identity_sha256"],
        "selected_count": len(selection.candidates),
        "selected_rows_file": selected_rows_path.name,
        "selected_rows_sha256": selected_rows_sha256,
        "selected_set_root_sha256": selection.manifest["selected_set_root_sha256"],
        "identity_derivative_plan_sha256": overlay.manifest[
            "identity_derivative_plan_sha256"
        ],
        "content_retained": False,
        "raw_token_ids_retained": False,
    }
    _validate_closed_schema(
        selection_preimage,
        schema_path=preimage_schema_path,
        reason="minimal320_selection_preimage_schema_mismatch",
    )
    selection_preimage_sha256 = _exclusive_json(
        selection_preimage_path,
        selection_preimage,
    )
    manifest = {
        "schema_version": PREIMAGE_MANIFEST_SCHEMA_VERSION,
        "contract_sha256": contract_sha256,
        "source_identity_sha256": inventory.source_identity_sha256,
        "preimage_schema_sha256": preimage_schema_sha256,
        "candidate_schema_sha256": candidate_schema_sha256,
        "overlay_manifest_file": overlay.manifest_path.name,
        "overlay_manifest_sha256": overlay_manifest_sha256,
        "candidate_manifest_file": candidate_manifest_path.name,
        "candidate_manifest_sha256": candidate_manifest_sha256,
        "selection_manifest_file": selection_manifest_path.name,
        "selection_manifest_sha256": selection_manifest_sha256,
        "selection_preimage_file": selection_preimage_path.name,
        "selection_preimage_sha256": selection_preimage_sha256,
        "candidate_count": len(ordered),
        "selected_count": len(selection.candidates),
        "candidate_set_root_sha256": candidate_manifest["candidate_set_root_sha256"],
        "selected_set_root_sha256": selection.manifest["selected_set_root_sha256"],
        "selector_identity_sha256": selection.manifest["selector_identity_sha256"],
        "identity_derivative_plan_sha256": overlay.manifest[
            "identity_derivative_plan_sha256"
        ],
        "content_retained": False,
        "raw_token_ids_retained": False,
    }
    _validate_closed_schema(
        manifest,
        schema_path=preimage_schema_path,
        reason="minimal320_preimage_manifest_schema_mismatch",
    )
    _exclusive_json(preimage_manifest_path, manifest)
    return SelectorPreimages(
        directory=directory,
        candidate_rows_path=candidate_rows_path,
        candidate_manifest_path=candidate_manifest_path,
        selected_rows_path=selected_rows_path,
        selection_manifest_path=selection_manifest_path,
        selection_preimage_path=selection_preimage_path,
        preimage_manifest_path=preimage_manifest_path,
        manifest=manifest,
    )


def load_selector_preimages(
    directory: Path,
) -> SelectorPreimages:
    """Construct an external reload handle from canonical physical paths."""

    canonical = directory.absolute()
    if canonical.name != "minimal320_selector_preimages_v2":
        raise batch.AdapterError("minimal320_preimage_directory_identity_drift")
    preimage_manifest_path = canonical / "selector_preimage_manifest.json"
    raw = _physical_read(
        preimage_manifest_path,
        stop=canonical,
        reason="minimal320_preimage_manifest_snapshot_drift",
    )
    _authenticate_sha256_sidecar(
        preimage_manifest_path,
        raw,
        stop=canonical,
        reason="minimal320_preimage_manifest_sidecar_drift",
    )
    value = closed._strict_json(
        raw,
        reason="minimal320_preimage_manifest_snapshot_drift",
    )
    if not isinstance(value, dict):
        raise batch.AdapterError("minimal320_preimage_manifest_snapshot_drift")
    return SelectorPreimages(
        directory=canonical,
        candidate_rows_path=canonical / "candidate_universe.jsonl",
        candidate_manifest_path=(canonical / "candidate_universe_manifest.json"),
        selected_rows_path=canonical / "selected_candidates_320.jsonl",
        selection_manifest_path=canonical / "selection_manifest.json",
        selection_preimage_path=canonical / "selection_preimage.json",
        preimage_manifest_path=preimage_manifest_path,
        manifest=dict(value),
    )


def authenticate_selector_preimages(
    preimages: SelectorPreimages,
    *,
    overlay: AuthenticatedMetadataOverlay,
    inventory: batch.SourceInventory,
    config: batch.AdapterConfig,
    contract_sha256: str,
    candidate_schema_path: Path = DEFAULT_CANDIDATE_SCHEMA_PATH,
    preimage_schema_path: Path = DEFAULT_PREIMAGE_SCHEMA_PATH,
) -> tuple[tuple[SelectorCandidate, ...], Minimal320Selection]:
    """Reload every selector artifact and reject reorder, omission, or spoofing."""

    authenticate_metadata_overlay(
        overlay,
        inventory=inventory,
        config=config,
    )
    directory = preimages.directory.absolute()
    canonical_paths = {
        "candidate_rows_path": directory / "candidate_universe.jsonl",
        "candidate_manifest_path": (directory / "candidate_universe_manifest.json"),
        "selected_rows_path": (directory / "selected_candidates_320.jsonl"),
        "selection_manifest_path": (directory / "selection_manifest.json"),
        "selection_preimage_path": directory / "selection_preimage.json",
        "preimage_manifest_path": (directory / "selector_preimage_manifest.json"),
    }
    if (
        directory.name != "minimal320_selector_preimages_v2"
        or directory != overlay.directory.absolute()
        or any(
            Path(getattr(preimages, field)).absolute() != expected
            for field, expected in canonical_paths.items()
        )
    ):
        raise batch.AdapterError("minimal320_preimage_canonical_path_drift")
    candidate_schema, candidate_schema_sha256 = _load_schema_physical(
        candidate_schema_path
    )
    _preimage_schema, preimage_schema_sha256 = _load_schema_physical(
        preimage_schema_path
    )

    def load_json_artifact(
        path: Path,
        *,
        reason: str,
    ) -> tuple[dict[str, Any], str]:
        raw = _physical_read(
            path,
            stop=directory,
            reason=reason,
        )
        digest = _authenticate_sha256_sidecar(
            path,
            raw,
            stop=directory,
            reason=reason,
        )
        value = closed._strict_json(raw, reason=reason)
        if not isinstance(value, dict):
            raise batch.AdapterError(reason)
        return value, digest

    manifest, _manifest_sha256 = load_json_artifact(
        preimages.preimage_manifest_path,
        reason="minimal320_preimage_manifest_snapshot_drift",
    )
    _validate_closed_schema(
        manifest,
        schema_path=preimage_schema_path,
        reason="minimal320_preimage_manifest_schema_mismatch",
    )
    if (
        manifest != preimages.manifest
        or manifest["contract_sha256"] != contract_sha256
        or manifest["source_identity_sha256"] != inventory.source_identity_sha256
        or manifest["preimage_schema_sha256"] != preimage_schema_sha256
        or manifest["candidate_schema_sha256"] != candidate_schema_sha256
    ):
        raise batch.AdapterError("minimal320_preimage_manifest_binding_drift")
    candidate_manifest, candidate_manifest_sha256 = load_json_artifact(
        preimages.candidate_manifest_path,
        reason="minimal320_candidate_manifest_snapshot_drift",
    )
    _validate_closed_schema(
        candidate_manifest,
        schema_path=preimage_schema_path,
        reason="minimal320_candidate_manifest_schema_mismatch",
    )
    selection_manifest, selection_manifest_sha256 = load_json_artifact(
        preimages.selection_manifest_path,
        reason="minimal320_selection_manifest_snapshot_drift",
    )
    selection_preimage, selection_preimage_sha256 = load_json_artifact(
        preimages.selection_preimage_path,
        reason="minimal320_selection_preimage_snapshot_drift",
    )
    _validate_closed_schema(
        selection_preimage,
        schema_path=preimage_schema_path,
        reason="minimal320_selection_preimage_schema_mismatch",
    )
    overlay_manifest_raw = _physical_read(
        overlay.manifest_path,
        stop=directory,
        reason="minimal320_overlay_manifest_snapshot_drift",
    )
    overlay_manifest_sha256 = hashlib.sha256(overlay_manifest_raw).hexdigest()
    if (
        manifest["candidate_manifest_sha256"] != candidate_manifest_sha256
        or manifest["selection_manifest_sha256"] != selection_manifest_sha256
        or manifest["selection_preimage_sha256"] != selection_preimage_sha256
        or manifest["overlay_manifest_sha256"] != overlay_manifest_sha256
        or candidate_manifest["overlay_manifest_sha256"] != overlay_manifest_sha256
        or candidate_manifest["overlay_binding_sha256"]
        != overlay.manifest["overlay_binding_sha256"]
        or candidate_manifest["identity_derivative_plan_sha256"]
        != overlay.manifest["identity_derivative_plan_sha256"]
        or selection_preimage["identity_derivative_plan_sha256"]
        != overlay.manifest["identity_derivative_plan_sha256"]
        or selection_preimage["candidate_manifest_sha256"] != candidate_manifest_sha256
        or selection_preimage["selection_manifest_sha256"] != selection_manifest_sha256
    ):
        raise batch.AdapterError("minimal320_preimage_cross_artifact_binding_drift")
    candidates = _load_candidate_jsonl(
        preimages.candidate_rows_path,
        stop=directory,
        expected_count=int(candidate_manifest["candidate_count"]),
        expected_sha256=str(candidate_manifest["candidate_rows_sha256"]),
        expected_root_sha256=str(candidate_manifest["candidate_set_root_sha256"]),
        candidate_schema=candidate_schema,
        require_unique_idempotency=False,
    )
    selected_rows = _load_candidate_jsonl(
        preimages.selected_rows_path,
        stop=directory,
        expected_count=int(selection_preimage["selected_count"]),
        expected_sha256=str(selection_preimage["selected_rows_sha256"]),
        expected_root_sha256=str(selection_preimage["selected_set_root_sha256"]),
        candidate_schema=candidate_schema,
        require_unique_idempotency=True,
    )
    expected_selection = select_minimal_320(
        candidates,
        contract_sha256=contract_sha256,
        source_identity_sha256=inventory.source_identity_sha256,
    )
    if (
        tuple(item.as_dict() for item in selected_rows)
        != tuple(item.as_dict() for item in expected_selection.candidates)
        or selection_manifest != expected_selection.manifest
        or selection_preimage["selector_identity_sha256"]
        != expected_selection.manifest["selector_identity_sha256"]
        or manifest["candidate_set_root_sha256"]
        != candidate_manifest["candidate_set_root_sha256"]
        or manifest["selected_set_root_sha256"]
        != expected_selection.manifest["selected_set_root_sha256"]
        or manifest["selector_identity_sha256"]
        != expected_selection.manifest["selector_identity_sha256"]
        or manifest["candidate_count"] != len(candidates)
        or manifest["selected_count"] != len(selected_rows)
    ):
        raise batch.AdapterError("minimal320_preimage_rederived_selection_drift")
    return candidates, expected_selection


def assess_legacy_reuse(
    selection: Minimal320Selection,
    *,
    evidence: ReuseClosureEvidence,
    authenticated_idempotency_keys: Sequence[str],
) -> dict[str, Any]:
    """Record closure evidence while always quarantining the historical partial."""

    selected_keys = {item.idempotency_key for item in selection.candidates}
    authenticated = set(authenticated_idempotency_keys)
    closures = {
        "full_wal_chain_and_sequence": evidence.full_wal_chain_and_sequence,
        "runtime_hmac_receipts": evidence.runtime_hmac_receipts,
        "source_identity_and_heldout": evidence.source_identity_and_heldout,
        "bundle_membership_and_selector_metadata": (
            evidence.bundle_membership_and_selector_metadata
        ),
        "uncertain_zero": evidence.uncertain_zero,
        "selected_keys_authenticated": selected_keys <= authenticated,
    }
    closure_complete = all(closures.values())
    return {
        "schema_version": (
            "anchor.gemma3-chat-unbalanced-v2.teacher-alignment-minimal320."
            "legacy-reuse-decision.v1"
        ),
        "decision": "fresh_vnext_320_required",
        "legacy_disposition": "historical_quarantine_only",
        "eligible": False,
        "closure_complete_for_forensics": closure_complete,
        "selected_records": len(selected_keys),
        "closures": closures,
        "content_retained": False,
        "raw_token_ids_retained": False,
        "credential_retained": False,
    }
