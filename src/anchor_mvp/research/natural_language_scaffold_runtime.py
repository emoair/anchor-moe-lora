"""Fail-closed runtime gates for a two-request natural-language scaffold.

The module is intentionally a small control-plane primitive.  It owns no
model, tensor, tokenizer, network client, or provider state.  A consumer may
use it immediately before its backend call to enforce four boundaries:

* a planner request always uses the base model;
* one validated scaffold may activate one adapter on the next request only;
* planner-private KV capabilities never cross a request boundary; and
* experiment-arm labels never substitute for adapter artifact attestation.

Token positions are also forbidden while tokenizer identity is unbound.  In
particular, synthetic token counts are never promoted into token indices.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol, TypeAlias, runtime_checkable

from anchor_mvp.research.hierarchical_kv import AdapterPlacement


NATURAL_LANGUAGE_SCAFFOLD_RUNTIME_VERSION = (
    "anchor.natural-language-scaffold-runtime.v1"
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PRIVATE_KV_KEYS = frozenset(
    {
        "capability",
        "kv_capability",
        "kv_handle",
        "lease_id",
        "private_branch_id",
        "private_page_digest",
    }
)
_TOKEN_COORDINATE_KEYS = frozenset(
    {
        "end_token",
        "input_ids",
        "invocation_token_ids",
        "position_ids",
        "start_token",
        "token_id",
        "token_ids",
        "token_index",
        "token_indices",
    }
)
_DECOUPLED_PLACEMENTS = frozenset(
    {
        AdapterPlacement.DECOUPLED_FROZEN_FACT_READOUT,
        AdapterPlacement.CROSS_ATTENTION_QUERY,
    }
)
_RECEIPT_CAPABILITY = object()


class ScaffoldRuntimeError(ValueError):
    """A request failed a control-plane gate before backend execution."""


class AdapterProfileLabel(str, Enum):
    """Experiment labels; these values are never execution capabilities."""

    Q_ONLY = "q_only"
    Q_PLUS_O = "q_plus_o"
    WIDE_LORA = "wide_lora"


@dataclass(frozen=True)
class TokenizerBinding:
    """All-or-nothing tokenizer identity for token-coordinate materialization."""

    tokenizer_sha256: str | None = None
    chat_template_sha256: str | None = None
    trigger_text_sha256: str | None = None
    ordered_token_ids_sha256: str | None = None

    def __post_init__(self) -> None:
        values = (
            self.tokenizer_sha256,
            self.chat_template_sha256,
            self.trigger_text_sha256,
            self.ordered_token_ids_sha256,
        )
        if all(value is None for value in values):
            return
        if any(value is None for value in values):
            raise ScaffoldRuntimeError(
                "tokenizer identity must be entirely bound or entirely unbound"
            )
        for name, value in (
            ("tokenizer_sha256", self.tokenizer_sha256),
            ("chat_template_sha256", self.chat_template_sha256),
            ("trigger_text_sha256", self.trigger_text_sha256),
            ("ordered_token_ids_sha256", self.ordered_token_ids_sha256),
        ):
            _require_sha256(value, name)

    @property
    def is_bound(self) -> bool:
        return self.tokenizer_sha256 is not None

    @classmethod
    def unbound(cls) -> TokenizerBinding:
        return cls()


@dataclass(frozen=True)
class _ArtifactFileSnapshot:
    path: Path
    sha256: str
    identity: tuple[int, int, int, int]

    def verify_current(self) -> None:
        _data, current = _read_stable_file(
            self.path,
            "verified adapter artifact",
        )
        if current.identity != self.identity or current.sha256 != self.sha256:
            raise ScaffoldRuntimeError(
                "verified adapter artifact bytes/identity changed after verification"
            )


@dataclass(frozen=True, init=False)
class AdapterArtifactVerificationReceipt:
    """Evidence emitted by a trusted artifact/release-lock verifier.

    Merely copying hashes into :class:`AdapterAttestation` is not sufficient to
    execute an adapter.  Until a real verifier has authenticated the artifact
    bytes, tensor inventory, and release lock, no instance can be constructed.
    The private capability prevents callers from promoting booleans into trust.
    """

    adapter_id: str
    adapter_sha256: str
    tensor_inventory_sha256: str
    base_revision_sha256: str
    converter_sha256: str
    release_lock_sha256: str
    target_modules: tuple[str, ...]
    rank: int
    alpha: float
    placement: AdapterPlacement
    _snapshots: tuple[_ArtifactFileSnapshot, ...] = field(repr=False)
    _capability: object = field(repr=False)

    def __init__(
        self,
        *,
        adapter_id: str,
        adapter_sha256: str,
        tensor_inventory_sha256: str,
        base_revision_sha256: str,
        converter_sha256: str,
        release_lock_sha256: str,
        target_modules: tuple[str, ...],
        rank: int,
        alpha: float,
        placement: AdapterPlacement,
        snapshots: tuple[_ArtifactFileSnapshot, ...],
        _capability: object,
    ) -> None:
        if _capability is not _RECEIPT_CAPABILITY:
            raise ScaffoldRuntimeError(
                "adapter verification receipts may only be created by verifier"
            )
        for name, value in locals().items():
            if name not in {"self", "_capability", "snapshots"}:
                object.__setattr__(self, name, value)
        object.__setattr__(self, "_snapshots", snapshots)
        object.__setattr__(self, "_capability", _capability)

    def validate_for(
        self,
        attestation: AdapterAttestation,
        *,
        expected_release_lock_sha256: str,
    ) -> None:
        expected = (
            attestation.adapter_id,
            attestation.adapter_sha256,
            attestation.tensor_inventory_sha256,
            attestation.base_revision_sha256,
            attestation.converter_sha256,
            attestation.target_modules,
            attestation.rank,
            float(attestation.alpha),
            attestation.placement,
        )
        observed = (
            self.adapter_id,
            self.adapter_sha256,
            self.tensor_inventory_sha256,
            self.base_revision_sha256,
            self.converter_sha256,
            self.target_modules,
            self.rank,
            float(self.alpha),
            self.placement,
        )
        if observed != expected:
            raise ScaffoldRuntimeError(
                "adapter verification receipt does not match attestation"
            )
        if self.release_lock_sha256 != expected_release_lock_sha256:
            raise ScaffoldRuntimeError(
                "adapter release lock does not match runtime authority"
            )
        for snapshot in self._snapshots:
            snapshot.verify_current()


def verify_adapter_artifact_files(
    *,
    adapter_path: str | Path,
    tensor_inventory_path: str | Path,
    release_lock_path: str | Path,
    expected_release_lock_sha256: str,
) -> AdapterArtifactVerificationReceipt:
    """Authenticate three stable physical files and mint an opaque receipt."""

    _require_sha256(expected_release_lock_sha256, "expected_release_lock_sha256")
    adapter_bytes, adapter_snapshot = _read_stable_file(
        Path(adapter_path),
        "adapter artifact",
    )
    inventory_bytes, inventory_snapshot = _read_stable_file(
        Path(tensor_inventory_path),
        "tensor inventory",
    )
    release_bytes, release_snapshot = _read_stable_file(
        Path(release_lock_path),
        "release lock",
    )
    if release_snapshot.sha256 != expected_release_lock_sha256:
        raise ScaffoldRuntimeError("release lock physical SHA-256 mismatch")
    try:
        inventory = json.loads(inventory_bytes.decode("utf-8"))
        release = json.loads(release_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ScaffoldRuntimeError(
            "adapter inventory/release lock must be UTF-8 JSON"
        ) from exc
    if not isinstance(inventory, Mapping) or not isinstance(release, Mapping):
        raise ScaffoldRuntimeError("adapter inventory/release lock must be objects")
    inventory_keys = {
        "schema_version",
        "adapter_id",
        "adapter_sha256",
        "target_modules",
        "rank",
        "alpha",
        "base_revision_sha256",
        "converter_sha256",
        "placement",
    }
    release_keys = inventory_keys | {
        "tensor_inventory_sha256",
        "execution_authorized",
    }
    if set(inventory) != inventory_keys or set(release) != release_keys:
        raise ScaffoldRuntimeError("adapter inventory/release lock shape is not closed")
    if inventory.get("schema_version") != "anchor.adapter-tensor-inventory-receipt.v1":
        raise ScaffoldRuntimeError("adapter tensor inventory schema version invalid")
    if release.get("schema_version") != "anchor.adapter-release-lock-receipt.v1":
        raise ScaffoldRuntimeError("adapter release lock schema version invalid")
    if release.get("execution_authorized") is not True:
        raise ScaffoldRuntimeError("adapter release lock does not authorize execution")
    adapter_sha256 = hashlib.sha256(adapter_bytes).hexdigest()
    inventory_sha256 = hashlib.sha256(inventory_bytes).hexdigest()
    shared_fields = inventory_keys - {"schema_version"}
    if (
        inventory.get("adapter_sha256") != adapter_sha256
        or release.get("adapter_sha256") != adapter_sha256
        or release.get("tensor_inventory_sha256") != inventory_sha256
        or any(release.get(key) != inventory.get(key) for key in shared_fields)
    ):
        raise ScaffoldRuntimeError(
            "adapter bytes, inventory, and release lock are not cross-bound"
        )
    modules = inventory.get("target_modules")
    if not isinstance(modules, list):
        raise ScaffoldRuntimeError("target_modules must be a JSON array")
    try:
        placement = AdapterPlacement(str(inventory["placement"]))
    except ValueError as exc:
        raise ScaffoldRuntimeError("adapter placement is invalid") from exc
    adapter_id = _require_identifier(inventory.get("adapter_id"), "adapter_id")
    base_revision_sha256 = _require_sha256(
        inventory.get("base_revision_sha256"),
        "base_revision_sha256",
    )
    converter_sha256 = _require_sha256(
        inventory.get("converter_sha256"),
        "converter_sha256",
    )
    rank = inventory.get("rank")
    alpha = inventory.get("alpha")
    attestation_probe = AdapterAttestation(
        adapter_id=adapter_id,
        adapter_sha256=adapter_sha256,
        tensor_inventory_sha256=inventory_sha256,
        target_modules=tuple(modules),
        rank=rank,
        alpha=alpha,
        base_revision_sha256=base_revision_sha256,
        converter_sha256=converter_sha256,
        placement=placement,
    )
    return AdapterArtifactVerificationReceipt(
        adapter_id=attestation_probe.adapter_id,
        adapter_sha256=attestation_probe.adapter_sha256,
        tensor_inventory_sha256=attestation_probe.tensor_inventory_sha256,
        base_revision_sha256=attestation_probe.base_revision_sha256,
        converter_sha256=attestation_probe.converter_sha256,
        release_lock_sha256=release_snapshot.sha256,
        target_modules=attestation_probe.target_modules,
        rank=attestation_probe.rank,
        alpha=attestation_probe.alpha,
        placement=attestation_probe.placement,
        snapshots=(adapter_snapshot, inventory_snapshot, release_snapshot),
        _capability=_RECEIPT_CAPABILITY,
    )


@dataclass(frozen=True)
class AdapterAttestation:
    """Authenticated facts about one concrete adapter artifact."""

    adapter_id: str
    adapter_sha256: str
    tensor_inventory_sha256: str
    target_modules: tuple[str, ...]
    rank: int
    alpha: float
    base_revision_sha256: str
    converter_sha256: str
    placement: AdapterPlacement
    verification_receipt: AdapterArtifactVerificationReceipt | None = field(
        default=None,
        repr=False,
    )

    def __post_init__(self) -> None:
        _require_identifier(self.adapter_id, "adapter_id")
        for name, value in (
            ("adapter_sha256", self.adapter_sha256),
            ("tensor_inventory_sha256", self.tensor_inventory_sha256),
            ("base_revision_sha256", self.base_revision_sha256),
            ("converter_sha256", self.converter_sha256),
        ):
            _require_sha256(value, name)
        if not isinstance(self.target_modules, tuple) or not self.target_modules:
            raise ScaffoldRuntimeError("target_modules must be a non-empty tuple")
        if any(
            not isinstance(module, str) or not module.strip()
            for module in self.target_modules
        ):
            raise ScaffoldRuntimeError("target_modules must be non-empty strings")
        if len(set(self.target_modules)) != len(self.target_modules):
            raise ScaffoldRuntimeError("target_modules must not contain duplicates")
        if (
            isinstance(self.rank, bool)
            or not isinstance(self.rank, int)
            or self.rank < 1
        ):
            raise ScaffoldRuntimeError("rank must be a positive integer")
        if (
            isinstance(self.alpha, bool)
            or not isinstance(self.alpha, (int, float))
            or not math.isfinite(float(self.alpha))
            or self.alpha <= 0
        ):
            raise ScaffoldRuntimeError("alpha must be a positive finite number")
        if not isinstance(self.placement, AdapterPlacement):
            raise ScaffoldRuntimeError("placement must be an AdapterPlacement")
        if self.verification_receipt is not None and not isinstance(
            self.verification_receipt,
            AdapterArtifactVerificationReceipt,
        ):
            raise ScaffoldRuntimeError(
                "verification_receipt must be an AdapterArtifactVerificationReceipt"
            )


@dataclass(frozen=True)
class AdapterSelection:
    """A profile label paired with the artifact evidence needed to execute it."""

    profile_label: AdapterProfileLabel
    attestation: AdapterAttestation | None
    exact_shared_kv: bool = False

    def validate(
        self,
        *,
        expected_release_lock_sha256: str | None = None,
    ) -> AdapterAttestation:
        if not isinstance(self.profile_label, AdapterProfileLabel):
            raise ScaffoldRuntimeError("profile_label must be an AdapterProfileLabel")
        if self.attestation is None:
            raise ScaffoldRuntimeError(
                "adapter profile label is not adapter artifact attestation"
            )
        modules = frozenset(self.attestation.target_modules)
        if self.profile_label is AdapterProfileLabel.Q_ONLY:
            expected = frozenset({"q_proj"})
            if modules != expected:
                raise ScaffoldRuntimeError(
                    "q_only label does not match attested target modules"
                )
        elif self.profile_label is AdapterProfileLabel.Q_PLUS_O:
            expected = frozenset({"q_proj", "o_proj"})
            if modules != expected:
                raise ScaffoldRuntimeError(
                    "q_plus_o label does not match attested target modules"
                )
        elif self.profile_label is AdapterProfileLabel.WIDE_LORA:
            if "q_proj" not in modules or len(modules) < 3:
                raise ScaffoldRuntimeError(
                    "wide_lora label requires an attested explicit wide target inventory"
                )
        else:  # pragma: no cover - Enum currently makes this unreachable
            raise ScaffoldRuntimeError("unsupported adapter profile label")

        if self.exact_shared_kv:
            if self.profile_label is not AdapterProfileLabel.Q_ONLY:
                raise ScaffoldRuntimeError(
                    "exact shared KV is unavailable to q_plus_o and wide_lora profiles"
                )
            if self.attestation.placement not in _DECOUPLED_PLACEMENTS:
                raise ScaffoldRuntimeError(
                    "q_only inside a decoder KV producer is not exact-share safe"
                )
        if self.attestation.verification_receipt is None:
            raise ScaffoldRuntimeError(
                "adapter attestation has no verified artifact/release-lock receipt"
            )
        if expected_release_lock_sha256 is None:
            raise ScaffoldRuntimeError(
                "runtime release-lock authority is required for adapter execution"
            )
        _require_sha256(
            expected_release_lock_sha256,
            "expected_release_lock_sha256",
        )
        self.attestation.verification_receipt.validate_for(
            self.attestation,
            expected_release_lock_sha256=expected_release_lock_sha256,
        )
        return self.attestation


@dataclass(frozen=True)
class PlannerScaffold:
    """Public planner output that may arm exactly one subsequent request."""

    task_bundle_sha256: str
    target_expert_id: str
    natural_language_scaffold: str
    trigger_text: str
    trigger_text_sha256: str
    structured_plan: Mapping[str, Any]

    def __post_init__(self) -> None:
        _require_sha256(self.task_bundle_sha256, "task_bundle_sha256")
        _require_identifier(self.target_expert_id, "target_expert_id")
        _require_bounded_text(
            self.natural_language_scaffold,
            "natural_language_scaffold",
            maximum=262_144,
        )
        _require_bounded_text(self.trigger_text, "trigger_text", maximum=8_192)
        expected_trigger_sha = hashlib.sha256(
            self.trigger_text.encode("utf-8")
        ).hexdigest()
        if self.trigger_text_sha256 != expected_trigger_sha:
            raise ScaffoldRuntimeError(
                "trigger_text_sha256 does not authenticate trigger_text"
            )
        if not isinstance(self.structured_plan, Mapping):
            raise ScaffoldRuntimeError("structured_plan must be a mapping")
        forbidden = _find_forbidden_keys(self.structured_plan, _PRIVATE_KV_KEYS)
        if forbidden:
            raise ScaffoldRuntimeError(
                "planner scaffold contains private KV control fields: "
                + ", ".join(sorted(forbidden))
            )
        object.__setattr__(self, "structured_plan", _freeze_json(self.structured_plan))

    @property
    def committed_scaffold_sha256(self) -> str:
        return hashlib.sha256(
            self.natural_language_scaffold.encode("utf-8")
        ).hexdigest()


@dataclass(frozen=True, init=False)
class ScaffoldReencodeReceipt:
    """Adapter-off re-encode and input-token scan evidence for committed R1 text."""

    planner_request_id: str
    task_bundle_sha256: str
    committed_scaffold_sha256: str
    trigger_text_sha256: str
    new_request_input_sha256: str
    tokenizer_sha256: str
    chat_template_sha256: str
    ordered_token_ids_sha256: str
    base_revision_sha256: str
    trigger_occurrences: int
    adapter_state: str = "off"
    _capability: object = field(repr=False)

    def __init__(
        self,
        *,
        planner_request_id: str,
        task_bundle_sha256: str,
        committed_scaffold_sha256: str,
        trigger_text_sha256: str,
        new_request_input_sha256: str,
        tokenizer_sha256: str,
        chat_template_sha256: str,
        ordered_token_ids_sha256: str,
        base_revision_sha256: str,
        trigger_occurrences: int,
        adapter_state: str,
        _capability: object,
    ) -> None:
        if _capability is not _RECEIPT_CAPABILITY:
            raise ScaffoldRuntimeError(
                "scaffold re-encode receipts may only be created by verifier"
            )
        for name, value in locals().items():
            if name not in {"self", "_capability"}:
                object.__setattr__(self, name, value)
        object.__setattr__(self, "_capability", _capability)
        _require_identifier(self.planner_request_id, "planner_request_id")
        for name, value in (
            ("task_bundle_sha256", self.task_bundle_sha256),
            ("committed_scaffold_sha256", self.committed_scaffold_sha256),
            ("trigger_text_sha256", self.trigger_text_sha256),
            ("new_request_input_sha256", self.new_request_input_sha256),
            ("tokenizer_sha256", self.tokenizer_sha256),
            ("chat_template_sha256", self.chat_template_sha256),
            ("ordered_token_ids_sha256", self.ordered_token_ids_sha256),
            ("base_revision_sha256", self.base_revision_sha256),
        ):
            _require_sha256(value, name)
        if isinstance(self.trigger_occurrences, bool) or not isinstance(
            self.trigger_occurrences,
            int,
        ):
            raise ScaffoldRuntimeError("trigger_occurrences must be an integer")

    def validate_commit(
        self,
        *,
        scaffold: PlannerScaffold,
        planner_request_id: str,
        base_revision_sha256: str,
    ) -> None:
        if (
            self.planner_request_id != planner_request_id
            or self.task_bundle_sha256 != scaffold.task_bundle_sha256
            or self.committed_scaffold_sha256 != scaffold.committed_scaffold_sha256
            or self.trigger_text_sha256 != scaffold.trigger_text_sha256
            or self.base_revision_sha256 != base_revision_sha256
            or self.adapter_state != "off"
        ):
            raise ScaffoldRuntimeError(
                "scaffold re-encode receipt does not match planner lineage"
            )
        if self._capability is not _RECEIPT_CAPABILITY or self.trigger_occurrences != 1:
            raise ScaffoldRuntimeError(
                "scaffold re-encode/token scan is not runtime verified"
            )

    def validate_dispatch(
        self,
        *,
        scaffold: PlannerScaffold,
        prompt: str,
        tokenizer_binding: TokenizerBinding,
        token_materialization: Mapping[str, Any] | None,
    ) -> None:
        if not tokenizer_binding.is_bound:
            raise ScaffoldRuntimeError(
                "expert activation requires a bound tokenizer identity"
            )
        if (
            tokenizer_binding.tokenizer_sha256 != self.tokenizer_sha256
            or tokenizer_binding.chat_template_sha256 != self.chat_template_sha256
            or tokenizer_binding.trigger_text_sha256 != self.trigger_text_sha256
            or tokenizer_binding.ordered_token_ids_sha256
            != self.ordered_token_ids_sha256
        ):
            raise ScaffoldRuntimeError(
                "tokenizer identity does not match scaffold re-encode receipt"
            )
        if hashlib.sha256(prompt.encode("utf-8")).hexdigest() != (
            self.new_request_input_sha256
        ):
            raise ScaffoldRuntimeError(
                "new request input does not match scaffold re-encode receipt"
            )
        if (
            scaffold.natural_language_scaffold not in prompt
            or prompt.count(scaffold.trigger_text) != 1
        ):
            raise ScaffoldRuntimeError(
                "committed scaffold and exactly one trigger must be in new request input"
            )
        if not isinstance(token_materialization, Mapping):
            raise ScaffoldRuntimeError(
                "expert activation requires token materialization evidence"
            )
        expected_materialization = {
            "new_request_input_sha256": self.new_request_input_sha256,
            "ordered_token_ids_sha256": self.ordered_token_ids_sha256,
            "trigger_text_sha256": self.trigger_text_sha256,
        }
        if any(
            token_materialization.get(key) != value
            for key, value in expected_materialization.items()
        ):
            raise ScaffoldRuntimeError(
                "token materialization does not match scaffold re-encode receipt"
            )


def verify_scaffold_reencode(
    *,
    planner_request_id: str,
    scaffold: PlannerScaffold,
    prompt: str,
    tokenizer_binding: TokenizerBinding,
    ordered_token_ids: Sequence[int],
    base_revision_sha256: str,
) -> ScaffoldReencodeReceipt:
    """Scan the exact new input and mint an opaque adapter-off receipt."""

    _require_identifier(planner_request_id, "planner_request_id")
    _require_bounded_text(prompt, "prompt", maximum=2_000_000)
    _require_sha256(base_revision_sha256, "base_revision_sha256")
    if not isinstance(tokenizer_binding, TokenizerBinding) or not (
        tokenizer_binding.is_bound
    ):
        raise ScaffoldRuntimeError(
            "scaffold re-encode verification requires a bound tokenizer"
        )
    if isinstance(ordered_token_ids, (str, bytes, bytearray)) or not isinstance(
        ordered_token_ids,
        Sequence,
    ):
        raise ScaffoldRuntimeError("ordered_token_ids must be an integer sequence")
    token_ids = tuple(ordered_token_ids)
    if not token_ids or any(
        isinstance(token_id, bool) or not isinstance(token_id, int) or token_id < 0
        for token_id in token_ids
    ):
        raise ScaffoldRuntimeError(
            "ordered_token_ids must contain non-negative integers"
        )
    token_ids_sha256 = hashlib.sha256(
        json.dumps(list(token_ids), separators=(",", ":")).encode("ascii")
    ).hexdigest()
    if tokenizer_binding.ordered_token_ids_sha256 != token_ids_sha256:
        raise ScaffoldRuntimeError("ordered token IDs do not match tokenizer binding")
    if tokenizer_binding.trigger_text_sha256 != scaffold.trigger_text_sha256:
        raise ScaffoldRuntimeError("tokenizer trigger identity mismatch")
    trigger_occurrences = prompt.count(scaffold.trigger_text)
    if scaffold.natural_language_scaffold not in prompt or trigger_occurrences != 1:
        raise ScaffoldRuntimeError(
            "committed scaffold and exactly one trigger must be in new request input"
        )
    return ScaffoldReencodeReceipt(
        planner_request_id=planner_request_id,
        task_bundle_sha256=scaffold.task_bundle_sha256,
        committed_scaffold_sha256=scaffold.committed_scaffold_sha256,
        trigger_text_sha256=scaffold.trigger_text_sha256,
        new_request_input_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        tokenizer_sha256=str(tokenizer_binding.tokenizer_sha256),
        chat_template_sha256=str(tokenizer_binding.chat_template_sha256),
        ordered_token_ids_sha256=token_ids_sha256,
        base_revision_sha256=base_revision_sha256,
        trigger_occurrences=trigger_occurrences,
        adapter_state="off",
        _capability=_RECEIPT_CAPABILITY,
    )


@dataclass(frozen=True)
class PrivateKVLease:
    """One request-local planner capability, never part of a scaffold."""

    owner_request_id: str
    task_bundle_sha256: str
    private_branch_id: str
    capability: str

    def __post_init__(self) -> None:
        _require_identifier(self.owner_request_id, "owner_request_id")
        _require_sha256(self.task_bundle_sha256, "task_bundle_sha256")
        _require_identifier(self.private_branch_id, "private_branch_id")
        _require_identifier(self.capability, "capability")


@dataclass(frozen=True)
class BackendRequest:
    """Validated request passed to a model backend only after every gate passes."""

    request_id: str
    task_bundle_sha256: str
    role: str
    prompt: str
    requested_expert_id: str | None
    adapter_id: str | None
    adapter_sha256: str | None
    profile_label: str | None
    private_kv: PrivateKVLease | None = field(repr=False)
    schema_version: str = NATURAL_LANGUAGE_SCAFFOLD_RUNTIME_VERSION


BackendResult: TypeAlias = Any


@runtime_checkable
class ScaffoldBackend(Protocol):
    def __call__(self, request: BackendRequest) -> BackendResult:
        """Execute one already-validated request."""


@dataclass(frozen=True)
class _PendingActivation:
    scaffold: PlannerScaffold
    selection: AdapterSelection
    reencode_receipt: ScaffoldReencodeReceipt


@dataclass(frozen=True)
class _PlannerLineage:
    request_id: str
    task_bundle_sha256: str


class NaturalLanguageScaffoldRuntime:
    """Single-slot, next-request-only adapter activation gate."""

    def __init__(
        self,
        backend: ScaffoldBackend | Callable[[BackendRequest], Any],
        *,
        base_revision_sha256: str,
        expected_release_lock_sha256: str,
    ) -> None:
        if not callable(backend):
            raise TypeError("backend must be callable")
        _require_sha256(base_revision_sha256, "base_revision_sha256")
        _require_sha256(
            expected_release_lock_sha256,
            "expected_release_lock_sha256",
        )
        self._backend = backend
        self._base_revision_sha256 = base_revision_sha256
        self._expected_release_lock_sha256 = expected_release_lock_sha256
        self._pending: _PendingActivation | None = None
        self._planner_lineage: _PlannerLineage | None = None

    @property
    def slot_is_armed(self) -> bool:
        return self._pending is not None

    def reset_slot(self) -> None:
        """Clear an unused activation without executing it."""

        self._pending = None
        self._planner_lineage = None

    def dispatch_planner(
        self,
        *,
        request_id: str,
        task_bundle_sha256: str,
        prompt: str,
        tokenizer_binding: TokenizerBinding,
        token_materialization: Mapping[str, Any] | None = None,
        private_kv: PrivateKVLease | None = None,
    ) -> BackendResult:
        """Dispatch R1 on the base model; it can never activate an adapter."""

        if self._pending is not None:
            raise ScaffoldRuntimeError(
                "cannot dispatch a planner while a next-request slot is armed"
            )
        if self._planner_lineage is not None:
            raise ScaffoldRuntimeError(
                "previous planner output must be committed or reset first"
            )
        _validate_common_request(request_id, task_bundle_sha256, prompt)
        _validate_token_materialization(tokenizer_binding, token_materialization)
        if private_kv is not None:
            if private_kv.owner_request_id != request_id:
                raise ScaffoldRuntimeError("planner private KV owner mismatch")
            if private_kv.task_bundle_sha256 != task_bundle_sha256:
                raise ScaffoldRuntimeError("planner private KV task mismatch")
        request = BackendRequest(
            request_id=request_id,
            task_bundle_sha256=task_bundle_sha256,
            role="planner",
            prompt=prompt,
            requested_expert_id=None,
            adapter_id=None,
            adapter_sha256=None,
            profile_label=None,
            private_kv=private_kv,
        )
        result = self._backend(request)
        self._planner_lineage = _PlannerLineage(
            request_id=request_id,
            task_bundle_sha256=task_bundle_sha256,
        )
        return result

    def arm_next_request(
        self,
        *,
        scaffold: PlannerScaffold,
        selection: AdapterSelection,
        reencode_receipt: ScaffoldReencodeReceipt,
    ) -> None:
        """Validate public R1 output and arm one, and only one, later call."""

        if self._pending is not None:
            raise ScaffoldRuntimeError("next-request adapter slot is already armed")
        lineage = self._planner_lineage
        if lineage is None:
            raise ScaffoldRuntimeError(
                "next-request activation requires a successful planner request"
            )
        if lineage.task_bundle_sha256 != scaffold.task_bundle_sha256:
            raise ScaffoldRuntimeError("planner/scaffold task bundle mismatch")
        if not isinstance(reencode_receipt, ScaffoldReencodeReceipt):
            raise ScaffoldRuntimeError(
                "reencode_receipt must be a ScaffoldReencodeReceipt"
            )
        selection.validate(
            expected_release_lock_sha256=self._expected_release_lock_sha256
        )
        reencode_receipt.validate_commit(
            scaffold=scaffold,
            planner_request_id=lineage.request_id,
            base_revision_sha256=self._base_revision_sha256,
        )
        self._pending = _PendingActivation(
            scaffold=scaffold,
            selection=selection,
            reencode_receipt=reencode_receipt,
        )
        self._planner_lineage = None

    def dispatch_next_request(
        self,
        *,
        request_id: str,
        task_bundle_sha256: str,
        prompt: str,
        tokenizer_binding: TokenizerBinding,
        requested_expert_id: str | None = None,
        token_materialization: Mapping[str, Any] | None = None,
        private_kv: PrivateKVLease | None = None,
    ) -> BackendResult:
        """Consume the slot before validating R2, so failures cannot leak to R3."""

        pending = self._pending
        self._pending = None

        _validate_common_request(request_id, task_bundle_sha256, prompt)
        if private_kv is not None:
            raise ScaffoldRuntimeError(
                "planner private KV cannot cross a request boundary"
            )
        _validate_token_materialization(tokenizer_binding, token_materialization)

        if pending is None:
            if requested_expert_id is not None:
                raise ScaffoldRuntimeError(
                    "expert request has no validated next-request activation"
                )
            request = BackendRequest(
                request_id=request_id,
                task_bundle_sha256=task_bundle_sha256,
                role="base",
                prompt=prompt,
                requested_expert_id=None,
                adapter_id=None,
                adapter_sha256=None,
                profile_label=None,
                private_kv=None,
            )
            return self._backend(request)

        scaffold = pending.scaffold
        attestation = pending.selection.validate(
            expected_release_lock_sha256=self._expected_release_lock_sha256
        )
        if task_bundle_sha256 != scaffold.task_bundle_sha256:
            raise ScaffoldRuntimeError("next-request task bundle mismatch")
        if requested_expert_id != scaffold.target_expert_id:
            raise ScaffoldRuntimeError("next-request expert mismatch")
        if request_id == pending.reencode_receipt.planner_request_id:
            raise ScaffoldRuntimeError("planner and expert request IDs must differ")
        if attestation.base_revision_sha256 != self._base_revision_sha256:
            raise ScaffoldRuntimeError("adapter/base revision mismatch")
        pending.reencode_receipt.validate_dispatch(
            scaffold=scaffold,
            prompt=prompt,
            tokenizer_binding=tokenizer_binding,
            token_materialization=token_materialization,
        )

        request = BackendRequest(
            request_id=request_id,
            task_bundle_sha256=task_bundle_sha256,
            role="expert",
            prompt=prompt,
            requested_expert_id=requested_expert_id,
            adapter_id=attestation.adapter_id,
            adapter_sha256=attestation.adapter_sha256,
            profile_label=pending.selection.profile_label.value,
            private_kv=None,
        )
        return self._backend(request)


def _validate_common_request(
    request_id: str,
    task_bundle_sha256: str,
    prompt: str,
) -> None:
    _require_identifier(request_id, "request_id")
    _require_sha256(task_bundle_sha256, "task_bundle_sha256")
    _require_bounded_text(prompt, "prompt", maximum=2_000_000)


def _stat_identity(value: os.stat_result) -> tuple[int, int, int, int]:
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns)


def _read_stable_file(
    path: Path,
    label: str,
) -> tuple[bytes, _ArtifactFileSnapshot]:
    try:
        if not path.is_file() or path.is_symlink():
            raise ScaffoldRuntimeError(f"{label} must be a regular non-symlink file")
        with path.open("rb") as handle:
            before = os.fstat(handle.fileno())
            data = handle.read()
            after = os.fstat(handle.fileno())
        current = path.stat()
    except OSError as exc:
        raise ScaffoldRuntimeError(f"{label} could not be authenticated") from exc
    identity = _stat_identity(after)
    if (
        _stat_identity(before) != identity
        or _stat_identity(current) != identity
        or len(data) != after.st_size
        or path.is_symlink()
    ):
        raise ScaffoldRuntimeError(f"{label} changed during authentication")
    snapshot = _ArtifactFileSnapshot(
        path=path.resolve(),
        sha256=hashlib.sha256(data).hexdigest(),
        identity=identity,
    )
    return data, snapshot


def _validate_token_materialization(
    binding: TokenizerBinding,
    materialization: Mapping[str, Any] | None,
) -> None:
    if not isinstance(binding, TokenizerBinding):
        raise ScaffoldRuntimeError("tokenizer_binding must be a TokenizerBinding")
    if materialization is None:
        return
    if not isinstance(materialization, Mapping):
        raise ScaffoldRuntimeError("token_materialization must be a mapping")
    if not binding.is_bound:
        forbidden = _find_forbidden_keys(
            materialization,
            _TOKEN_COORDINATE_KEYS,
        )
        if forbidden:
            raise ScaffoldRuntimeError(
                "tokenizer is unbound; token indices and ids are forbidden: "
                + ", ".join(sorted(forbidden))
            )


def _find_forbidden_keys(value: Any, forbidden: frozenset[str]) -> set[str]:
    found: set[str] = set()
    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            if not isinstance(raw_key, str):
                raise ScaffoldRuntimeError("control-plane mapping keys must be strings")
            normalized = raw_key.strip().lower()
            if normalized in forbidden:
                found.add(normalized)
            found.update(_find_forbidden_keys(nested, forbidden))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested in value:
            found.update(_find_forbidden_keys(nested, forbidden))
    return found


def _freeze_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ScaffoldRuntimeError("structured_plan contains non-finite float")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ScaffoldRuntimeError("structured_plan keys must be strings")
            frozen[key] = _freeze_json(nested)
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json(nested) for nested in value)
    raise ScaffoldRuntimeError("structured_plan must contain JSON-compatible values")


def _require_identifier(value: object, field_name: str) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value.encode("utf-8")) > 1_024
    ):
        raise ScaffoldRuntimeError(f"{field_name} must be a non-empty bounded string")
    return value


def _require_bounded_text(value: object, field_name: str, *, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value.strip()
        or len(value.encode("utf-8")) > maximum
    ):
        raise ScaffoldRuntimeError(f"{field_name} must be non-empty and bounded")
    return value


def _require_sha256(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ScaffoldRuntimeError(
            f"{field_name} must be 64 lowercase hexadecimal characters"
        )
    return value
