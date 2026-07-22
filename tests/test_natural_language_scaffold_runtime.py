from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from anchor_mvp.research.hierarchical_kv import AdapterPlacement
from anchor_mvp.research.natural_language_scaffold_runtime import (
    AdapterArtifactVerificationReceipt,
    AdapterAttestation,
    AdapterProfileLabel,
    AdapterSelection,
    BackendRequest,
    NaturalLanguageScaffoldRuntime,
    PlannerScaffold,
    PrivateKVLease,
    ScaffoldReencodeReceipt,
    ScaffoldRuntimeError,
    TokenizerBinding,
    verify_adapter_artifact_files,
    verify_scaffold_reencode,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class RecordingBackend:
    def __init__(self) -> None:
        self.calls: list[BackendRequest] = []

    def __call__(self, request: BackendRequest) -> dict[str, Any]:
        self.calls.append(request)
        return {"request_id": request.request_id, "role": request.role}


def _scaffold(*, plan: dict[str, Any] | None = None) -> PlannerScaffold:
    trigger = "<|activate:builder|>"
    return PlannerScaffold(
        task_bundle_sha256=_sha("bundle"),
        target_expert_id="builder",
        natural_language_scaffold="Implement the validated calculator plan.",
        trigger_text=trigger,
        trigger_text_sha256=hashlib.sha256(trigger.encode("utf-8")).hexdigest(),
        structured_plan=plan or {"steps": [{"id": "P1", "action": "build"}]},
    )


def _attestation(
    *,
    modules: tuple[str, ...] = ("q_proj",),
    placement: AdapterPlacement = AdapterPlacement.DECOUPLED_FROZEN_FACT_READOUT,
    receipt: AdapterArtifactVerificationReceipt | None = None,
) -> AdapterAttestation:
    return AdapterAttestation(
        adapter_id="adapter-builder-r16",
        adapter_sha256=receipt.adapter_sha256 if receipt else _sha("adapter"),
        tensor_inventory_sha256=(
            receipt.tensor_inventory_sha256 if receipt else _sha("tensor-inventory")
        ),
        target_modules=modules,
        rank=16,
        alpha=32.0,
        base_revision_sha256=_sha("base"),
        converter_sha256=_sha("converter"),
        placement=placement,
        verification_receipt=receipt,
    )


def _selection(
    *,
    attestation: AdapterAttestation | None = None,
    profile: AdapterProfileLabel = AdapterProfileLabel.Q_ONLY,
    exact_shared_kv: bool = True,
) -> AdapterSelection:
    return AdapterSelection(
        profile_label=profile,
        attestation=attestation if attestation is not None else _attestation(),
        exact_shared_kv=exact_shared_kv,
    )


def _runtime(
    backend: RecordingBackend,
    *,
    release_lock_sha256: str = _sha("unavailable-release-lock"),
) -> NaturalLanguageScaffoldRuntime:
    return NaturalLanguageScaffoldRuntime(
        backend,
        base_revision_sha256=_sha("base"),
        expected_release_lock_sha256=release_lock_sha256,
    )


def _bound_tokenizer(scaffold: PlannerScaffold) -> TokenizerBinding:
    return TokenizerBinding(
        tokenizer_sha256=_sha("tokenizer"),
        chat_template_sha256=_sha("chat-template"),
        trigger_text_sha256=scaffold.trigger_text_sha256,
        ordered_token_ids_sha256=hashlib.sha256(b"[1,2,3]").hexdigest(),
    )


def _expert_prompt(scaffold: PlannerScaffold) -> str:
    return f"{scaffold.natural_language_scaffold}\n{scaffold.trigger_text}\nExecute."


def _reencode_receipt(
    scaffold: PlannerScaffold,
    prompt: str,
) -> ScaffoldReencodeReceipt:
    return verify_scaffold_reencode(
        planner_request_id="R1",
        scaffold=scaffold,
        prompt=prompt,
        tokenizer_binding=_bound_tokenizer(scaffold),
        ordered_token_ids=(1, 2, 3),
        base_revision_sha256=_sha("base"),
    )


def _verified_attestation(
    tmp_path: Path,
) -> tuple[AdapterAttestation, str]:
    adapter = tmp_path / "adapter.bin"
    inventory_path = tmp_path / "tensor_inventory.json"
    release_path = tmp_path / "release_lock.json"
    adapter.write_bytes(b"minimal adapter artifact bytes")
    adapter_sha256 = hashlib.sha256(adapter.read_bytes()).hexdigest()
    inventory = {
        "schema_version": "anchor.adapter-tensor-inventory-receipt.v1",
        "adapter_id": "adapter-builder-r16",
        "adapter_sha256": adapter_sha256,
        "target_modules": ["q_proj"],
        "rank": 16,
        "alpha": 32.0,
        "base_revision_sha256": _sha("base"),
        "converter_sha256": _sha("converter"),
        "placement": AdapterPlacement.DECOUPLED_FROZEN_FACT_READOUT.value,
    }
    inventory_path.write_text(
        json.dumps(inventory, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    release = {
        **inventory,
        "schema_version": "anchor.adapter-release-lock-receipt.v1",
        "tensor_inventory_sha256": hashlib.sha256(
            inventory_path.read_bytes()
        ).hexdigest(),
        "execution_authorized": True,
    }
    release_path.write_text(
        json.dumps(release, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    release_sha256 = hashlib.sha256(release_path.read_bytes()).hexdigest()
    receipt = verify_adapter_artifact_files(
        adapter_path=adapter,
        tensor_inventory_path=inventory_path,
        release_lock_path=release_path,
        expected_release_lock_sha256=release_sha256,
    )
    return _attestation(receipt=receipt), release_sha256


def _materialization(receipt: ScaffoldReencodeReceipt) -> dict[str, str]:
    return {
        "new_request_input_sha256": receipt.new_request_input_sha256,
        "ordered_token_ids_sha256": receipt.ordered_token_ids_sha256,
        "trigger_text_sha256": receipt.trigger_text_sha256,
    }


def test_alora_activation_requires_all_verified_receipts_and_is_one_shot(
    tmp_path: Path,
) -> None:
    backend = RecordingBackend()
    attestation, release_sha256 = _verified_attestation(tmp_path)
    runtime = _runtime(backend, release_lock_sha256=release_sha256)
    scaffold = _scaffold()
    binding = _bound_tokenizer(scaffold)
    prompt = _expert_prompt(scaffold)
    receipt = _reencode_receipt(scaffold, prompt)

    runtime.dispatch_planner(
        request_id="R1",
        task_bundle_sha256=_sha("bundle"),
        prompt="Plan the task.",
        tokenizer_binding=TokenizerBinding.unbound(),
    )
    runtime.arm_next_request(
        scaffold=scaffold,
        selection=_selection(attestation=attestation),
        reencode_receipt=receipt,
    )
    assert runtime.slot_is_armed

    runtime.dispatch_next_request(
        request_id="R2",
        task_bundle_sha256=_sha("bundle"),
        prompt=prompt,
        tokenizer_binding=binding,
        requested_expert_id="builder",
        token_materialization=_materialization(receipt),
    )
    assert not runtime.slot_is_armed

    runtime.dispatch_next_request(
        request_id="R3",
        task_bundle_sha256=_sha("bundle"),
        prompt="A new independent request.",
        tokenizer_binding=TokenizerBinding.unbound(),
    )

    assert [call.role for call in backend.calls] == ["planner", "expert", "base"]
    assert [call.adapter_id for call in backend.calls] == [
        None,
        "adapter-builder-r16",
        None,
    ]
    assert [call.profile_label for call in backend.calls] == [
        None,
        "q_only",
        None,
    ]


def test_cannot_arm_without_successful_planner_request() -> None:
    backend = RecordingBackend()
    runtime = _runtime(backend)
    scaffold = _scaffold()
    prompt = _expert_prompt(scaffold)

    with pytest.raises(ScaffoldRuntimeError, match="successful planner"):
        runtime.arm_next_request(
            scaffold=scaffold,
            selection=_selection(),
            reencode_receipt=_reencode_receipt(scaffold, prompt),
        )
    assert backend.calls == []


def test_missing_reencode_receipt_cannot_arm() -> None:
    backend = RecordingBackend()
    runtime = _runtime(backend)
    scaffold = _scaffold()
    runtime.dispatch_planner(
        request_id="R1",
        task_bundle_sha256=scaffold.task_bundle_sha256,
        prompt="Plan.",
        tokenizer_binding=TokenizerBinding.unbound(),
    )
    backend.calls.clear()

    with pytest.raises(ScaffoldRuntimeError, match="must be a ScaffoldReencodeReceipt"):
        runtime.arm_next_request(
            scaffold=scaffold,
            selection=_selection(),
            reencode_receipt=None,  # type: ignore[arg-type]
        )
    assert backend.calls == []


def test_unverified_adapter_attestation_cannot_arm() -> None:
    backend = RecordingBackend()
    runtime = _runtime(backend)
    scaffold = _scaffold()
    prompt = _expert_prompt(scaffold)
    runtime.dispatch_planner(
        request_id="R1",
        task_bundle_sha256=scaffold.task_bundle_sha256,
        prompt="Plan.",
        tokenizer_binding=TokenizerBinding.unbound(),
    )
    backend.calls.clear()

    with pytest.raises(ScaffoldRuntimeError, match="no verified artifact"):
        runtime.arm_next_request(
            scaffold=scaffold,
            selection=_selection(),
            reencode_receipt=_reencode_receipt(scaffold, prompt),
        )
    assert backend.calls == []


def test_callers_cannot_promote_booleans_or_private_fields_into_receipts() -> None:
    scaffold = _scaffold()
    with pytest.raises(ScaffoldRuntimeError, match="only be created by verifier"):
        ScaffoldReencodeReceipt(
            planner_request_id="R1",
            task_bundle_sha256=scaffold.task_bundle_sha256,
            committed_scaffold_sha256=scaffold.committed_scaffold_sha256,
            trigger_text_sha256=scaffold.trigger_text_sha256,
            new_request_input_sha256=_sha("prompt"),
            tokenizer_sha256=_sha("tokenizer"),
            chat_template_sha256=_sha("chat-template"),
            ordered_token_ids_sha256=_sha("tokens"),
            base_revision_sha256=_sha("base"),
            trigger_occurrences=1,
            adapter_state="off",
            _capability=object(),
        )
    with pytest.raises(ScaffoldRuntimeError, match="only be created by verifier"):
        AdapterArtifactVerificationReceipt(
            adapter_id="adapter-builder-r16",
            adapter_sha256=_sha("adapter"),
            tensor_inventory_sha256=_sha("inventory"),
            base_revision_sha256=_sha("base"),
            converter_sha256=_sha("converter"),
            release_lock_sha256=_sha("release"),
            target_modules=("q_proj",),
            rank=16,
            alpha=32.0,
            placement=AdapterPlacement.DECOUPLED_FROZEN_FACT_READOUT,
            snapshots=(),
            _capability=object(),
        )


def test_verified_adapter_file_identity_drift_fails_before_expert(
    tmp_path: Path,
) -> None:
    backend = RecordingBackend()
    attestation, release_sha256 = _verified_attestation(tmp_path)
    runtime = _runtime(backend, release_lock_sha256=release_sha256)
    scaffold = _scaffold()
    prompt = _expert_prompt(scaffold)
    runtime.dispatch_planner(
        request_id="R1",
        task_bundle_sha256=scaffold.task_bundle_sha256,
        prompt="Plan.",
        tokenizer_binding=TokenizerBinding.unbound(),
    )
    backend.calls.clear()
    adapter_path = tmp_path / "adapter.bin"
    adapter_path.write_bytes(b"x" * adapter_path.stat().st_size)

    with pytest.raises(ScaffoldRuntimeError, match="identity changed"):
        runtime.arm_next_request(
            scaffold=scaffold,
            selection=_selection(attestation=attestation),
            reencode_receipt=_reencode_receipt(scaffold, prompt),
        )
    assert backend.calls == []


def test_bad_r2_consumes_slot_and_does_not_reach_backend(tmp_path: Path) -> None:
    backend = RecordingBackend()
    attestation, release_sha256 = _verified_attestation(tmp_path)
    runtime = _runtime(backend, release_lock_sha256=release_sha256)
    scaffold = _scaffold()
    good_prompt = _expert_prompt(scaffold)
    receipt = _reencode_receipt(scaffold, good_prompt)
    runtime.dispatch_planner(
        request_id="R1",
        task_bundle_sha256=scaffold.task_bundle_sha256,
        prompt="Plan.",
        tokenizer_binding=TokenizerBinding.unbound(),
    )
    runtime.arm_next_request(
        scaffold=scaffold,
        selection=_selection(attestation=attestation),
        reencode_receipt=receipt,
    )
    backend.calls.clear()

    with pytest.raises(ScaffoldRuntimeError, match="bound tokenizer"):
        runtime.dispatch_next_request(
            request_id="R2",
            task_bundle_sha256=_sha("bundle"),
            prompt="Execute.",
            tokenizer_binding=TokenizerBinding.unbound(),
            requested_expert_id="builder",
            token_materialization=_materialization(receipt),
        )

    assert not runtime.slot_is_armed
    assert backend.calls == []

    runtime.dispatch_next_request(
        request_id="R3",
        task_bundle_sha256=_sha("bundle"),
        prompt="Base after rejected R2.",
        tokenizer_binding=TokenizerBinding.unbound(),
    )
    assert [call.adapter_id for call in backend.calls] == [None]


def test_fake_observed_trigger_cannot_replace_real_new_input_scan() -> None:
    backend = RecordingBackend()
    runtime = _runtime(backend)
    scaffold = _scaffold()
    prompt_without_trigger = scaffold.natural_language_scaffold + "\nExecute."
    runtime.dispatch_planner(
        request_id="R1",
        task_bundle_sha256=scaffold.task_bundle_sha256,
        prompt="Plan.",
        tokenizer_binding=TokenizerBinding.unbound(),
    )
    backend.calls.clear()

    with pytest.raises(ScaffoldRuntimeError, match="exactly one trigger"):
        verify_scaffold_reencode(
            planner_request_id="R1",
            scaffold=scaffold,
            prompt=prompt_without_trigger,
            tokenizer_binding=_bound_tokenizer(scaffold),
            ordered_token_ids=(1, 2, 3),
            base_revision_sha256=_sha("base"),
        )
    assert backend.calls == []


def test_planner_private_kv_never_crosses_request_without_commit(
    tmp_path: Path,
) -> None:
    backend = RecordingBackend()
    attestation, release_sha256 = _verified_attestation(tmp_path)
    runtime = _runtime(backend, release_lock_sha256=release_sha256)
    scaffold = _scaffold()
    prompt = _expert_prompt(scaffold)
    receipt = _reencode_receipt(scaffold, prompt)
    lease = PrivateKVLease(
        owner_request_id="R1",
        task_bundle_sha256=_sha("bundle"),
        private_branch_id="planner-private-branch",
        capability="request-local-secret",
    )
    runtime.dispatch_planner(
        request_id="R1",
        task_bundle_sha256=_sha("bundle"),
        prompt="Plan privately.",
        tokenizer_binding=TokenizerBinding.unbound(),
        private_kv=lease,
    )
    assert backend.calls[-1].private_kv is lease

    runtime.arm_next_request(
        scaffold=scaffold,
        selection=_selection(attestation=attestation),
        reencode_receipt=receipt,
    )
    before = len(backend.calls)
    with pytest.raises(ScaffoldRuntimeError, match="cannot cross"):
        runtime.dispatch_next_request(
            request_id="R2",
            task_bundle_sha256=_sha("bundle"),
            prompt=prompt,
            tokenizer_binding=_bound_tokenizer(scaffold),
            requested_expert_id="builder",
            token_materialization=_materialization(receipt),
            private_kv=lease,
        )
    assert len(backend.calls) == before
    assert not runtime.slot_is_armed

    assert not runtime.slot_is_armed


def test_planner_scaffold_rejects_private_kv_control_fields() -> None:
    backend = RecordingBackend()

    with pytest.raises(ScaffoldRuntimeError, match="private KV"):
        _scaffold(plan={"steps": [], "metadata": {"private_branch_id": "secret"}})
    assert backend.calls == []


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("token_index", 42),
        ("invocation_token_ids", [1, 2, 3]),
        ("position_ids", [0, 1, 2]),
    ],
)
def test_unbound_tokenizer_emits_no_token_indices_and_does_no_backend_work(
    field_name: str,
    field_value: object,
) -> None:
    backend = RecordingBackend()
    runtime = _runtime(backend)

    with pytest.raises(ScaffoldRuntimeError, match="tokenizer is unbound"):
        runtime.dispatch_next_request(
            request_id="R1",
            task_bundle_sha256=_sha("bundle"),
            prompt="Base request.",
            tokenizer_binding=TokenizerBinding.unbound(),
            token_materialization={"nested": {field_name: field_value}},
        )
    assert backend.calls == []


def test_profile_label_is_not_adapter_attestation() -> None:
    with pytest.raises(ScaffoldRuntimeError, match="not adapter artifact"):
        AdapterSelection(
            profile_label=AdapterProfileLabel.Q_ONLY,
            attestation=None,
            exact_shared_kv=True,
        ).validate()


@pytest.mark.parametrize(
    ("profile", "modules"),
    [
        (AdapterProfileLabel.Q_ONLY, ("q_proj", "o_proj")),
        (AdapterProfileLabel.Q_PLUS_O, ("q_proj",)),
        (AdapterProfileLabel.WIDE_LORA, ("q_proj", "o_proj")),
    ],
)
def test_profile_label_must_match_attested_tensor_inventory(
    profile: AdapterProfileLabel,
    modules: tuple[str, ...],
) -> None:
    with pytest.raises(ScaffoldRuntimeError, match="label"):
        AdapterSelection(
            profile_label=profile,
            attestation=_attestation(modules=modules),
            exact_shared_kv=False,
        ).validate()


def test_q_only_naive_decoder_cannot_claim_exact_shared_kv() -> None:
    with pytest.raises(ScaffoldRuntimeError, match="not exact-share safe"):
        _selection(
            attestation=_attestation(placement=AdapterPlacement.NAIVE_DECODER_IN_STACK)
        ).validate()
