from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest

from anchor_mvp.research import qwen_qo_memory_ablation_audit as audit
from anchor_mvp.training.config import ConfigError


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / audit.CONFIG_PATH


def test_config_and_adapter_are_strictly_authenticated() -> None:
    config = audit.load_config(CONFIG)
    authenticated = audit.authenticate_adapter(config)
    assert authenticated["receipt"]["profile"] == "q_plus_o"
    assert authenticated["receipt"]["lora"]["trainable_parameters"] == 1_376_256
    assert set(authenticated["file_sha256"]) == audit._ADAPTER_FILES
    audit.assert_adapter_unchanged(authenticated)


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("claims", "formal", True),
        ("claims", "training_authorized", True),
        ("dataset", "heldout_allowed", True),
        ("adapter", "source_profile", "q_only"),
        ("audit", "sequence_length", 1024),
        ("audit", "modes", ["full"]),
    ],
)
def test_contract_drift_fails_closed(section: str, field: str, value: object) -> None:
    config = audit.load_config(CONFIG)
    config[section][field] = value
    with pytest.raises(ConfigError):
        audit.validate_config(config)


def test_ood_generator_is_deterministic_disjoint_and_body_free_in_inventory() -> None:
    config = audit.load_config(CONFIG)
    dataset = audit.synth.load_dataset(config)
    first = audit.build_ood_examples(20260723)
    second = audit.build_ood_examples(20260723)
    assert first == second
    assert len(first) == 20
    assert len({item.source_bundle_id for item in first}) == 4
    assert {item.source_bundle_id for item in first}.isdisjoint(
        {item.source_bundle_id for item in (*dataset.train, *dataset.eval_proxy)}
    )
    inventory = audit._body_digest_inventory(dataset, first)
    assert inventory == {
        "algorithm": "sha256_utf8_exact_body_and_compact_sorted_rows_v1",
        "records": 20,
        "source_bundles": 4,
        "rows_sha256": inventory["rows_sha256"],
        "exact_body_overlap_count": 0,
        "source_bundle_overlap_count": 0,
        "raw_bodies_emitted": False,
    }
    assert all(key not in inventory for key in ("prompt", "target", "body"))


class _ScalingModule:
    def __init__(self, value: float = 2.0) -> None:
        self.scaling = {"default": value}


class _FakePeft:
    def __init__(self) -> None:
        self.q = [_ScalingModule() for _ in range(28)]
        self.o = [_ScalingModule() for _ in range(28)]
        self.adapter_disabled = False

    def named_modules(self):
        yield "", self
        for index, module in enumerate(self.q):
            yield f"model.layers.{index}.self_attn.q_proj", module
        for index, module in enumerate(self.o):
            yield f"model.layers.{index}.self_attn.o_proj", module

    @contextmanager
    def disable_adapter(self):
        self.adapter_disabled = True
        try:
            yield
        finally:
            self.adapter_disabled = False


@pytest.mark.parametrize(
    ("mode", "q_value", "o_value"),
    [
        ("full", 2.0, 2.0),
        ("q_only_contribution", 2.0, 0.0),
        ("o_only_contribution", 0.0, 2.0),
    ],
)
def test_contribution_view_zeroes_only_requested_delta_and_restores(
    mode: str, q_value: float, o_value: float
) -> None:
    model = _FakePeft()
    with audit.contribution_view(model, mode):
        assert {item.scaling["default"] for item in model.q} == {q_value}
        assert {item.scaling["default"] for item in model.o} == {o_value}
    assert {item.scaling["default"] for item in (*model.q, *model.o)} == {2.0}


def test_adapter_off_uses_peft_context_and_restores() -> None:
    model = _FakePeft()
    with audit.contribution_view(model, "adapter_off"):
        assert model.adapter_disabled is True
    assert model.adapter_disabled is False


def test_preflight_is_body_free_and_loads_no_model_gpu_provider_or_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = audit.load_config(CONFIG)
    isolated_output = (
        ROOT
        / "artifacts"
        / "diagnostics"
        / f".qo-memory-audit-preflight-test-{tmp_path.name}"
    )
    assert not isolated_output.exists()
    monkeypatch.setattr(audit, "_output_path", lambda _config: isolated_output)
    report = audit.build_preflight(config)
    assert report["ready"] is True
    assert report["dataset"]["train_records"] == 80
    assert report["dataset"]["eval_proxy_records"] == 20
    assert report["ood_proxy"]["records"] == 20
    assert report["modes"] == list(audit.MODES)
    assert report["audit"] == {
        "model_loads": 0,
        "gpu_requests": 0,
        "provider_requests": 0,
        "network_requests": 0,
        "heldout_reads": 0,
        "protected_body_reads": 0,
    }
    assert "materialized_prompt" not in audit._canonical_json_bytes(report).decode(
        "utf-8"
    )
