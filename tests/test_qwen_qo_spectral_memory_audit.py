from __future__ import annotations

from collections import Counter
import copy
import json
from pathlib import Path

from jsonschema import Draft202012Validator
import pytest

from anchor_mvp.research import qwen_qo_spectral_memory_audit as audit
from anchor_mvp.training.config import ConfigError


ROOT = Path(__file__).resolve().parents[1]


def test_config_is_canonical_and_fail_closed() -> None:
    config = audit.load_config(ROOT / audit.CONFIG_PATH)
    assert config["schema_version"] == audit.CONFIG_VERSION
    assert config["metrics"]["top_k_energy"] == [1, 2, 4]
    assert config["metrics"]["token_group_size"] == 128
    assert config["claims"]["memorization_proven"] is False
    assert config["claims"]["exploit_code_memorization_tested"] is False
    assert config["dataset"]["eval_proxy_reads_allowed"] is False
    assert config["dataset"]["heldout_reads_allowed"] is False
    with pytest.raises(ConfigError, match="config must remain exactly"):
        audit.load_config(ROOT / "configs" / "research" / "other.yaml")
    drifted = copy.deepcopy(config)
    drifted["model"]["unexpected"] = True
    with pytest.raises(ConfigError, match="model fields drifted"):
        audit._validate_config(drifted)


def test_published_schemas_are_valid_draft_2020_12() -> None:
    for relative in (audit.CONFIG_SCHEMA_PATH, audit.RECEIPT_SCHEMA_PATH):
        schema = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        Draft202012Validator.check_schema(schema)


def test_thin_qr_singular_values_match_dense_delta() -> None:
    torch = pytest.importorskip("torch")
    generator = torch.Generator(device="cpu").manual_seed(1731)
    a = torch.randn(3, 7, generator=generator, dtype=torch.float64)
    b = torch.randn(5, 3, generator=generator, dtype=torch.float64)
    observed = audit._thin_delta_singular_values(a, b, 1.75)
    expected = torch.linalg.svdvals((b @ a) * 1.75)
    assert torch.allclose(
        observed, expected[: observed.numel()], atol=1e-11, rtol=1e-11
    )


def test_spectral_metrics_use_svdvals_and_known_rank_statistics() -> None:
    torch = pytest.importorskip("torch")
    values = torch.tensor([4.0, 2.0, 1.0], dtype=torch.float64)
    result = audit._spectrum_metrics(values, [1, 2, 4])
    assert result["frobenius_norm"] == pytest.approx(21**0.5)
    assert result["spectral_norm_svdvals_max"] == 4.0
    assert result["stable_rank"] == pytest.approx(21 / 16)
    assert result["top_k_energy_fraction"]["top_1"] == pytest.approx(16 / 21)
    assert result["top_k_energy_fraction"]["top_2"] == pytest.approx(20 / 21)
    assert result["top_k_energy_fraction"]["top_4"] == 1.0
    source = (ROOT / audit.IMPLEMENTATION_PATH).read_text(encoding="utf-8")
    assert "torch.linalg.matrix_norm(" not in source
    assert "torch.linalg.svdvals" in source


def test_module_spectrum_covers_all_layers_and_hashes_distribution() -> None:
    torch = pytest.importorskip("torch")
    pairs = {
        layer: (
            torch.tensor([[1.0 + layer / 100, 0.0], [0.0, 0.5]]),
            torch.tensor([[1.0, 0.0], [0.0, 0.25]]),
        )
        for layer in range(28)
    }
    result = audit._module_spectrum(pairs, rank=2, alpha=4, top_k=[1, 2, 4])
    assert result["layers"] == 28
    assert [item["layer"] for item in result["per_layer"]] == list(range(28))
    assert result["total_delta_energy"] > 0
    assert (
        len(result["layer_energy_distribution"]["ordered_layer_energy_fraction_sha256"])
        == 64
    )


def test_token_group_selection_is_deterministic_and_control_disjoint() -> None:
    target = Counter({10: 9, 11: 8, 12: 2, 13: 1})
    prompt = Counter({20: 10, 21: 8, 10: 4, 22: 3})
    kwargs = {
        "special_ids": {0},
        "vocab_size": 64,
        "group_size": 2,
        "random_seed": "audit-test-seed",
    }
    first = audit.select_token_groups(target, prompt, **kwargs)
    second = audit.select_token_groups(target, prompt, **kwargs)
    assert first == second
    assert first["target_frequent"] == [10, 11]
    assert first["prompt_control"] == [20, 21]
    assert first["target_low_frequency"] == [13, 12]
    assert not set(first["deterministic_random_vocab"]) & (set(target) | set(prompt))


def test_projection_energy_matches_direct_projector() -> None:
    torch = pytest.importorskip("torch")
    o_pairs = {
        layer: (
            torch.ones(2, 4),
            torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0], [0.0, 0.0]]),
        )
        for layer in range(28)
    }
    embeddings = torch.eye(4)
    groups = {
        "target_frequent": [0],
        "prompt_control": [1],
        "target_low_frequency": [2],
        "deterministic_random_vocab": [1],
    }
    result = audit._subspace_alignment(
        o_pairs, embeddings, {index: index for index in range(4)}, groups, 4
    )
    assert result["groups"]["target_frequent"]["mean_projection_energy"] == 1.0
    assert result["groups"]["prompt_control"]["mean_projection_energy"] == 1.0
    assert result["groups"]["target_low_frequency"]["mean_projection_energy"] == 0.0
    assert (
        result["groups"]["deterministic_random_vocab"]["mean_projection_energy"] == 1.0
    )


def test_atomic_publish_rejects_existing_destination(tmp_path: Path) -> None:
    diagnostics = tmp_path / "diagnostics"
    path, digest = audit.publish_receipt(
        {"diagnostic": True},
        diagnostics / "receipt-v1",
        diagnostics_root=diagnostics,
    )
    assert path.read_bytes() == b'{\n  "diagnostic": true\n}\n'
    assert (path.parent / "receipt.json.sha256").read_text(encoding="ascii") == (
        f"{digest}  receipt.json\n"
    )
    with pytest.raises(audit.SpectralMemoryAuditError, match="must be a new"):
        audit.publish_receipt(
            {"diagnostic": True},
            diagnostics / "receipt-v1",
            diagnostics_root=diagnostics,
        )
