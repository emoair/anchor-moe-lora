from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from anchor_mvp.training import qwen_synthetic_scaffold_diagnostic as diagnostic
from anchor_mvp.training.config import ConfigError, _expand_env


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / diagnostic.CONFIG_PATH


def _raw_config() -> dict[str, object]:
    value = yaml.safe_load(CONFIG.read_text("utf-8"))
    assert isinstance(value, dict)
    result = _expand_env(value)
    result["_config_path"] = str(CONFIG)
    return result


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        assert tokenize is True
        assert messages[0]["role"] == "user"
        rendered = f"U:{messages[0]['content']}\nA:"
        if len(messages) == 2:
            assert messages[1]["role"] == "assistant"
            rendered += messages[1]["content"] + "<eos>"
        elif not add_generation_prompt:
            raise AssertionError("prompt-only view needs a generation marker")
        return list(rendered.encode("utf-8"))


def _example(
    *, prompt: str = "plan", target: str = '{"ok":true}'
) -> diagnostic.ScaffoldExample:
    return diagnostic.ScaffoldExample(
        record_id="synthetic-record",
        split="train",
        variant="json_only",
        source_bundle_id="synthetic-bundle",
        prompt=prompt,
        target=target,
    )


def _receipt_value() -> dict[str, object]:
    return {
        "schema_version": diagnostic.PREFLIGHT_VERSION,
        "status": "passed_tokenizer_only_diagnostic_preflight",
        "ready": True,
        "identity": {},
        "max_steps": 2,
        "precision": {},
        "dataset": {},
        "tokenizer": {},
        "token_lengths": {},
        "model_identity": {},
        "output_path": "artifacts/diagnostics/adapter",
        "gates": {"ready": True},
        "claims": {
            "diagnostic_only": True,
            "training_authorized": False,
            "formal": False,
            "eval_proxy_is_heldout": False,
            "diagnostic_execution_user_requested": False,
        },
        "audit": {
            "protected_body_reads": 0,
            "heldout_reads": 0,
            "provider_requests": 0,
            "network_requests": 0,
            "model_loads": 0,
            "gpu_requests": 0,
            "tokenizer_loads": 1,
        },
    }


def test_config_is_strict_qonly_low_resource_and_blocked() -> None:
    config = _raw_config()
    diagnostic.validate_config(config)
    assert config["lora"] == {
        "profile": "q_only",
        "rank": 4,
        "alpha": 8,
        "dropout": 0.0,
        "bias": "none",
        "target_modules": ["q_proj"],
    }
    assert config["training"] == {
        "allowed_max_steps": [2, 20],
        "default_max_steps": 2,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "sequence_length": 512,
        "learning_rate": 0.00005,
        "seed": 1337,
        "gradient_checkpointing": True,
        "eval_before_after": True,
    }
    assert config["claims"] == {
        "diagnostic_only": True,
        "training_authorized": False,
        "formal": False,
        "eval_proxy_is_heldout": False,
    }


@pytest.mark.parametrize("value", [0, 1, 3, 19, 21, 100])
def test_only_two_or_twenty_steps_are_accepted(value: int) -> None:
    config = _raw_config()
    with pytest.raises(ConfigError, match="exactly 2 or 20"):
        diagnostic._max_steps(config, value)
    assert diagnostic._max_steps(config, 2) == 2
    assert diagnostic._max_steps(config, 20) == 20


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("claims", "training_authorized"), True),
        (("claims", "formal"), True),
        (("dataset", "heldout_allowed"), True),
        (("dataset", "protected_source_paths_allowed"), True),
        (("lora", "target_modules"), ["q_proj", "o_proj"]),
        (("training", "sequence_length"), 1024),
        (("precision", "tf32"), False),
    ],
)
def test_authority_scope_and_resource_drift_fail_closed(
    path: tuple[str, str], value: object
) -> None:
    config = _raw_config()
    section = config[path[0]]
    assert isinstance(section, dict)
    section[path[1]] = value
    with pytest.raises(ConfigError):
        diagnostic.validate_config(config)


def test_tokenization_masks_prompt_and_keeps_complete_target() -> None:
    encoded = diagnostic.tokenize_example(
        FakeTokenizer(), _example(), sequence_length=512
    )
    prompt_tokens = encoded["prompt_tokens"]
    assert encoded["labels"][:prompt_tokens] == [-100] * prompt_tokens
    assert encoded["labels"][prompt_tokens:] == encoded["input_ids"][prompt_tokens:]
    assert encoded["target_tokens"] > 0
    assert encoded["full_tokens"] <= 512


def test_transformers_batch_encoding_shape_is_accepted() -> None:
    assert diagnostic._token_ids(
        {"input_ids": [1, 2, 3], "attention_mask": [1, 1, 1]}
    ) == [1, 2, 3]
    with pytest.raises(RuntimeError, match="unexpected token fields"):
        diagnostic._token_ids({"input_ids": [1], "secret_ids": [2]})


def test_any_full_or_target_truncation_is_fail_closed() -> None:
    example = _example(prompt="p" * 30, target="t" * 30)
    with pytest.raises(RuntimeError, match="would truncate target/full view"):
        diagnostic.tokenize_example(FakeTokenizer(), example, sequence_length=32)


def test_token_length_preflight_covers_train_and_eval() -> None:
    train = _example(prompt="short", target="target")
    eval_proxy = diagnostic.ScaffoldExample(
        record_id="eval-record",
        split="eval_proxy",
        variant="json_only",
        source_bundle_id="eval-bundle",
        prompt="slightly longer",
        target="target",
    )
    dataset = diagnostic.ScaffoldDataset(
        manifest={},
        manifest_sha256="0" * 64,
        partition_sha256={},
        train=(train,),
        eval_proxy=(eval_proxy,),
    )
    report = diagnostic.token_length_preflight(
        FakeTokenizer(), dataset, sequence_length=512
    )
    assert report["records"] == 2
    assert report["target_truncation_detected"] is False
    assert report["truncated_records"] == 0
    inventory = report["token_view_digest_inventory"]
    assert inventory["records"] == 2
    assert len(inventory["digest_rows"]) == 2
    assert inventory["raw_token_ids_emitted"] is False
    assert "input_ids" not in json.dumps(inventory)
    assert set(report["runtime_versions"]) == {"transformers", "tokenizers"}


def test_current_synthetic_fixture_loader_uses_only_bound_partitions() -> None:
    config = _raw_config()
    dataset_config = config["dataset"]
    assert isinstance(dataset_config, dict)
    assert dataset_config["expected_manifest_sha256"] == _sha(
        REPO_ROOT / f"{diagnostic.DATASET_ROOT}/manifest.json"
    )
    assert dataset_config["expected_record_schema_sha256"] == _sha(
        REPO_ROOT / diagnostic.RECORD_SCHEMA_PATH
    )
    assert dataset_config["expected_manifest_schema_sha256"] == _sha(
        REPO_ROOT / diagnostic.MANIFEST_SCHEMA_PATH
    )
    diagnostic.validate_config(config)
    dataset = diagnostic.load_dataset(config)
    assert len(dataset.train) == 80
    assert len(dataset.eval_proxy) == 20
    assert {item.split for item in dataset.train} == {"train"}
    assert {item.split for item in dataset.eval_proxy} == {"eval_proxy"}
    assert {item.source_bundle_id for item in dataset.train}.isdisjoint(
        {item.source_bundle_id for item in dataset.eval_proxy}
    )


def test_output_is_isolated_by_step_count() -> None:
    config = _raw_config()
    two = diagnostic._output_path(config, 2)
    twenty = diagnostic._output_path(config, 20)
    assert two != twenty
    assert two.name.endswith("step2")
    assert twenty.name.endswith("step20")
    assert two.parent == REPO_ROOT / "artifacts" / "diagnostics"


def test_preflight_receipt_is_atomic_authenticated_and_no_replace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(diagnostic.qdiag, "_project_root_from_module", lambda: tmp_path)
    output = Path(diagnostic.PREFLIGHT_OUTPUT_ROOT) / "step2"
    receipt_path, receipt_sha256 = diagnostic.publish_preflight_receipt(
        _receipt_value(), output
    )
    assert receipt_path == tmp_path / output / diagnostic.PREFLIGHT_FILENAME
    assert receipt_path.read_bytes().endswith(b"\n")
    assert (
        receipt_path.with_name(diagnostic.PREFLIGHT_SIDECAR_FILENAME).read_text("ascii")
        == f"{receipt_sha256}  {diagnostic.PREFLIGHT_FILENAME}\n"
    )
    authenticated = diagnostic.authenticate_preflight_receipt(
        receipt_path, receipt_sha256
    )
    diagnostic._require_preflight_match(authenticated, _receipt_value())
    (receipt_path.parent / "unexpected.txt").write_bytes(b"unexpected")
    with pytest.raises(ConfigError, match="exact file inventory drifted"):
        diagnostic.authenticate_preflight_receipt(receipt_path, receipt_sha256)
    (receipt_path.parent / "unexpected.txt").unlink()
    with pytest.raises(ConfigError, match="already exists"):
        diagnostic.publish_preflight_receipt(_receipt_value(), output)


def test_preflight_receipt_wrong_sha_and_malformed_sidecar_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(diagnostic.qdiag, "_project_root_from_module", lambda: tmp_path)
    output = Path(diagnostic.PREFLIGHT_OUTPUT_ROOT) / "step2"
    receipt_path, receipt_sha256 = diagnostic.publish_preflight_receipt(
        _receipt_value(), output
    )
    with pytest.raises(ConfigError, match="SHA-256 mismatch"):
        diagnostic.authenticate_preflight_receipt(receipt_path, "0" * 64)
    receipt_path.with_name(diagnostic.PREFLIGHT_SIDECAR_FILENAME).write_text(
        f"{receipt_sha256} preflight.json\n", encoding="ascii"
    )
    with pytest.raises(ConfigError, match="missing or malformed"):
        diagnostic.authenticate_preflight_receipt(receipt_path, receipt_sha256)


def test_preflight_receipt_semantic_drift_and_snapshot_replacement_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(diagnostic.qdiag, "_project_root_from_module", lambda: tmp_path)
    output = Path(diagnostic.PREFLIGHT_OUTPUT_ROOT) / "step2"
    receipt_path, receipt_sha256 = diagnostic.publish_preflight_receipt(
        _receipt_value(), output
    )
    authenticated = diagnostic.authenticate_preflight_receipt(
        receipt_path, receipt_sha256
    )
    drifted = _receipt_value()
    drifted["max_steps"] = 20
    with pytest.raises(ConfigError, match="does not exactly match"):
        diagnostic._require_preflight_match(authenticated, drifted)
    receipt_path.write_text(json.dumps(_receipt_value()) + "\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="authenticated file changed"):
        authenticated.assert_unchanged()


def test_preflight_paths_and_cli_arguments_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(diagnostic.qdiag, "_project_root_from_module", lambda: tmp_path)
    with pytest.raises(ConfigError, match="must be a child"):
        diagnostic._preflight_output_path("artifacts/diagnostics/outside")

    monkeypatch.undo()
    with pytest.raises(SystemExit):
        diagnostic.main(["--config", str(CONFIG), "--dry-run"])
    with pytest.raises(SystemExit):
        diagnostic.main(["--config", str(CONFIG), "--execute"])


def test_final_diagnostic_receipt_sidecar_detects_tampering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    adapter_hashes = {
        "adapter_config.json": "1" * 64,
        "adapter_model.safetensors": "2" * 64,
    }
    monkeypatch.setattr(
        diagnostic.qdiag,
        "_validate_saved_adapter",
        lambda *args, **kwargs: dict(adapter_hashes),
    )
    receipt = {
        "schema_version": diagnostic.SCHEMA_VERSION,
        "status": "passed_diagnostic_only",
        "adapter_artifact_sha256": adapter_hashes,
    }
    receipt_bytes = diagnostic._canonical_json_bytes(receipt)
    receipt_sha256 = hashlib.sha256(receipt_bytes).hexdigest()
    (tmp_path / "adapter_config.json").write_bytes(b"adapter config")
    (tmp_path / "adapter_model.safetensors").write_bytes(b"weights")
    (tmp_path / "diagnostic_receipt.json").write_bytes(receipt_bytes)
    (tmp_path / "diagnostic_receipt.json.sha256").write_bytes(
        f"{receipt_sha256}  diagnostic_receipt.json\n".encode("ascii")
    )
    diagnostic._validate_diagnostic_artifact(
        tmp_path,
        expected_adapter_hashes=adapter_hashes,
        expected_receipt_sha256=receipt_sha256,
    )
    (tmp_path / "README.md").write_bytes(b"must not ship")
    with pytest.raises(ConfigError, match="exact file inventory drifted"):
        diagnostic._validate_diagnostic_artifact(
            tmp_path,
            expected_adapter_hashes=adapter_hashes,
            expected_receipt_sha256=receipt_sha256,
        )
    (tmp_path / "README.md").unlink()
    (tmp_path / "diagnostic_receipt.json").write_bytes(receipt_bytes + b" ")
    with pytest.raises(RuntimeError, match="authentication failed"):
        diagnostic._validate_diagnostic_artifact(
            tmp_path,
            expected_adapter_hashes=adapter_hashes,
            expected_receipt_sha256=receipt_sha256,
        )


def test_failed_post_rename_validation_removes_published_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    preflight_calls = 0
    real_preflight_validator = diagnostic._validate_preflight_directory

    def fail_second_preflight(*args: object, **kwargs: object) -> None:
        nonlocal preflight_calls
        preflight_calls += 1
        if preflight_calls == 2:
            raise RuntimeError("simulated final preflight drift")
        real_preflight_validator(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(diagnostic.qdiag, "_project_root_from_module", lambda: tmp_path)
    monkeypatch.setattr(
        diagnostic, "_validate_preflight_directory", fail_second_preflight
    )
    output = Path(diagnostic.PREFLIGHT_OUTPUT_ROOT) / "rollback"
    with pytest.raises(RuntimeError, match="simulated final preflight drift"):
        diagnostic.publish_preflight_receipt(_receipt_value(), output)
    assert not (tmp_path / output).exists()

    diagnostic_calls = 0

    def fail_second_diagnostic(*args: object, **kwargs: object) -> None:
        nonlocal diagnostic_calls
        diagnostic_calls += 1
        if diagnostic_calls == 2:
            raise RuntimeError("simulated final adapter drift")

    monkeypatch.setattr(
        diagnostic, "_validate_diagnostic_artifact", fail_second_diagnostic
    )
    staging = tmp_path / "diagnostic-staging"
    output_path = tmp_path / "diagnostic-final"
    staging.mkdir()
    with pytest.raises(RuntimeError, match="simulated final adapter drift"):
        diagnostic._publish_verified_diagnostic(
            staging,
            output_path,
            expected_adapter_hashes={},
            expected_receipt_sha256="0" * 64,
        )
    assert not output_path.exists()
