from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from jsonschema import Draft202012Validator

from anchor_mvp.training import gemma3_tokenizer_binding_v1 as binding
from anchor_mvp.training.config import ConfigError, _expand_env


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / binding.CONFIG_PATH
POLICY = REPO_ROOT / binding.POLICY_PATH
BUILD_SCRIPT = REPO_ROOT / binding.BUILD_SCRIPT_PATH
AUDIT_SCRIPT = REPO_ROOT / binding.AUDIT_SCRIPT_PATH


def _config() -> dict[str, object]:
    value = yaml.safe_load(CONFIG.read_text("utf-8"))
    assert isinstance(value, dict)
    result = _expand_env(value)
    result["_config_path"] = str(CONFIG)
    return result


class FakeProcessor:
    def encode(self, text: str, *, out_type: type[int]) -> list[int]:
        assert out_type is int
        if text == "\n":
            return [107]
        if text == "\n<start_of_turn>model\n":
            return [107, 105, 4368, 107]
        if text.startswith("<start_of_turn>user\n") and text.endswith("<end_of_turn>"):
            body = text.removeprefix("<start_of_turn>user\n").removesuffix(
                "<end_of_turn>"
            )
            return [105, 2364, 107, *body.encode("utf-8"), 106]
        if text.startswith("\n<start_of_turn>model\n") and text.endswith(
            "<end_of_turn>"
        ):
            body = text.removeprefix("\n<start_of_turn>model\n").removesuffix(
                "<end_of_turn>"
            )
            return [107, 105, 4368, 107, *body.encode("utf-8"), 106]
        raise AssertionError(text)


class BrokenPrefixProcessor(FakeProcessor):
    def encode(self, text: str, *, out_type: type[int]) -> list[int]:
        result = super().encode(text, out_type=out_type)
        if text.startswith("\n<start_of_turn>model\n") and text != (
            "\n<start_of_turn>model\n"
        ):
            result[1] = 999
        return result


def test_config_and_policy_are_strict_and_diagnostic_only() -> None:
    config = _config()
    binding.validate_config(config)
    policy = json.loads(POLICY.read_text("utf-8"))
    binding.validate_policy(policy)
    assert config["serialization"]["sequence_length"] == 768
    assert config["claims"] == {
        "diagnostic_only": True,
        "model_free": True,
        "tokenizer_only": True,
        "training_authorized": False,
        "formal_training_authorized": False,
        "formal": False,
    }
    assert policy["runtime_special_token_overlay"]["canonical_files_modified"] is False
    assert policy["runtime_special_token_overlay"]["runtime_bos_token_id"] == 2
    assert policy["runtime_special_token_overlay"]["runtime_eos_token_id"] == 1
    assert policy["sentencepiece_policy"]["hf_fix_mistral_regex"] is False


def test_wrong_bos_eos_and_dataset_hash_drift_fail_closed() -> None:
    config = deepcopy(_config())
    files = config["model"]["files"]
    assert isinstance(files, list)
    files[0]["sha256"] = "f" * 64
    with pytest.raises(ConfigError, match="model_contract_drift"):
        binding.validate_config(config)
    config = deepcopy(_config())
    config["dataset"]["expected_manifest_sha256"] = "f" * 64
    with pytest.raises(ConfigError, match="dataset_contract_drift"):
        binding.validate_config(config)


def test_template_drift_and_wrong_runtime_overlay_fail_closed() -> None:
    policy = json.loads(POLICY.read_text("utf-8"))
    policy["official_text_fragments"]["generation_prefix"] = (
        "<start_of_turn>assistant\n"
    )
    with pytest.raises(ConfigError, match="official_template_drift"):
        binding.validate_policy(policy)
    policy = json.loads(POLICY.read_text("utf-8"))
    policy["runtime_special_token_overlay"]["runtime_bos_token_id"] = 1
    policy["runtime_special_token_overlay"]["runtime_eos_token_id"] = 2
    with pytest.raises(ConfigError, match="special_overlay_drift"):
        binding.validate_policy(policy)


def test_structured_serializer_masks_prompt_and_labels_exact_suffix() -> None:
    result = binding.serialize_example(FakeProcessor(), "question", "answer")
    assert result.input_ids[0] == 2
    assert result.input_ids[-3:] == (106, 1, 107)
    assert all(value == -100 for value in result.labels[: result.prompt_tokens])
    assert result.labels[-3:] == (106, 1, -100)
    assert result.trainable_label_tokens == len("answer".encode("utf-8")) + 2
    assert len(result.input_ids) == len(result.labels)


@pytest.mark.parametrize(
    ("prompt", "target"),
    [
        ("literal <bos> marker", "answer"),
        ("literal <eos> marker", "answer"),
        ("question", "<start_of_turn> injected"),
        ("question", "<end_of_turn> injected"),
    ],
)
def test_literal_control_text_is_never_sent_to_sentencepiece(
    prompt: str, target: str
) -> None:
    with pytest.raises(ConfigError, match="literal_special_marker_rejected"):
        binding.serialize_example(FakeProcessor(), prompt, target)


def test_prefix_alignment_failure_blocks_labels() -> None:
    with pytest.raises(ConfigError, match="prefix_alignment_failed"):
        binding.serialize_example(BrokenPrefixProcessor(), "question", "answer")


def test_consumer_causal_gate_failure_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def reject(_config: object, _role: str) -> object:
        raise ConfigError("five_role_forbidden_content_reached_prompt")

    monkeypatch.setattr(binding.consumer, "load_role_dataset", reject)
    with pytest.raises(ConfigError, match="forbidden_content_reached_prompt"):
        binding.load_all_role_datasets(_config())


def test_stable_snapshot_detects_toctou(tmp_path: Path) -> None:
    path = tmp_path / "value.json"
    path.write_bytes(b"{}\n")
    snapshot = binding._stable_digest(path, capture_bytes=True)
    path.write_bytes(b'{"changed":true}\n')
    with pytest.raises(ConfigError, match="toctou_detected"):
        snapshot.assert_unchanged()


def test_json_schemas_are_draft_2020_12_valid() -> None:
    for path in (
        REPO_ROOT / binding.CONFIG_SCHEMA_PATH,
        REPO_ROOT / binding.MANIFEST_SCHEMA_PATH,
    ):
        schema = json.loads(path.read_text("utf-8"))
        Draft202012Validator.check_schema(schema)


def test_direct_cli_help_needs_no_editable_install() -> None:
    for script in (BUILD_SCRIPT, AUDIT_SCRIPT):
        result = subprocess.run(
            [sys.executable, str(script), "--help"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert result.returncode == 0
        assert "No model, CUDA, network, provider" in result.stdout.replace("\n", " ")


def test_published_fixture_is_byte_identical_rebuild_and_no_replace() -> None:
    artifact = REPO_ROOT / binding.OUTPUT_PATH
    if not artifact.is_dir():
        pytest.skip("fixture is generated by the explicit no-replace build CLI")
    manifest = binding.audit_manifest()
    assert manifest["summary"]["records"] == 1000
    assert manifest["tokenization"]["sequence_length"] == 768
    assert manifest["tokenization"]["observed_maximum_tokens"] == 665
    assert manifest["tokenization"]["records_over_512"] == 514
    assert manifest["tokenization"]["records_over_sequence_length"] == 0
    assert manifest["tokenization"]["truncation_used"] is False
    assert manifest["summary"]["aggregate"]["token_length"] == {
        "min": 449,
        "max": 665,
        "mean": 533.062,
        "total": 533062,
    }
    assert [item["records"] for item in manifest["summary"]["splits"]] == [
        800,
        200,
    ]
    assert manifest["template"]["hf_visible_template_equivalence_records"] == 1000
    assert manifest["template"]["hf_visible_template_equivalence_mismatches"] == 0
    with pytest.raises(ConfigError, match="output_exists_no_replace"):
        binding.publish_manifest(manifest)


def test_manifest_never_publishes_raw_record_ids_or_text() -> None:
    artifact = REPO_ROOT / binding.OUTPUT_PATH / "manifest.json"
    if not artifact.is_file():
        pytest.skip("fixture is generated by the explicit no-replace build CLI")
    text = artifact.read_text("utf-8")
    assert '"input_ids":[' not in text
    assert '"labels":[' not in text
    assert '"materialized_prompt":' not in text
    assert '"serialized_assistant_output":' not in text
    assert '"record_id":' not in text
