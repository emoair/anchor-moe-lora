import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_3080ti_safe_profile_is_conservative():
    profiles = json.loads(
        (ROOT / "configs" / "serving" / "profiles.json").read_text(encoding="utf-8")
    )["profiles"]
    safe = profiles["3080ti-safe"]

    assert safe["max_model_len"] == 1024
    assert safe["allowed_max_model_len"] == [1024, 2048]
    assert safe["max_num_seqs"] == 1
    assert safe["max_loras"] == 1
    assert safe["max_cpu_loras"] == 6
    assert safe["max_lora_rank"] == 16
    assert safe["enforce_eager"] is True
    assert safe["enable_prefix_caching"] is False
    assert safe["enable_chunked_prefill"] is False
    assert safe["language_model_only"] is True
    assert safe["speculative_decoding"] is False
    assert safe["kv_cache_dtype"] == "auto"
    assert "quantization" not in safe
    assert "load_format" not in safe


def test_throughput_profile_requires_gate_and_keeps_memory_boundaries():
    profiles = json.loads(
        (ROOT / "configs" / "serving" / "profiles.json").read_text(encoding="utf-8")
    )["profiles"]
    throughput = profiles["throughput"]

    assert throughput["gate"]
    assert throughput["enforce_eager"] is False
    assert throughput["enable_prefix_caching"] is True
    assert throughput["enable_chunked_prefill"] is True
    assert throughput["max_num_seqs"] == 1
    assert throughput["max_loras"] == 1
    assert throughput["kv_cache_dtype"] == "auto"
    assert throughput["speculative_decoding"] is False


def test_wsl_launcher_encodes_profile_and_no_fp8_override():
    launcher = (ROOT / "scripts" / "serve" / "start_vllm_wsl.sh").read_text(
        encoding="utf-8"
    )

    assert "--enforce-eager" in launcher
    assert "--no-enable-prefix-caching" in launcher
    assert "--no-enable-chunked-prefill" in launcher
    assert "--enable-prefix-caching" in launcher
    assert "--enable-chunked-prefill" in launcher
    assert "--max-lora-rank 16" in launcher
    assert "--kv-cache-dtype auto" in launcher
    assert "--language-model-only" in launcher
    assert "fp8" not in launcher.lower()
    assert "speculative-config" not in launcher.split("# No speculative-config", 1)[0]


def test_vram_probe_records_capacity_and_process_usage():
    probe = (ROOT / "scripts" / "serve" / "probe_vram_wsl.sh").read_text(
        encoding="utf-8"
    )

    assert "memory.total" in probe
    assert "memory.used" in probe
    assert "memory.free" in probe
    assert "query-compute-apps" in probe


def test_quantization_modes_are_explicit_and_profile_independent():
    payload = json.loads(
        (ROOT / "configs" / "serving" / "quantization_modes.json").read_text(
            encoding="utf-8"
        )
    )
    modes = payload["modes"]

    assert payload["independent_from_execution_profile"] is True
    assert modes["bitsandbytes"]["quantization"] == "bitsandbytes"
    assert modes["bitsandbytes"]["load_format"] == "bitsandbytes"
    assert modes["bitsandbytes"]["requires_prequantized_checkpoint"] is False
    assert modes["compressed-tensors"]["quantization"] == "compressed-tensors"
    assert modes["compressed-tensors"]["load_format"] == "auto"
    assert modes["compressed-tensors"]["requires_prequantized_checkpoint"] is True


def test_serving_config_pins_canonical_base_and_defaults_local_bnb():
    config = json.loads(
        (ROOT / "configs" / "serving" / "vllm.example.json").read_text(
            encoding="utf-8"
        )
    )

    assert config["canonical_base_model"] == "google/gemma-4-12B"
    assert len(config["canonical_revision"]) == 40
    assert config["served_model_path"] == "models/google-gemma-4-12B-base"
    assert config["default_quantization"] == "bitsandbytes"
    assert config["default_load_format"] == "bitsandbytes"
    assert set(config["adapters"]) == {
        "planner",
        "tool_policy",
        "frontend",
        "review",
        "security",
        "mixed",
    }


def test_launcher_declares_both_validated_load_paths_without_single_qlora_binding():
    bash = (ROOT / "scripts" / "serve" / "start_vllm_wsl.sh").read_text(
        encoding="utf-8"
    )
    powershell = (ROOT / "scripts" / "serve" / "start_vllm.ps1").read_text(
        encoding="utf-8"
    )

    assert 'quantization="bitsandbytes"' in bash
    assert 'load_format="bitsandbytes"' in bash
    assert 'load_format="auto"' in bash
    assert '--load-format "$load_format"' in bash
    assert "bitsandbytes in-flight quantization requires" in powershell
    assert "compressed-tensors requires LoadFormat=auto" in powershell
    assert "models\\google-gemma-4-12B-base" in powershell
    assert "qlora-adapter" not in bash.lower()
    assert "qlora_adapter" not in bash.lower()
