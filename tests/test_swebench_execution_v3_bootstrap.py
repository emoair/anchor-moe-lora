from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "tooling" / "bootstrap_swebench_harness_v3.ps1"


def test_bootstrap_is_pinned_explicit_and_never_pulls_images() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    assert "f7bbbb2ccdf479001d6467c9e34af59e44a840f9" in source
    assert 'ExpectedVersion = "4.1.0"' in source
    assert "ConfirmNetwork" in source
    assert "--no-tags --depth 1 origin $Revision" in source
    assert "build_swebench_execution_attestation.py" in source
    assert "/var/lib/anchor/keys/official-eval-hmac-v1" in source
    assert "secrets.token_bytes(64)" in source
    assert "chmod 600" in source
    assert "chown 0:0" in source
    assert "official_eval_receipt_key=ready" in source
    assert "Get-Content" not in source
    assert "podman pull" not in source.casefold()
    assert "podman system prune" not in source.casefold()
    assert "docker pull" not in source.casefold()
