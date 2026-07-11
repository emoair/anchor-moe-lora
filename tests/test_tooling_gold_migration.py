import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/tooling/migrate_legacy_tool_gold.py"


def test_legacy_migration_is_dry_by_default_and_preserves_source(tmp_path):
    completed = {
        "sample_id": "ok",
        "success": True,
        "public_outcome": {"status": "completed"},
    }
    failed = {"sample_id": "bad", "success": False, "public_outcome": None}
    source = tmp_path / "legacy.jsonl"
    original = "\n".join(json.dumps(row) for row in (failed, completed)) + "\n"
    source.write_text(original, encoding="utf-8")
    attempts = tmp_path / "attempts.jsonl"
    accepted = tmp_path / "accepted.jsonl"
    command = [
        sys.executable,
        str(SCRIPT),
        "--source",
        str(source),
        "--attempts-output",
        str(attempts),
        "--accepted-output",
        str(accepted),
    ]

    dry_run = subprocess.run(command, capture_output=True, text=True, check=False)
    assert dry_run.returncode == 0
    assert "DRY RUN" in dry_run.stdout
    assert not attempts.exists()
    assert not accepted.exists()

    migration = subprocess.run(
        [*command, "--confirm"], capture_output=True, text=True, check=False
    )
    assert migration.returncode == 0
    assert source.read_text(encoding="utf-8") == original
    assert len(attempts.read_text(encoding="utf-8").splitlines()) == 2
    accepted_rows = [
        json.loads(line) for line in accepted.read_text(encoding="utf-8").splitlines()
    ]
    assert [row["sample_id"] for row in accepted_rows] == ["ok"]
