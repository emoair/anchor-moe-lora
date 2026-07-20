from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import threading
from urllib.request import urlopen

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "observability" / "distillation_dashboard.py"
SPEC = importlib.util.spec_from_file_location("distillation_dashboard", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
dashboard = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = dashboard
SPEC.loader.exec_module(dashboard)


def _append_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _record(
    seed_id: str,
    *,
    usage: bool = True,
    attempts: bool = True,
    model: str | None = None,
    cache_usage: bool = False,
) -> dict:
    provider: dict = {"protocol": "fixture"}
    if usage:
        public_usage = {"input_tokens": 3, "output_tokens": 5, "total_tokens": 8}
        if cache_usage:
            public_usage.update({"cache_read_tokens": 0, "cache_write_tokens": 0})
        provider["completion"] = {"usage": public_usage}
    if attempts:
        provider["attempts"] = {"wire_attempts": 1, "retry_count": 0}
    if model is not None:
        provider.update(
            {
                "model": model,
                "protocol": "openai",
                "base_url": "https://custom.example/v1",
            }
        )
    return {
        "id": f"record-{seed_id}",
        "input": {
            "prompt": "DO-NOT-RETURN-PROMPT",
            "absolute_path": r"C:\Users\private\secret-workspace",
        },
        "messages": [
            {"role": "user", "content": "DO-NOT-RETURN-MESSAGE"},
            {"role": "assistant", "content": "DO-NOT-RETURN-CODE"},
        ],
        "output": {"code": "DO-NOT-RETURN-CODE", "key": "sk-secret-fixture"},
        "provenance": {
            "seed_id": seed_id,
            "teacher": {
                "generation_params": {
                    "max_output_tokens_total": 1_000,
                    "max_requests": 100,
                },
                "provider": provider,
            },
        },
    }


def _fixture_shard(tmp_path: Path) -> Path:
    shard = tmp_path / "private-shard-directory"
    _append_jsonl(
        shard / "seeds.jsonl",
        [
            {"seed_id": "seed-a", "request": "DO-NOT-RETURN-SEED-BODY"},
            {"seed_id": "seed-b"},
        ],
    )
    for stage, filename in dashboard.STAGE_FILES.items():
        del stage
        _append_jsonl(shard / filename, [_record("seed-a")])
    _append_jsonl(
        shard / "data_plan.jsonl",
        [_record("seed-b", usage=False, attempts=False)],
    )
    _append_jsonl(
        shard / "automation" / "attempts.jsonl",
        [
            {
                "error_class": "ProviderRateLimit",
                "task_type": "frontend",
                "seed_id": "DO-NOT-RETURN-ATTEMPT-SEED",
                "teacher_content": "DO-NOT-RETURN-ATTEMPT-CONTENT",
            }
        ],
    )
    status = {
        "state": "running",
        "quota_epoch": {
            "requests_used": 12,
            "output_tokens_used": 40,
            "max_requests": 100,
            "max_output_tokens_total": 1_000,
        },
        "quota_history": [
            {"requests_used": 8, "output_tokens_used": 60, "closed_at": "fixture"}
        ],
        "audit_ledger": {
            "requests_total": 20,
            "output_tokens_total": 100,
            "secret": "DO-NOT-RETURN-STATUS-CONTENT",
        },
        "usage_checkpoint_policy": {"maximum_seconds": 5},
    }
    (shard / "automation" / "status.json").write_text(
        json.dumps(status), encoding="utf-8"
    )
    return shard


def test_selective_scanner_materializes_only_whitelisted_metadata() -> None:
    raw = json.dumps(_record("seed-a"), ensure_ascii=False).encode()
    metadata = dashboard.scan_metadata(raw, dashboard.RECORD_PATHS)

    assert metadata[("provenance", "seed_id")] == "seed-a"
    assert (
        metadata[
            (
                "provenance",
                "teacher",
                "provider",
                "completion",
                "usage",
                "total_tokens",
            )
        ]
        == 8
    )
    serialized = repr(metadata)
    assert "DO-NOT-RETURN" not in serialized
    assert "sk-secret" not in serialized


def test_snapshot_is_content_free_and_marks_unknown_usage(tmp_path: Path) -> None:
    shard = _fixture_shard(tmp_path)
    engine = dashboard.DashboardEngine([("fixture", shard)])

    snapshot = engine.snapshot()
    public = snapshot["shards"][0]

    assert public["state"] == "running"
    assert public["complete_chains"] == {
        "value": 1,
        "exact": True,
        "unknown_rows": 0,
        "source": "seed_id_intersection",
    }
    assert public["stages"]["plan"]["rows"] == 2
    assert public["tokens"]["input"]["value"] == 15
    assert public["tokens"]["input"]["exact"] is False
    assert public["tokens"]["input"]["unknown_rows"] == 15
    assert public["tokens"]["output"] == {
        "value": 100,
        "exact": True,
        "unknown_rows": 0,
        "source": "audit_ledger_checkpoint",
    }
    assert public["retained_stage_tokens"]["output"]["value"] == 25
    assert public["retained_stage_tokens"]["output"]["exact"] is False
    assert public["retained_stage_tokens"]["output"]["source"] == (
        "retained_stage_provider_usage_subtotal"
    )
    assert public["wire_attempts"]["value"] == 5
    assert public["wire_attempts"]["exact"] is False
    assert public["requests"]["value"] == 20
    assert public["requests"]["exact"] is True
    assert public["budget"]["request_percent"]["value"] == 12.0
    assert public["errors"]["by_type"] == {"ProviderRateLimit": 1}

    serialized = json.dumps(snapshot, ensure_ascii=False)
    assert str(tmp_path) not in serialized
    assert "DO-NOT-RETURN" not in serialized
    assert "sk-secret" not in serialized
    assert "private-shard-directory" not in serialized


def test_invalid_json_reports_only_line_and_hash(tmp_path: Path) -> None:
    shard = _fixture_shard(tmp_path)
    invalid = b'{"messages":["DO-NOT-RETURN-BROKEN"]'
    with (shard / "data_review.jsonl").open("ab") as handle:
        handle.write(invalid + b"\n")

    snapshot = dashboard.DashboardEngine([("fixture", shard)]).snapshot()
    errors = snapshot["shards"][0]["errors"]["invalid_json_lines"]

    assert errors[-1] == {
        "source": "review",
        "line": 2,
        "sha256": hashlib.sha256(invalid).hexdigest(),
    }
    assert "DO-NOT-RETURN-BROKEN" not in json.dumps(snapshot)


def test_seed_rejections_expose_only_content_free_reason_codes(tmp_path: Path) -> None:
    shard = _fixture_shard(tmp_path)
    _append_jsonl(
        shard / "seed_rejections.jsonl",
        [
            {
                "seed_index": 991,
                "error_class": "ValueError",
                "reason": "seed contains active payload material",
                "raw_response_sha256": "DO-NOT-RETURN-RESPONSE-HASH",
                "content_retained": False,
                "observed_at": "2026-07-13T12:34:56+00:00",
            },
            {
                "seed_index": 992,
                "error_class": "DataValidationError",
                "reason": "DO-NOT-RETURN-FREE-FORM-REASON",
                "raw_response_sha256": "DO-NOT-RETURN-SECOND-HASH",
                "content_retained": False,
                "observed_at": "2026-07-13T12:35:56Z",
            },
            {
                "error_class": "ValueError",
                "reason": "seed contains credential-like material",
                "content_retained": True,
                "observed_at": "not-a-time",
            },
        ],
    )

    snapshot = dashboard.DashboardEngine([("fixture", shard)]).snapshot()
    public = snapshot["shards"][0]["seed_rejections"]

    assert public["value"] == 3
    assert public["exact"] is True
    assert public["content_retained"] is False
    assert public["by_reason"] == {
        "active_payload_material": 1,
        "metadata_policy_violation": 1,
        "unclassified_validation": 1,
    }
    assert public["recent"][0] == {
        "reason": "metadata_policy_violation",
        "error_class": "ValueError",
        "observed_at": None,
    }
    serialized = json.dumps(snapshot, ensure_ascii=False)
    assert "DO-NOT-RETURN" not in serialized
    assert "991" not in serialized


def test_audit_ledger_totals_override_current_epoch_and_retained_subtotals(
    tmp_path: Path,
) -> None:
    shard = tmp_path / "multi-epoch-shard"
    _append_jsonl(
        shard / "seeds.jsonl", [{"seed_id": f"seed-{index}"} for index in range(3)]
    )
    _append_jsonl(
        shard / "seed_rejections.jsonl",
        [
            {
                "error_class": "ValueError",
                "reason": "seed contains active payload material",
                "content_retained": False,
            }
        ],
    )
    _append_jsonl(shard / "data_plan.jsonl", [_record("seed-0")])
    status = shard / "automation" / "status.json"
    status.parent.mkdir(parents=True)
    status.write_text(
        json.dumps(
            {
                "state": "running",
                "quota_epoch": {
                    "requests_used": 9,
                    "output_tokens_used": 90,
                    "max_requests": 100,
                    "max_output_tokens_total": 1_000,
                },
                "quota_history": [
                    {"requests_used": 1_244, "output_tokens_used": 2_111_959}
                ],
                "audit_ledger": {
                    "requests_total": 1_253,
                    "output_tokens_total": 2_112_049,
                },
            }
        ),
        encoding="utf-8",
    )

    public = dashboard.DashboardEngine([("multi", shard)]).snapshot()["shards"][0]

    assert public["requests"]["value"] == 1_253
    assert public["requests"]["source"] == "audit_ledger_checkpoint"
    assert public["tokens"]["output"]["value"] == 2_112_049
    assert public["tokens"]["output"]["exact"] is True
    assert public["tokens"]["input"]["value"] == 3
    assert public["tokens"]["input"]["exact"] is False
    assert public["retained_stage_tokens"]["output"]["value"] == 5
    assert public["budget"]["request_percent"]["value"] == 9.0
    assert public["budget"]["request_percent"]["source"] == "current_quota_epoch"
    assert public["budget"]["output_token_percent"]["value"] == 9.0
    assert public["seed_rejections"]["value"] == 1


def test_current_epoch_is_never_reported_as_cumulative_usage(tmp_path: Path) -> None:
    shard = tmp_path / "legacy-status-shard"
    _append_jsonl(shard / "seeds.jsonl", [{"seed_id": "seed-0"}])
    _append_jsonl(shard / "data_plan.jsonl", [_record("seed-0")])
    status = shard / "automation" / "status.json"
    status.parent.mkdir(parents=True)
    status.write_text(
        json.dumps(
            {
                "state": "running",
                "quota_epoch": {
                    "requests_used": 8,
                    "output_tokens_used": 80,
                    "max_requests": 100,
                    "max_output_tokens_total": 1_000,
                },
            }
        ),
        encoding="utf-8",
    )

    snapshot = dashboard.DashboardEngine([("legacy", shard)]).snapshot()
    public = snapshot["shards"][0]

    assert public["requests"] == {
        "value": None,
        "exact": False,
        "unknown_rows": 0,
        "source": "audit_ledger_checkpoint",
    }
    assert public["tokens"]["output"] == {
        "value": None,
        "exact": False,
        "unknown_rows": 0,
        "source": "audit_ledger_checkpoint",
    }
    assert public["budget"]["request_percent"]["value"] == 8.0
    assert public["budget"]["output_token_percent"]["value"] == 8.0
    assert snapshot["totals"]["requests"]["value"] is None
    assert snapshot["totals"]["requests"]["exact"] is False


def test_incremental_reader_only_reads_appended_bytes(tmp_path: Path) -> None:
    path = tmp_path / "data_plan.jsonl"
    first = json.dumps(_record("seed-a"), separators=(",", ":")).encode() + b"\n"
    path.write_bytes(first)
    reader = dashboard.IncrementalJsonl(path, "plan", "stage")

    assert reader.refresh() is True
    first_bytes = reader.bytes_read_total
    assert first_bytes == len(first)
    assert reader.refresh() is False
    assert reader.bytes_read_total == first_bytes

    second = json.dumps(_record("seed-b"), separators=(",", ":")).encode() + b"\n"
    with path.open("ab") as handle:
        handle.write(second)
    assert reader.refresh() is True
    assert reader.bytes_read_total == first_bytes + len(second)
    assert reader.aggregate.rows == 2


def test_partial_line_waits_for_newline(tmp_path: Path) -> None:
    path = tmp_path / "data_frontend.jsonl"
    encoded = json.dumps(_record("seed-a"), separators=(",", ":")).encode()
    path.write_bytes(encoded)
    reader = dashboard.IncrementalJsonl(path, "frontend", "stage")

    reader.refresh()
    assert reader.aggregate.rows == 0
    with path.open("ab") as handle:
        handle.write(b"\n")
    reader.refresh()
    assert reader.aggregate.rows == 1


def test_http_api_is_read_only_content_free_and_no_store(tmp_path: Path) -> None:
    shard = _fixture_shard(tmp_path)
    engine = dashboard.DashboardEngine([("fixture", shard)])
    server = dashboard.DashboardServer(("127.0.0.1", 0), engine, b"<p>fixture</p>")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        with urlopen(f"http://127.0.0.1:{port}/api/snapshot", timeout=5) as response:
            payload = response.read().decode("utf-8")
            assert response.headers["Cache-Control"] == "no-store"
            assert response.headers["X-Content-Type-Options"] == "nosniff"
        parsed = json.loads(payload)
        assert parsed["privacy"]["content_free"] is True
        assert str(tmp_path) not in payload
        assert "DO-NOT-RETURN" not in payload
        assert "sk-secret" not in payload
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_http_catalog_api_is_read_only_pinned_and_secret_free(tmp_path: Path) -> None:
    catalog = dashboard.CatalogService(state_dir=tmp_path / "catalog-state")
    engine = dashboard.DashboardEngine([], catalog=catalog)
    server = dashboard.DashboardServer(("127.0.0.1", 0), engine, b"<p>x</p>")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        port = server.server_address[1]
        with urlopen(f"http://127.0.0.1:{port}/api/catalog", timeout=5) as response:
            payload = response.read().decode("utf-8")
        parsed = json.loads(payload)
        assert parsed["content_safe"] is True
        assert parsed["secrets_read"] is False
        assert parsed["provenance"]["source_tag"] == "v3.16.5"
        assert parsed["update_status"]["automatic_apply"] is False
        assert str(tmp_path) not in payload
        assert "api_key" not in payload.casefold()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_snapshot_pinned_cost_never_guesses_missing_cache_usage(tmp_path: Path) -> None:
    shard = tmp_path / "priced-shard"
    _append_jsonl(shard / "seeds.jsonl", [{"seed_id": "seed-a"}])
    for filename in dashboard.STAGE_FILES.values():
        _append_jsonl(
            shard / filename,
            [_record("seed-a", model="gpt-5.5-low", cache_usage=True)],
        )
    catalog = dashboard.CatalogService(state_dir=tmp_path / "catalog-state")

    exact = dashboard.DashboardEngine([("priced", shard)], catalog=catalog).snapshot()[
        "shards"
    ][0]["pinned_cost"]
    assert exact["known"] is True
    assert exact["exact"] is True
    assert exact["canonical_model_id"] == "gpt-5.5"
    assert exact["total"] == "0.000825"

    _append_jsonl(
        shard / "data_plan.jsonl",
        [_record("seed-b", model="gpt-5.5-low", cache_usage=False)],
    )
    unknown = dashboard.DashboardEngine(
        [("priced", shard)], catalog=catalog
    ).snapshot()["shards"][0]["pinned_cost"]
    assert unknown["known"] is False
    assert unknown["reason"] == "cache_read_usage_unknown"
    assert unknown["total"] is None


def test_parse_shards_returns_only_operator_label_publicly(tmp_path: Path) -> None:
    shard = _fixture_shard(tmp_path)
    parsed = dashboard.parse_shards([f"public-01={shard}"])
    snapshot = dashboard.DashboardEngine(parsed).snapshot()

    assert snapshot["shards"][0]["label"] == "public-01"
    assert str(shard) not in json.dumps(snapshot)


def test_bundled_page_uses_safe_dom_updates_and_local_api_only() -> None:
    asset = (
        ROOT / "scripts" / "observability" / "dashboard_assets" / "index.html"
    ).read_text(encoding="utf-8")

    assert 'fetch("/api/snapshot"' in asset
    assert "textContent" in asset
    assert "replaceChildren" in asset
    assert "innerHTML" not in asset
    assert not re.search(r"<(?:script|link)\b[^>]+(?:src|href)=[\"']https?://", asset)


def test_bundled_page_keeps_provider_controls_manual_and_route_claims_exact() -> None:
    asset = (
        ROOT / "scripts" / "observability" / "dashboard_assets" / "index.html"
    ).read_text(encoding="utf-8")

    for control_id in (
        "base-url",
        "protocol",
        "model-id",
        "force-model",
        "reasoning-enabled",
        "reasoning-effort",
        "concurrency",
        "formal-max-tasks",
        "max-retries",
        "reconnect-attempts",
        "network-route",
        "route-component-state",
        "bank-gate-state",
        "execution-contract-state",
        "official-evaluation-state",
        "live-start-state",
        "container-route-state",
        "language-routing-state",
        "zh-localization-state",
        "formal-runtime-state",
        "formal-progress-state",
        "formal-speed-state",
        "formal-eta-state",
        "formal-error-state",
        "control-target",
    ):
        assert f'id="{control_id}"' in asset
    assert '<input id="concurrency" type="number" min="1" step="1" value="1" required>' in asset
    assert 'id="formal-max-tasks" type="number" min="1" max="19008"' in asset
    assert 'id="concurrency" type="number" min="1" max=' not in asset
    assert '<option value="max" selected>max</option>' in asset
    assert 'postControl("/api/control/formal-start"' in asset
    assert 'postControl("/api/control/formal-start", formalRunPayload())' in asset
    assert 'cap=${whole(currentFormal.max_tasks)}' in asset
    assert 'postControl("/api/control/formal-continue"' in asset
    assert 'postControl("/api/control/formal-stop"' in asset
    assert 'fetch("/api/control/formal-status"' in asset
    assert 'postControl("/api/control/start", newRunPayload())' in asset
    assert 'postControl("/api/control/continue"' in asset
    assert 'postControl("/api/control/stop"' in asset
    assert 'postControl("/api/control/models"' in asset
    assert "does not pin physical NIC" in asset
    assert "不会锁定物理网卡" in asset
    assert "component evidence is not E2E readiness" in asset
    assert "组件证明不等于端到端就绪" in asset
    assert "assignment only; not translated body text" in asset
    assert "仅为路由分配，不代表中文正文已完成" in asset
    assert "observed npm-only v2" not in asset
    assert "当前仅证明 npm v2" not in asset
    assert "SAFE PAUSE (GRACEFUL STOP)" in asset
    assert "Formal full-bank lifecycle controls are not wired" not in asset
    assert "正式全题库的开始/暂停/继续/停止尚未接入" not in asset
    assert "DISTILLATION RUN / WORKLOAD" in asset
    assert "蒸馏运行 / 工作负载" in asset
    assert "content-safe observability · local control" in asset
    assert "内容安全监控 · 本地控制" in asset


def test_formal_ui_hides_untrusted_or_disconnected_live_telemetry() -> None:
    asset = (
        ROOT / "scripts" / "observability" / "dashboard_assets" / "index.html"
    ).read_text(encoding="utf-8")

    assert (
        "const telemetryTrusted = currentFormal.telemetry_trusted === true;"
        in asset
    )
    assert 'formalProgressStateNode.textContent = t("formal.telemetry_untrusted")' in asset
    assert 'status.historical_unbound' in asset
    assert 'status.stale_status' in asset
    assert 'status.untrusted_status' in asset
    assert "telemetry_trusted: false" in asset
    assert "completed_tasks: null" in asset
    assert "stage_counts: {}" in asset
    assert "tasks_per_minute: null" in asset
    assert "provider_output_tokens_per_second: null" in asset
    assert "eta_seconds: null" in asset


def test_formal_status_response_separates_train_and_official_gate_labels() -> None:
    asset = (
        ROOT / "scripts" / "observability" / "dashboard_assets" / "index.html"
    ).read_text(encoding="utf-8")

    assert "function renderFormalGates(gates, executionStatus = {})" in asset
    assert 'id="official-evaluation-state"' in asset
    assert "official heldout evaluation (non-blocking)" in asset
    assert "官方 heldout 评测（不阻塞蒸馏）" in asset
    assert (
        "renderFormalGates(gates, (lastOptions || {}).formal_execution || {});"
        in asset
    )
    assert "renderFormalGates(formalGates, formalExecution);" in asset
    for field in (
        "component_ready",
        "bank_ready",
        "execution_contract_ready",
        "live_start_allowed",
    ):
        assert f"value.{field} === true" in asset
        assert f"value.{field} === false" in asset
    assert "execution.official_evaluation_contract_ready === true" in asset
    assert "execution.official_evaluation_contract_ready === false" in asset
    assert "clearKeyNode.disabled = currentFormal.can_stop === true;" in asset
    assert "clearKeyNode.disabled = !enabled || currentControl.can_stop === true;" in asset


def test_bundled_page_freezes_last_snapshot_and_classifies_disconnects() -> None:
    asset = (
        ROOT / "scripts" / "observability" / "dashboard_assets" / "index.html"
    ).read_text(encoding="utf-8")

    assert "Disconnected from dashboard backend" in asset
    assert "已与面板后端断开连接" in asset
    assert "let lastSnapshot = null;" in asset
    assert "connectionHealth.lastSuccessAt = now;" in asset
    assert "connectionHealth.failures += 1;" in asset
    assert "connectionHealth.nextRetryAt = now + POLL_INTERVAL_MS;" in asset
    assert (
        'if (!validSnapshot(snapshot)) throw { diagnosticReason: "invalid_schema" };'
        in asset
    )
    for reason in (
        "http_client_error",
        "http_server_error",
        "network_unreachable",
        "invalid_json",
        "invalid_schema",
    ):
        assert f'"{reason}"' in asset
        assert f'"reason.{reason}"' in asset
    poll_source = asset.split("async function poll()", 1)[1].split(
        'refreshNode.addEventListener("click", poll)', 1
    )[0]
    failure_branch = re.search(
        r"} catch \(error\) \{(.*?)\n\s*} finally \{", poll_source, re.DOTALL
    )
    assert failure_branch is not None
    assert "renderConnectionDiagnostics(lastSnapshot)" in failure_branch.group(1)
    assert "renderConnectionHeader(lastSnapshot)" in failure_branch.group(1)
    assert "renderSummary(" not in failure_branch.group(1)
    assert "renderShards(" not in failure_branch.group(1)
    assert "lastSnapshot =" not in failure_branch.group(1)
    assert "lastSnapshot = null" not in poll_source


def test_bundled_page_has_finite_bilingual_diagnostic_reasons() -> None:
    asset = (
        ROOT / "scripts" / "observability" / "dashboard_assets" / "index.html"
    ).read_text(encoding="utf-8")
    for reason in sorted(dashboard.DIAGNOSTIC_REASON_CODES):
        assert asset.count(f'"reason.{reason}"') == 2
    assert "Running normally; rolling window warming" in asset
    assert "运行正常；滚动窗口预热" in asset
    assert "not applicable · external process is observed read-only" in asset
    assert "不适用 · 外部进程仅作只读观察" in asset


def test_bundled_page_inline_javascript_is_valid(tmp_path: Path) -> None:
    node = shutil.which("node")
    if node is None:
        pytest.skip("Node.js is not installed")
    asset = (
        ROOT / "scripts" / "observability" / "dashboard_assets" / "index.html"
    ).read_text(encoding="utf-8")
    scripts = re.findall(r"<script(?:\s[^>]*)?>(.*?)</script>", asset, re.DOTALL)
    assert len(scripts) == 1
    source = tmp_path / "dashboard-inline.js"
    source.write_text(scripts[0], encoding="utf-8")

    completed = subprocess.run(
        [node, "--check", str(source)],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def test_bundled_page_has_complete_persisted_bilingual_dictionary() -> None:
    asset = (
        ROOT / "scripts" / "observability" / "dashboard_assets" / "index.html"
    ).read_text(encoding="utf-8")
    dictionaries = re.search(
        r"en: Object\.freeze\(\{(.*?)\}\),\s*"
        r'"zh-CN": Object\.freeze\(\{(.*?)\}\)\s*\}\);',
        asset,
        re.DOTALL,
    )

    assert dictionaries is not None
    english = set(re.findall(r'^\s*"([^"]+)":', dictionaries.group(1), re.MULTILINE))
    chinese = set(re.findall(r'^\s*"([^"]+)":', dictionaries.group(2), re.MULTILINE))
    assert english
    assert english == chinese
    used = set(
        re.findall(
            r'data-i18n(?:-title|-aria-label)?="([^"]+)"',
            asset,
        )
    )
    used.update(re.findall(r'\bt\("([^"]+)"', asset))
    assert used <= english
    assert 'id="language-toggle"' in asset
    assert 'const LANGUAGE_STORAGE_KEY = "anchor.dashboard.language"' in asset
    assert "navigator.language" in asset
    assert "window.localStorage.setItem(LANGUAGE_STORAGE_KEY, language)" in asset


def test_bundled_page_has_persisted_three_state_theme_control() -> None:
    asset = (
        ROOT / "scripts" / "observability" / "dashboard_assets" / "index.html"
    ).read_text(encoding="utf-8")

    assert '<html lang="en" data-theme="system">' in asset
    assert 'id="theme-toggle"' in asset
    assert '<label class="visually-hidden" for="theme-toggle"' in asset
    for mode in ("system", "light", "dark"):
        assert f'<option value="{mode}" data-i18n="theme.{mode}">' in asset
    assert 'const THEME_STORAGE_KEY = "anchor.dashboard.theme";' in asset
    assert 'Object.freeze(["system", "light", "dark"])' in asset
    assert "window.localStorage.getItem(THEME_STORAGE_KEY)" in asset
    assert "window.localStorage.setItem(THEME_STORAGE_KEY, theme)" in asset
    assert 'return "system";' in asset
    assert "document.documentElement.dataset.theme = theme;" in asset
    assert 'themeToggleNode.addEventListener("change"' in asset


def test_bundled_page_theme_is_monochrome_accessible_and_system_aware() -> None:
    asset = (
        ROOT / "scripts" / "observability" / "dashboard_assets" / "index.html"
    ).read_text(encoding="utf-8")

    assert '@media (prefers-color-scheme: light)' in asset
    assert 'html[data-theme="system"]' in asset
    assert 'html[data-theme="light"]' in asset
    assert 'html[data-theme="dark"]' in asset
    assert '<meta name="color-scheme" content="light dark">' in asset
    assert "--accent: #ffffff;" in asset
    assert "--accent: #111111;" in asset
    assert "#73f6a5" not in asset
    assert "radial-gradient" not in asset
    assert "button:focus-visible" in asset
    assert "select:focus-visible" in asset
    assert "@media (max-width: 480px)" in asset


def test_bundled_page_surfaces_provider_scoped_glm52_pricing() -> None:
    asset = (
        ROOT / "scripts" / "observability" / "dashboard_assets" / "index.html"
    ).read_text(encoding="utf-8")

    assert "function formatCatalogPrice(pricing)" in asset
    assert "subscription quota · marginal token price UNKNOWN" in asset
    assert "订阅额度 · Token 边际价未知" in asset
    assert "{currency}/1M · in {input} · out {output} · cache hit {cacheRead}" in asset
    assert (
        "{currency}/百万 Token · 输入 {input} · 输出 {output} · 缓存命中 {cacheRead}"
        in asset
    )


def test_rolling_rates_report_requests_tokens_and_stage_rows(tmp_path: Path) -> None:
    shard = tmp_path / "rate-shard"
    _append_jsonl(shard / "seeds.jsonl", [{"seed_id": "seed-a"}])
    for filename in dashboard.STAGE_FILES.values():
        _append_jsonl(shard / filename, [_record("seed-a")])
    status_path = shard / "automation" / "status.json"
    status_path.parent.mkdir(parents=True)
    status_path.write_text(
        json.dumps(
            {
                "state": "running",
                "quota_epoch": {
                    "requests_used": 5,
                    "output_tokens_used": 25,
                    "max_requests": 100,
                    "max_output_tokens_total": 1_000,
                },
                "audit_ledger": {"requests_total": 5, "output_tokens_total": 25},
            }
        ),
        encoding="utf-8",
    )
    monitor = dashboard.ShardMonitor("rate-fixture", shard)
    monitor.refresh(100.0)

    _append_jsonl(shard / "data_plan.jsonl", [_record("seed-b")])
    status_path.write_text(
        json.dumps(
            {
                "state": "running",
                "quota_epoch": {
                    "requests_used": 6,
                    "output_tokens_used": 30,
                    "max_requests": 100,
                    "max_output_tokens_total": 1_000,
                },
                "audit_ledger": {"requests_total": 6, "output_tokens_total": 30},
            }
        ),
        encoding="utf-8",
    )
    monitor.refresh(160.0)
    rates = monitor.public()["rates"]

    assert rates["requests_per_minute"]["value"] == 1.0
    assert rates["requests_per_minute"]["exact"] is True
    assert rates["requests_per_minute"]["source"] == "rolling_audit_ledger_60s"
    assert rates["wire_attempts_per_minute"]["value"] == 1.0
    assert rates["provider_output_tokens_per_second"]["value"] == pytest.approx(
        5 / 60, abs=1e-6
    )
    assert rates["provider_output_tokens_per_second"]["source"] == (
        "rolling_audit_ledger_60s"
    )
    assert rates["retained_tokens_per_second"]["output"]["value"] == pytest.approx(
        5 / 60, abs=1e-6
    )
    assert rates["retained_tokens_per_second"]["output"]["source"] == (
        "rolling_retained_rows_60s"
    )
    assert rates["stage_rows_per_minute"]["plan"]["value"] == 1.0
    assert rates["stage_rows_per_minute"]["total"]["value"] == 1.0


def test_cold_start_is_reported_as_normal_warming_not_generic_unknown(
    tmp_path: Path,
) -> None:
    shard = _fixture_shard(tmp_path)
    monitor = dashboard.ShardMonitor("cold-start", shard)

    monitor.refresh(100.0)
    first = monitor.public()

    assert first["state"] == "running"
    assert first["rates"]["requests_per_minute"]["value"] is None
    assert first["diagnostics"]["summary"] == "normal_warming"
    assert first["diagnostics"]["reason_codes"] == ["telemetry_cold_start"]

    monitor.refresh(102.1)
    second = monitor.public()
    assert second["rates"]["requests_per_minute"]["value"] == 0.0
    assert second["diagnostics"]["summary"] == "normal_warming"
    assert second["diagnostics"]["reason_codes"] == ["telemetry_warming"]


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("cooldown", {"provider_cooldown", "rate_limit"}),
        ("provider_quota_exhausted", {"quota"}),
        ("client_deadline", {"client_deadline"}),
    ],
)
def test_diagnostic_workload_reasons_are_finite_and_content_free(
    tmp_path: Path, state: str, expected: set[str]
) -> None:
    shard = _fixture_shard(tmp_path)
    status_path = shard / "automation" / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status.update(
        {
            "state": state,
            "updated_at": "2026-07-14T01:02:03+00:00",
            "cooldown_until": "2026-07-14T01:05:03+00:00",
            "provider_body": "DO-NOT-RETURN-PROVIDER-BODY",
        }
    )
    status_path.write_text(json.dumps(status), encoding="utf-8")

    public = dashboard.DashboardEngine([("fixture", shard)]).snapshot()["shards"][0]
    reasons = set(public["diagnostics"]["reason_codes"])

    assert expected <= reasons
    assert reasons <= dashboard.DIAGNOSTIC_REASON_CODES
    serialized = json.dumps(public["diagnostics"], ensure_ascii=False)
    assert "DO-NOT-RETURN" not in serialized
    assert str(shard) not in serialized


def test_diagnostics_distinguish_stale_parse_and_missing_counter(
    tmp_path: Path,
) -> None:
    stale_shard = _fixture_shard(tmp_path / "stale")
    stale = dashboard.ShardMonitor("stale", stale_shard)
    stale.refresh(1.0)
    stale.status_reader.last_mtime = 100.0
    for reader in [
        stale.seed_reader,
        stale.rejection_reader,
        stale.attempt_reader,
        *stale.stage_readers.values(),
    ]:
        reader.last_mtime = 1000.0
    assert "status_stale" in stale.public()["diagnostics"]["reason_codes"]

    parse_shard = _fixture_shard(tmp_path / "parse")
    with (parse_shard / "data_security.jsonl").open("ab") as handle:
        handle.write(b'{"broken":true\n')
    parse_public = dashboard.DashboardEngine([("parse", parse_shard)]).snapshot()[
        "shards"
    ][0]
    assert "file_parse_error" in parse_public["diagnostics"]["reason_codes"]

    counter_shard = _fixture_shard(tmp_path / "counter")
    status_path = counter_shard / "automation" / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    status.pop("audit_ledger")
    status_path.write_text(json.dumps(status), encoding="utf-8")
    counter_public = dashboard.DashboardEngine([("counter", counter_shard)]).snapshot()[
        "shards"
    ][0]
    assert "unknown_counter" in counter_public["diagnostics"]["reason_codes"]


def test_connection_diagnostics_do_not_invent_external_reconnect(
    tmp_path: Path,
) -> None:
    shard = _fixture_shard(tmp_path)
    snapshot = dashboard.DashboardEngine([("external-c10", shard)]).snapshot()
    diagnostics = snapshot["diagnostics"]

    assert diagnostics["collector_alive"] is True
    assert diagnostics["process_alive"] is None
    assert diagnostics["ownership"] == "external_read_only"
    assert diagnostics["reconnect"] == {
        "applicable": False,
        "used": None,
        "maximum": None,
        "next_at": None,
    }
    assert diagnostics["last_exit"] == {"code": None, "signal": None}


def test_connection_diagnostics_expose_only_known_managed_exit_signal() -> None:
    shard = {
        "label": "managed-shard",
        "diagnostics": {
            "summary": "attention",
            "reason_codes": [],
            "observed_at": "2026-07-14T01:02:03+00:00",
        },
    }
    control = {
        "output_label": "managed-shard",
        "process_state": "failed",
        "exit_code": -9,
        "reconnect": {
            "used": 1,
            "maximum": 2,
            "next_at": "2026-07-14T01:02:13+00:00",
        },
    }

    public = dashboard._public_connection_diagnostics(
        [shard], control, observed_at="2026-07-14T01:02:04+00:00"
    )

    assert public["ownership"] == "managed"
    assert public["process_alive"] is False
    assert public["reason_codes"] == ["process_exit"]
    assert public["reconnect"]["next_at"] == "2026-07-14T01:02:13+00:00"
    assert public["last_exit"] == {"code": -9, "signal": 9}


def test_status_freshness_uses_checkpoint_aware_minimum_grace(tmp_path: Path) -> None:
    shard = _fixture_shard(tmp_path)
    monitor = dashboard.ShardMonitor("freshness", shard)
    monitor.refresh(1.0)
    monitor.status_reader.last_mtime = 100.0
    monitor.status_reader.metadata[("usage_checkpoint_policy", "maximum_seconds")] = 5
    readers = [
        monitor.seed_reader,
        monitor.rejection_reader,
        monitor.attempt_reader,
        *monitor.stage_readers.values(),
    ]
    for reader in readers:
        reader.last_mtime = 129.9
    assert monitor._status_is_fresh() is True
    for reader in readers:
        reader.last_mtime = 130.1
    assert monitor._status_is_fresh() is False

    monitor.status_reader.metadata[("usage_checkpoint_policy", "maximum_seconds")] = 20
    for reader in readers:
        reader.last_mtime = 164.9
    assert monitor._status_is_fresh() is True
    for reader in readers:
        reader.last_mtime = 165.1
    assert monitor._status_is_fresh() is False


def test_cli_defaults_to_ipv4_loopback(tmp_path: Path) -> None:
    shard = _fixture_shard(tmp_path)
    args = dashboard.build_parser().parse_args(["--shard", f"fixture={shard}"])

    assert args.host == "127.0.0.1"
    assert args.port == 8765
