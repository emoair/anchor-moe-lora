from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from html import escape
import json
import math
from pathlib import Path
from typing import Any, TypeGuard

from .metrics import compute_metrics
from .models import BenchmarkRecord, load_records_jsonl


PRIMARY_ORDER = ("base_matched_calls", "mixed_matched_calls", "c_pipeline")
BUDGET_MATCHED_ORDER = ("mixed_matched_calls", "d_budget_matched_pipeline")
AUXILIARY_ORDER = ("a_base", "b_mixed")
NA = "N/A"


@dataclass(frozen=True)
class ReportPaths:
    summary: Path
    metrics_csv: Path
    chart_svg: Path


def generate_report(
    records_path: str | Path,
    metrics_path: str | Path,
    output_dir: str | Path,
) -> ReportPaths:
    records_source = Path(records_path)
    metrics_source = Path(metrics_path)
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    records = load_records_jsonl(records_source)
    supplied_metrics = json.loads(metrics_source.read_text(encoding="utf-8"))
    recomputed_metrics = compute_metrics(records)
    order = _baseline_order(recomputed_metrics)
    discrepancies = _compare_metrics(supplied_metrics, recomputed_metrics)

    paths = ReportPaths(
        summary=destination / "summary.md",
        metrics_csv=destination / "metrics.csv",
        chart_svg=destination / "comparison.svg",
    )
    _write_metrics_csv(paths.metrics_csv, recomputed_metrics, order)
    paths.chart_svg.write_text(
        _render_svg(recomputed_metrics, order), encoding="utf-8", newline="\n"
    )
    paths.summary.write_text(
        _render_summary(
            records,
            recomputed_metrics,
            order,
            discrepancies,
            records_source,
            metrics_source,
        ),
        encoding="utf-8",
        newline="\n",
    )
    return paths


def _baseline_order(metrics: dict[str, dict[str, Any]]) -> list[str]:
    present = set(metrics)
    ordered = [name for name in PRIMARY_ORDER if name in present]
    ordered.extend(name for name in AUXILIARY_ORDER if name in present)
    ordered.extend(sorted(present - set(ordered)))
    return ordered


def _comparison_role(name: str) -> str:
    if name in PRIMARY_ORDER:
        return "primary causal"
    if name in AUXILIARY_ORDER:
        return "auxiliary single-call"
    return "secondary fairness"


CSV_COLUMNS = (
    "baseline",
    "comparison_role",
    "cases",
    "structural_pass_proxy_not_true_pass_at_1",
    "build_pass_at_1",
    "frontend_build_pass_at_1",
    "plan_quality_rate",
    "review_repair_rate",
    "tool_policy_accuracy",
    "tool_policy_approve_accuracy",
    "tool_policy_block_accuracy",
    "tool_policy_escalate_accuracy",
    "composite_success_rate",
    "cost_per_success_tokens",
    "tpr_valid_security",
    "fpr_valid_security",
    "operational_malicious_block_rate",
    "operational_benign_block_rate",
    "mean_latency_ms",
    "mean_total_tokens",
    "peak_vram_mb",
    "error_rate",
    "fail_closed_rate",
    "unknown_decisions",
    "valid_security_cases",
)


def _write_metrics_csv(
    path: Path, metrics: dict[str, dict[str, Any]], order: list[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for name in order:
            values = metrics[name]
            writer.writerow(
                {
                    "baseline": name,
                    "comparison_role": _comparison_role(name),
                    "cases": _csv_value(values.get("cases")),
                    "structural_pass_proxy_not_true_pass_at_1": _csv_value(
                        values.get("structural_pass_proxy", values.get("pass_at_1"))
                    ),
                    "build_pass_at_1": _csv_value(values.get("build_pass_at_1")),
                    "frontend_build_pass_at_1": _csv_value(
                        values.get("frontend_build_pass_at_1")
                    ),
                    "plan_quality_rate": _csv_value(values.get("plan_quality_rate")),
                    "review_repair_rate": _csv_value(values.get("review_repair_rate")),
                    "tool_policy_accuracy": _csv_value(values.get("tool_policy_accuracy")),
                    "tool_policy_approve_accuracy": _csv_value(
                        values.get("tool_policy_approve_accuracy")
                    ),
                    "tool_policy_block_accuracy": _csv_value(
                        values.get("tool_policy_block_accuracy")
                    ),
                    "tool_policy_escalate_accuracy": _csv_value(
                        values.get("tool_policy_escalate_accuracy")
                    ),
                    "composite_success_rate": _csv_value(
                        values.get("composite_success_rate")
                    ),
                    "cost_per_success_tokens": _csv_value(
                        values.get("cost_per_success_tokens")
                    ),
                    "tpr_valid_security": _csv_value(values.get("tpr_valid_security")),
                    "fpr_valid_security": _csv_value(values.get("fpr_valid_security")),
                    "operational_malicious_block_rate": _csv_value(
                        values.get("operational_malicious_block_rate")
                    ),
                    "operational_benign_block_rate": _csv_value(
                        values.get("operational_benign_block_rate")
                    ),
                    "mean_latency_ms": _csv_value(values.get("mean_latency_ms")),
                    "mean_total_tokens": _csv_value(values.get("mean_total_tokens")),
                    "peak_vram_mb": _csv_value(values.get("peak_vram_mb")),
                    "error_rate": _csv_value(values.get("error_rate")),
                    "fail_closed_rate": _csv_value(values.get("fail_closed_rate")),
                    "unknown_decisions": _csv_value(values.get("unknown_decisions")),
                    "valid_security_cases": _csv_value(
                        values.get("valid_security_cases")
                    ),
                }
            )


def _csv_value(value: Any) -> Any:
    return NA if value is None else value


def _render_summary(
    records: list[BenchmarkRecord],
    metrics: dict[str, dict[str, Any]],
    order: list[str],
    discrepancies: list[str],
    records_path: Path,
    metrics_path: Path,
) -> str:
    missing_primary = [name for name in PRIMARY_ORDER if name not in metrics]
    backends = sorted({record.backend for record in records})
    provenance = _provenance_summary(records)
    generated = datetime.now(timezone.utc).isoformat()

    lines = [
        "# Anchor-MoE-LoRA benchmark report",
        "",
        "> **Structural proxy warning:** the legacy marker score retained for diagnostics is "
        "not true Pass@1. Only `build_pass_at_1` backed by isolated build/test execution is "
        "reported as build Pass@1; an unset value is not an OpenCode/tool-verified result.",
        "",
        "## Audit envelope",
        "",
        f"- Generated UTC: `{generated}`",
        f"- Records: `{records_path}` (`sha256:{_file_hash(records_path)}`)",
        f"- Supplied metrics: `{metrics_path}` (`sha256:{_file_hash(metrics_path)}`)",
        f"- Record count: `{len(records)}`",
        f"- Backend labels: `{', '.join(backends) if backends else NA}`",
        "- Report values are recomputed from records; supplied metrics are used for audit comparison.",
        "",
        "## Primary causal comparison",
        "",
        "The primary comparison is `base_matched_calls` vs `mixed_matched_calls` vs "
        "`c_pipeline`. The held-out protocol holds the five-stage Planner -> Tool-Policy -> "
        "Frontend -> Domain Review -> Final Security prompts, call structure, and token caps "
        "constant. "
        "A/B single-call results are auxiliary product-shape baselines and do not by "
        "themselves prove expert isolation.",
        "",
    ]
    if missing_primary:
        lines.extend(
            [
                f"> Missing primary baselines: `{', '.join(missing_primary)}`. No complete causal comparison is available.",
                "",
            ]
        )
    lines.extend(_markdown_table(metrics, [name for name in PRIMARY_ORDER if name in metrics]))
    lines.extend(
        [
            "",
            "### Architecture differences",
            "",
            "| Arm | Frozen Q4 base | Adapter assignment | Calls | Runtime authority |",
            "| --- | --- | --- | ---: | --- |",
            "| A `base_matched_calls` | Native Gemma 4 12B Q4 | No LoRA; base handles all five stages | 5 | Deterministic local allowlist |",
            "| B `mixed_matched_calls` | Same Q4 base | One `mixed-all` LoRA reused at every stage | 5 | Deterministic local allowlist |",
            "| C `c_pipeline` | Same Q4 base | Five task-specific LoRAs selected by application routing | 5 | Deterministic local allowlist |",
            "| D `d_budget_matched_pipeline` | Same Q4 base | Five smaller routed LoRAs whose total trainable parameters equal B | 5 | Deterministic local allowlist |",
            "",
            "Model `APPROVE/BLOCK/ESCALATE` output is scored but never grants tool permission. "
            "The C arm is a task-routed adapter pipeline, not a learned neural MoE.",
            "",
            "### Native Q4 = 100 normalized comparison",
            "",
        ]
    )
    lines.extend(_normalized_primary_table(metrics))
    if all(name in metrics for name in BUDGET_MATCHED_ORDER):
        lines.extend(
            [
                "",
                "### Equal adapter-parameter budget: B versus D",
                "",
                "B and D have the same materialized trainable parameter count. B puts the whole budget in one mixed-data adapter; D partitions it across five routed specialists.",
                "",
            ]
        )
        lines.extend(_markdown_table(metrics, list(BUDGET_MATCHED_ORDER)))
    lines.extend(["", "## Auxiliary and secondary baselines", ""])
    secondary = [name for name in order if name not in PRIMARY_ORDER]
    lines.extend(_markdown_table(metrics, secondary))
    lines.extend(
        [
            "",
            "## Metric interpretation",
            "",
            "- `build Pass@1`: only a real isolated `npm run build` + `npm run test` success; generated HTML is inspected as data and never executed.",
            "- `structural proxy`: marker-only diagnostic retained for legacy runs; it is not true Pass@1.",
            "- `plan quality`: all frozen held-out planning concepts are present in the public plan.",
            "- `tool policy`: model APPROVE/BLOCK/ESCALATE accuracy; deterministic enforcement is measured separately and remains authoritative.",
            "- `review repair`: the known benign mutation was applied and the exact lost behavior was restored.",
            "- `composite`: plan, tool policy, review, security, and (for benign cases) sandbox build all succeed.",
            "- `cost/success`: observed total tokens divided by composite successes; it is a local token-cost proxy, not a currency claim.",
            "- `valid TPR/FPR`: security rates only where inference succeeded and emitted PASS/BLOCK.",
            "- `operational block`: user-visible block rate including fail-closed infrastructure errors.",
            "- `N/A`: no valid denominator or no VRAM sample; it is not zero.",
            "- Latency and tokens are means; VRAM is the peak aggregate NVIDIA compute-process sample.",
            "",
            "## Evaluator provenance",
            "",
        ]
    )
    lines.extend(f"- `{item}`" for item in provenance)
    lines.extend(["", "## Supplied-metrics reconciliation", ""])
    if discrepancies:
        lines.extend(f"- {item}" for item in discrepancies)
    else:
        lines.append("- Supplied metrics match recomputation for shared fields.")
    if len(backends) > 1:
        lines.extend(
            [
                "",
                "> Multiple backend labels are present. Metrics group by baseline; split backend runs before causal interpretation.",
            ]
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `metrics.csv`: machine-readable values with explicit `N/A`.",
            "- `comparison.svg`: dependency-free chart with valid-security and operational panels separated.",
            "",
        ]
    )
    return "\n".join(lines)


def _markdown_table(
    metrics: dict[str, dict[str, Any]], names: list[str]
) -> list[str]:
    header = (
        "| Baseline | Role | Build Pass@1 | Plan | Tool policy | Review repair | Composite | Valid TPR | Valid FPR | "
        "Latency ms | Tokens | VRAM MiB | Cost/success | Error |"
    )
    rows = [header, "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    if not names:
        rows.append("| N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
        return rows
    for name in names:
        item = metrics[name]
        rows.append(
            "| "
            + " | ".join(
                [
                    _md(name),
                    _comparison_role(name),
                    _format_ratio(item.get("build_pass_at_1")),
                    _format_ratio(item.get("plan_quality_rate")),
                    _format_ratio(item.get("tool_policy_accuracy")),
                    _format_ratio(item.get("review_repair_rate")),
                    _format_ratio(item.get("composite_success_rate")),
                    _format_ratio(item.get("tpr_valid_security")),
                    _format_ratio(item.get("fpr_valid_security")),
                    _format_number(item.get("mean_latency_ms")),
                    _format_number(item.get("mean_total_tokens")),
                    _format_number(item.get("peak_vram_mb")),
                    _format_number(item.get("cost_per_success_tokens")),
                    _format_ratio(item.get("error_rate")),
                ]
            )
            + " |"
        )
    return rows


def _normalized_primary_table(metrics: dict[str, dict[str, Any]]) -> list[str]:
    if any(name not in metrics for name in PRIMARY_ORDER):
        return ["Complete A/B/C primary records are required for normalization."]
    definitions = (
        ("Build Pass@1", "build_pass_at_1", "ratio"),
        ("Plan quality", "plan_quality_rate", "ratio"),
        ("Tool policy accuracy", "tool_policy_accuracy", "ratio"),
        ("Tool APPROVE accuracy", "tool_policy_approve_accuracy", "ratio"),
        ("Tool BLOCK accuracy", "tool_policy_block_accuracy", "ratio"),
        ("Tool ESCALATE accuracy", "tool_policy_escalate_accuracy", "ratio"),
        ("Review repair", "review_repair_rate", "ratio"),
        ("Security TPR", "tpr_valid_security", "ratio"),
        ("Security FPR (lower is better)", "fpr_valid_security", "ratio"),
        ("Composite", "composite_success_rate", "ratio"),
        ("E2E latency ms (lower is better)", "mean_latency_ms", "number"),
        ("Token use (lower is better)", "mean_total_tokens", "number"),
        ("Peak VRAM MiB (lower is better)", "peak_vram_mb", "number"),
        ("Cost/success tokens (lower is better)", "cost_per_success_tokens", "number"),
    )
    rows = [
        "| Metric | A native Q4 absolute | A index | B absolute | B delta vs A | B index | C absolute | C delta vs A | C index |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    base = metrics[PRIMARY_ORDER[0]]
    mixed = metrics[PRIMARY_ORDER[1]]
    routed = metrics[PRIMARY_ORDER[2]]
    for label, key, kind in definitions:
        a = base.get(key)
        b = mixed.get(key)
        c = routed.get(key)
        rows.append(
            "| "
            + " | ".join(
                [
                    label,
                    _format_metric(a, kind),
                    "100.0" if _is_number(a) else NA,
                    _format_metric(b, kind),
                    _format_delta(b, a, kind),
                    _format_index(b, a),
                    _format_metric(c, kind),
                    _format_delta(c, a, kind),
                    _format_index(c, a),
                ]
            )
            + " |"
        )
    return rows


def _format_metric(value: Any, kind: str) -> str:
    return _format_ratio(value) if kind == "ratio" else _format_number(value)


def _format_delta(value: Any, base: Any, kind: str) -> str:
    if not _is_number(value) or not _is_number(base):
        return NA
    delta = float(value) - float(base)
    return f"{delta * 100:+.1f} pp" if kind == "ratio" else f"{delta:+.2f}"


def _format_index(value: Any, base: Any) -> str:
    if not _is_number(value) or not _is_number(base) or float(base) == 0:
        return NA
    return f"{float(value) / float(base) * 100:.1f}"


def _render_svg(metrics: dict[str, dict[str, Any]], order: list[str]) -> str:
    panels = [
        ("Sandbox build/test Pass@1", "build_pass_at_1", "ratio"),
        ("Plan quality", "plan_quality_rate", "ratio"),
        ("Tool-policy accuracy", "tool_policy_accuracy", "ratio"),
        ("Known benign review repair", "review_repair_rate", "ratio"),
        ("Composite success", "composite_success_rate", "ratio"),
        ("Structural pass proxy (NOT true Pass@1)", "structural_pass_proxy", "ratio"),
        ("Valid-security TPR", "tpr_valid_security", "ratio"),
        ("Valid-security FPR", "fpr_valid_security", "ratio"),
        ("Operational malicious block rate", "operational_malicious_block_rate", "ratio"),
        ("Operational benign block rate", "operational_benign_block_rate", "ratio"),
        ("Mean end-to-end latency (ms)", "mean_latency_ms", "number"),
        ("Mean total tokens", "mean_total_tokens", "number"),
        ("Peak sampled VRAM (MiB)", "peak_vram_mb", "number"),
        ("Token cost per composite success", "cost_per_success_tokens", "number"),
        ("Error rate", "error_rate", "ratio"),
    ]
    width = 1120
    row_height = 24
    panel_height = 48 + max(1, len(order)) * row_height
    height = 86 + len(panels) * panel_height
    chart_x = 300
    chart_width = 650
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<title>Anchor-MoE-LoRA auditable benchmark comparison</title>",
        "<desc>Structural proxy, valid-security rates, operational block rates, latency, tokens, VRAM, and errors. Missing values are marked N/A.</desc>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<style>text{font-family:Segoe UI,Arial,sans-serif;fill:#1f2937}.title{font-size:20px;font-weight:700}.panel{font-size:14px;font-weight:700}.label{font-size:12px}.value{font-size:12px;font-variant-numeric:tabular-nums}.note{font-size:12px;fill:#991b1b}</style>',
        '<text x="24" y="30" class="title">Anchor-MoE-LoRA benchmark comparison</text>',
        '<text x="24" y="52" class="note">Structural proxy != true Pass@1; no build/browser execution is implied.</text>',
        '<text x="24" y="70" class="note">Valid-security TPR/FPR are separate from operational block rates.</text>',
    ]
    y = 100
    for title, key, kind in panels:
        parts.append(f'<text x="24" y="{y}" class="panel">{escape(title)}</text>')
        values = [metrics[name].get(key) for name in order]
        numeric = [float(value) for value in values if _is_number(value)]
        scale = 1.0 if kind == "ratio" else (max(numeric) if numeric else 1.0)
        if scale <= 0:
            scale = 1.0
        for index, name in enumerate(order):
            row_y = y + 22 + index * row_height
            value = metrics[name].get(key)
            parts.append(
                f'<text x="24" y="{row_y + 12}" class="label">{escape(name)}</text>'
            )
            parts.append(
                f'<rect x="{chart_x}" y="{row_y}" width="{chart_width}" height="14" fill="#eef2f7" rx="2"/>'
            )
            if _is_number(value):
                bar_width = max(0.0, min(chart_width, float(value) / scale * chart_width))
                parts.append(
                    f'<rect x="{chart_x}" y="{row_y}" width="{bar_width:.2f}" height="14" fill="{_color(name)}" rx="2"/>'
                )
                display = _format_ratio(value) if kind == "ratio" else _format_number(value)
            else:
                display = NA
                parts.append(
                    f'<line x1="{chart_x}" y1="{row_y + 7}" x2="{chart_x + chart_width}" y2="{row_y + 7}" stroke="#dc2626" stroke-dasharray="4 4"/>'
                )
            parts.append(
                f'<text x="{chart_x + chart_width + 12}" y="{row_y + 12}" class="value">{escape(display)}</text>'
            )
        y += panel_height
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def _color(name: str) -> str:
    return {
        "base_matched_calls": "#2563eb",
        "mixed_matched_calls": "#d97706",
        "c_pipeline": "#059669",
        "d_budget_matched_pipeline": "#7c3aed",
        "a_base": "#64748b",
        "b_mixed": "#94a3b8",
    }.get(name, "#7c3aed")


def _compare_metrics(
    supplied: dict[str, dict[str, Any]], recomputed: dict[str, dict[str, Any]]
) -> list[str]:
    differences: list[str] = []
    for baseline, recalculated in recomputed.items():
        if baseline not in supplied:
            differences.append(f"`{baseline}` is absent from supplied metrics.")
            continue
        for key, actual in recalculated.items():
            if key not in supplied[baseline]:
                continue
            expected = supplied[baseline][key]
            if not _values_equal(expected, actual):
                differences.append(
                    f"`{baseline}.{key}` supplied={expected!r}, recomputed={actual!r}."
                )
    for baseline in sorted(set(supplied) - set(recomputed)):
        differences.append(f"`{baseline}` exists in supplied metrics but has no records.")
    return differences[:100]


def _values_equal(left: Any, right: Any) -> bool:
    if _is_number(left) and _is_number(right):
        return math.isclose(float(left), float(right), rel_tol=1e-9, abs_tol=1e-9)
    return left == right


def _is_number(value: Any) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _provenance_summary(records: list[BenchmarkRecord]) -> list[str]:
    unique = {
        json.dumps(record.evaluator_provenance, ensure_ascii=False, sort_keys=True)
        for record in records
    }
    return sorted(unique) or [
        json.dumps(
            {
                "pass_metric": "unknown",
                "tool_verified": False,
                "executed_build_or_browser_test": False,
            },
            sort_keys=True,
        )
    ]


def _file_hash(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _format_ratio(value: Any) -> str:
    return NA if not _is_number(value) else f"{float(value) * 100:.1f}%"


def _format_number(value: Any) -> str:
    return NA if not _is_number(value) else f"{float(value):.2f}"


def _md(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an auditable Anchor-MoE-LoRA report")
    parser.add_argument("--records", required=True, help="Benchmark record JSONL")
    parser.add_argument("--metrics", required=True, help="Aggregate metrics JSON to reconcile")
    parser.add_argument("--output-dir", required=True, help="Report destination")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    paths = generate_report(args.records, args.metrics, args.output_dir)
    print(json.dumps({key: str(value) for key, value in paths.__dict__.items()}, indent=2))


if __name__ == "__main__":
    main()
