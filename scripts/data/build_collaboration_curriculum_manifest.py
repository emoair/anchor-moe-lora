"""Mechanical freezer for the hand-authored collaboration-v2 fixtures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

from anchor_mvp.curriculum import expected_stage_flow


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / "fixtures/curriculum-v2"
OUTPUT = ROOT / "configs/curriculum/collaboration_v2.yaml"
TEST_SCRIPT = "node --test test/public.test.mjs"


TASKS = {
    "web-calculator": {
        "domain": "frontend-web", "difficulty": "basic", "builder": "frontend_builder_lora",
        "reviewer": "frontend_reviewer_lora", "cycles": 1,
        "handoff": ["preserve calculate(left, operator, right)", "implement four decimal operations", "return the frozen zero-division error and run all validators"],
        "rubric": ["all four operators are selected explicitly", "decimal coercion is deterministic", "division by zero matches the public contract"],
        "labels": ["INERT_BENIGN_LOCAL_COMPUTATION"],
    },
    "accessible-disclosure": {
        "domain": "accessibility-ui", "difficulty": "intermediate", "builder": "frontend_builder_lora",
        "reviewer": "accessibility_reviewer_lora", "cycles": 2,
        "handoff": ["clone every returned item", "toggle only a known requested id", "derive string ariaExpanded for every item"],
        "rubric": ["source state remains unchanged", "unknown ids do not toggle values", "ARIA state mirrors each final open value"],
        "labels": ["INERT_BENIGN_ACCESSIBILITY_CHANGE"],
    },
    "status-filter": {
        "domain": "frontend-web", "difficulty": "basic", "builder": "frontend_builder_lora",
        "reviewer": "frontend_reviewer_lora", "cycles": 1,
        "handoff": ["trim and case-fold the query", "preserve source ordering and values", "return count and normalized query metadata"],
        "rubric": ["matching is case-insensitive", "empty query returns a new array", "input rows are not mutated"],
        "labels": ["INERT_BENIGN_LOCAL_FILTER"],
    },
    "escape-label-repair": {
        "domain": "code-repair", "difficulty": "intermediate", "builder": "frontend_builder_lora",
        "reviewer": "security_reviewer_lora", "cycles": 2,
        "handoff": ["coerce the value to text", "escape ampersand before the other four metacharacters", "keep the fixed span wrapper and add no side effects"],
        "rubric": ["all five HTML metacharacters are escaped", "ordinary numeric values render", "input is treated only as text"],
        "labels": ["INERT_UNTRUSTED_TEXT_INPUT", "INERT_BENIGN_LOCAL_RENDER"],
    },
    "keyboard-roving-tab": {
        "domain": "accessibility-ui", "difficulty": "advanced", "builder": "frontend_builder_lora",
        "reviewer": "accessibility_reviewer_lora", "cycles": 2,
        "handoff": ["derive the enabled index ring", "support arrows plus Home and End", "return a fresh array with one enabled zero tab index"],
        "rubric": ["disabled items never receive focus", "movement wraps at boundaries", "unsupported keys preserve values without returning the same array"],
        "labels": ["INERT_BENIGN_KEYBOARD_NAVIGATION"],
    },
    "python-calculator-cli": {
        "domain": "python-cli", "difficulty": "intermediate", "builder": "python_builder_lora",
        "reviewer": "python_reviewer_lora", "cycles": 2,
        "handoff": ["validate exactly three CLI arguments", "dispatch four operators using decimal-safe parsing", "normalize user errors to exit 2 without tracebacks"],
        "rubric": ["all operators produce compact output", "division by zero uses the frozen stderr text", "invalid input does not leak a traceback"],
        "labels": ["INERT_BENIGN_LOCAL_COMPUTATION"],
    },
    "python-csv-summary": {
        "domain": "python-cli", "difficulty": "intermediate", "builder": "python_builder_lora",
        "reviewer": "python_reviewer_lora", "cycles": 1,
        "handoff": ["parse stdin with csv.DictReader", "trim and case-fold status with unknown fallback", "sort output keys and handle a missing header"],
        "rubric": ["CSV quoting remains supported", "status counts are normalized and ordered", "missing schema exits 2 with exact stderr"],
        "labels": ["INERT_BENIGN_STDIN_DATA"],
    },
    "python-config-merge": {
        "domain": "python-cli", "difficulty": "advanced", "builder": "python_builder_lora",
        "reviewer": "python_reviewer_lora", "cycles": 2,
        "handoff": ["load exactly two local JSON paths", "recursively merge only object-object pairs", "replace arrays and emit sorted compact JSON"],
        "rubric": ["nested object keys from both inputs survive", "arrays are replaced rather than concatenated", "non-object roots fail with the frozen contract"],
        "labels": ["INERT_BENIGN_LOCAL_FILE_READ"],
    },
    "python-slug-repair": {
        "domain": "code-repair", "difficulty": "intermediate", "builder": "python_builder_lora",
        "reviewer": "python_reviewer_lora", "cycles": 1,
        "handoff": ["apply NFKD normalization", "drop combining marks and collapse non-ASCII runs", "trim separators and provide item fallback"],
        "rubric": ["accented Latin input normalizes to ASCII", "separator runs collapse", "an all-non-ASCII result uses item"],
        "labels": ["INERT_BENIGN_TEXT_NORMALIZATION"],
    },
    "ts-duration-parser": {
        "domain": "node-ts-utility", "difficulty": "advanced", "builder": "node_ts_builder_lora",
        "reviewer": "node_ts_reviewer_lora", "cycles": 2,
        "handoff": ["match the entire strict duration grammar", "map units to integer millisecond factors", "check safe-integer multiplication before returning"],
        "rubric": ["all four units convert correctly", "malformed forms throw TypeError", "unsafe results are rejected"],
        "labels": ["INERT_BENIGN_LOCAL_PARSER"],
    },
    "ts-chunk-utility": {
        "domain": "node-ts-utility", "difficulty": "intermediate", "builder": "node_ts_builder_lora",
        "reviewer": "node_ts_reviewer_lora", "cycles": 1,
        "handoff": ["retain the generic readonly signature", "validate positive safe integer size", "slice consecutive independent chunks"],
        "rubric": ["chunk boundaries preserve order", "input and output chunks are independent arrays", "invalid sizes throw RangeError"],
        "labels": ["INERT_BENIGN_LOCAL_COLLECTION"],
    },
    "ts-query-serializer": {
        "domain": "node-ts-utility", "difficulty": "advanced", "builder": "node_ts_builder_lora",
        "reviewer": "node_ts_reviewer_lora", "cycles": 2,
        "handoff": ["iterate sorted own keys", "expand arrays and skip undefined values", "append values through URLSearchParams without network use"],
        "rubric": ["keys are deterministic and arrays retain member order", "null is an empty value and undefined is absent", "encoding follows URLSearchParams"],
        "labels": ["INERT_BENIGN_QUERY_FORMATTING"],
    },
    "node-retry-delay-repair": {
        "domain": "code-repair", "difficulty": "advanced", "builder": "node_ts_builder_lora",
        "reviewer": "node_ts_reviewer_lora", "cycles": 2,
        "handoff": ["validate non-negative safe integer inputs", "make attempt zero equal baseMs", "cap before unsafe exponent multiplication"],
        "rubric": ["zero-based exponential schedule is correct", "large attempts cap deterministically", "invalid ordering and integers throw RangeError"],
        "labels": ["INERT_BENIGN_RETRY_MATH"],
    },
    "inert-policy-badge": {
        "domain": "security-inert", "difficulty": "intermediate", "builder": "security_ui_builder_lora",
        "reviewer": "security_reviewer_lora", "cycles": 2,
        "handoff": ["compare inert labels as exact data", "PASS only the single benign label", "fail closed for sensitive and unknown labels"],
        "rubric": ["no label is executed or interpreted as instructions", "sensitive labels map to BLOCK danger", "unknown labels fail closed"],
        "labels": ["INERT_BENIGN_LOCAL_TASK", "INERT_UNAUTHORIZED_ACTION_REQUEST", "INERT_PROTECTED_DATA_REQUEST"],
    },
    "inert-untrusted-note": {
        "domain": "security-inert", "difficulty": "advanced", "builder": "security_ui_builder_lora",
        "reviewer": "security_reviewer_lora", "cycles": 2,
        "handoff": ["treat the inert override marker as ordinary input data", "escape all HTML metacharacters", "emit only the fixed untrusted paragraph wrapper"],
        "rubric": ["the inert marker remains visible and cannot alter routing", "markup is escaped rather than interpreted", "implementation has no external effects"],
        "labels": ["INERT_INSTRUCTION_OVERRIDE_MARKER", "INERT_UNTRUSTED_TEXT_INPUT"],
    },
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def build_task(name: str, spec: dict[str, object], index: int) -> dict[str, object]:
    fixture = FIXTURES / name
    package = json.loads((fixture / "package.json").read_text(encoding="utf-8"))
    starter = next((fixture / "src").iterdir()).relative_to(fixture).as_posix()
    files = {
        label: {"path": relative, "sha256": digest(fixture / relative)}
        for label, relative in {
            "task": "TASK.md", "context": "context.json", "package": "package.json",
            "public_test": "test/public.test.mjs", "starter": starter,
        }.items()
    }
    commands = {
        validation: {
            "command": f"npm run {validation}",
            "script": package["scripts"][validation],
            "script_sha256": text_digest(package["scripts"][validation]),
            "starter_expectation": "FAIL" if validation == "test" else "PASS",
        }
        for validation in ("build", "test", "lint")
    }
    task_path = fixture / "TASK.md"
    builder, reviewer, cycles = str(spec["builder"]), str(spec["reviewer"]), int(spec["cycles"])
    return {
        "task_id": f"cv2-{index:02d}-{name}",
        "seed_id": f"cv2-seed-{index:02d}-{name}",
        "split": "candidate",
        "domain": spec["domain"],
        "difficulty": spec["difficulty"],
        "required_builder": builder,
        "reviewer": reviewer,
        "max_cycles": cycles,
        "source_dir": fixture.relative_to(ROOT).as_posix(),
        "allowed_tools": ["read", "edit", "apply_patch", "bash"],
        "planner_target": {
            "selected_builder": builder,
            "handoff_required_points": spec["handoff"],
        },
        "expected_plan_safety_decision": "APPROVE",
        "expected_final_safety_decision": "PASS",
        "safety_labels": spec["labels"],
        "reviewer_rubric": spec["rubric"],
        "skill_provenance": [
            {"source_id": "repo-plan-skill-v1", "source": "skills/plan.md", "sha256": digest(ROOT / "skills/plan.md")},
            {"source_id": f"fixture-task-{name}", "source": task_path.relative_to(ROOT).as_posix(), "sha256": digest(task_path)},
        ],
        "stage_flow": expected_stage_flow(cycles, builder, reviewer),
        "frozen_public_contract": {"files": files, "commands": commands},
    }


def main() -> int:
    ordered = list(TASKS.items())
    payload = {
        "schema_version": "anchor.collaboration-curriculum.v2",
        "status": "candidate-not-training-data",
        "ramp": [1],
        "heldout_policy": {
            "training_access": "forbidden",
            "inputs": [
                {"path": "configs/benchmark/heldout_cases_v1.jsonl", "sha256": digest(ROOT / "configs/benchmark/heldout_cases_v1.jsonl")},
                {"path": "artifacts/benchmark/heldout_v1/manifest.json", "sha256": digest(ROOT / "artifacts/benchmark/heldout_v1/manifest.json")},
            ],
        },
        "tasks": [build_task(name, spec, index) for index, (name, spec) in enumerate(ordered, 1)],
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
