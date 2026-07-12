"""Validation for continuous multi-adapter collaboration curricula.

The curriculum is deliberately separate from generated training data.  It
freezes executable public contracts and the public-output lineage that a
teacher session must follow before a capture can become a candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Mapping, Sequence

import yaml


CURRICULUM_SCHEMA_VERSION = "anchor.collaboration-curriculum.v2"
REVIEW_SCHEMA_VERSION = "anchor.domain-review-verdict.v2"
BUILDER_CAPTURE_SCHEMA_VERSION = "anchor.session-training-candidate.v1"
ALLOWED_TOOLS = frozenset({"read", "edit", "apply_patch", "bash"})
REQUIRED_VALIDATIONS = ("build", "test", "lint")
REQUIRED_DOMAINS = frozenset(
    {
        "frontend-web",
        "python-cli",
        "node-ts-utility",
        "code-repair",
        "accessibility-ui",
        "security-inert",
    }
)
INERT_SECURITY_LABEL = re.compile(r"^INERT_[A-Z0-9_]{3,80}$")
HASH = re.compile(r"^[0-9a-f]{64}$")


class CurriculumValidationError(ValueError):
    """A candidate curriculum is unsafe, stale, or structurally ambiguous."""


@dataclass(frozen=True)
class FixtureRun:
    task_id: str
    validation: str
    command: str
    exit_code: int
    expected_exit: str
    stdout_sha256: str


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalized(value: str) -> str:
    return " ".join(value.casefold().split())


def _similar(left: str, right: str) -> bool:
    a, b = _normalized(left), _normalized(right)
    if not a or not b:
        return False
    if a == b or (len(a) >= 24 and a in b) or (len(b) >= 24 and b in a):
        return True
    if min(len(a), len(b)) / max(len(a), len(b)) < 0.55:
        return False
    return SequenceMatcher(None, a, b, autojunk=False).ratio() >= 0.82


def _load_mapping(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise CurriculumValidationError(f"{path} must contain an object")
    return {str(key): item for key, item in value.items()}


def _within_repo(repo: Path, relative: object, *, label: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise CurriculumValidationError(f"{label} must be a non-empty relative path")
    candidate = (repo / relative).resolve()
    try:
        candidate.relative_to(repo)
    except ValueError as exc:
        raise CurriculumValidationError(f"{label} escapes the repository") from exc
    return candidate


def _heldout_needles(repo: Path, policy: Mapping[str, Any]) -> tuple[set[str], list[str]]:
    identifiers: set[str] = set()
    requirements: list[str] = []
    paths = policy.get("inputs")
    if not isinstance(paths, list) or not paths:
        raise CurriculumValidationError("heldout_policy.inputs must be a non-empty list")
    for index, raw in enumerate(paths):
        if not isinstance(raw, Mapping):
            raise CurriculumValidationError(f"heldout input {index} must be an object")
        path = _within_repo(repo, raw.get("path"), label=f"heldout input {index}")
        expected = str(raw.get("sha256", ""))
        if not path.is_file() or not HASH.fullmatch(expected):
            raise CurriculumValidationError(f"heldout input {index} is missing or unhashed")
        if _sha256_file(path) != expected:
            raise CurriculumValidationError(f"heldout input hash drift: {path}")
        if path.suffix != ".jsonl":
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            if not isinstance(record, Mapping):
                raise CurriculumValidationError(f"heldout row in {path} is not an object")
            for key in ("case_id", "seed_id", "case_family", "namespace", "seed_namespace"):
                if record.get(key):
                    identifiers.add(str(record[key]).casefold())
            if record.get("requirement"):
                requirements.append(str(record["requirement"]))
    return identifiers, requirements


def expected_stage_flow(
    max_cycles: int,
    required_builder: str = "required_builder",
    required_reviewer: str = "required_reviewer",
) -> list[dict[str, object]]:
    """Return the only accepted public-output lineage for one continuous seed."""

    if max_cycles not in (1, 2):
        raise CurriculumValidationError("max_cycles must be 1 or 2 for the MVP curriculum")
    stages: list[dict[str, object]] = [
        {
            "id": "planner",
            "adapter": "planner_lora",
            "input_refs": ["context"],
            "output_schema": "anchor.planner-handoff.v2",
        },
        {
            "id": "plan_safety",
            "adapter": "safety_tool_policy_lora",
            "input_refs": ["context", "planner.output"],
            "output_schema": "anchor.safety-tool-decision.v2",
        },
        {
            "id": "builder_1",
            "adapter": required_builder,
            "input_refs": ["context", "planner.output", "plan_safety.output"],
            "output_schema": BUILDER_CAPTURE_SCHEMA_VERSION,
        },
        {
            "id": "domain_review_1",
            "adapter": required_reviewer,
            "input_refs": [
                "context",
                "planner.output",
                "builder_1.public_output",
                "builder_1.tool_trace",
                "builder_1.diff",
                "builder_1.validators",
            ],
            "output_schema": REVIEW_SCHEMA_VERSION,
        },
    ]
    final_builder = "builder_1"
    final_review = "domain_review_1"
    if max_cycles == 2:
        stages.extend(
            [
                {
                    "id": "builder_2",
                    "adapter": required_builder,
                    "input_refs": [
                        "context",
                        "planner.output",
                        "plan_safety.output",
                        "builder_1.public_output",
                        "domain_review_1.output",
                    ],
                    "output_schema": BUILDER_CAPTURE_SCHEMA_VERSION,
                },
                {
                    "id": "domain_review_2",
                    "adapter": required_reviewer,
                    "input_refs": [
                        "context",
                        "planner.output",
                        "builder_2.public_output",
                        "builder_2.tool_trace",
                        "builder_2.diff",
                        "builder_2.validators",
                    ],
                    "output_schema": REVIEW_SCHEMA_VERSION,
                },
            ]
        )
        final_builder, final_review = "builder_2", "domain_review_2"
    stages.append(
        {
            "id": "final_safety",
            "adapter": "safety_tool_policy_lora",
            "input_refs": [
                "context",
                "planner.output",
                "plan_safety.output",
                f"{final_builder}.public_output",
                f"{final_builder}.tool_trace",
                f"{final_builder}.diff",
                f"{final_builder}.validators",
                f"{final_review}.output",
            ],
            "output_schema": "anchor.final-safety-decision.v2",
        }
    )
    return stages


def _validate_task(
    task: Mapping[str, Any], repo: Path, *, identifiers: set[str], requirements: Sequence[str]
) -> None:
    task_id = str(task.get("task_id", ""))
    seed_id = str(task.get("seed_id", ""))
    if not re.fullmatch(r"cv2-[a-z0-9-]{3,80}", task_id):
        raise CurriculumValidationError(f"invalid task_id: {task_id!r}")
    if not re.fullmatch(r"cv2-seed-[a-z0-9-]{3,80}", seed_id):
        raise CurriculumValidationError(f"invalid seed_id for {task_id}")
    if task.get("split") != "candidate":
        raise CurriculumValidationError(f"{task_id} must be candidate split")
    domain = str(task.get("domain", ""))
    difficulty = str(task.get("difficulty", ""))
    if domain not in REQUIRED_DOMAINS or difficulty not in {"basic", "intermediate", "advanced"}:
        raise CurriculumValidationError(f"{task_id} has invalid domain/difficulty")
    for field in ("required_builder", "reviewer"):
        if not re.fullmatch(r"[a-z][a-z0-9_]{2,63}", str(task.get(field, ""))):
            raise CurriculumValidationError(f"{task_id} has invalid {field}")
    max_cycles = task.get("max_cycles")
    if not isinstance(max_cycles, int):
        raise CurriculumValidationError(f"{task_id} max_cycles must be an integer")
    if task.get("stage_flow") != expected_stage_flow(
        max_cycles, str(task.get("required_builder")), str(task.get("reviewer"))
    ):
        raise CurriculumValidationError(f"{task_id} stage_flow breaks continuous lineage")

    planner = task.get("planner_target")
    if not isinstance(planner, Mapping):
        raise CurriculumValidationError(f"{task_id} requires planner_target")
    if planner.get("selected_builder") != task.get("required_builder"):
        raise CurriculumValidationError(f"{task_id} planner does not select required_builder")
    points = planner.get("handoff_required_points")
    if not isinstance(points, list) or len(points) < 3 or any(not str(item).strip() for item in points):
        raise CurriculumValidationError(f"{task_id} needs at least three concrete handoff points")
    rubric = task.get("reviewer_rubric")
    if not isinstance(rubric, list) or len(rubric) < 3 or any(not str(item).strip() for item in rubric):
        raise CurriculumValidationError(f"{task_id} reviewer_rubric is underspecified")
    labels = task.get("safety_labels")
    if not isinstance(labels, list) or not labels or any(
        not INERT_SECURITY_LABEL.fullmatch(str(item)) for item in labels
    ):
        raise CurriculumValidationError(f"{task_id} safety labels must be inert labels")
    tools = task.get("allowed_tools")
    if not isinstance(tools, list) or not {str(item) for item in tools}.issubset(ALLOWED_TOOLS):
        raise CurriculumValidationError(f"{task_id} contains a non-allowlisted tool")
    if "bash" not in tools or not {"read", "edit"}.intersection(tools):
        raise CurriculumValidationError(f"{task_id} cannot produce an executable tool trace")

    provenance = task.get("skill_provenance")
    if not isinstance(provenance, list) or not provenance:
        raise CurriculumValidationError(f"{task_id} requires skill provenance")
    for item in provenance:
        if not isinstance(item, Mapping) or not HASH.fullmatch(str(item.get("sha256", ""))):
            raise CurriculumValidationError(f"{task_id} has invalid skill provenance")
        source = _within_repo(repo, item.get("source"), label=f"{task_id} skill source")
        if not source.is_file() or _sha256_file(source) != item.get("sha256"):
            raise CurriculumValidationError(f"{task_id} skill provenance drift: {source}")

    fixture = _within_repo(repo, task.get("source_dir"), label=f"{task_id} source_dir")
    if not fixture.is_dir():
        raise CurriculumValidationError(f"{task_id} fixture directory is missing")
    contracts = task.get("frozen_public_contract")
    if not isinstance(contracts, Mapping):
        raise CurriculumValidationError(f"{task_id} lacks frozen_public_contract")
    files = contracts.get("files")
    commands = contracts.get("commands")
    if not isinstance(files, Mapping) or not isinstance(commands, Mapping):
        raise CurriculumValidationError(f"{task_id} frozen contract is malformed")
    required_files = {"task", "context", "package", "public_test", "starter"}
    if set(files) != required_files:
        raise CurriculumValidationError(f"{task_id} frozen file set must be {sorted(required_files)}")
    for label, raw in files.items():
        if not isinstance(raw, Mapping):
            raise CurriculumValidationError(f"{task_id} file contract {label} is malformed")
        path = _within_repo(fixture, raw.get("path"), label=f"{task_id} {label}")
        expected = str(raw.get("sha256", ""))
        if not path.is_file() or not HASH.fullmatch(expected) or _sha256_file(path) != expected:
            raise CurriculumValidationError(f"{task_id} frozen file drift: {path}")
    if set(commands) != set(REQUIRED_VALIDATIONS):
        raise CurriculumValidationError(f"{task_id} must freeze build/test/lint")
    package = json.loads((fixture / str(files["package"]["path"])).read_text(encoding="utf-8"))
    scripts = package.get("scripts", {})
    for name in REQUIRED_VALIDATIONS:
        raw = commands.get(name)
        if not isinstance(raw, Mapping):
            raise CurriculumValidationError(f"{task_id} missing {name} command contract")
        command = str(raw.get("command", ""))
        if command != f"npm run {name}" or raw.get("script") != scripts.get(name):
            raise CurriculumValidationError(f"{task_id} {name} command drift")
        expected_command_hash = _sha256_bytes(str(scripts[name]).encode("utf-8"))
        if raw.get("script_sha256") != expected_command_hash:
            raise CurriculumValidationError(f"{task_id} {name} script hash drift")
        if raw.get("starter_expectation") not in {"PASS", "FAIL"}:
            raise CurriculumValidationError(f"{task_id} {name} starter expectation is invalid")

    searchable = json.dumps(task, ensure_ascii=False, sort_keys=True)
    searchable += "\n" + "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in sorted(fixture.rglob("*"))
        if path.is_file()
    )
    folded = searchable.casefold()
    if any(identifier in folded for identifier in identifiers):
        raise CurriculumValidationError(f"{task_id} leaks a heldout identifier")
    if any(_similar(searchable, requirement) for requirement in requirements):
        raise CurriculumValidationError(f"{task_id} resembles a heldout requirement")
    active_markers = ("<script src=http", "powershell -enc", "curl http", "wget http")
    if any(marker in folded for marker in active_markers):
        raise CurriculumValidationError(f"{task_id} contains an active payload marker")


def validate_curriculum(manifest_path: str | Path, repo_root: str | Path) -> dict[str, Any]:
    repo = Path(repo_root).resolve()
    manifest = _load_mapping(Path(manifest_path))
    if manifest.get("schema_version") != CURRICULUM_SCHEMA_VERSION:
        raise CurriculumValidationError("curriculum schema_version mismatch")
    if manifest.get("status") != "candidate-not-training-data":
        raise CurriculumValidationError("curriculum must remain candidate-not-training-data")
    ramp = manifest.get("ramp")
    if (
        not isinstance(ramp, list)
        or not ramp
        or any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in ramp)
    ):
        raise CurriculumValidationError("curriculum ramp must contain positive integers")
    policy = manifest.get("heldout_policy")
    if not isinstance(policy, Mapping) or policy.get("training_access") != "forbidden":
        raise CurriculumValidationError("heldout policy must forbid training access")
    identifiers, requirements = _heldout_needles(repo, policy)
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or len(tasks) != 15:
        raise CurriculumValidationError("curriculum must contain exactly 15 candidates")
    task_ids: set[str] = set()
    seed_ids: set[str] = set()
    domains: set[str] = set()
    for raw in tasks:
        if not isinstance(raw, Mapping):
            raise CurriculumValidationError("every curriculum task must be an object")
        _validate_task(raw, repo, identifiers=identifiers, requirements=requirements)
        task_id, seed_id = str(raw["task_id"]), str(raw["seed_id"])
        if task_id in task_ids or seed_id in seed_ids:
            raise CurriculumValidationError("task_id and seed_id must be unique")
        task_ids.add(task_id)
        seed_ids.add(seed_id)
        domains.add(str(raw["domain"]))
    if not REQUIRED_DOMAINS.issubset(domains):
        raise CurriculumValidationError("curriculum does not cover every required domain")
    return manifest


def run_fixture_contracts(
    manifest: Mapping[str, Any], repo_root: str | Path, *, timeout_seconds: float = 20.0
) -> tuple[FixtureRun, ...]:
    """Execute each frozen offline command and verify its starter expectation."""

    repo = Path(repo_root).resolve()
    npm = "npm.cmd" if __import__("os").name == "nt" else "npm"
    results: list[FixtureRun] = []
    for task in manifest["tasks"]:
        fixture = _within_repo(repo, task["source_dir"], label=f"{task['task_id']} source_dir")
        commands = task["frozen_public_contract"]["commands"]
        for name in REQUIRED_VALIDATIONS:
            completed = subprocess.run(
                [npm, "run", name],
                cwd=fixture,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_seconds,
                check=False,
                shell=False,
            )
            expected = str(commands[name]["starter_expectation"])
            actual = "PASS" if completed.returncode == 0 else "FAIL"
            if actual != expected:
                raise CurriculumValidationError(
                    f"{task['task_id']} {name}: expected {expected}, got {actual}"
                )
            results.append(
                FixtureRun(
                    task_id=str(task["task_id"]),
                    validation=name,
                    command=f"npm run {name}",
                    exit_code=completed.returncode,
                    expected_exit=expected,
                    stdout_sha256=_sha256_bytes(completed.stdout.encode("utf-8")),
                )
            )
    return tuple(results)
