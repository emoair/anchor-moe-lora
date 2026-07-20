from __future__ import annotations

from dataclasses import asdict
from difflib import SequenceMatcher
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Iterable, Iterator, Mapping

from .models import BenchmarkCase, load_cases_jsonl
from .segment_protocol import (
    ARTIFACT_PROTOCOL,
    SEGMENT_CONTRACT_VERSION,
    SEGMENTED_REVIEW_PROTOCOL,
    SegmentContract,
    SegmentProtocolError,
)


MANIFEST_SCHEMA = "anchor.heldout-manifest.v1"
LEAK_AUDIT_SCHEMA = "anchor.leak-audit.v1"
PRIMARY_BASELINES = ("base_matched_calls", "mixed_matched_calls", "c_pipeline")
BUDGET_MATCHED_BASELINE = "d_budget_matched_pipeline"
ADAPTIVE_BASELINE = "e_adaptive_pareto_pipeline"
ADAPTIVE_BUDGET_BASELINE = "f_adaptive_budget_matched_pipeline"
PRIMARY_STAGES = ("planner", "tool_policy", "frontend", "review", "security")
_ACTIVE_PAYLOAD = re.compile(
    r"<\s*script\b|javascript\s*:|on(?:error|load|click)\s*=|"
    r"\beval\s*\(|\bdocument\.cookie\b|\bfetch\s*\(|"
    r"(?:https?|wss?)://|\b(?:powershell|cmd\.exe|curl|wget)\b",
    re.IGNORECASE,
)
_INERT_LABEL = re.compile(r"^INERT_[A-Z0-9_]{4,80}$")


class HeldoutGateError(ValueError):
    """A held-out integrity or leakage gate failed closed."""


def normalized_text(value: str) -> str:
    return " ".join(re.findall(r"[\w]+", value.casefold(), flags=re.UNICODE))


def digest_text(value: str) -> str:
    return sha256(normalized_text(value).encode("utf-8")).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tree_sha256(path: str | Path) -> str:
    root = Path(path).resolve()
    if not root.is_dir():
        raise HeldoutGateError("held-out fixture is missing")
    digest = sha256()
    for item in sorted(root.rglob("*")):
        if item.is_symlink():
            raise HeldoutGateError("held-out fixtures may not contain symlinks")
        if not item.is_file():
            continue
        relative = item.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(bytes.fromhex(file_sha256(item)))
    return digest.hexdigest()


def validate_heldout_cases(cases: Iterable[BenchmarkCase]) -> list[BenchmarkCase]:
    checked = list(cases)
    if not checked:
        raise HeldoutGateError("held-out case set is empty")
    ids: set[str] = set()
    seed_ids: set[str] = set()
    request_hashes: set[str] = set()
    namespaces = {case.namespace for case in checked}
    seed_namespaces = {case.seed_namespace for case in checked}
    if len(namespaces) != 1 or not next(iter(namespaces)).startswith("anchor-heldout-"):
        raise HeldoutGateError("held-out cases require one dedicated namespace")
    if len(seed_namespaces) != 1 or not next(iter(seed_namespaces)).startswith("anchor-ho-"):
        raise HeldoutGateError("held-out cases require one dedicated seed namespace")
    for case in checked:
        if case.split != "heldout":
            raise HeldoutGateError("every held-out case must declare split=heldout")
        if not case.case_id.startswith(f"{case.namespace}:"):
            raise HeldoutGateError("case_id is outside the held-out namespace")
        if not case.seed_id.startswith(f"{case.seed_namespace}:"):
            raise HeldoutGateError("seed_id is outside the held-out seed namespace")
        if not case.case_family.startswith("hf-v1-"):
            raise HeldoutGateError("case_family must use the hf-v1 namespace")
        if case.case_id in ids or case.seed_id in seed_ids:
            raise HeldoutGateError("held-out case_id and seed_id values must be unique")
        request_hash = digest_text(case.requirement)
        if request_hash in request_hashes:
            raise HeldoutGateError("duplicate normalized held-out requirement")
        ids.add(case.case_id)
        seed_ids.add(case.seed_id)
        request_hashes.add(request_hash)
        expected = case.expected_security_decision
        if expected not in {"PASS", "BLOCK"}:
            raise HeldoutGateError("expected_security_decision must be PASS or BLOCK")
        if (expected == "BLOCK") != case.malicious:
            raise HeldoutGateError("malicious flag and security expectation disagree")
        if not _INERT_LABEL.fullmatch(case.security_intent_label):
            raise HeldoutGateError("security cases must use inert intent labels")
        if _ACTIVE_PAYLOAD.search(case.requirement):
            raise HeldoutGateError("active payload-like content is forbidden in held-out cases")
        mutation = case.review_mutation
        if mutation.get("kind") != "remove_literal_marker":
            raise HeldoutGateError("review mutation must be remove_literal_marker")
        marker = mutation.get("marker", "")
        if not marker or len(marker) > 200 or _ACTIVE_PAYLOAD.search(marker):
            raise HeldoutGateError("review mutation marker is missing or unsafe")
        if not mutation.get("known_benign_defect"):
            raise HeldoutGateError("review mutation requires a known benign defect")
        if not case.plan_required_concepts or any(
            not str(item).strip() for item in case.plan_required_concepts
        ):
            raise HeldoutGateError("plan quality requires non-empty held-out concepts")
        if not case.tool_proposal_labels or any(
            not re.fullmatch(r"INERT_TOOL_[A-Z0-9_]{4,80}", str(item))
            for item in case.tool_proposal_labels
        ):
            raise HeldoutGateError("tool proposals must be inert labels")
        deterministic = deterministic_tool_policy(case.tool_proposal_labels)
        if case.expected_tool_policy_decision not in {"APPROVE", "BLOCK", "ESCALATE"}:
            raise HeldoutGateError("tool-policy expectation must be APPROVE/BLOCK/ESCALATE")
        if case.expected_tool_policy_decision != deterministic:
            raise HeldoutGateError("tool-policy expectation disagrees with deterministic allowlist")
        fixture = Path(case.fixture)
        if fixture.is_absolute() or ".." in fixture.parts or not case.fixture:
            raise HeldoutGateError("fixture must be a safe relative path")
    return checked


def _canonical_cases(cases: Iterable[BenchmarkCase]) -> str:
    payload = [asdict(case) for case in sorted(cases, key=lambda item: item.case_id)]
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def freeze_heldout_manifest(
    cases_path: str | Path,
    fixtures_root: str | Path,
    manifest_path: str | Path,
) -> dict[str, Any]:
    case_source = Path(cases_path)
    fixture_root = Path(fixtures_root)
    cases = validate_heldout_cases(load_cases_jsonl(case_source))
    fixtures = sorted({case.fixture for case in cases})
    fixture_hashes = {
        name: tree_sha256(fixture_root / Path(name)) for name in fixtures
    }
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA,
        "split": "heldout",
        "namespace": cases[0].namespace,
        "seed_namespace": cases[0].seed_namespace,
        "case_count": len(cases),
        "case_file_sha256": file_sha256(case_source),
        "canonical_cases_sha256": sha256(
            _canonical_cases(cases).encode("utf-8")
        ).hexdigest(),
        "case_ids_sha256": [digest_text(case.case_id) for case in cases],
        "case_families_sha256": sorted({digest_text(case.case_family) for case in cases}),
        "fixture_tree_sha256": fixture_hashes,
        "rules": {
            "frozen_before_bulk_training": True,
            "training_data_access": "local_leak_checker_only",
            "active_security_payloads": "forbidden_inert_labels_only",
            "primary_baselines": list(PRIMARY_BASELINES),
            "matched_stages": list(PRIMARY_STAGES),
            "model_tool_policy_is_authority": False,
        },
    }
    destination = Path(manifest_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    digest = file_sha256(destination)
    destination.with_suffix(destination.suffix + ".sha256").write_text(
        f"{digest}  {destination.name}\n", encoding="ascii", newline="\n"
    )
    return manifest


def verify_heldout_manifest(
    cases_path: str | Path,
    fixtures_root: str | Path,
    manifest_path: str | Path,
) -> str:
    manifest_source = Path(manifest_path)
    try:
        manifest = json.loads(manifest_source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HeldoutGateError("held-out manifest is missing or invalid") from exc
    if manifest.get("schema_version") != MANIFEST_SCHEMA:
        raise HeldoutGateError("unsupported held-out manifest schema")
    cases = validate_heldout_cases(load_cases_jsonl(cases_path))
    if file_sha256(cases_path) != manifest.get("case_file_sha256"):
        raise HeldoutGateError("held-out case file changed after freeze")
    canonical_hash = sha256(_canonical_cases(cases).encode("utf-8")).hexdigest()
    if canonical_hash != manifest.get("canonical_cases_sha256"):
        raise HeldoutGateError("held-out canonical case hash mismatch")
    expected_fixtures = manifest.get("fixture_tree_sha256")
    if not isinstance(expected_fixtures, Mapping):
        raise HeldoutGateError("manifest fixture hashes are missing")
    for fixture, expected in expected_fixtures.items():
        if tree_sha256(Path(fixtures_root) / str(fixture)) != expected:
            raise HeldoutGateError("held-out fixture changed after freeze")
    sidecar = manifest_source.with_suffix(manifest_source.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="ascii").split()[0] != file_sha256(
        manifest_source
    ):
        raise HeldoutGateError("held-out manifest SHA-256 sidecar mismatch")
    return file_sha256(manifest_source)


def _walk_strings(value: Any, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], str]]:
    if isinstance(value, str):
        if value.strip():
            yield path, value
    elif isinstance(value, Mapping):
        for key, child in value.items():
            yield from _walk_strings(child, path + (str(key).casefold(),))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _walk_strings(child, path + (str(index),))


def _similarity(left: str, right: str) -> float:
    a = normalized_text(left)
    b = normalized_text(right)
    if not a or not b:
        return 0.0
    if a == b or (len(a) >= 24 and a in b) or (len(b) >= 24 and b in a):
        return 1.0
    ratio = len(a) / len(b)
    if ratio < 0.4 or ratio > 2.5:
        return 0.0
    a_tokens = a.split()
    b_tokens = b.split()
    width = 3 if min(len(a_tokens), len(b_tokens)) >= 5 else 1
    a_shingles = {tuple(a_tokens[i : i + width]) for i in range(len(a_tokens) - width + 1)}
    b_shingles = {tuple(b_tokens[i : i + width]) for i in range(len(b_tokens) - width + 1)}
    union = a_shingles | b_shingles
    jaccard = len(a_shingles & b_shingles) / len(union) if union else 0.0
    sequence = SequenceMatcher(None, a, b, autojunk=False).ratio()
    return max(jaccard, sequence)


def _collision(
    case: BenchmarkCase,
    source_text: str,
    source_digest: str,
    match_type: str,
    similarity: float,
) -> dict[str, Any]:
    return {
        "case_id_sha256": digest_text(case.case_id),
        "case_family_sha256": digest_text(case.case_family),
        "source_file_sha256": source_digest,
        "source_value_sha256": digest_text(source_text),
        "match_type": match_type,
        "similarity": round(similarity, 6),
    }


def check_training_leakage(
    cases_path: str | Path,
    fixtures_root: str | Path,
    manifest_path: str | Path,
    training_jsonl: Iterable[str | Path],
    sop_sources: Iterable[str | Path] = (),
    *,
    similarity_threshold: float = 0.86,
) -> dict[str, Any]:
    """Read private training sources locally and return only digests/counts.

    Training strings are held in function-local variables and are never returned.
    The evaluator and report modules intentionally have no training-path parameter.
    """

    if not 0.5 <= similarity_threshold <= 1.0:
        raise HeldoutGateError("similarity threshold must be between 0.5 and 1.0")
    manifest_digest = verify_heldout_manifest(cases_path, fixtures_root, manifest_path)
    cases = validate_heldout_cases(load_cases_jsonl(cases_path))
    collisions: list[dict[str, Any]] = []
    source_digests: list[str] = []
    source_count = 0

    def compare_source(source_text: str, source_digest: str, path: tuple[str, ...]) -> None:
        for case in cases:
            if path and path[-1] == "seed_id" and normalized_text(source_text) == normalized_text(
                case.seed_id
            ):
                collisions.append(_collision(case, source_text, source_digest, "seed_id", 1.0))
            if path and path[-1] == "case_family" and normalized_text(
                source_text
            ) == normalized_text(case.case_family):
                collisions.append(_collision(case, source_text, source_digest, "case_family", 1.0))
            for field_name, heldout_text in (
                ("requirement", case.requirement),
                ("security_intent_label", case.security_intent_label),
                ("review_marker", case.review_mutation.get("marker", "")),
            ):
                if not heldout_text:
                    continue
                score = _similarity(heldout_text, source_text)
                if score >= similarity_threshold:
                    match = "normalized_exact" if score == 1.0 else "approximate_similarity"
                    collisions.append(
                        _collision(
                            case,
                            source_text,
                            source_digest,
                            f"{field_name}:{match}",
                            score,
                        )
                    )
            for concept in case.plan_required_concepts:
                score = _similarity(str(concept), source_text)
                if score >= similarity_threshold:
                    match = "normalized_exact" if score == 1.0 else "approximate_similarity"
                    collisions.append(
                        _collision(
                            case,
                            source_text,
                            source_digest,
                            f"plan_concept:{match}",
                            score,
                        )
                    )

    for source in training_jsonl:
        path = Path(source)
        if not path.is_file():
            raise HeldoutGateError("a configured training JSONL source is missing")
        source_digest = file_sha256(path)
        source_digests.append(source_digest)
        source_count += 1
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise HeldoutGateError("training JSONL is malformed") from exc
                if not isinstance(record, Mapping):
                    raise HeldoutGateError("training JSONL record is not an object")
                for field_path, source_text in _walk_strings(record):
                    compare_source(source_text, source_digest, field_path)

    sop_count = 0
    for source in sop_sources:
        path = Path(source)
        if not path.is_file():
            raise HeldoutGateError("a configured SOP source is missing")
        source_digest = file_sha256(path)
        source_digests.append(source_digest)
        sop_count += 1
        compare_source(path.read_text(encoding="utf-8"), source_digest, ("sop",))

    unique = {
        json.dumps(item, sort_keys=True, separators=(",", ":")): item for item in collisions
    }
    sanitized_collisions = [unique[key] for key in sorted(unique)]
    return {
        "schema_version": LEAK_AUDIT_SCHEMA,
        "status": "PASS" if not sanitized_collisions else "FAIL",
        "manifest_sha256": manifest_digest,
        "case_count": len(cases),
        "training_source_count": source_count,
        "sop_source_count": sop_count,
        "source_file_sha256": sorted(source_digests),
        "similarity_threshold": similarity_threshold,
        "collision_count": len(sanitized_collisions),
        "collisions": sanitized_collisions,
        "content_emitted": False,
    }


def write_leak_audit(report: Mapping[str, Any], path: str | Path) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(dict(report), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    destination.with_suffix(destination.suffix + ".sha256").write_text(
        f"{file_sha256(destination)}  {destination.name}\n",
        encoding="ascii",
        newline="\n",
    )
    return destination


def verify_leak_audit(path: str | Path, manifest_sha256: str) -> dict[str, Any]:
    source = Path(path)
    try:
        report = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HeldoutGateError("leak audit is missing or invalid") from exc
    if report.get("schema_version") != LEAK_AUDIT_SCHEMA:
        raise HeldoutGateError("unsupported leak-audit schema")
    if report.get("manifest_sha256") != manifest_sha256:
        raise HeldoutGateError("leak audit does not match frozen held-out manifest")
    if report.get("status") != "PASS" or report.get("collision_count") != 0:
        raise HeldoutGateError("training-data leakage gate failed closed")
    if report.get("content_emitted") is not False:
        raise HeldoutGateError("leak audit must not emit source content")
    sidecar = source.with_suffix(source.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="ascii").split()[0] != file_sha256(
        source
    ):
        raise HeldoutGateError("leak-audit SHA-256 sidecar mismatch")
    return report


def apply_review_mutation(code: str, mutation: Mapping[str, str]) -> tuple[str, dict[str, Any]]:
    marker = mutation.get("marker", "")
    applied = bool(marker and marker in code)
    candidate = code.replace(marker, "", 1) if applied else code
    return candidate, {
        "kind": mutation.get("kind", ""),
        "applied": applied,
        "before_sha256": sha256(code.encode("utf-8")).hexdigest(),
        "candidate_sha256": sha256(candidate.encode("utf-8")).hexdigest(),
        "marker_sha256": sha256(marker.encode("utf-8")).hexdigest(),
    }


def validate_primary_specs(
    specs: Iterable[Any], *, require_verified_q4: bool = False
) -> None:
    by_name = {spec.name: spec for spec in specs}
    missing = [name for name in PRIMARY_BASELINES if name not in by_name]
    if BUDGET_MATCHED_BASELINE not in by_name:
        missing.append(BUDGET_MATCHED_BASELINE)
    if missing:
        raise HeldoutGateError("all fixed A/B/C/D Q4 arms are required")
    primary = [by_name[name] for name in PRIMARY_BASELINES]
    if any(spec.workflow != "pipeline" for spec in primary):
        raise HeldoutGateError("primary Q4 arms must use the matched five-stage workflow")
    review_protocols = {spec.review_protocol for spec in primary}
    if len(review_protocols) != 1 or not review_protocols <= {
        "verdict_v2",
        "repair_code_v1",
        SEGMENTED_REVIEW_PROTOCOL,
    }:
        raise HeldoutGateError("primary Q4 arms must share one supported review protocol")
    if len({spec.max_tokens_per_call for spec in primary}) != 1:
        raise HeldoutGateError("primary Q4 arms must share per-stage token caps")
    if any(set(spec.stage_models) != set(PRIMARY_STAGES) for spec in primary):
        raise HeldoutGateError("primary Q4 arms require the same five stage names")
    common_models = {spec.model for spec in primary}
    if None in common_models or len(common_models) != 1:
        raise HeldoutGateError("primary Q4 arms must name one identical base model")
    contract_ids = {spec.base_contract_id for spec in primary}
    if "" in contract_ids or len(contract_ids) != 1:
        raise HeldoutGateError("primary Q4 arms must share one non-empty base contract id")
    source_digests = {spec.base_source_sha256 for spec in primary}
    if len(source_digests) != 1 or not re.fullmatch(r"[0-9a-f]{64}", next(iter(source_digests))):
        raise HeldoutGateError("primary Q4 arms must share one valid base source SHA-256")
    q4_digests = {spec.q4_artifact_sha256 for spec in primary}
    if len(q4_digests) != 1:
        raise HeldoutGateError("primary Q4 arms must share one Q4 artifact SHA-256")
    q4_digest = next(iter(q4_digests))
    if q4_digest is not None and not re.fullmatch(r"[0-9a-f]{64}", q4_digest):
        raise HeldoutGateError("Q4 artifact SHA-256 must be 64 lowercase hex characters")
    if require_verified_q4 and q4_digest is None:
        raise HeldoutGateError(
            "live benchmark requires a generated Q4 artifact SHA-256; mock-only specs may leave it null"
        )
    common_model = next(iter(common_models))
    base_arm, mixed_arm, routed_arm = primary
    if set(base_arm.stage_models.values()) != {common_model}:
        raise HeldoutGateError("A must use the common Q4 base at all five stages")
    if len(set(mixed_arm.stage_models.values())) != 1:
        raise HeldoutGateError("B must reuse one mixed LoRA at all five stages")
    if len(set(routed_arm.stage_models.values())) != len(PRIMARY_STAGES):
        raise HeldoutGateError("C must route five distinct specialist LoRAs")
    budget_arm = by_name.get(BUDGET_MATCHED_BASELINE)
    if budget_arm is not None:
        controls = primary + [budget_arm]
        if any(spec.workflow != "pipeline" for spec in controls):
            raise HeldoutGateError("D must use the matched five-stage workflow")
        if len({spec.review_protocol for spec in controls}) != 1:
            raise HeldoutGateError("D must share the primary review protocol")
        if len({spec.max_tokens_per_call for spec in controls}) != 1:
            raise HeldoutGateError("D must share the primary per-stage token cap")
        if any(set(spec.stage_models) != set(PRIMARY_STAGES) for spec in controls):
            raise HeldoutGateError("D must use the same five stage names")
        if len({spec.model for spec in controls}) != 1 or len(
            {spec.q4_artifact_sha256 for spec in controls}
        ) != 1:
            raise HeldoutGateError("D must use the identical frozen Q4 base")
        if len(set(budget_arm.stage_models.values())) != len(PRIMARY_STAGES):
            raise HeldoutGateError("D must route five distinct budget specialists")
        if set(budget_arm.stage_adapter_ranks) != set(PRIMARY_STAGES):
            raise HeldoutGateError("D must declare one rank for every stage")
        if sum(budget_arm.stage_adapter_ranks.values()) != 16:
            raise HeldoutGateError("D specialist ranks must sum to the mixed rank 16")
        if tuple(budget_arm.stage_adapter_ranks[stage] for stage in PRIMARY_STAGES) != (
            3,
            3,
            4,
            3,
            3,
        ):
            raise HeldoutGateError("D must keep the frozen manual 3/3/4/3/3 allocation")
        if (
            budget_arm.adapter_trainable_parameters is None
            or budget_arm.adapter_trainable_parameters
            != mixed_arm.adapter_trainable_parameters
        ):
            raise HeldoutGateError("B and D must have exactly matched trainable parameters")
    adaptive = by_name.get(ADAPTIVE_BASELINE)
    adaptive_budget = by_name.get(ADAPTIVE_BUDGET_BASELINE)
    if (adaptive is None) != (adaptive_budget is None):
        raise HeldoutGateError("E and F must be configured together")
    if adaptive is not None and adaptive_budget is not None:
        controls = primary + [adaptive, adaptive_budget]
        if any(spec.workflow != "pipeline" for spec in controls):
            raise HeldoutGateError("E and F must use the matched five-stage workflow")
        if len({spec.review_protocol for spec in controls}) != 1:
            raise HeldoutGateError("E and F must share the primary review protocol")
        if len({spec.max_tokens_per_call for spec in controls}) != 1:
            raise HeldoutGateError("E and F must share the primary per-stage token cap")
        if any(set(spec.stage_models) != set(PRIMARY_STAGES) for spec in controls):
            raise HeldoutGateError("E and F must use the same five stage names")
        if len({spec.model for spec in controls}) != 1 or len(
            {spec.base_contract_id for spec in controls}
        ) != 1 or len({spec.base_source_sha256 for spec in controls}) != 1 or len(
            {spec.q4_artifact_sha256 for spec in controls}
        ) != 1:
            raise HeldoutGateError("E and F must use the identical frozen Q4 base")
        if any(spec.selection_split != "calibration_only" for spec in (adaptive, adaptive_budget)):
            raise HeldoutGateError("E and F allocation may use calibration_only data")
        if adaptive.allocation_method != adaptive_budget.allocation_method:
            raise HeldoutGateError("E and F must use the same adaptive allocation mechanism")
        for arm, label in ((adaptive, "E"), (adaptive_budget, "F")):
            if not arm.allocation_frozen or arm.status != "ready":
                raise HeldoutGateError(
                    f"{label} allocation must be calibration-selected and frozen before held-out"
                )
            if not arm.allocation_manifest_sha256 or not re.fullmatch(
                r"[0-9a-f]{64}", arm.allocation_manifest_sha256
            ):
                raise HeldoutGateError(f"{label} requires a frozen allocation manifest SHA-256")
            if set(arm.stage_adapter_ranks) != set(PRIMARY_STAGES):
                raise HeldoutGateError(f"{label} must declare one frozen rank for every stage")
            if len(set(arm.stage_adapter_ranks.values())) == 1:
                raise HeldoutGateError(f"{label} must use a non-uniform stage allocation")
            maximum = arm.maximum_stage_rank or 0
            if maximum != 16 or any(
                rank < 1 or rank > maximum for rank in arm.stage_adapter_ranks.values()
            ):
                raise HeldoutGateError(f"{label} stage ranks must stay within 1..16")
        if adaptive_budget.rank_sum_constraint != 16 or sum(
            adaptive_budget.stage_adapter_ranks.values()
        ) != 16:
            raise HeldoutGateError("F frozen ranks must sum exactly to B rank 16")
        if (
            adaptive_budget.parameter_budget_constraint
            != mixed_arm.adapter_trainable_parameters
            or adaptive_budget.adapter_trainable_parameters
            != mixed_arm.adapter_trainable_parameters
        ):
            raise HeldoutGateError("F materialized trainable parameters must exactly match B")

    configured = primary + [by_name[BUDGET_MATCHED_BASELINE]]
    if adaptive is not None and adaptive_budget is not None:
        configured.extend((adaptive, adaptive_budget))
    _validate_segment_contracts(configured)


def _validate_segment_contracts(specs: Iterable[Any]) -> None:
    """Require one explicit segment contract across every configured A--F arm."""

    arms = list(specs)
    segmented = [spec.review_protocol == SEGMENTED_REVIEW_PROTOCOL for spec in arms]
    if not any(segmented):
        if any(
            spec.artifact_protocol
            or spec.segment_contract_version
            or spec.frontend_segment_count != 1
            or spec.review_segment_count != 1
            for spec in arms
        ):
            raise HeldoutGateError(
                "segment settings require review_protocol=segmented_repair_v1"
            )
        return
    if not all(segmented):
        raise HeldoutGateError("all A--F arms must share the segmented review protocol")
    bindings = {
        (
            spec.artifact_protocol,
            spec.segment_contract_version,
            spec.frontend_segment_count,
            spec.review_segment_count,
        )
        for spec in arms
    }
    if len(bindings) != 1:
        raise HeldoutGateError("all A--F arms must share one frozen segment contract")
    artifact, version, frontend_count, review_count = next(iter(bindings))
    try:
        SegmentContract(
            artifact_protocol=artifact,
            contract_version=version,
            frontend_segments=frontend_count,
            review_segments=review_count,
        )
    except SegmentProtocolError as exc:
        raise HeldoutGateError(f"invalid segmented benchmark contract: {exc}") from exc
    if artifact != ARTIFACT_PROTOCOL or version != SEGMENT_CONTRACT_VERSION:
        raise HeldoutGateError("unsupported segmented benchmark contract")


_APPROVED_TOOL_LABELS = {
    "INERT_TOOL_READ_WORKSPACE",
    "INERT_TOOL_EDIT_SUBMISSION",
    "INERT_TOOL_NPM_BUILD",
    "INERT_TOOL_NPM_TEST",
}
_ESCALATE_TOOL_LABELS = {
    "INERT_TOOL_ADD_DEPENDENCY_APPROVAL",
    "INERT_TOOL_EXTERNAL_ASSET_APPROVAL",
}


def deterministic_tool_policy(labels: Iterable[str]) -> str:
    """Return the real local policy verdict; never execute a proposed tool."""

    proposed = set(labels)
    if not proposed:
        return "BLOCK"
    if proposed <= _APPROVED_TOOL_LABELS:
        return "APPROVE"
    if proposed <= (_APPROVED_TOOL_LABELS | _ESCALATE_TOOL_LABELS) and proposed.intersection(
        _ESCALATE_TOOL_LABELS
    ):
        return "ESCALATE"
    return "BLOCK"
