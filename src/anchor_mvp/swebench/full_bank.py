"""Reproducible, train-only SWE-bench full-bank staging and publication.

Readiness is deliberately split into three independent gates:

* launch readiness validates immutable local inputs and component attestations;
* training readiness validates completed Gold, localization, and tool results;
* publication readiness validates the metadata-only public export.

Runtime outputs are never launch prerequisites.  That separation makes an
offline reboot-time preflight possible without claiming that a run already
produced the evidence it is about to create.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from ..data.provider import reject_inline_secrets, validate_base_url
from ..tooling.tool_contract import EXECUTION_TOOL_CONTRACT_V3_VERSION
from .partition import ALLOWLIST_SCHEMA_VERSION, file_sha256
from .schema import (
    CHAIN_STAGES,
    SWEBenchValidationError,
    canonical_json,
    clean_problem_statement,
    digest_string_set,
    validate_dataset_id,
    validate_immutable_revision,
    validate_instance_id,
    validate_repository,
)


FULL_BANK_CONFIG_SCHEMA_VERSION = "anchor.swebench-full-bank-config.v2"
FULL_BANK_MANIFEST_SCHEMA_VERSION = "anchor.swebench-full-bank-manifest.v2"
FULL_BANK_PREFLIGHT_SCHEMA_VERSION = "anchor.swebench-full-bank-preflight.v1"
PUBLICATION_MANIFEST_SCHEMA_VERSION = "anchor.swebench-publication-manifest.v1"
CANDIDATE_TASK_SCHEMA_VERSION = "anchor.swebench-candidate-task.v1"
CANDIDATE_WORK_ORDER_SCHEMA_VERSION = "anchor.swebench-candidate-work-order.v1"

OPEN_CODE_BUNDLE_SCHEMA_VERSION = "anchor.patched-opencode.bundle.v1"
OPEN_CODE_PLATFORM_SCHEMA_VERSION = "anchor.patched-opencode.platform.v1"
CCSWITCH_ROUTE_SCHEMA_VERSION = "anchor.ccswitch-route-manifest.v1"
TRAINING_GATE_SCHEMAS = {
    "gold_manifest": "anchor.swebench-gold-manifest.v1",
    "zh_cn_localization_manifest": (
        "anchor.swebench-zh-cn-localization-manifest.v1"
    ),
    "real_tool_results_manifest": (
        "anchor.swebench-real-tool-results-manifest.v1"
    ),
}

FIFTY_MIB = 50 * 1024 * 1024
SOURCE_COLUMNS = (
    "repo",
    "instance_id",
    "base_commit",
    "problem_statement",
)
UPSTREAM_ORACLE_COLUMNS = frozenset(
    {
        "patch",
        "test_patch",
        "hints_text",
        "fail_to_pass",
        "pass_to_pass",
        "tests",
        "test_cases",
    }
)
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_SAFE_ALIAS = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_PUBLIC_SECRET = re.compile(
    r"(?:"
    r"sk-[A-Za-z0-9_-]{20,}|"
    r"(?<![A-Za-z0-9])ark-[0-9A-Fa-f]{8}-"
    r"(?:[0-9A-Fa-f]{4}-){3}[0-9A-Fa-f]{12}"
    r"(?:-[A-Za-z0-9_-]+)?(?![A-Za-z0-9_-])|"
    r"AKIA[0-9A-Z]{16}|"
    r"gh[pousr]_[A-Za-z0-9]{20,}|"
    r"xox[baprs]-[A-Za-z0-9-]{20,}|"
    r"-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----"
    r")"
)
_FORBIDDEN_PUBLIC_KEYS = frozenset(
    {
        "patch",
        "gold_patch",
        "gold_solution",
        "test",
        "test_patch",
        "hint",
        "hints",
        "hints_text",
        "fail_to_pass",
        "pass_to_pass",
        "tests",
        "test_cases",
        "oracle",
        "oracle_text",
        "heldout",
        "held_out",
        "heldout_registry",
        "api_key",
        "api_key_env",
        "api_token",
        "access_token",
        "authorization",
        "credentials",
        "key",
        "password",
        "private_key",
        "secret",
        "token",
    }
)
_STAGE_EXECUTION = {
    "planner": "teacher-json",
    "tool_policy": "teacher-json",
    "domain_builder": "controlled-opencode-sandbox",
    "domain_review": "teacher-json",
    "security": "teacher-json",
}
_OUTPUT_SCHEMAS = {
    "planner": "anchor.swebench-planner-output.v1",
    "tool_policy": "anchor.swebench-tool-policy-output.v1",
    "domain_builder": "controlled-opencode-export+real-tool-results",
    "domain_review": "anchor.swebench-domain-review-output.v1",
    "security": "anchor.swebench-security-output.v1",
}


@dataclass(frozen=True)
class ProviderProfile:
    alias: str
    provider: str
    protocol: str
    base_url: str
    api_key_env: str
    model_id: str | None
    user_agent: str
    reasoning_effort: str
    discover_models: bool
    force_manual_model: bool
    discovery_failure_policy: str

    def public_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "protocol": self.protocol,
            "base_url": self.base_url,
            "api_key_env": self.api_key_env,
            "model_id": self.model_id,
            "user_agent": self.user_agent,
            "reasoning_effort": self.reasoning_effort,
            "discover_models": self.discover_models,
            "force_manual_model": self.force_manual_model,
            "discovery_failure_policy": self.discovery_failure_policy,
        }


@dataclass(frozen=True)
class StageRoute:
    default_provider: str
    frontend_provider: str | None
    execution: str


@dataclass(frozen=True)
class FullBankConfig:
    project_root: Path
    config_path: Path
    dataset_id: str
    dataset_revision: str
    source_parquet: Path
    source_parquet_sha256: str
    expected_rows: int
    output_dir: Path
    public_manifest_path: Path
    validation_numerator: int
    validation_denominator: int
    split_algorithm: str
    locales: tuple[str, ...]
    locale_algorithm: str
    require_localized_text_before_live: bool
    shard_rows: int
    max_file_bytes: int
    raw_source_publishable: bool
    audited_export_dir: Path
    gate_paths: Mapping[str, Mapping[str, Path]]
    providers: Mapping[str, ProviderProfile]
    stage_routes: Mapping[str, StageRoute]
    frontend_keywords: tuple[str, ...]
    formal_enabled: bool
    formal_required_effort: str
    formal_required_providers: tuple[str, ...]
    require_real_sandbox: bool
    capture_tool_calls: bool
    capture_tool_results: bool
    capture_hidden_chain_of_thought: bool
    capture_public_reasoning_summary: bool

    @classmethod
    def load(cls, project_root: Path, path: Path) -> "FullBankConfig":
        import yaml

        root = project_root.resolve()
        config_path = path.resolve()
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if not isinstance(raw, Mapping):
            raise SWEBenchValidationError("full-bank config root must be an object")
        expected_top = {
            "schema_version",
            "source",
            "output_dir",
            "public_manifest",
            "split",
            "bilingual",
            "publication",
            "gates",
            "providers",
            "stage_routes",
            "classification",
            "formal_profile",
        }
        if set(raw) != expected_top:
            raise SWEBenchValidationError("full-bank config has unexpected fields")
        if raw.get("schema_version") != FULL_BANK_CONFIG_SCHEMA_VERSION:
            raise SWEBenchValidationError("unsupported full-bank config schema")

        source = _exact_mapping(
            raw.get("source"),
            "source",
            {
                "dataset_id",
                "dataset_revision",
                "split",
                "parquet",
                "parquet_sha256",
                "expected_rows",
            },
        )
        if source.get("split") != "train":
            raise SWEBenchValidationError("full-bank source must be split=train")
        dataset_id = validate_dataset_id(_text(source.get("dataset_id"), "dataset_id"))
        dataset_revision = validate_immutable_revision(
            _text(source.get("dataset_revision"), "dataset_revision")
        )
        parquet_sha = _text(source.get("parquet_sha256"), "parquet_sha256").lower()
        if not _SHA256.fullmatch(parquet_sha):
            raise SWEBenchValidationError("source parquet_sha256 must be lowercase SHA-256")
        expected_rows = _positive_int(source.get("expected_rows"), "expected_rows")
        source_parquet = _project_path(root, source.get("parquet"), "source.parquet")
        output_dir = _project_path(root, raw.get("output_dir"), "output_dir")
        public_manifest_path = _project_path(
            root, raw.get("public_manifest"), "public_manifest"
        )
        if output_dir == source_parquet or output_dir in source_parquet.parents:
            raise SWEBenchValidationError("output_dir must not contain the raw parquet")

        split = _exact_mapping(
            raw.get("split"),
            "split",
            {"algorithm", "validation_numerator", "validation_denominator"},
        )
        split_algorithm = _text(split.get("algorithm"), "split.algorithm")
        if split_algorithm != "per_repo_sha256_rank_v1":
            raise SWEBenchValidationError("unsupported full-bank split algorithm")
        numerator = _positive_int(
            split.get("validation_numerator"), "validation_numerator"
        )
        denominator = _positive_int(
            split.get("validation_denominator"), "validation_denominator"
        )
        if numerator >= denominator:
            raise SWEBenchValidationError("validation fraction must be below one")

        bilingual = _exact_mapping(
            raw.get("bilingual"),
            "bilingual",
            {"locales", "assignment", "require_localized_text_before_live"},
        )
        locale_values = bilingual.get("locales")
        if (
            not isinstance(locale_values, list)
            or len(locale_values) != 2
            or set(locale_values) != {"en-US", "zh-CN"}
        ):
            raise SWEBenchValidationError("bilingual.locales must be en-US and zh-CN")
        locale_algorithm = _text(bilingual.get("assignment"), "bilingual.assignment")
        if locale_algorithm != "global_sha256_rank_alternating_v1":
            raise SWEBenchValidationError("unsupported bilingual assignment algorithm")
        require_localized = bilingual.get("require_localized_text_before_live")
        if require_localized is not True:
            raise SWEBenchValidationError(
                "formal bilingual bank must require localized text before live use"
            )

        publication = _exact_mapping(
            raw.get("publication"),
            "publication",
            {
                "shard_rows",
                "max_file_bytes",
                "raw_source_publishable",
                "audited_export_dir",
            },
        )
        shard_rows = _positive_int(publication.get("shard_rows"), "shard_rows")
        max_file_bytes = _positive_int(
            publication.get("max_file_bytes"), "max_file_bytes"
        )
        if max_file_bytes > FIFTY_MIB:
            raise SWEBenchValidationError("publication max_file_bytes may not exceed 50 MiB")
        if publication.get("raw_source_publishable") is not False:
            raise SWEBenchValidationError("raw source parquet must remain non-publishable")
        audited_export_dir = _project_path(
            root, publication.get("audited_export_dir"), "audited_export_dir"
        )
        if "artifacts" in {part.casefold() for part in audited_export_dir.parts}:
            raise SWEBenchValidationError(
                "audited public export must not reuse the ignored artifacts tree"
            )

        gate_raw = _exact_mapping(
            raw.get("gates"), "gates", {"launch", "training", "publication"}
        )
        launch_gate_raw = _exact_mapping(
            gate_raw.get("launch"),
            "gates.launch",
            {"opencode_bundle_manifest", "ccswitch_route_manifest"},
        )
        training_gate_raw = _exact_mapping(
            gate_raw.get("training"),
            "gates.training",
            {
                "gold_manifest",
                "zh_cn_localization_manifest",
                "real_tool_results_manifest",
            },
        )
        publication_gate_raw = _exact_mapping(
            gate_raw.get("publication"),
            "gates.publication",
            {"mit_attribution_file"},
        )
        gate_paths: dict[str, dict[str, Path]] = {
            "launch": {
                name: _project_path(root, value, f"gates.launch.{name}")
                for name, value in launch_gate_raw.items()
            },
            "training": {
                name: _project_path(root, value, f"gates.training.{name}")
                for name, value in training_gate_raw.items()
            },
            "publication": {
                name: _project_path(root, value, f"gates.publication.{name}")
                for name, value in publication_gate_raw.items()
            },
        }
        attribution_path = gate_paths["publication"]["mit_attribution_file"]
        if attribution_path != audited_export_dir / "ATTRIBUTION.md":
            raise SWEBenchValidationError(
                "publication attribution must be audited_export_dir/ATTRIBUTION.md"
            )
        for group in ("launch", "training"):
            if any(
                path == audited_export_dir or audited_export_dir in path.parents
                for path in gate_paths[group].values()
            ):
                raise SWEBenchValidationError(
                    f"{group} evidence must remain outside the public export"
                )

        providers_raw = raw.get("providers")
        if not isinstance(providers_raw, Mapping) or not providers_raw:
            raise SWEBenchValidationError("providers must be a non-empty object")
        providers: dict[str, ProviderProfile] = {}
        provider_fields = {
            "provider",
            "protocol",
            "base_url",
            "api_key_env",
            "model_id",
            "user_agent",
            "reasoning_effort",
            "discover_models",
            "force_manual_model",
            "discovery_failure_policy",
        }
        for raw_alias, raw_profile in providers_raw.items():
            alias = _safe_alias(raw_alias, "provider alias")
            profile = _exact_mapping(raw_profile, f"providers.{alias}", provider_fields)
            reject_inline_secrets(profile)
            protocol = _text(profile.get("protocol"), f"providers.{alias}.protocol")
            if protocol not in {"openai", "openai_responses", "anthropic"}:
                raise SWEBenchValidationError("unsupported provider protocol")
            api_key_env = _text(
                profile.get("api_key_env"), f"providers.{alias}.api_key_env"
            )
            if not _ENV_NAME.fullmatch(api_key_env):
                raise SWEBenchValidationError("api_key_env is not a valid env name")
            model_value = profile.get("model_id")
            model_id = None if model_value is None else _text(model_value, "model_id")
            discover = _bool(profile.get("discover_models"), "discover_models")
            force_manual = _bool(
                profile.get("force_manual_model"), "force_manual_model"
            )
            failure_policy = _text(
                profile.get("discovery_failure_policy"), "discovery_failure_policy"
            )
            if failure_policy not in {"fail_closed", "require_manual_model"}:
                raise SWEBenchValidationError("unsupported discovery failure policy")
            if (force_manual or not discover) and model_id is None:
                raise SWEBenchValidationError(
                    "manual/undiscovered provider profiles require model_id"
                )
            providers[alias] = ProviderProfile(
                alias=alias,
                provider=_text(profile.get("provider"), "provider"),
                protocol=protocol,
                base_url=validate_base_url(_text(profile.get("base_url"), "base_url")),
                api_key_env=api_key_env,
                model_id=model_id,
                user_agent=_text(profile.get("user_agent"), "user_agent"),
                reasoning_effort=_text(
                    profile.get("reasoning_effort"), "reasoning_effort"
                ).casefold(),
                discover_models=discover,
                force_manual_model=force_manual,
                discovery_failure_policy=failure_policy,
            )

        routes_raw = raw.get("stage_routes")
        if not isinstance(routes_raw, Mapping) or set(routes_raw) != set(CHAIN_STAGES):
            raise SWEBenchValidationError("stage_routes must define exactly five stages")
        stage_routes: dict[str, StageRoute] = {}
        for stage in CHAIN_STAGES:
            route = _exact_mapping(
                routes_raw.get(stage),
                f"stage_routes.{stage}",
                {"default_provider", "frontend_provider", "execution"},
            )
            default_provider = _safe_alias(route.get("default_provider"), "route provider")
            frontend_value = route.get("frontend_provider")
            frontend_provider = (
                None
                if frontend_value is None
                else _safe_alias(frontend_value, "frontend route provider")
            )
            execution = _text(route.get("execution"), "route.execution")
            if execution != _STAGE_EXECUTION[stage]:
                raise SWEBenchValidationError(
                    f"{stage} execution must be {_STAGE_EXECUTION[stage]}"
                )
            for alias in (default_provider, frontend_provider):
                if alias is not None and alias not in providers:
                    raise SWEBenchValidationError("stage route references unknown provider")
            stage_routes[stage] = StageRoute(
                default_provider=default_provider,
                frontend_provider=frontend_provider,
                execution=execution,
            )

        classification = _exact_mapping(
            raw.get("classification"), "classification", {"frontend_keywords"}
        )
        keyword_values = classification.get("frontend_keywords")
        if not isinstance(keyword_values, list) or not keyword_values:
            raise SWEBenchValidationError("frontend_keywords must be non-empty")
        keywords = tuple(
            sorted({_text(value, "frontend keyword").casefold() for value in keyword_values})
        )

        formal = _exact_mapping(
            raw.get("formal_profile"),
            "formal_profile",
            {
                "enabled",
                "required_reasoning_effort",
                "required_provider_aliases",
                "require_real_sandbox",
                "capture_tool_calls",
                "capture_tool_results",
                "capture_hidden_chain_of_thought",
                "capture_public_reasoning_summary",
            },
        )
        formal_enabled = _bool(formal.get("enabled"), "formal_profile.enabled")
        required_effort = _text(
            formal.get("required_reasoning_effort"), "required_reasoning_effort"
        ).casefold()
        required_aliases_raw = formal.get("required_provider_aliases")
        if not isinstance(required_aliases_raw, list) or not required_aliases_raw:
            raise SWEBenchValidationError("formal profile requires provider aliases")
        required_aliases = tuple(
            _safe_alias(alias, "required provider alias") for alias in required_aliases_raw
        )
        if len(set(required_aliases)) != len(required_aliases):
            raise SWEBenchValidationError("formal provider aliases repeat")
        for alias in required_aliases:
            if alias not in providers:
                raise SWEBenchValidationError("formal profile references unknown provider")
        require_real_sandbox = _bool(
            formal.get("require_real_sandbox"), "require_real_sandbox"
        )
        capture_calls = _bool(formal.get("capture_tool_calls"), "capture_tool_calls")
        capture_results = _bool(
            formal.get("capture_tool_results"), "capture_tool_results"
        )
        capture_hidden = _bool(
            formal.get("capture_hidden_chain_of_thought"),
            "capture_hidden_chain_of_thought",
        )
        capture_public = _bool(
            formal.get("capture_public_reasoning_summary"),
            "capture_public_reasoning_summary",
        )
        if formal_enabled:
            if required_effort != "max":
                raise SWEBenchValidationError(
                    "formal full-bank required_reasoning_effort must be max"
                )
            referenced = {
                alias
                for route in stage_routes.values()
                for alias in (route.default_provider, route.frontend_provider)
                if alias is not None
            }
            if not set(required_aliases).issubset(referenced):
                raise SWEBenchValidationError(
                    "every formal provider must be used by a stage route"
                )
            if any(providers[alias].reasoning_effort != "max" for alias in referenced):
                raise SWEBenchValidationError(
                    "every formal GLM/Kimi request must preserve reasoning_effort=max"
                )
            if not require_real_sandbox or not capture_calls or not capture_results:
                raise SWEBenchValidationError(
                    "formal full-bank requires real sandbox tool calls and tool results"
                )
            if capture_hidden or not capture_public:
                raise SWEBenchValidationError(
                    "capture structured public rationale, never hidden chain-of-thought"
                )

        return cls(
            project_root=root,
            config_path=config_path,
            dataset_id=dataset_id,
            dataset_revision=dataset_revision,
            source_parquet=source_parquet,
            source_parquet_sha256=parquet_sha,
            expected_rows=expected_rows,
            output_dir=output_dir,
            public_manifest_path=public_manifest_path,
            validation_numerator=numerator,
            validation_denominator=denominator,
            split_algorithm=split_algorithm,
            locales=tuple(str(item) for item in locale_values),
            locale_algorithm=locale_algorithm,
            require_localized_text_before_live=require_localized,
            shard_rows=shard_rows,
            max_file_bytes=max_file_bytes,
            raw_source_publishable=False,
            audited_export_dir=audited_export_dir,
            gate_paths=gate_paths,
            providers=providers,
            stage_routes=stage_routes,
            frontend_keywords=keywords,
            formal_enabled=formal_enabled,
            formal_required_effort=required_effort,
            formal_required_providers=required_aliases,
            require_real_sandbox=require_real_sandbox,
            capture_tool_calls=capture_calls,
            capture_tool_results=capture_results,
            capture_hidden_chain_of_thought=capture_hidden,
            capture_public_reasoning_summary=capture_public,
        )


@dataclass(frozen=True)
class FullBankBuildResult:
    manifest_path: Path
    manifest: Mapping[str, Any]


def build_full_bank(config: FullBankConfig) -> FullBankBuildResult:
    """Build local candidate shards and a content-free fail-closed manifest."""

    if not config.source_parquet.is_file():
        raise SWEBenchValidationError("pinned train parquet is missing")
    observed_sha = file_sha256(config.source_parquet)
    if observed_sha != config.source_parquet_sha256:
        raise SWEBenchValidationError("pinned train parquet SHA-256 mismatch")

    rows, upstream_schema = _read_projected_train_rows(config)
    if len(rows) != config.expected_rows:
        raise SWEBenchValidationError("train parquet cardinality differs from config")
    ids = [row["instance_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise SWEBenchValidationError("train parquet repeats instance_id")

    validation_ids = _validation_ids(
        rows,
        numerator=config.validation_numerator,
        denominator=config.validation_denominator,
    )
    locale_by_id = _locale_assignments(rows, config.locales)
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)

    source_rows: list[dict[str, Any]] = []
    candidate_tasks: list[dict[str, Any]] = []
    candidate_orders: list[dict[str, Any]] = []
    repository_counts: Counter[str] = Counter()
    partition_counts: Counter[str] = Counter()
    locale_counts: Counter[str] = Counter()
    domain_counts: Counter[str] = Counter()
    provider_stage_counts: Counter[str] = Counter()

    sorted_rows = sorted(rows, key=lambda item: (item["repo"].casefold(), item["instance_id"]))
    for row in sorted_rows:
        instance_id = row["instance_id"]
        repo = row["repo"]
        partition = "validation" if instance_id in validation_ids else "train"
        locale = locale_by_id[instance_id]
        domain_label = _domain_label(row, config.frontend_keywords)
        task_id = _task_id(config, row)
        providers = _providers_for_task(config, domain_label)
        # This is the complete public projection. Dataset binding lives in the
        # manifest rather than being repeated in every content-bearing row.
        source_row = {
            "repo": repo,
            "instance_id": instance_id,
            "base_commit": row["base_commit"],
            "problem_statement": row["problem_statement"],
        }
        source_rows.append(source_row)
        candidate = {
            "schema_version": CANDIDATE_TASK_SCHEMA_VERSION,
            "task_id": task_id,
            "eligibility_status": "candidate_train_only",
            "source": {
                "dataset_id": config.dataset_id,
                "dataset_revision": config.dataset_revision,
                "split": "train",
                "derived_partition": partition,
                "instance_id": instance_id,
                "repo": repo,
                "base_commit": row["base_commit"],
            },
            "public_input": {"problem_statement": row["problem_statement"]},
            "bilingual": {
                "source_locale": "en-US",
                "requested_locale": locale,
                "localization_status": (
                    "source_ready" if locale == "en-US" else "translation_required"
                ),
            },
            "routing_contract": {
                "domain_label": domain_label,
                "providers_by_stage": providers,
                "reasoning_effort": config.formal_required_effort,
            },
            "chain_contract": {
                "stages": list(CHAIN_STAGES),
                "dependency_order": "strict",
                "review_revision_edge": "domain_review:REVISE->domain_builder",
                "real_sandbox_required_for_builder": True,
            },
        }
        candidate_tasks.append(candidate)
        candidate_orders.extend(_work_orders(candidate, config))
        repository_counts[repo] += 1
        partition_counts[partition] += 1
        locale_counts[locale] += 1
        domain_counts[domain_label] += 1
        for stage, alias in providers.items():
            provider_stage_counts[f"{stage}:{alias}"] += 1

    train_ids = sorted(set(ids) - validation_ids)
    validation_ids_sorted = sorted(validation_ids)
    allowlist_dir = output / "allowlists"
    allowlist_dir.mkdir(parents=True, exist_ok=True)
    training_allowlist = allowlist_dir / "train.json"
    validation_allowlist = allowlist_dir / "validation-from-train.json"
    _atomic_json(
        training_allowlist,
        {
            "schema_version": ALLOWLIST_SCHEMA_VERSION,
            "dataset_id": config.dataset_id,
            "dataset_revision": config.dataset_revision,
            "split": "train",
            "instance_ids": train_ids,
        },
    )
    _atomic_json(
        validation_allowlist,
        {
            "schema_version": ALLOWLIST_SCHEMA_VERSION,
            "dataset_id": config.dataset_id,
            "dataset_revision": config.dataset_revision,
            "split": "train",
            "instance_ids": validation_ids_sorted,
        },
    )

    files: list[dict[str, Any]] = []
    source_path = output / "source-metadata.train.jsonl"
    files.extend(
        _write_jsonl_shards(
            source_rows,
            path=source_path,
            output_root=output,
            shard_rows=len(source_rows),
            max_file_bytes=config.max_file_bytes,
        )
    )
    files.extend(
        _write_jsonl_shards(
            candidate_tasks,
            path=output / "candidate-tasks" / "tasks.jsonl",
            output_root=output,
            shard_rows=config.shard_rows,
            max_file_bytes=config.max_file_bytes,
        )
    )
    files.extend(
        _write_jsonl_shards(
            candidate_orders,
            path=output / "candidate-work-orders" / "work-orders.jsonl",
            output_root=output,
            shard_rows=config.shard_rows * len(CHAIN_STAGES),
            max_file_bytes=config.max_file_bytes,
        )
    )
    for path, records in (
        (training_allowlist, len(train_ids)),
        (validation_allowlist, len(validation_ids_sorted)),
    ):
        files.append(_file_entry(path, output, records))

    source_gate = _source_gate_from_build(
        config,
        observed_sha=observed_sha,
        observed_rows=len(rows),
        schema_valid=set(SOURCE_COLUMNS).issubset(upstream_schema),
    )
    launch_gate = _gate_group(
        {
            "source_train_parquet": source_gate,
            "formal_route_contract": _formal_route_contract_gate(config),
            **_component_launch_gates(config),
        }
    )
    training_gate = _gate_group(_training_output_gates(config))
    credential_redaction_count = sum(
        str(row["problem_statement"]).count("[REDACTED_CREDENTIAL]")
        for row in source_rows
    )
    publication_check = _write_public_export(
        config,
        source_rows=source_rows,
        candidate_tasks=candidate_tasks,
        candidate_orders=candidate_orders,
        train_ids=train_ids,
        validation_ids=validation_ids_sorted,
        repository_count=len(repository_counts),
        locale_counts=locale_counts,
        credential_redaction_count=credential_redaction_count,
    )
    publication_gate = _gate_group({"audited_export": publication_check})
    gate_groups = {
        "launch": launch_gate,
        "training": training_gate,
        "publication": publication_gate,
    }
    manifest: dict[str, Any] = {
        "schema_version": FULL_BANK_MANIFEST_SCHEMA_VERSION,
        "source": {
            "dataset_id": config.dataset_id,
            "dataset_revision": config.dataset_revision,
            "split": "train",
            "parquet_sha256": observed_sha,
            "parquet_bytes": config.source_parquet.stat().st_size,
            "parquet_publishable": False,
            "row_count": len(rows),
            "unique_instance_count": len(set(ids)),
            "repository_count": len(repository_counts),
            "upstream_field_names": upstream_schema,
            "projected_fields": list(SOURCE_COLUMNS),
            "oracle_fields_projected": False,
            "credential_redaction_count": credential_redaction_count,
        },
        "derived_split": {
            "algorithm": config.split_algorithm,
            "validation_numerator": config.validation_numerator,
            "validation_denominator": config.validation_denominator,
            "train_count": len(train_ids),
            "validation_count": len(validation_ids_sorted),
            "train_ids_sha256": digest_string_set(train_ids),
            "validation_ids_sha256": digest_string_set(validation_ids_sorted),
            "partitions_disjoint": not bool(set(train_ids) & set(validation_ids_sorted)),
            "complete_coverage": len(train_ids) + len(validation_ids_sorted) == len(rows),
            "validation_origin": "official-train-only",
        },
        "bilingual": {
            "assignment": config.locale_algorithm,
            "counts": dict(sorted(locale_counts.items())),
            "zh_cn_rows_require_translation": locale_counts.get("zh-CN", 0),
            "translation_manifest_present": training_gate["checks"][
                "zh_cn_localization_manifest"
            ]["present"],
        },
        "routing": {
            "providers": {
                alias: profile.public_dict()
                for alias, profile in sorted(config.providers.items())
            },
            "provider_stage_counts": dict(sorted(provider_stage_counts.items())),
            "domain_counts": dict(sorted(domain_counts.items())),
            "strict_stage_order": list(CHAIN_STAGES),
            "work_order_count": len(candidate_orders),
            "work_orders_per_task": len(CHAIN_STAGES),
            "real_sandbox_builder": True,
            "tool_call_capture_required": config.capture_tool_calls,
            "tool_result_capture_required": config.capture_tool_results,
            "hidden_chain_of_thought_captured": config.capture_hidden_chain_of_thought,
            "public_reasoning_summary_captured": config.capture_public_reasoning_summary,
        },
        "repository_counts": dict(sorted(repository_counts.items())),
        "partition_counts": dict(sorted(partition_counts.items())),
        "publication": {
            "strict_max_file_bytes": config.max_file_bytes,
            "all_local_staging_files_below_limit": all(
                int(item["bytes"]) < config.max_file_bytes for item in files
            ),
            "raw_source_excluded": True,
            "audited_export_dir": config.audited_export_dir.relative_to(
                config.project_root
            ).as_posix(),
            "local_staging_files": sorted(
                files, key=lambda item: str(item["path"])
            ),
            "public_export_manifest": publication_check.get("manifest_path"),
            "public_file_count": publication_check.get("file_count", 0),
        },
        "gates": gate_groups,
        "missing_gates": {
            name: group["missing"] for name, group in gate_groups.items()
        },
        "unvalidated_gates": {
            name: group["invalid"] for name, group in gate_groups.items()
        },
        "launch_ready": launch_gate["ready"],
        "training_ready": training_gate["ready"],
        "publication_ready": publication_gate["ready"],
        "sandbox_results_claimed": training_gate["checks"][
            "real_tool_results_manifest"
        ]["validated"],
        "heldout_bodies_read": False,
        "launch_gate_requires_runtime_outputs": False,
        "config_sha256": file_sha256(config.config_path),
    }
    if not manifest["publication"]["all_local_staging_files_below_limit"]:
        raise SWEBenchValidationError("a local staging file reached 50 MiB")
    manifest_path = output / "manifest.json"
    _atomic_json(manifest_path, manifest)
    _atomic_json(config.public_manifest_path, manifest)
    return FullBankBuildResult(manifest_path=manifest_path, manifest=manifest)


def refresh_hash_only_manifest_from_public(
    config: FullBankConfig,
) -> FullBankBuildResult:
    """Refresh the hash-only snapshot from the audited public manifest.

    The public ``manifest.json`` is the sole authority for the publication
    inventory.  This path deliberately does not parse any JSONL payload or
    source parquet: payload bytes are only streamed through SHA-256 while file
    sizes are read from metadata.  Record cardinality is checked from the
    authoritative inventory against its top-level counts.
    """

    public_manifest_path = config.audited_export_dir / "manifest.json"
    if not public_manifest_path.is_file():
        raise SWEBenchValidationError("public bank manifest is missing")
    try:
        public_bytes = public_manifest_path.read_bytes()
        public_manifest = json.loads(public_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SWEBenchValidationError("public bank manifest is unreadable") from exc
    if not isinstance(public_manifest, Mapping):
        raise SWEBenchValidationError("public bank manifest is not an object")
    public_sha256 = sha256(public_bytes).hexdigest()
    if (
        public_manifest.get("schema_version")
        != PUBLICATION_MANIFEST_SCHEMA_VERSION
        or public_manifest.get("dataset_id") != config.dataset_id
        or public_manifest.get("dataset_revision") != config.dataset_revision
        or public_manifest.get("source_split") != "train"
        or public_manifest.get("train_only") is not True
        or public_manifest.get("source_parquet_sha256")
        != config.source_parquet_sha256
        or public_manifest.get("raw_source_included") is not False
        or public_manifest.get("publication_ready") is not True
    ):
        raise SWEBenchValidationError("public bank manifest binding is invalid")

    counts = public_manifest.get("counts")
    expected_work_orders = config.expected_rows * len(CHAIN_STAGES)
    required_count_names = (
        "tasks",
        "work_orders",
        "repositories",
        "derived_train",
        "derived_validation_from_train",
    )
    if not isinstance(counts, Mapping) or any(
        isinstance(counts.get(name), bool)
        or not isinstance(counts.get(name), int)
        or int(counts[name]) < 0
        for name in required_count_names
    ):
        raise SWEBenchValidationError("public bank manifest counts are invalid")
    if (
        counts["tasks"] != config.expected_rows
        or counts["work_orders"] != expected_work_orders
        or counts["derived_train"] + counts["derived_validation_from_train"]
        != config.expected_rows
    ):
        raise SWEBenchValidationError("public bank manifest counts are invalid")

    raw_inventory = public_manifest.get("files")
    if not isinstance(raw_inventory, list):
        raise SWEBenchValidationError("public bank payload inventory is missing")
    expected_shards = (
        (config.expected_rows + config.shard_rows - 1) // config.shard_rows
    )
    order_shard_rows = config.shard_rows * len(CHAIN_STAGES)
    expected_order_shards = (
        (expected_work_orders + order_shard_rows - 1) // order_shard_rows
    )
    expected_payload_count = expected_shards * 2 + expected_order_shards + 3
    if len(raw_inventory) != expected_payload_count:
        raise SWEBenchValidationError("public bank payload count is invalid")

    inventory: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    record_totals = {
        "source": 0,
        "tasks": 0,
        "work_orders": 0,
        "derived_train": 0,
        "derived_validation": 0,
        "attribution": 0,
    }
    category_files = Counter[str]()
    for raw_item in raw_inventory:
        if not isinstance(raw_item, Mapping) or set(raw_item) != {
            "path",
            "sha256",
            "bytes",
            "records",
        }:
            raise SWEBenchValidationError("public bank payload inventory item is invalid")
        relative = raw_item.get("path")
        digest = raw_item.get("sha256")
        byte_count = raw_item.get("bytes")
        record_count = raw_item.get("records")
        if (
            not isinstance(relative, str)
            or relative in seen_paths
            or relative == "manifest.json"
            or not _allowed_publication_path(relative)
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not isinstance(digest, str)
            or not _SHA256.fullmatch(digest)
            or isinstance(byte_count, bool)
            or not isinstance(byte_count, int)
            or byte_count < 0
            or byte_count >= config.max_file_bytes
            or isinstance(record_count, bool)
            or not isinstance(record_count, int)
            or record_count < 0
        ):
            raise SWEBenchValidationError("public bank payload inventory item is invalid")
        seen_paths.add(relative)
        payload = config.audited_export_dir / Path(relative)
        if payload.is_symlink() or not payload.is_file():
            raise SWEBenchValidationError("public bank payload is missing or unsafe")
        if payload.stat().st_size != byte_count or file_sha256(payload) != digest:
            raise SWEBenchValidationError("public bank payload binding mismatch")

        if relative.startswith("source-metadata.train"):
            category = "source"
        elif relative.startswith("candidate-tasks/"):
            category = "tasks"
        elif relative.startswith("candidate-work-orders/"):
            category = "work_orders"
        elif relative == "allowlists/train.json":
            category = "derived_train"
        elif relative == "allowlists/validation-from-train.json":
            category = "derived_validation"
        elif relative == "ATTRIBUTION.md":
            category = "attribution"
        else:  # pragma: no cover - guarded by the publication path allowlist
            raise SWEBenchValidationError("public bank payload path is invalid")
        record_totals[category] += record_count
        category_files[category] += 1
        inventory.append(
            {
                "path": relative,
                "sha256": digest,
                "bytes": byte_count,
                "records": record_count,
            }
        )

    actual_payloads = {
        path.relative_to(config.audited_export_dir).as_posix()
        for path in config.audited_export_dir.rglob("*")
        if path.is_file() and path != public_manifest_path
    }
    if seen_paths != actual_payloads:
        raise SWEBenchValidationError("public bank payload paths do not match inventory")
    if category_files != Counter(
        {
            "source": expected_shards,
            "tasks": expected_shards,
            "work_orders": expected_order_shards,
            "derived_train": 1,
            "derived_validation": 1,
            "attribution": 1,
        }
    ):
        raise SWEBenchValidationError("public bank payload shard counts are invalid")
    if record_totals != {
        "source": config.expected_rows,
        "tasks": config.expected_rows,
        "work_orders": expected_work_orders,
        "derived_train": counts["derived_train"],
        "derived_validation": counts["derived_validation_from_train"],
        "attribution": 0,
    }:
        raise SWEBenchValidationError("public bank payload record counts are invalid")
    if file_sha256(public_manifest_path) != public_sha256:
        raise SWEBenchValidationError("public bank manifest changed during refresh")

    if not config.public_manifest_path.is_file():
        raise SWEBenchValidationError("hash-only bank manifest is missing")
    try:
        hash_only = _load_json_object(config.public_manifest_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SWEBenchValidationError("hash-only bank manifest is unreadable") from exc
    source = hash_only.get("source")
    if (
        hash_only.get("schema_version") != FULL_BANK_MANIFEST_SCHEMA_VERSION
        or not isinstance(source, Mapping)
        or source.get("dataset_id") != config.dataset_id
        or source.get("dataset_revision") != config.dataset_revision
        or source.get("split") != "train"
        or source.get("row_count") != config.expected_rows
    ):
        raise SWEBenchValidationError("hash-only bank manifest binding is invalid")

    raw_publication = hash_only.get("publication")
    raw_gates = hash_only.get("gates")
    raw_missing_gates = hash_only.get("missing_gates")
    raw_unvalidated_gates = hash_only.get("unvalidated_gates")
    if not all(
        isinstance(value, Mapping)
        for value in (
            raw_publication,
            raw_gates,
            raw_missing_gates,
            raw_unvalidated_gates,
        )
    ):
        raise SWEBenchValidationError("hash-only bank manifest structure is invalid")
    publication = dict(raw_publication)
    publication.update(
        {
            "public_export_manifest": public_manifest_path.relative_to(
                config.project_root
            ).as_posix(),
            "public_manifest_sha256": public_sha256,
            "payload_file_count": len(inventory),
            "public_file_count": len(inventory) + 1,
            "payload_inventory": sorted(inventory, key=lambda item: item["path"]),
            "counts": dict(counts),
        }
    )
    hash_only["publication"] = publication
    publication_check = {
        "path": config.audited_export_dir.relative_to(config.project_root).as_posix(),
        "manifest_path": public_manifest_path.relative_to(
            config.project_root
        ).as_posix(),
        "present": True,
        "validated": True,
        "sha256": public_sha256,
        "file_count": len(inventory) + 1,
        "strict_max_file_bytes": config.max_file_bytes,
        "errors": [],
    }
    gates = dict(raw_gates)
    gates["publication"] = _gate_group({"audited_export": publication_check})
    hash_only["gates"] = gates
    hash_only["publication_ready"] = public_manifest["publication_ready"]
    missing_gates = dict(raw_missing_gates)
    missing_gates["publication"] = []
    hash_only["missing_gates"] = missing_gates
    unvalidated_gates = dict(raw_unvalidated_gates)
    unvalidated_gates["publication"] = []
    hash_only["unvalidated_gates"] = unvalidated_gates
    _atomic_json(config.public_manifest_path, hash_only)
    return FullBankBuildResult(
        manifest_path=config.public_manifest_path,
        manifest=hash_only,
    )


def preflight_full_bank(config: FullBankConfig) -> dict[str, Any]:
    """Run a read-only, offline readiness check.

    This function hashes local files and reads manifest metadata.  It never
    loads a credential, contacts a provider, starts OpenCode, or requires any
    output that the next distillation run is expected to produce.
    """

    source_gate = _inspect_source_gate(config)
    launch_gate = _gate_group(
        {
            "source_train_parquet": source_gate,
            "formal_route_contract": _formal_route_contract_gate(config),
            **_component_launch_gates(config),
        }
    )
    training_gate = _gate_group(_training_output_gates(config))
    publication_gate = _gate_group(
        {"audited_export": _audit_publication(config)}
    )
    groups = {
        "launch": launch_gate,
        "training": training_gate,
        "publication": publication_gate,
    }
    return {
        "schema_version": FULL_BANK_PREFLIGHT_SCHEMA_VERSION,
        "offline": True,
        "provider_requests": 0,
        "gpu_required": False,
        "source_bodies_printed": False,
        "gates": groups,
        "launch_ready": launch_gate["ready"],
        "training_ready": training_gate["ready"],
        "publication_ready": publication_gate["ready"],
        "launch_gate_requires_runtime_outputs": False,
    }


def _source_gate_from_build(
    config: FullBankConfig,
    *,
    observed_sha: str,
    observed_rows: int,
    schema_valid: bool,
) -> dict[str, Any]:
    errors: list[str] = []
    if observed_sha != config.source_parquet_sha256:
        errors.append("sha256_mismatch")
    if observed_rows != config.expected_rows:
        errors.append("row_count_mismatch")
    if not schema_valid:
        errors.append("projected_columns_missing")
    return {
        "path": _relative_project_path(config, config.source_parquet),
        "present": True,
        "validated": not errors,
        "sha256": observed_sha,
        "row_count": observed_rows,
        "source_split": "train",
        "errors": errors,
    }


def _inspect_source_gate(config: FullBankConfig) -> dict[str, Any]:
    path = config.source_parquet
    if not path.is_file():
        return {
            "path": _relative_project_path(config, path),
            "present": False,
            "validated": False,
            "sha256": None,
            "row_count": None,
            "source_split": "train",
            "errors": ["missing"],
        }
    errors: list[str] = []
    observed_sha = file_sha256(path)
    if observed_sha != config.source_parquet_sha256:
        errors.append("sha256_mismatch")
    observed_rows: int | None = None
    try:
        import pyarrow.parquet as pq

        parquet = pq.ParquetFile(path)
        observed_rows = parquet.metadata.num_rows
        names = {field.name for field in parquet.schema_arrow}
        if not set(SOURCE_COLUMNS).issubset(names):
            errors.append("projected_columns_missing")
    except (ImportError, OSError, ValueError):
        errors.append("parquet_metadata_unreadable")
    if observed_rows != config.expected_rows:
        errors.append("row_count_mismatch")
    return {
        "path": _relative_project_path(config, path),
        "present": True,
        "validated": not errors,
        "sha256": observed_sha,
        "row_count": observed_rows,
        "source_split": "train",
        "errors": sorted(set(errors)),
    }


def _formal_route_contract_gate(config: FullBankConfig) -> dict[str, Any]:
    referenced = {
        alias
        for route in config.stage_routes.values()
        for alias in (route.default_provider, route.frontend_provider)
        if alias is not None
    }
    errors: list[str] = []
    if not config.formal_enabled:
        errors.append("formal_profile_disabled")
    if len(config.formal_required_providers) < 2:
        errors.append("two_formal_providers_required")
    if not set(config.formal_required_providers).issubset(referenced):
        errors.append("formal_provider_not_routed")
    if any(config.providers[alias].reasoning_effort != "max" for alias in referenced):
        errors.append("reasoning_effort_not_max")
    if config.formal_required_effort != "max":
        errors.append("formal_effort_not_max")
    return {
        "path": None,
        "present": True,
        "validated": not errors,
        "sha256": None,
        "provider_count": len(referenced),
        "errors": errors,
    }


def _component_launch_gates(config: FullBankConfig) -> dict[str, dict[str, Any]]:
    launch_paths = config.gate_paths["launch"]
    return {
        "opencode_bundle_manifest": _validate_opencode_bundle(
            config, launch_paths["opencode_bundle_manifest"]
        ),
        "ccswitch_route_manifest": _validate_ccswitch_route(
            config, launch_paths["ccswitch_route_manifest"]
        ),
    }


def _validate_opencode_bundle(
    config: FullBankConfig, path: Path
) -> dict[str, Any]:
    status = _path_gate_base(config, path)
    if not path.is_file():
        status["errors"] = ["missing"]
        return status
    errors: list[str] = []
    try:
        value = _load_json_object(path)
        if value.get("schema_version") != OPEN_CODE_BUNDLE_SCHEMA_VERSION:
            errors.append("schema_mismatch")
        source = value.get("source")
        if not isinstance(source, Mapping):
            errors.append("source_contract_missing")
        else:
            contract = source.get("tool_contract")
            if (
                source.get("tool_contract_version")
                != EXECUTION_TOOL_CONTRACT_V3_VERSION
                or not isinstance(contract, Mapping)
                or contract.get("version") != EXECUTION_TOOL_CONTRACT_V3_VERSION
            ):
                errors.append("tool_contract_mismatch")
        platforms = value.get("platforms")
        if not isinstance(platforms, Mapping) or set(platforms) != {
            "windows-x64",
            "linux-x64",
        }:
            errors.append("platform_set_mismatch")
        else:
            for target in ("windows-x64", "linux-x64"):
                entry = platforms.get(target)
                if not isinstance(entry, Mapping):
                    errors.append(f"{target}_entry_invalid")
                    continue
                manifest_file = _relative_artifact_file(
                    path.parent, entry.get("manifest")
                )
                manifest_sha = entry.get("manifest_sha256")
                if manifest_file is None:
                    errors.append(f"{target}_manifest_missing")
                elif not isinstance(manifest_sha, str) or file_sha256(
                    manifest_file
                ) != manifest_sha:
                    errors.append(f"{target}_manifest_hash_mismatch")
                else:
                    platform_value = _load_json_object(manifest_file)
                    if (
                        platform_value.get("schema_version")
                        != OPEN_CODE_PLATFORM_SCHEMA_VERSION
                        or platform_value.get("target") != target
                    ):
                        errors.append(f"{target}_manifest_schema_mismatch")
                binary = entry.get("binary")
                if not isinstance(binary, Mapping):
                    errors.append(f"{target}_binary_invalid")
                    continue
                binary_file = _relative_artifact_file(
                    path.parent, binary.get("path")
                )
                binary_sha = binary.get("sha256")
                if binary_file is None:
                    errors.append(f"{target}_binary_missing")
                elif not isinstance(binary_sha, str) or file_sha256(
                    binary_file
                ) != binary_sha:
                    errors.append(f"{target}_binary_hash_mismatch")
    except (OSError, ValueError, json.JSONDecodeError):
        errors.append("manifest_unreadable")
    status.update(
        {
            "present": True,
            "validated": not errors,
            "sha256": file_sha256(path),
            "errors": sorted(set(errors)),
        }
    )
    return status


def _validate_ccswitch_route(
    config: FullBankConfig, path: Path
) -> dict[str, Any]:
    status = _path_gate_base(config, path)
    if not path.is_file():
        status["errors"] = ["missing"]
        return status
    errors: list[str] = []
    try:
        value = _load_json_object(path)
        if value.get("schema_version") != CCSWITCH_ROUTE_SCHEMA_VERSION:
            errors.append("schema_mismatch")
        if value.get("ready") is not True:
            errors.append("route_not_ready")
        if value.get("secret_persisted") is not False:
            errors.append("secret_persistence_not_disabled")
        route = value.get("route")
        if (
            not isinstance(route, Mapping)
            or route.get("app_type") != "anchor-opencode"
            or not str(route.get("base_url", "")).startswith("http://127.0.0.1:")
            or route.get("content_free_health_status") is not True
        ):
            errors.append("route_contract_mismatch")
        patch = value.get("patch")
        if not isinstance(patch, Mapping):
            errors.append("patch_attestation_missing")
        else:
            patch_file = _project_artifact_file(config, patch.get("path"))
            if patch_file is None:
                errors.append("patch_file_missing")
            elif file_sha256(patch_file) != patch.get("sha256"):
                errors.append("patch_hash_mismatch")
        binary = value.get("binary")
        if not isinstance(binary, Mapping):
            errors.append("binary_attestation_missing")
        else:
            binary_file = _project_artifact_file(config, binary.get("path"))
            if binary_file is None:
                errors.append("binary_missing")
            elif file_sha256(binary_file) != binary.get("sha256"):
                errors.append("binary_hash_mismatch")
        verified = value.get("verified_tests")
        if (
            not isinstance(verified, list)
            or not verified
            or any(
                not isinstance(item, Mapping) or item.get("status") != "passed"
                for item in verified
            )
        ):
            errors.append("component_tests_incomplete")
        if _PUBLIC_SECRET.search(json.dumps(value, ensure_ascii=False)):
            errors.append("credential_pattern_detected")
    except (OSError, ValueError, json.JSONDecodeError):
        errors.append("manifest_unreadable")
    status.update(
        {
            "present": True,
            "validated": not errors,
            "sha256": file_sha256(path),
            "errors": sorted(set(errors)),
        }
    )
    return status


def _training_output_gates(
    config: FullBankConfig,
) -> dict[str, dict[str, Any]]:
    return {
        name: _validate_training_gate(config, name, path)
        for name, path in sorted(config.gate_paths["training"].items())
    }


def _validate_training_gate(
    config: FullBankConfig, name: str, path: Path
) -> dict[str, Any]:
    status = _path_gate_base(config, path)
    if not path.is_file():
        status["errors"] = ["missing"]
        return status
    errors: list[str] = []
    record_count: int | None = None
    try:
        value = _load_json_object(path)
        if value.get("schema_version") != TRAINING_GATE_SCHEMAS[name]:
            errors.append("schema_mismatch")
        source = value.get("source")
        if not isinstance(source, Mapping) or dict(source) != {
            "dataset_id": config.dataset_id,
            "dataset_revision": config.dataset_revision,
            "split": "train",
            "parquet_sha256": config.source_parquet_sha256,
        }:
            errors.append("source_binding_mismatch")
        if value.get("complete") is not True:
            errors.append("not_complete")
        if value.get("train_only") is not True:
            errors.append("not_train_only")
        if value.get("contains_heldout") is not False:
            errors.append("heldout_exclusion_missing")
        if name == "gold_manifest" and value.get("gold_records") is not True:
            errors.append("gold_attestation_missing")
        if (
            name == "zh_cn_localization_manifest"
            and value.get("locale") != "zh-CN"
        ):
            errors.append("locale_mismatch")
        if (
            name == "real_tool_results_manifest"
            and value.get("real_tool_results") is not True
        ):
            errors.append("real_tool_results_attestation_missing")
        record_count = value.get("record_count")
        if (
            isinstance(record_count, bool)
            or not isinstance(record_count, int)
            or record_count < 1
        ):
            errors.append("record_count_invalid")
            record_count = None
        files = value.get("files")
        if not isinstance(files, list) or not files:
            errors.append("file_inventory_missing")
        else:
            seen: set[str] = set()
            for item in files:
                if not isinstance(item, Mapping):
                    errors.append("file_inventory_invalid")
                    continue
                raw_path = item.get("path")
                artifact = _project_artifact_file(config, raw_path)
                if not isinstance(raw_path, str) or raw_path in seen:
                    errors.append("file_inventory_invalid")
                    continue
                seen.add(raw_path)
                if artifact is None:
                    errors.append("referenced_file_missing")
                elif file_sha256(artifact) != item.get("sha256"):
                    errors.append("referenced_file_hash_mismatch")
                records = item.get("records")
                if (
                    isinstance(records, bool)
                    or not isinstance(records, int)
                    or records < 1
                ):
                    errors.append("referenced_file_record_count_invalid")
    except (OSError, ValueError, json.JSONDecodeError):
        errors.append("manifest_unreadable")
    status.update(
        {
            "present": True,
            "validated": not errors,
            "sha256": file_sha256(path),
            "record_count": record_count,
            "errors": sorted(set(errors)),
        }
    )
    return status


def _write_public_export(
    config: FullBankConfig,
    *,
    source_rows: Sequence[Mapping[str, Any]],
    candidate_tasks: Sequence[Mapping[str, Any]],
    candidate_orders: Sequence[Mapping[str, Any]],
    train_ids: Sequence[str],
    validation_ids: Sequence[str],
    repository_count: int,
    locale_counts: Mapping[str, int],
    credential_redaction_count: int,
) -> dict[str, Any]:
    root = config.audited_export_dir
    root.mkdir(parents=True, exist_ok=True)
    _prune_generated_publication_files(root)

    files: list[dict[str, Any]] = []
    files.extend(
        _write_jsonl_shards(
            source_rows,
            path=root / "source-metadata.train.jsonl",
            output_root=root,
            shard_rows=config.shard_rows,
            max_file_bytes=config.max_file_bytes,
        )
    )
    files.extend(
        _write_jsonl_shards(
            candidate_tasks,
            path=root / "candidate-tasks" / "tasks.jsonl",
            output_root=root,
            shard_rows=config.shard_rows,
            max_file_bytes=config.max_file_bytes,
        )
    )
    files.extend(
        _write_jsonl_shards(
            candidate_orders,
            path=root / "candidate-work-orders" / "work-orders.jsonl",
            output_root=root,
            shard_rows=config.shard_rows * len(CHAIN_STAGES),
            max_file_bytes=config.max_file_bytes,
        )
    )

    allowlist_dir = root / "allowlists"
    train_allowlist = allowlist_dir / "train.json"
    validation_allowlist = allowlist_dir / "validation-from-train.json"
    _atomic_json(
        train_allowlist,
        {
            "schema_version": ALLOWLIST_SCHEMA_VERSION,
            "dataset_id": config.dataset_id,
            "dataset_revision": config.dataset_revision,
            "split": "train",
            "instance_ids": list(train_ids),
        },
    )
    _atomic_json(
        validation_allowlist,
        {
            "schema_version": ALLOWLIST_SCHEMA_VERSION,
            "dataset_id": config.dataset_id,
            "dataset_revision": config.dataset_revision,
            "split": "train",
            "instance_ids": list(validation_ids),
        },
    )
    files.extend(
        [
            _file_entry(train_allowlist, root, len(train_ids)),
            _file_entry(validation_allowlist, root, len(validation_ids)),
        ]
    )

    attribution = config.gate_paths["publication"]["mit_attribution_file"]
    _atomic_text(attribution, _attribution_text(config))
    files.append(_file_entry(attribution, root, 0))
    files.sort(key=lambda item: str(item["path"]))

    public_manifest_path = root / "manifest.json"
    public_manifest: dict[str, Any] = {
        "schema_version": PUBLICATION_MANIFEST_SCHEMA_VERSION,
        "dataset_id": config.dataset_id,
        "dataset_revision": config.dataset_revision,
        "source_split": "train",
        "train_only": True,
        "source_parquet_sha256": config.source_parquet_sha256,
        "raw_source_included": False,
        "source_fields": list(SOURCE_COLUMNS),
        "counts": {
            "tasks": len(source_rows),
            "work_orders": len(candidate_orders),
            "repositories": repository_count,
            "derived_train": len(train_ids),
            "derived_validation_from_train": len(validation_ids),
            "locales": dict(sorted(locale_counts.items())),
        },
        "attribution": {
            "path": "ATTRIBUTION.md",
            "upstream_project": "SWE-bench/SWE-bench",
            "upstream_repository": "https://github.com/SWE-bench/SWE-bench",
            "upstream_license": "MIT",
        },
        "files": files,
        "safety": {
            "structured_field_scan_passed": True,
            "credential_scan_passed": True,
            "credential_redaction_count": credential_redaction_count,
            "source_scope_scan_passed": True,
            "strict_size_limit_bytes": config.max_file_bytes,
        },
        "publication_ready": True,
    }
    _atomic_json(public_manifest_path, public_manifest)
    status = _audit_publication(config)
    if not status["validated"]:
        public_manifest["publication_ready"] = False
        public_manifest["audit_errors"] = list(status["errors"])
        _atomic_json(public_manifest_path, public_manifest)
        status = _audit_publication(config)
    return status


def _audit_publication(config: FullBankConfig) -> dict[str, Any]:
    root = config.audited_export_dir
    manifest_path = root / "manifest.json"
    status = _path_gate_base(config, root)
    status["manifest_path"] = _relative_project_path(config, manifest_path)
    if not root.is_dir():
        status["errors"] = ["missing"]
        return status
    errors: list[str] = []
    actual_files = sorted(
        path for path in root.rglob("*") if path.is_file() or path.is_symlink()
    )
    record_counts: dict[str, int] = {}
    for path in actual_files:
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            errors.append(f"symlink_not_allowed:{relative}")
            continue
        if not _allowed_publication_path(relative):
            errors.append(f"unexpected_file:{relative}")
            continue
        if path.stat().st_size >= config.max_file_bytes:
            errors.append(f"file_size_limit:{relative}")
        if relative == "ATTRIBUTION.md":
            try:
                text = path.read_text(encoding="utf-8")
            except (OSError, UnicodeError):
                errors.append("attribution_unreadable")
                continue
            if text != _attribution_text(config):
                errors.append("attribution_mismatch")
            if _PUBLIC_SECRET.search(text):
                errors.append("credential_pattern:ATTRIBUTION.md")
            record_counts[relative] = 0
            continue
        try:
            if path.suffix == ".jsonl":
                # Split only on the JSONL delimiter. str.splitlines() also
                # treats U+2028/U+2029 inside valid JSON strings as delimiters.
                values = [
                    json.loads(line)
                    for line in path.read_text(encoding="utf-8").split("\n")
                    if line
                ]
            else:
                values = [json.loads(path.read_text(encoding="utf-8"))]
        except (OSError, UnicodeError, json.JSONDecodeError):
            errors.append(f"json_unreadable:{relative}")
            continue
        if relative.startswith("source-metadata.train"):
            if any(
                not isinstance(value, Mapping)
                or set(value) != set(SOURCE_COLUMNS)
                for value in values
            ):
                errors.append(f"source_projection_mismatch:{relative}")
        elif relative.startswith("candidate-tasks/"):
            if any(
                not isinstance(value, Mapping)
                or value.get("schema_version") != CANDIDATE_TASK_SCHEMA_VERSION
                or not isinstance(value.get("source"), Mapping)
                or value["source"].get("split") != "train"
                for value in values
            ):
                errors.append(f"candidate_task_binding_mismatch:{relative}")
        elif relative.startswith("candidate-work-orders/"):
            if any(
                not isinstance(value, Mapping)
                or value.get("schema_version") != CANDIDATE_WORK_ORDER_SCHEMA_VERSION
                for value in values
            ):
                errors.append(f"work_order_schema_mismatch:{relative}")
        for value in values:
            _scan_public_value(value, relative, errors)
        if path.suffix == ".jsonl":
            record_counts[relative] = len(values)
        elif relative.startswith("allowlists/") and values:
            allowlist = values[0]
            if (
                not isinstance(allowlist, Mapping)
                or allowlist.get("dataset_id") != config.dataset_id
                or allowlist.get("dataset_revision") != config.dataset_revision
                or allowlist.get("split") != "train"
                or not isinstance(allowlist.get("instance_ids"), list)
            ):
                errors.append(f"allowlist_binding_mismatch:{relative}")
                record_counts[relative] = 0
            else:
                record_counts[relative] = len(allowlist["instance_ids"])
        else:
            record_counts[relative] = 1

    manifest: dict[str, Any] | None = None
    if not manifest_path.is_file():
        errors.append("publication_manifest_missing")
    else:
        try:
            manifest = _load_json_object(manifest_path)
        except (OSError, ValueError, json.JSONDecodeError):
            errors.append("publication_manifest_unreadable")
        if manifest is not None:
            if (
                manifest.get("schema_version")
                != PUBLICATION_MANIFEST_SCHEMA_VERSION
                or manifest.get("dataset_id") != config.dataset_id
                or manifest.get("dataset_revision") != config.dataset_revision
                or manifest.get("source_split") != "train"
                or manifest.get("train_only") is not True
                or manifest.get("source_parquet_sha256")
                != config.source_parquet_sha256
                or manifest.get("raw_source_included") is not False
                or manifest.get("publication_ready") is not True
            ):
                errors.append("publication_manifest_binding_mismatch")
            inventory = manifest.get("files")
            reported: dict[str, Mapping[str, Any]] = {}
            if not isinstance(inventory, list):
                errors.append("publication_inventory_missing")
            else:
                for item in inventory:
                    if (
                        not isinstance(item, Mapping)
                        or not isinstance(item.get("path"), str)
                        or item["path"] in reported
                    ):
                        errors.append("publication_inventory_invalid")
                        continue
                    reported[str(item["path"])] = item
                payload_paths = {
                    path.relative_to(root).as_posix()
                    for path in actual_files
                    if path.is_file()
                    and path.relative_to(root).as_posix() != "manifest.json"
                }
                if set(reported) != payload_paths:
                    errors.append("publication_inventory_path_mismatch")
                for relative, item in reported.items():
                    payload = root / Path(relative)
                    if not payload.is_file():
                        continue
                    if (
                        item.get("sha256") != file_sha256(payload)
                        or item.get("bytes") != payload.stat().st_size
                        or item.get("records") != record_counts.get(relative)
                    ):
                        errors.append(f"publication_inventory_mismatch:{relative}")

    status.update(
        {
            "present": True,
            "validated": not errors,
            "sha256": file_sha256(manifest_path) if manifest_path.is_file() else None,
            "file_count": len(actual_files),
            "strict_max_file_bytes": config.max_file_bytes,
            "errors": sorted(set(errors)),
        }
    )
    return status


def _scan_public_value(value: Any, relative: str, errors: list[str]) -> None:
    if isinstance(value, Mapping):
        for raw_key, nested in value.items():
            key = str(raw_key).casefold().replace("-", "_")
            if key in _FORBIDDEN_PUBLIC_KEYS:
                errors.append(f"forbidden_field:{relative}")
            if key == "split" and nested != "train":
                errors.append(f"non_train_split:{relative}")
            _scan_public_value(nested, relative, errors)
        return
    if isinstance(value, list):
        for nested in value:
            _scan_public_value(nested, relative, errors)
        return
    if isinstance(value, str):
        if _PUBLIC_SECRET.search(value):
            errors.append(f"credential_pattern:{relative}")
        folded = value.casefold()
        if any(
            marker in folded
            for marker in (
                "swe-bench/swe-bench_lite",
                "swe-bench/swe-bench_verified",
            )
        ):
            errors.append(f"non_train_dataset_reference:{relative}")


def _attribution_text(config: FullBankConfig) -> str:
    return (
        "# SWE-bench attribution\n\n"
        "This train-only public metadata projection is derived from "
        "`SWE-bench/SWE-bench` at immutable revision "
        f"`{config.dataset_revision}`.\n\n"
        "Upstream project: https://github.com/SWE-bench/SWE-bench  \n"
        "Upstream software license: MIT (`SPDX-License-Identifier: MIT`).\n\n"
        "The export contains only public issue metadata and deterministic "
        "routing records. The original parquet and runtime outputs are not "
        "redistributed. The MIT notice identifies the SWE-bench software "
        "source; it does not relicense issue text or the source repositories, "
        "whose applicable terms remain in force.\n"
    )


def _prune_generated_publication_files(root: Path) -> None:
    if not root.is_dir():
        return
    for path in sorted(root.rglob("*"), reverse=True):
        if path.is_file() and _allowed_publication_path(
            path.relative_to(root).as_posix()
        ):
            path.unlink()
        elif path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass


def _allowed_publication_path(relative: str) -> bool:
    if relative in {
        "ATTRIBUTION.md",
        "manifest.json",
        "allowlists/train.json",
        "allowlists/validation-from-train.json",
    }:
        return True
    return bool(
        re.fullmatch(
            r"source-metadata\.train(?:-\d{5}-of-\d{5})?\.jsonl",
            relative,
        )
        or re.fullmatch(
            r"candidate-tasks/tasks(?:-\d{5}-of-\d{5})?\.jsonl",
            relative,
        )
        or re.fullmatch(
            r"candidate-work-orders/work-orders(?:-\d{5}-of-\d{5})?\.jsonl",
            relative,
        )
    )


def _gate_group(checks: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    normalized = {name: dict(value) for name, value in sorted(checks.items())}
    missing = sorted(
        name for name, value in normalized.items() if not value.get("present")
    )
    invalid = sorted(
        name
        for name, value in normalized.items()
        if value.get("present") and not value.get("validated")
    )
    return {
        "ready": not missing and not invalid,
        "missing": missing,
        "invalid": invalid,
        "checks": normalized,
    }


def _path_gate_base(config: FullBankConfig, path: Path) -> dict[str, Any]:
    return {
        "path": _relative_project_path(config, path),
        "present": path.exists(),
        "validated": False,
        "sha256": None,
        "errors": [],
    }


def _relative_project_path(config: FullBankConfig, path: Path) -> str:
    return path.resolve().relative_to(config.project_root).as_posix()


def _relative_artifact_file(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        return None
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError:
        return None
    return candidate if candidate.is_file() else None


def _project_artifact_file(config: FullBankConfig, value: Any) -> Path | None:
    return _relative_artifact_file(config.project_root, value)


def _load_json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError("manifest must be a JSON object")
    return value


def _read_projected_train_rows(
    config: FullBankConfig,
) -> tuple[list[dict[str, str]], list[str]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SWEBenchValidationError(
            "full-bank build requires pyarrow; install the project training extra"
        ) from exc

    parquet = pq.ParquetFile(config.source_parquet)
    schema_names = [field.name for field in parquet.schema_arrow]
    if not set(SOURCE_COLUMNS).issubset(schema_names):
        raise SWEBenchValidationError("train parquet lacks required projected fields")
    # Column projection is intentional: patch, test_patch, hints, and oracle test
    # labels never enter Python objects or derived output files.
    table = pq.read_table(config.source_parquet, columns=list(SOURCE_COLUMNS))
    rows: list[dict[str, str]] = []
    for raw in table.to_pylist():
        instance_id = validate_instance_id(_text(raw.get("instance_id"), "instance_id"))
        repo = validate_repository(_text(raw.get("repo"), "repo"))
        base_commit = _text(raw.get("base_commit"), "base_commit")
        if not re.fullmatch(r"[0-9a-fA-F]{40}", base_commit):
            raise SWEBenchValidationError("base_commit must be a 40-character hash")
        raw_problem = raw.get("problem_statement")
        if not isinstance(raw_problem, str):
            raise SWEBenchValidationError("problem_statement must be a string")
        problem_statement = _PUBLIC_SECRET.sub(
            "[REDACTED_CREDENTIAL]", clean_problem_statement(raw_problem)
        )
        rows.append(
            {
                "instance_id": instance_id,
                "repo": repo,
                "base_commit": base_commit.lower(),
                "problem_statement": problem_statement,
            }
        )
    return rows, schema_names


def _validation_ids(
    rows: Sequence[Mapping[str, str]], *, numerator: int, denominator: int
) -> set[str]:
    by_repo: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for row in rows:
        repo = row["repo"]
        instance_id = row["instance_id"]
        rank = sha256(f"{repo}\0{instance_id}".encode("utf-8")).hexdigest()
        by_repo[repo].append((rank, instance_id))
    selected: set[str] = set()
    for items in by_repo.values():
        items.sort()
        count = len(items)
        quota = (count * numerator + denominator // 2) // denominator
        quota = max(1, min(count - 1, quota)) if count > 1 else 0
        selected.update(instance_id for _, instance_id in items[:quota])
    return selected


def _locale_assignments(
    rows: Sequence[Mapping[str, str]], locales: Sequence[str]
) -> dict[str, str]:
    ranked = sorted(
        (
            sha256(
                f"locale\0{row['repo']}\0{row['instance_id']}".encode("utf-8")
            ).hexdigest(),
            row["instance_id"],
        )
        for row in rows
    )
    return {
        instance_id: locales[index % len(locales)]
        for index, (_, instance_id) in enumerate(ranked)
    }


def _domain_label(row: Mapping[str, str], keywords: Sequence[str]) -> str:
    haystack = f"{row['repo']}\n{row['problem_statement']}".casefold()
    tokens = set(re.findall(r"[a-z0-9]+", haystack))
    matched = any(
        (keyword in haystack if " " in keyword else keyword in tokens)
        for keyword in keywords
    )
    return "frontend" if matched else "general"


def _providers_for_task(config: FullBankConfig, domain_label: str) -> dict[str, str]:
    selected: dict[str, str] = {}
    for stage in CHAIN_STAGES:
        route = config.stage_routes[stage]
        selected[stage] = (
            route.frontend_provider
            if domain_label == "frontend" and route.frontend_provider is not None
            else route.default_provider
        )
    return selected


def _task_id(config: FullBankConfig, row: Mapping[str, str]) -> str:
    digest = sha256(
        canonical_json(
            {
                "dataset_id": config.dataset_id,
                "dataset_revision": config.dataset_revision,
                "instance_id": row["instance_id"],
                "repo": row["repo"],
                "base_commit": row["base_commit"],
            }
        ).encode("utf-8")
    ).hexdigest()
    return f"swe-full-v1:{digest}"


def _work_orders(
    task: Mapping[str, Any], config: FullBankConfig
) -> list[dict[str, Any]]:
    task_id = str(task["task_id"])
    routing = task["routing_contract"]
    providers = routing["providers_by_stage"]
    previous = task_id
    orders: list[dict[str, Any]] = []
    for stage in CHAIN_STAGES:
        record_id = "swe-full-stage-v1:" + sha256(
            canonical_json(
                {"task_id": task_id, "stage": stage, "upstream": previous}
            ).encode("utf-8")
        ).hexdigest()
        orders.append(
            {
                "schema_version": CANDIDATE_WORK_ORDER_SCHEMA_VERSION,
                "record_id": record_id,
                "task_id": task_id,
                "stage": stage,
                "upstream_record_ids": [previous],
                "status": "pending_distillation",
                "provider_alias": providers[stage],
                "reasoning_effort": config.formal_required_effort,
                "execution": config.stage_routes[stage].execution,
                "required_output_schema": _OUTPUT_SCHEMAS[stage],
                "input_contract": _input_contract(stage),
                "review_revision_edge": (
                    "REVISE->domain_builder" if stage == "domain_review" else None
                ),
            }
        )
        previous = record_id
    return orders


def _input_contract(stage: str) -> list[str]:
    return {
        "planner": [
            "problem_statement",
            "controlled_workspace_inventory",
            "selected_builder_contract",
        ],
        "tool_policy": ["planner_json", "proposed_tool_calls"],
        "domain_builder": [
            "approved_plan",
            "approved_tool_policy",
            "disposable_sandbox",
        ],
        "domain_review": [
            "builder_diff",
            "real_tool_calls",
            "real_tool_results",
            "build_test_evidence",
        ],
        "security": [
            "reviewed_diff",
            "revision_history",
            "build_test_evidence",
        ],
    }[stage]


def _write_jsonl_shards(
    rows: Iterable[Mapping[str, Any]],
    *,
    path: Path,
    output_root: Path,
    shard_rows: int,
    max_file_bytes: int,
) -> list[dict[str, Any]]:
    encoded = [(canonical_json(dict(row)) + "\n").encode("utf-8") for row in rows]
    if any(len(line) >= max_file_bytes for line in encoded):
        raise SWEBenchValidationError("one derived JSONL row reaches 50 MiB")
    path.parent.mkdir(parents=True, exist_ok=True)
    _prune_shard_family(path)
    groups: list[list[bytes]] = []
    current: list[bytes] = []
    current_bytes = 0
    for line in encoded:
        if current and (
            len(current) >= shard_rows or current_bytes + len(line) >= max_file_bytes
        ):
            groups.append(current)
            current = []
            current_bytes = 0
        current.append(line)
        current_bytes += len(line)
    if current:
        groups.append(current)
    entries: list[dict[str, Any]] = []
    for index, group in enumerate(groups):
        shard = (
            path
            if len(groups) == 1
            else path.with_name(f"{path.stem}-{index:05d}-of-{len(groups):05d}{path.suffix}")
        )
        temporary = shard.with_name(shard.name + ".tmp")
        try:
            with temporary.open("wb") as handle:
                for line in group:
                    handle.write(line)
            temporary.replace(shard)
        finally:
            temporary.unlink(missing_ok=True)
        if shard.stat().st_size >= max_file_bytes:
            raise SWEBenchValidationError("derived shard is not below 50 MiB")
        entries.append(_file_entry(shard, output_root, len(group)))
    return entries


def _file_entry(path: Path, output_root: Path, records: int) -> dict[str, Any]:
    return {
        "path": path.relative_to(output_root).as_posix(),
        "records": records,
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value, encoding="utf-8", newline="\n")
    temporary.replace(path)


def _prune_shard_family(path: Path) -> None:
    pattern = re.compile(
        rf"^{re.escape(path.stem)}(?:-\d{{5}}-of-\d{{5}})?{re.escape(path.suffix)}$"
    )
    for candidate in path.parent.iterdir():
        if candidate.is_file() and pattern.fullmatch(candidate.name):
            candidate.unlink()


def _exact_mapping(
    value: Any, label: str, fields: set[str]
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != fields:
        raise SWEBenchValidationError(f"{label} has unexpected fields")
    return value


def _project_path(root: Path, value: Any, label: str) -> Path:
    candidate = (root / _text(value, label)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise SWEBenchValidationError(f"{label} must stay inside project root") from exc
    return candidate


def _text(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise SWEBenchValidationError(f"{label} must be a non-empty trimmed string")
    return value


def _safe_alias(value: Any, label: str) -> str:
    candidate = _text(value, label)
    if not _SAFE_ALIAS.fullmatch(candidate):
        raise SWEBenchValidationError(f"{label} is not a safe alias")
    return candidate


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise SWEBenchValidationError(f"{label} must be a positive integer")
    return value


def _bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise SWEBenchValidationError(f"{label} must be boolean")
    return value


__all__ = [
    "CANDIDATE_TASK_SCHEMA_VERSION",
    "CANDIDATE_WORK_ORDER_SCHEMA_VERSION",
    "FIFTY_MIB",
    "FULL_BANK_CONFIG_SCHEMA_VERSION",
    "FULL_BANK_MANIFEST_SCHEMA_VERSION",
    "FULL_BANK_PREFLIGHT_SCHEMA_VERSION",
    "PUBLICATION_MANIFEST_SCHEMA_VERSION",
    "FullBankBuildResult",
    "FullBankConfig",
    "ProviderProfile",
    "StageRoute",
    "build_full_bank",
    "preflight_full_bank",
]
