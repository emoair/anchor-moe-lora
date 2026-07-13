from __future__ import annotations

from difflib import unified_diff
from pathlib import Path

from .config import DEFAULT_PROVIDER, OpenCodeProvider, write_opencode_config
from .models import (
    FileChange,
    GoldRecord,
    SampleSpec,
    ToolTraceEntry,
    sample_contract_sha256,
)
from .policy import ToolPolicy
from .runner import AgentExecutor
from .validation import run_validations_with_output
from .workspace import WorkspaceManager, diff_snapshots, snapshot_files


def _capture_final_diff(
    source: Path, workspace: Path, changes: tuple[FileChange, ...]
) -> tuple[dict[str, object], ...]:
    """Render the harness-observed file changes for controlled session capture."""

    result: list[dict[str, object]] = []
    for change in changes:
        before_path = source / change.path
        after_path = workspace / change.path
        before_bytes = (
            before_path.read_bytes() if change.before_sha256 is not None else b""
        )
        after_bytes = (
            after_path.read_bytes() if change.after_sha256 is not None else b""
        )
        try:
            before_text = before_bytes.decode("utf-8")
            after_text = after_bytes.decode("utf-8")
        except UnicodeDecodeError:
            patch = f"Binary files a/{change.path} and b/{change.path} differ"
            additions = deletions = 0
        else:
            fromfile = (
                "/dev/null" if change.operation == "added" else f"a/{change.path}"
            )
            tofile = (
                "/dev/null" if change.operation == "deleted" else f"b/{change.path}"
            )
            raw_patch_lines = unified_diff(
                before_text.splitlines(keepends=True),
                after_text.splitlines(keepends=True),
                fromfile=fromfile,
                tofile=tofile,
            )
            patch_lines: list[str] = []
            for raw_line in raw_patch_lines:
                has_line_ending = raw_line.endswith(("\n", "\r"))
                line = raw_line.rstrip("\r\n")
                patch_lines.append(line)
                if (
                    not has_line_ending
                    and line.startswith(("+", "-", " "))
                    and not line.startswith(("+++", "---"))
                ):
                    patch_lines.append(r"\ No newline at end of file")
            if not patch_lines:
                # Empty-file creation/deletion and EOF-newline-only changes still need
                # explicit evidence instead of falling through as final_diff_missing.
                patch_lines = [f"--- {fromfile}", f"+++ {tofile}"]
            patch = "\n".join(patch_lines)
            additions = sum(
                line.startswith("+") and not line.startswith("+++")
                for line in patch_lines
            )
            deletions = sum(
                line.startswith("-") and not line.startswith("---")
                for line in patch_lines
            )
        result.append(
            {
                "file": change.path,
                "patch": patch,
                "additions": additions,
                "deletions": deletions,
                "status": change.operation,
            }
        )
    return tuple(result)


class ToolingHarness:
    def __init__(
        self,
        workspace_root: str | Path,
        executor: AgentExecutor,
        *,
        policy: ToolPolicy | None = None,
        retain_workspace: bool = False,
    ) -> None:
        self.workspaces = WorkspaceManager(workspace_root)
        self.executor = executor
        self.policy = policy or ToolPolicy()
        self.retain_workspace = retain_workspace

    def _provider(self) -> OpenCodeProvider:
        provider = getattr(self.executor, "provider", DEFAULT_PROVIDER)
        if not isinstance(provider, OpenCodeProvider):
            raise ValueError("executor provider is not an audited OpenCodeProvider")
        return provider

    def run_sample(self, sample: SampleSpec) -> GoldRecord:
        workspace = self.workspaces.prepare(sample.sample_id, sample.source_dir)
        try:
            config_path = write_opencode_config(
                workspace / ".anchor" / "opencode.json",
                self.policy,
                provider=self._provider(),
            )
            before = snapshot_files(workspace)
            expected_protected = dict(sample.protected_files)
            expected_contract = dict(sample.protected_files + sample.input_files)
            preflight_contract_errors = (
                ("fixture_contract_hash_mismatch",)
                if any(
                    before.get(path) != digest
                    for path, digest in expected_contract.items()
                )
                else ()
            )
            execution = self.executor.run(
                sample_id=sample.sample_id,
                prompt=sample.prompt,
                workspace=workspace,
                config_path=config_path,
                policy=self.policy,
            )
            after_agent = snapshot_files(workspace)
            changes = diff_snapshots(before, after_agent)
            final_diff = _capture_final_diff(
                sample.source_dir.resolve(), workspace, changes
            )
            protected_change_errors = (
                ("protected_fixture_modified",)
                if any(change.path in expected_protected for change in changes)
                else ()
            )
            try:
                validations, validation_trace, validation_capture = (
                    run_validations_with_output(workspace, self.policy)
                )
                validation_errors: tuple[str, ...] = ()
            except ValueError:
                validations = ()
                validation_trace = ()
                validation_capture = ()
                validation_errors = ("invalid_package_manifest",)

            capture_errors: tuple[str, ...] = ()
            finalize_capture = getattr(self.executor, "finalize_capture", None)
            if callable(finalize_capture):
                captured, capture_code = finalize_capture(
                    execution=execution,
                    sample_id=sample.sample_id,
                    workspace=workspace,
                    validators=validation_capture,
                    final_diff=final_diff,
                    skill_provenance=sample.skill_provenance,
                )
                if not captured:
                    capture_errors = (
                        capture_code or "controlled_session_capture_failed",
                    )

            offset = len(execution.trace)
            resequenced_validation_trace = tuple(
                ToolTraceEntry(
                    sequence=offset + index,
                    source=item.source,
                    tool=item.tool,
                    status=item.status,
                    command=item.command,
                    command_sha256=item.command_sha256,
                    exit_code=item.exit_code,
                    duration_ms=item.duration_ms,
                    output_sha256=item.output_sha256,
                )
                for index, item in enumerate(validation_trace, 1)
            )
            trace = execution.trace + resequenced_validation_trace
            by_name = {item.name: item for item in validations}
            required_passed = all(
                name in by_name
                and by_name[name].script_present
                and by_name[name].status == "PASS"
                for name in sample.required_validations
            )
            present_validations_passed = all(
                item.status == "PASS" for item in validations if item.script_present
            )
            errors = tuple(
                dict.fromkeys(
                    execution.error_codes
                    + validation_errors
                    + preflight_contract_errors
                    + protected_change_errors
                    + capture_errors
                )
            )
            if sample.requires_changes and not changes:
                errors = tuple(dict.fromkeys(errors + ("no_changes",)))
            if execution.public_outcome is None:
                errors = tuple(dict.fromkeys(errors + ("public_outcome_missing",)))
            elif execution.public_outcome.status != "completed":
                errors = tuple(
                    dict.fromkeys(errors + ("public_outcome_not_completed",))
                )
            success = (
                execution.exit_code == 0
                and not execution.timed_out
                and execution.rejected_events == 0
                and not errors
                and execution.public_outcome is not None
                and execution.public_outcome.status == "completed"
                and required_passed
                and present_validations_passed
            )
            return GoldRecord(
                sample_id=sample.sample_id,
                backend=self.executor.backend_name,
                success=success,
                workspace_id=workspace.name,
                max_iterations=self.policy.max_iterations,
                timeout_seconds=self.policy.timeout_seconds,
                agent_exit_code=execution.exit_code,
                timed_out=execution.timed_out,
                duration_ms=execution.duration_ms
                + sum(item.duration_ms for item in validations),
                validations=validations,
                tool_trace=trace,
                changed_files=changes,
                task_bundle_sha256=sample_contract_sha256(sample),
                agent_stdout_sha256=execution.stdout_sha256,
                agent_stderr_sha256=execution.stderr_sha256,
                skill_provenance=sample.skill_provenance,
                public_outcome=execution.public_outcome,
                rejected_events=execution.rejected_events,
                error_codes=errors,
            )
        finally:
            if not self.retain_workspace:
                self.workspaces.cleanup(workspace)
