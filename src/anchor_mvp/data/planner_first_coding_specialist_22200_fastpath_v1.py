"""Non-live, planner-first fast-path scheduler for the 22,200 campaign.

This is intentionally a scheduling and receipt primitive, not a teacher
client.  A later provider controller may use it only after the source and
planner-release gates in :mod:`planner_first_coding_specialist_22200_v1` have
been met.  The online path verifies only the previous chain tip and the
current group delta; final release performs the one complete replay.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
from typing import Iterable, Sequence

from .planner_first_coding_specialist_22200_v1 import (
    CampaignValidationError,
    PlannerEvidence,
    canonical_json,
    validate_planner_evidence,
)


SCHEDULER_SCHEMA_VERSION = "anchor.planner-first-coding-specialist-fastpath.v1"
GROUP_SIZE = 30
TERMINAL_STATES = frozenset({"accepted", "rejected", "quarantine"})
ROUTE_COMMIT_SCHEMA_VERSION = "anchor.planner-route-commit.v1"
_HASH_LENGTH = 64


class SchedulerValidationError(CampaignValidationError):
    """Raised when the planner-first fast path cannot safely advance."""


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _HASH_LENGTH
        and all(char in "0123456789abcdef" for char in value)
    )


def _canonical_hash(value: object) -> str:
    return _sha256(canonical_json(value))


def _hmac_hex(key: bytes, value: object) -> str:
    return hmac.new(key, canonical_json(value), hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class RouteCommit:
    object_id_sha256: str
    route_commit_sha256: str
    selected_expert: str
    planner_freeze_receipt_sha256: str

    def validate(self) -> None:
        if not all(
            _is_hash(value)
            for value in (
                self.object_id_sha256,
                self.route_commit_sha256,
                self.planner_freeze_receipt_sha256,
            )
        ):
            raise SchedulerValidationError("route_commit_sha256_required")
        if self.selected_expert not in {
            "coding_specialist",
            "tool_policy",
            "domain_review",
            "security_gate",
        }:
            raise SchedulerValidationError("route_commit_selected_expert_invalid")


@dataclass(frozen=True)
class GroupTerminal:
    object_id_sha256: str
    terminal_state: str
    receipt_sha256: str
    provider_input_tokens: int | None
    provider_output_tokens: int | None

    def canonical_row(self) -> dict[str, object]:
        if not _is_hash(self.object_id_sha256) or not _is_hash(self.receipt_sha256):
            raise SchedulerValidationError("terminal_hash_required")
        if self.terminal_state not in TERMINAL_STATES:
            raise SchedulerValidationError("terminal_state_invalid")
        for label, value in (
            ("provider_input_tokens", self.provider_input_tokens),
            ("provider_output_tokens", self.provider_output_tokens),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise SchedulerValidationError(f"{label}_invalid")
        return {
            "object_id_sha256": self.object_id_sha256,
            "provider_input_tokens": self.provider_input_tokens,
            "provider_output_tokens": self.provider_output_tokens,
            "receipt_sha256": self.receipt_sha256,
            "terminal_state": self.terminal_state,
        }


@dataclass(frozen=True)
class DeltaCheckpoint:
    group_index: int
    previous_chain_tip_sha256: str
    delta_root_sha256: str
    chain_tip_sha256: str
    hmac_sha256: str
    accepted_count: int
    rejected_count: int
    quarantine_count: int
    provider_input_tokens: int | None
    provider_output_tokens: int | None
    observed_duration_seconds: float | None

    def public_dict(self) -> dict[str, object]:
        return {
            "accepted_count": self.accepted_count,
            "chain_tip_sha256": self.chain_tip_sha256,
            "delta_root_sha256": self.delta_root_sha256,
            "group_index": self.group_index,
            "hmac_sha256": self.hmac_sha256,
            "observed_duration_seconds": self.observed_duration_seconds,
            "previous_chain_tip_sha256": self.previous_chain_tip_sha256,
            "provider_input_tokens": self.provider_input_tokens,
            "provider_output_tokens": self.provider_output_tokens,
            "quarantine_count": self.quarantine_count,
            "rejected_count": self.rejected_count,
        }


class PlannerFirstFastPath:
    """Serial state machine with bounded online work and final full replay."""

    def __init__(
        self,
        *,
        planner_bootstrap_ids: Sequence[str],
        expert_object_ids: Sequence[str],
        runtime_hmac_key: bytes,
        group_size: int = GROUP_SIZE,
    ) -> None:
        if not runtime_hmac_key:
            raise SchedulerValidationError("runtime_hmac_key_required")
        if not isinstance(group_size, int) or group_size < 1:
            raise SchedulerValidationError("positive_group_size_required")
        self._planner_bootstrap_ids = self._unique_hashes(
            planner_bootstrap_ids, "planner_bootstrap_ids"
        )
        self._expert_object_ids = self._unique_hashes(
            expert_object_ids, "expert_object_ids"
        )
        if set(self._planner_bootstrap_ids).intersection(self._expert_object_ids):
            raise SchedulerValidationError(
                "planner_and_expert_source_ids_must_be_disjoint"
            )
        self._key = bytes(runtime_hmac_key)
        self._group_size = group_size
        self._planner_evidence: PlannerEvidence | None = None
        self._routes: dict[str, RouteCommit] = {}
        self._terminal: dict[str, GroupTerminal] = {}
        self._checkpoints: list[DeltaCheckpoint] = []
        self._checkpoint_rows: list[tuple[dict[str, object], ...]] = []
        self._chain_tip = _sha256(b"anchor.planner-first-fastpath.v1:genesis")
        self._full_replay_count = 0
        self._online_delta_rows_verified = 0

    @staticmethod
    def _unique_hashes(values: Sequence[str], label: str) -> tuple[str, ...]:
        result = tuple(values)
        if not result or any(not _is_hash(value) for value in result):
            raise SchedulerValidationError(f"{label}_must_be_nonempty_sha256_values")
        if len(set(result)) != len(result):
            raise SchedulerValidationError(f"{label}_must_be_unique")
        return result

    @property
    def planner_frozen(self) -> bool:
        return self._planner_evidence is not None

    @property
    def full_replay_count(self) -> int:
        return self._full_replay_count

    @property
    def online_delta_rows_verified(self) -> int:
        return self._online_delta_rows_verified

    def freeze_planner(self, evidence: PlannerEvidence) -> None:
        if self._planner_evidence is not None:
            raise SchedulerValidationError("planner_freeze_is_immutable")
        validate_planner_evidence(evidence)
        self._planner_evidence = evidence

    def register_route_commits(self, route_commits: Iterable[RouteCommit]) -> None:
        if self._planner_evidence is None:
            raise SchedulerValidationError("planner_must_freeze_before_route_commit")
        staged = tuple(route_commits)
        if not staged:
            raise SchedulerValidationError("route_commit_batch_required")
        for commit in staged:
            commit.validate()
            if (
                commit.planner_freeze_receipt_sha256
                != self._planner_evidence.freeze_receipt_sha256
            ):
                raise SchedulerValidationError("route_commit_freeze_lineage_mismatch")
            if commit.object_id_sha256 not in self._expert_object_ids:
                raise SchedulerValidationError("route_commit_unknown_expert_object")
            if commit.object_id_sha256 in self._routes:
                raise SchedulerValidationError("route_commit_is_immutable")
        if len({commit.object_id_sha256 for commit in staged}) != len(staged):
            raise SchedulerValidationError("route_commit_batch_duplicates")
        self._routes.update({commit.object_id_sha256: commit for commit in staged})

    def commit_group(
        self,
        terminal_rows: Sequence[GroupTerminal],
        *,
        observed_duration_seconds: float | None,
    ) -> DeltaCheckpoint:
        """Commit one fully terminal group; online work is O(size of this group)."""

        if self._planner_evidence is None:
            raise SchedulerValidationError("planner_must_freeze_before_expert_group")
        rows = tuple(terminal_rows)
        if not rows or len(rows) > self._group_size:
            raise SchedulerValidationError("terminal_group_size_invalid")
        if len({row.object_id_sha256 for row in rows}) != len(rows):
            raise SchedulerValidationError("terminal_group_duplicates")
        for row in rows:
            row.canonical_row()
            if row.object_id_sha256 not in self._routes:
                raise SchedulerValidationError(
                    "expert_group_requires_frozen_route_commit"
                )
            if row.object_id_sha256 in self._terminal:
                raise SchedulerValidationError("terminal_row_is_immutable")
        if observed_duration_seconds is not None and (
            isinstance(observed_duration_seconds, bool)
            or not isinstance(observed_duration_seconds, (float, int))
            or observed_duration_seconds <= 0
        ):
            raise SchedulerValidationError("observed_duration_seconds_invalid")
        canonical_rows = [
            row.canonical_row()
            for row in sorted(rows, key=lambda item: item.object_id_sha256)
        ]
        delta_root = _canonical_hash(canonical_rows)
        terminal_counts = {
            state: sum(row.terminal_state == state for row in rows)
            for state in TERMINAL_STATES
        }
        input_values = [row.provider_input_tokens for row in rows]
        output_values = [row.provider_output_tokens for row in rows]
        input_total = sum(value for value in input_values if value is not None)
        output_total = sum(value for value in output_values if value is not None)
        input_unknown = any(value is None for value in input_values)
        output_unknown = any(value is None for value in output_values)
        checkpoint_payload = {
            "accepted_count": terminal_counts["accepted"],
            "delta_root_sha256": delta_root,
            "group_index": len(self._checkpoints),
            "previous_chain_tip_sha256": self._chain_tip,
            "provider_input_tokens": None if input_unknown else input_total,
            "provider_output_tokens": None if output_unknown else output_total,
            "quarantine_count": terminal_counts["quarantine"],
            "rejected_count": terminal_counts["rejected"],
        }
        chain_tip = _canonical_hash(checkpoint_payload)
        checkpoint_payload["chain_tip_sha256"] = chain_tip
        checkpoint = DeltaCheckpoint(
            group_index=checkpoint_payload["group_index"],
            previous_chain_tip_sha256=self._chain_tip,
            delta_root_sha256=delta_root,
            chain_tip_sha256=chain_tip,
            hmac_sha256=_hmac_hex(self._key, checkpoint_payload),
            accepted_count=terminal_counts["accepted"],
            rejected_count=terminal_counts["rejected"],
            quarantine_count=terminal_counts["quarantine"],
            provider_input_tokens=checkpoint_payload["provider_input_tokens"],
            provider_output_tokens=checkpoint_payload["provider_output_tokens"],
            observed_duration_seconds=(
                float(observed_duration_seconds)
                if observed_duration_seconds is not None
                else None
            ),
        )
        self._checkpoints.append(checkpoint)
        self._checkpoint_rows.append(tuple(canonical_rows))
        self._terminal.update({row.object_id_sha256: row for row in rows})
        self._chain_tip = chain_tip
        self._online_delta_rows_verified += len(rows)
        return checkpoint

    def group_throughput(self, checkpoint: DeltaCheckpoint) -> dict[str, object]:
        """Return exact provider-usage rates only for a complete, known group."""

        if checkpoint not in self._checkpoints:
            raise SchedulerValidationError("checkpoint_not_owned_by_scheduler")
        if (
            checkpoint.observed_duration_seconds is None
            or checkpoint.provider_input_tokens is None
            or checkpoint.provider_output_tokens is None
        ):
            return {
                "input_tokens_per_second": None,
                "output_tokens_per_second": None,
                "semantics": "UNKNOWN_provider_usage_or_duration_missing",
            }
        return {
            "input_tokens_per_second": checkpoint.provider_input_tokens
            / checkpoint.observed_duration_seconds,
            "output_tokens_per_second": checkpoint.provider_output_tokens
            / checkpoint.observed_duration_seconds,
            "semantics": "complete_group_wall_clock_provider_usage",
        }

    def final_full_replay(self) -> dict[str, object]:
        """Recompute the entire chain once before any release claim."""

        if self._planner_evidence is None:
            raise SchedulerValidationError("planner_freeze_missing_for_final_replay")
        expected_tip = _sha256(b"anchor.planner-first-fastpath.v1:genesis")
        replay_rows: list[dict[str, object]] = []
        replay_object_ids: set[str] = set()
        for expected_index, checkpoint in enumerate(self._checkpoints):
            if (
                checkpoint.group_index != expected_index
                or checkpoint.previous_chain_tip_sha256 != expected_tip
            ):
                raise SchedulerValidationError("checkpoint_sequence_or_tip_mismatch")
            try:
                delta_rows = self._checkpoint_rows[expected_index]
            except IndexError as error:
                raise SchedulerValidationError(
                    "checkpoint_delta_rows_missing"
                ) from error
            if _canonical_hash(list(delta_rows)) != checkpoint.delta_root_sha256:
                raise SchedulerValidationError("checkpoint_delta_root_mismatch")
            current_ids = {str(row["object_id_sha256"]) for row in delta_rows}
            if len(current_ids) != len(delta_rows) or replay_object_ids.intersection(
                current_ids
            ):
                raise SchedulerValidationError("checkpoint_terminal_inventory_overlap")
            replay_object_ids.update(current_ids)
            current_counts = {
                state: sum(row["terminal_state"] == state for row in delta_rows)
                for state in TERMINAL_STATES
            }
            input_values = [row["provider_input_tokens"] for row in delta_rows]
            output_values = [row["provider_output_tokens"] for row in delta_rows]
            current_input = (
                None
                if any(value is None for value in input_values)
                else sum(int(value) for value in input_values)
            )
            current_output = (
                None
                if any(value is None for value in output_values)
                else sum(int(value) for value in output_values)
            )
            payload = {
                "accepted_count": current_counts["accepted"],
                "delta_root_sha256": checkpoint.delta_root_sha256,
                "group_index": checkpoint.group_index,
                "previous_chain_tip_sha256": checkpoint.previous_chain_tip_sha256,
                "provider_input_tokens": current_input,
                "provider_output_tokens": current_output,
                "quarantine_count": current_counts["quarantine"],
                "rejected_count": current_counts["rejected"],
            }
            computed_tip = _canonical_hash(payload)
            signed_payload = {**payload, "chain_tip_sha256": computed_tip}
            if computed_tip != checkpoint.chain_tip_sha256 or not hmac.compare_digest(
                _hmac_hex(self._key, signed_payload), checkpoint.hmac_sha256
            ):
                raise SchedulerValidationError("checkpoint_hmac_or_tip_mismatch")
            expected_tip = computed_tip
            replay_rows.append(checkpoint.public_dict())
        if expected_tip != self._chain_tip:
            raise SchedulerValidationError("final_chain_tip_mismatch")
        if replay_object_ids != set(self._terminal):
            raise SchedulerValidationError("final_terminal_inventory_mismatch")
        self._full_replay_count += 1
        terminal_counts = {
            state: sum(row.terminal_state == state for row in self._terminal.values())
            for state in TERMINAL_STATES
        }
        return {
            "accepted_count": terminal_counts["accepted"],
            "checkpoint_count": len(self._checkpoints),
            "checkpoint_root_sha256": _canonical_hash(replay_rows),
            "full_replay_count": self._full_replay_count,
            "online_delta_rows_verified": self._online_delta_rows_verified,
            "planner_freeze_receipt_sha256": self._planner_evidence.freeze_receipt_sha256,
            "quarantine_count": terminal_counts["quarantine"],
            "rejected_count": terminal_counts["rejected"],
            "terminal_object_count": len(self._terminal),
        }


def complexity_model(
    *, object_count: int, group_size: int = GROUP_SIZE
) -> dict[str, int | str]:
    """State the operation model without pretending it is a provider benchmark."""

    if not isinstance(object_count, int) or object_count < 1:
        raise SchedulerValidationError("positive_object_count_required")
    if not isinstance(group_size, int) or group_size < 1:
        raise SchedulerValidationError("positive_group_size_required")
    group_count = (object_count + group_size - 1) // group_size
    return {
        "final_full_replay_rows": object_count,
        "group_count": group_count,
        "legacy_per_append_full_replay_row_visits": object_count
        * (object_count + 1)
        // 2,
        "online_delta_row_visits": object_count,
        "online_status_updates": group_count,
        "online_work": "O(1) chain-tip plus O(k) current group; final O(N) replay",
    }
