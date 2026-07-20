from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Iterator

from .models import PublicDecisionStep, PublicOutcome, ToolTraceEntry
from .policy import ToolPolicy


def digest_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()


def _digest_json(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return digest_text(encoded)


def classify_kimi_400_error(value: str) -> str:
    """Reduce a trusted Kimi HTTP 400 body to one fixed, content-free category."""

    lowered = value.casefold()
    if "invalid_url" in lowered or "provided url is invalid" in lowered:
        return "kimi_400_invalid_url"
    if "total message size" in lowered and "exceeds limit" in lowered:
        return "kimi_400_message_too_large"
    if "request exceeded model token limit" in lowered:
        return "kimi_400_token_limit"
    if "reasoning_content" in lowered and "missing" in lowered:
        return "kimi_400_missing_reasoning_content"
    if "unsupported image url" in lowered:
        return "kimi_400_unsupported_image_url"
    if "function name" in lowered and "duplicated" in lowered:
        return "kimi_400_duplicate_function_name"
    if "request was rejected" in lowered and "high risk" in lowered:
        return "kimi_400_high_risk_rejected"
    return "kimi_400_unknown"


def classify_error_text(value: str) -> tuple[str, ...]:
    lowered = value.lower()
    codes: list[str] = []
    if (
        "invalid_url" in lowered
        or "invalid url" in lowered
        or "missing scheme" in lowered
    ):
        codes.append("invalid_url")
    if (
        "context canceled" in lowered
        or "context cancelled" in lowered
        or "http 499" in lowered
    ):
        codes.append("client_cancelled")
    if (
        "rate limit" in lowered
        or "status code: 429" in lowered
        or "http 429" in lowered
    ):
        codes.append("rate_limited")
    if (
        "invalid authentication" in lowered
        or "status code: 401" in lowered
        or "http 401" in lowered
    ):
        codes.append("authentication_failed")
    if _is_missing_reasoning_content_400(lowered):
        codes.append("missing_reasoning_content")
    return tuple(codes)


_SAFE_ERROR_TOKEN = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,79}$")
_HTTP_400_SIGNAL = re.compile(
    r'(?:\bhttp(?:\s+status)?|\bstatus(?:\s*code)?|"status(?:code)?")\s*(?::|=)?\s*400\b'
)


def _is_missing_reasoning_content_400(lowered: str) -> bool:
    """Recognize the Kimi 400 contract error without returning provider text."""

    if not _HTTP_400_SIGNAL.search(lowered):
        return False
    has_reasoning_field = (
        "reasoning_content" in lowered or "reasoning content" in lowered
    )
    missing_signal = any(
        marker in lowered
        for marker in (
            "missing",
            "required",
            "must include",
            "must be included",
            "not provided",
            "not present",
        )
    )
    return has_reasoning_field and missing_signal


def classify_error_metadata(stdout: str, stderr: str) -> tuple[str, ...]:
    """Classify failures without retaining messages, model text, or tool contents."""

    codes = list(classify_error_text(stdout + "\n" + stderr))
    status_names = {"status", "statuscode", "httpstatus", "http_status"}
    code_names = {"code", "name", "errorcode", "error_code"}
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and str(event.get("type", "")).casefold() == "error":
            error = event.get("error")
            data = error.get("data") if isinstance(error, dict) else None
            if (
                isinstance(error, dict)
                and str(error.get("name", "")).casefold() == "apierror"
                and isinstance(data, dict)
            ):
                raw_status_code = data.get("statusCode")
                if isinstance(raw_status_code, (str, int, float)):
                    try:
                        status_code = int(raw_status_code)
                    except (TypeError, ValueError):
                        status_code = 0
                else:
                    status_code = 0
                if status_code == 400:
                    trusted_text = "\n".join(
                        value
                        for name in ("message", "responseBody")
                        if isinstance((value := data.get(name)), str)
                    )
                    codes.append(classify_kimi_400_error(trusted_text))
        for item in _walk_dicts(event):
            if str(item.get("type", "")).casefold() == "error":
                codes.append("agent_error_event")
            item_codes = {
                str(value).casefold()
                for key, value in item.items()
                if str(key).casefold() in code_names
                and isinstance(value, str)
                and _SAFE_ERROR_TOKEN.fullmatch(value)
            }
            for key, value in item.items():
                name = str(key).casefold()
                if name in status_names:
                    try:
                        status = int(value)
                    except (TypeError, ValueError):
                        continue
                    if status == 400:
                        codes.append(
                            "invalid_url"
                            if "invalid_url" in item_codes
                            else "invalid_request"
                        )
                    elif status == 401:
                        codes.append("authentication_failed")
                    elif status == 403:
                        codes.append("forbidden")
                    elif status == 404:
                        codes.append("endpoint_or_model_not_found")
                    elif status in {408, 499}:
                        codes.append("client_cancelled")
                    elif status == 429:
                        codes.append("rate_limited")
                    elif 500 <= status <= 599:
                        codes.append("upstream_server_error")
                elif (
                    name in code_names
                    and isinstance(value, str)
                    and _SAFE_ERROR_TOKEN.fullmatch(value)
                ):
                    normalized = value.casefold()
                    if normalized == "invalid_url":
                        codes.append("invalid_url")
                    else:
                        codes.append(f"agent_{normalized}")
    if stderr.strip():
        codes.append("agent_stderr_present")
    return tuple(dict.fromkeys(codes))


def _walk_dicts(value: Any) -> Iterator[dict[str, Any]]:
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from _walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from _walk_dicts(child)


def _json_candidate(value: str) -> dict[str, Any] | None:
    text = value.strip()
    if text.startswith("```json") and text.endswith("```"):
        text = text[7:-3].strip()
    if not (text.startswith("{") and text.endswith("}")):
        return None
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        return None
    return loaded if isinstance(loaded, dict) else None


def parse_public_outcome(stdout: str) -> PublicOutcome | None:
    """Extract only the explicit public work summary from OpenCode events.

    Arbitrary assistant text and reasoning blocks are ignored. The accepted object
    has a narrow schema and bounded public evidence fields.
    """

    accepted: PublicOutcome | None = None
    forbidden = {"thinking", "reasoning", "chain_of_thought", "cot"}
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if (
            not isinstance(event, dict)
            or str(event.get("type", "")).casefold() != "text"
        ):
            continue
        part = event.get("part")
        if not isinstance(part, dict) or str(part.get("type", "")).casefold() not in {
            "",
            "text",
        }:
            continue
        text = part.get("text")
        if not isinstance(text, str):
            continue
        for value in (text,):
            candidate = _json_candidate(value)
            if (
                not candidate
                or candidate.get("schema_version") != "anchor.public-outcome.v1"
            ):
                continue
            if forbidden.intersection(str(key).casefold() for key in candidate):
                continue
            status = candidate.get("status")
            trace = candidate.get("decision_trace")
            repairs = candidate.get("repair_summaries")
            summary = candidate.get("final_summary")
            if status not in {"completed", "blocked", "partial"}:
                continue
            if not isinstance(trace, list) or not 1 <= len(trace) <= 8:
                continue
            steps: list[PublicDecisionStep] = []
            valid = True
            for item in trace:
                if not isinstance(item, dict):
                    valid = False
                    break
                fields = tuple(
                    str(item.get(name, "")).strip()
                    for name in ("check", "evidence", "action")
                )
                if not all(fields) or any(len(field) > 600 for field in fields):
                    valid = False
                    break
                steps.append(PublicDecisionStep(*fields))
            if not valid or not isinstance(repairs, list) or len(repairs) > 8:
                continue
            clean_repairs = tuple(str(item).strip() for item in repairs)
            if any(not item or len(item) > 600 for item in clean_repairs):
                continue
            if (
                not isinstance(summary, str)
                or not summary.strip()
                or len(summary.strip()) > 1000
            ):
                continue
            accepted = PublicOutcome(
                status=status,
                decision_trace=tuple(steps),
                repair_summaries=clean_repairs,
                final_summary=summary.strip(),
            )
    return accepted


def _first_string(mapping: dict[str, Any], names: tuple[str, ...]) -> str | None:
    for name in names:
        value = mapping.get(name)
        if isinstance(value, str):
            return value
    return None


def _first_integer(mapping: dict[str, Any], names: tuple[str, ...]) -> int | None:
    for name in names:
        value = mapping.get(name)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _first_not_none(*values: int | None) -> int | None:
    return next((value for value in values if value is not None), None)


def parse_opencode_jsonl(
    stdout: str, policy: ToolPolicy
) -> tuple[tuple[ToolTraceEntry, ...], int]:
    """Reduce raw OpenCode events to safe metadata.

    Model text, thinking blocks, file contents, and arbitrary tool arguments are
    deliberately discarded. Unapproved commands are represented only by a hash.
    """

    trace: list[ToolTraceEntry] = []
    call_indexes: dict[str, int] = {}
    for line in stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        seen_without_id: set[tuple[str, str | None, str | None, int | None, str]] = (
            set()
        )
        for item in _walk_dicts(event):
            event_type = str(item.get("type", "")).lower()
            tool = _first_string(item, ("tool", "toolName", "tool_name", "name"))
            if not tool or not (
                event_type in {"tool", "tool_use", "tool_call", "tool_result"}
                or "state" in item
                or "input" in item
            ):
                continue
            tool = tool.lower()
            raw_state = item.get("state")
            state: dict[str, Any] = raw_state if isinstance(raw_state, dict) else {}
            raw_tool_input = item.get("input")
            tool_input: dict[str, Any] = (
                raw_tool_input if isinstance(raw_tool_input, dict) else {}
            )
            raw_state_input = state.get("input")
            state_input: dict[str, Any] = (
                raw_state_input if isinstance(raw_state_input, dict) else {}
            )
            actual_input = (
                raw_state_input
                if isinstance(raw_state_input, dict)
                else raw_tool_input
                if isinstance(raw_tool_input, dict)
                else None
            )
            input_sha256 = (
                _digest_json(actual_input) if actual_input is not None else None
            )
            command = (
                _first_string(item, ("command", "cmd"))
                or _first_string(tool_input, ("command", "cmd"))
                or _first_string(state_input, ("command", "cmd"))
            )
            exit_code = _first_not_none(
                _first_integer(item, ("exitCode", "exit_code", "code")),
                _first_integer(state, ("exitCode", "exit_code", "code")),
            )
            status = str(state.get("status", item.get("status", "observed")))
            safe_command: str | None = None
            command_hash: str | None = None
            allowed = policy.is_tool_allowed(tool)
            if command is not None:
                normalized = policy.normalize_command(command)
                allowed = allowed and policy.is_command_allowed(command)
                if allowed:
                    safe_command = normalized
                else:
                    command_hash = policy.command_digest(command)
            if not allowed:
                status = "rejected"
            raw_output = state.get("output", item.get("output"))
            output_sha256: str | None = None
            if isinstance(raw_output, str):
                output_sha256 = digest_text(raw_output)
            elif isinstance(raw_output, (dict, list)):
                output_sha256 = digest_text(
                    json.dumps(
                        raw_output,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                )
            call_id = _first_string(item, ("callID", "call_id", "id")) or _first_string(
                state, ("callID", "call_id", "id")
            )
            if call_id is None:
                key = (
                    tool,
                    safe_command or command_hash,
                    input_sha256,
                    exit_code,
                    status,
                )
                if key in seen_without_id:
                    continue
                seen_without_id.add(key)
            entry = ToolTraceEntry(
                sequence=len(trace) + 1,
                source="agent",
                tool=tool,
                status=status,
                input_sha256=input_sha256,
                command=safe_command,
                command_sha256=command_hash,
                exit_code=exit_code,
                output_sha256=output_sha256,
            )
            if call_id is None or call_id not in call_indexes:
                if call_id is not None:
                    call_indexes[call_id] = len(trace)
                trace.append(entry)
                continue

            index = call_indexes[call_id]
            previous = trace[index]
            trace[index] = ToolTraceEntry(
                sequence=previous.sequence,
                source="agent",
                tool=entry.tool,
                status=(
                    "rejected"
                    if "rejected" in {previous.status, entry.status}
                    else entry.status
                ),
                input_sha256=entry.input_sha256 or previous.input_sha256,
                command=entry.command or previous.command,
                command_sha256=entry.command_sha256 or previous.command_sha256,
                exit_code=(
                    entry.exit_code
                    if entry.exit_code is not None
                    else previous.exit_code
                ),
                output_sha256=entry.output_sha256 or previous.output_sha256,
            )
    rejected = sum(item.status == "rejected" for item in trace)
    return tuple(trace), rejected
