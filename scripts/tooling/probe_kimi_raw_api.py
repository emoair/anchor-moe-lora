from __future__ import annotations

import argparse
import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


URL = "https://api.kimi.com/coding/v1/chat/completions"
MODEL = "kimi-for-coding"


def payload(tool_choice: str) -> dict[str, object]:
    return {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": "Call the read_probe tool exactly once, then stop.",
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read_probe",
                    "description": "Read one harmless local probe marker.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                        },
                        "required": ["path"],
                        "additionalProperties": False,
                    },
                },
            }
        ],
        "tool_choice": tool_choice,
        "reasoning_effort": "medium",
        "max_tokens": 128,
        "stream": False,
    }


def safe_response_body(raw: bytes) -> object:
    text = raw.decode("utf-8", errors="replace")
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        return {"non_json_body_prefix": text[:500]}
    if isinstance(value, dict) and isinstance(value.get("error"), dict):
        error = value["error"]
        return {
            "error": {
                "type": error.get("type"),
                "code": error.get("code"),
                "message": error.get("message"),
            }
        }
    if isinstance(value, dict):
        choices = value.get("choices")
        return {
            "id_present": bool(value.get("id")),
            "model": value.get("model"),
            "choices_count": len(choices) if isinstance(choices, list) else None,
        }
    return {"response_type": type(value).__name__}


def probe(api_key: str, tool_choice: str, timeout: float) -> dict[str, object]:
    request = Request(
        URL,
        data=json.dumps(payload(tool_choice), separators=(",", ":")).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "anchor-moe-lora-raw-kimi-probe/1.0",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read()
            return {
                "tool_choice": tool_choice,
                "http_status": response.status,
                "response": safe_response_body(body),
            }
    except HTTPError as error:
        return {
            "tool_choice": tool_choice,
            "http_status": error.code,
            "response": safe_response_body(error.read()),
        }
    except URLError as error:
        return {
            "tool_choice": tool_choice,
            "transport_error": type(error.reason).__name__,
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Call the Kimi Code OpenAI-compatible endpoint without OpenCode."
    )
    parser.add_argument(
        "--mode",
        choices=("auto", "required", "both"),
        default="auto",
        help="Default to automatic choice; required/both are explicit negative diagnostics.",
    )
    parser.add_argument("--timeout", type=float, default=90.0)
    args = parser.parse_args()

    api_key = os.environ.get("KIMI_CODE_API_KEY", "").strip()
    if not api_key:
        print("KIMI_CODE_API_KEY is required in the current process.", file=sys.stderr)
        return 2

    choices = ("auto", "required") if args.mode == "both" else (args.mode,)
    results = [probe(api_key, choice, args.timeout) for choice in choices]
    print(
        json.dumps(
            {
                "url": URL,
                "model": MODEL,
                "thinking": {"reasoning_effort": "medium"},
                "results": results,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
