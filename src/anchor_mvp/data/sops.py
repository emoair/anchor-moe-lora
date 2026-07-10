"""Expert SOP loading for Markdown and a safe YAML subset."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path
import re
from typing import Any

from .schema import DataValidationError, ExpertSOP, TASK_TYPES, TaskType


def _yaml_scalar(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def _load_simple_yaml(text: str) -> dict[str, Any]:
    """Parse the flat keys and block scalar used by bundled SOPs.

    PyYAML is used when installed. This conservative fallback intentionally does
    not implement arbitrary YAML tags or object construction.
    """

    try:
        import yaml  # type: ignore[import-not-found]

        loaded = yaml.safe_load(text)
        if not isinstance(loaded, dict):
            raise DataValidationError("YAML SOP root must be a mapping")
        return loaded
    except ImportError:
        pass

    result: dict[str, Any] = {}
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        line = lines[index]
        index += 1
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = re.match(r"^([A-Za-z_][\w-]*):\s*(.*)$", line)
        if not match:
            raise DataValidationError(f"unsupported YAML line: {line!r}")
        key, raw_value = match.groups()
        if raw_value in ("|", ">"):
            block: list[str] = []
            while index < len(lines) and (not lines[index].strip() or lines[index].startswith((" ", "\t"))):
                block.append(lines[index].lstrip())
                index += 1
            result[key] = "\n".join(block).strip()
        else:
            result[key] = _yaml_scalar(raw_value)
    return result


def load_sop(path: str | Path, task_type: TaskType | None = None) -> ExpertSOP:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"SOP not found: {source}")
    raw = source.read_text(encoding="utf-8")
    suffix = source.suffix.casefold()
    if suffix in (".yaml", ".yml"):
        loaded = _load_simple_yaml(raw)
        inferred_task = str(loaded.get("task_type", task_type or "")).strip()
        content = str(loaded.get("procedure", loaded.get("content", ""))).strip()
        sop_id = str(loaded.get("sop_id", source.stem)).strip()
    elif suffix == ".md":
        frontmatter: dict[str, Any] = {}
        content = raw.strip()
        if raw.startswith("---\n") and "\n---\n" in raw[4:]:
            header, content = raw[4:].split("\n---\n", 1)
            frontmatter = _load_simple_yaml(header)
        inferred_task = str(frontmatter.get("task_type", task_type or source.stem)).strip()
        sop_id = str(frontmatter.get("sop_id", source.stem)).strip()
        content = content.strip()
    else:
        raise DataValidationError("SOP files must be .md, .yaml, or .yml")
    if inferred_task not in TASK_TYPES:
        raise DataValidationError(f"invalid or missing SOP task_type: {inferred_task!r}")
    if not content:
        raise DataValidationError("SOP content is empty")
    digest = sha256(content.encode("utf-8")).hexdigest()
    return ExpertSOP(
        sop_id=sop_id,
        task_type=inferred_task,  # type: ignore[arg-type]
        source=str(source),
        content=content,
        sha256=digest,
    )


def load_sop_directory(path: str | Path) -> dict[TaskType, ExpertSOP]:
    root = Path(path).expanduser().resolve()
    loaded: dict[TaskType, ExpertSOP] = {}
    for source in sorted(root.iterdir()):
        if source.suffix.casefold() not in (".md", ".yaml", ".yml"):
            continue
        sop = load_sop(source)
        if sop.task_type in loaded:
            raise DataValidationError(f"multiple SOPs found for {sop.task_type}")
        loaded[sop.task_type] = sop
    missing = set(TASK_TYPES).difference(loaded)
    if missing:
        raise DataValidationError(f"missing SOPs: {', '.join(sorted(missing))}")
    return loaded
