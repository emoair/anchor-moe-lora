"""Deterministic, benign-only candidate mutations for review training."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import re

from .schema import DataValidationError, stable_id


class MutationUnavailableError(DataValidationError):
    """No allowlisted benign mutation applies; never invent a security defect."""


@dataclass(frozen=True)
class MutationManifest:
    mutation_id: str
    rule: str
    path: str
    count: int
    sha256_before: str
    sha256_after: str
    known_benign_defect: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def mutate_frontend_code(code: str, *, source_record_id: str) -> tuple[str, MutationManifest]:
    """Apply the first deterministic allowlisted accessibility mutation."""

    if not code.strip():
        raise MutationUnavailableError("frontend source code is empty")
    before_hash = sha256(code.encode("utf-8")).hexdigest()

    # Preferred rule: remove exactly one literal aria-label, creating a local a11y defect.
    aria = re.compile(r"\s+aria-label\s*=\s*([\"'])[^\"']+\1", re.IGNORECASE)
    candidate, count = aria.subn("", code, count=1)
    if count:
        rule = "remove_literal_aria_label"
        defect = "One literal aria-label was removed; restore an accurate accessible name."
    else:
        # Safe fallback: degrade one semantic landmark while preserving balanced JSX/text.
        candidate, open_count = re.subn(r"<main(?=[\s>])", "<div", code, count=1, flags=re.IGNORECASE)
        if open_count:
            candidate, close_count = re.subn(
                r"</main\s*>", "</div>", candidate, count=1, flags=re.IGNORECASE
            )
            if close_count != 1:
                raise MutationUnavailableError("main landmark is not textually balanced")
            count = open_count + close_count
            rule = "semantic_main_to_div"
            defect = "A main landmark was replaced by a generic div; restore semantic main markup."
        else:
            candidate, open_count = re.subn(
                r"<h1(?=[\s>])", "<div", code, count=1, flags=re.IGNORECASE
            )
            if open_count:
                candidate, close_count = re.subn(
                    r"</h1\s*>", "</div>", candidate, count=1, flags=re.IGNORECASE
                )
                if close_count != 1:
                    raise MutationUnavailableError("h1 heading is not textually balanced")
                count = open_count + close_count
                rule = "semantic_h1_to_div"
                defect = "A page h1 was replaced by a generic div; restore the heading semantic."
            else:
                raise MutationUnavailableError(
                    "no allowlisted aria-label, main landmark, or h1 mutation applies"
                )

    after_hash = sha256(candidate.encode("utf-8")).hexdigest()
    if after_hash == before_hash:
        raise MutationUnavailableError("benign mutation did not change source code")
    identity = f"{source_record_id}\n{rule}\n{before_hash}\n{after_hash}"
    manifest = MutationManifest(
        mutation_id=stable_id("mutation", identity),
        rule=rule,
        path="output.code",
        count=count,
        sha256_before=before_hash,
        sha256_after=after_hash,
        known_benign_defect=defect,
    )
    return candidate, manifest
