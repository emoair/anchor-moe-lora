"""Shared immutable source and tokenizer gate for unbalanced-v2 runtimes.

This module is deliberately model-free.  It authenticates the historical
Producer candidate/release lineage directly from immutable Git objects, pins
the exact read-set and FINAL payload blobs, and only then permits body reads.
It never trusts the current Producer worktree, branch, upstream, or remote.

The tokenizer entry point snapshots ``tokenizer.model`` once, constructs
SentencePiece from those authenticated in-memory bytes, and applies the
checked-in Gemma chat-template policy.  It does not ask Hugging Face to infer
special-token or chat-template behavior from the exported tokenizer config.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Final

from . import gemma3_chat_unbalanced_v2_consumer_v1 as consumer
from . import gemma3_chat_unbalanced_v2_multiarm_q8_qlora_v1 as contract
from . import gemma3_tokenizer_binding_v1 as tokenizer_binding


SOURCE_SCHEMA_VERSION: Final = (
    "anchor.gemma3-chat-unbalanced-v2-immutable-git-source.v2"
)
TOKENIZER_SCHEMA_VERSION: Final = (
    "anchor.gemma3-chat-unbalanced-v2-authenticated-tokenizer.v2"
)

PRODUCER_CANDIDATE_COMMIT: Final = "c5080249aa103d6cac09f0a55e51982b68524480"
PRODUCER_RELEASE_COMMIT: Final = "304be2b86f82aad7ddade80fae04528bf2801977"
PRODUCER_RELEASE_TREE: Final = "9bf154d481623b696dbc8a4bdf276967710b7214"
PRODUCER_ARTIFACT_PREFIX: Final = (
    "fixtures/research/gemma3_chat_five_expert_qonly_unbalanced_distilled_v2_sharded_v3"
)
PRODUCER_REPOSITORY_ENV: Final = "ANCHOR_GEMMA3_UNBALANCED_V2_PRODUCER_REPOSITORY"
PRODUCER_REPOSITORY_DEFAULT: Final = "../anchor-moe-lora-gemma3-chat-final-r304be2"

CONSUMER_BINDING_PATH: Final = (
    "configs/research/gemma3_chat_unbalanced_v2_consumer_binding_sharded_v1.json"
)
CONSUMER_BINDING_PHYSICAL_SHA256: Final = (
    "9f18c4106d9ed74ecfa132e74628642ce76f9111125a4b3fdd18e0d003dc57c0"
)
CONSUMER_BINDING_CANONICAL_SHA256: Final = (
    "187a1f9309edec1fbdb9204f063bffdfe0c3cadd3a405a27f369e8150c57cbe4"
)
CONSUMER_PREFLIGHT_RECEIPT_PATH: Final = (
    "fixtures/research/"
    "gemma3_chat_unbalanced_v2_consumer_preflight_sharded_v1/receipt.json"
)
CONSUMER_PREFLIGHT_RECEIPT_SHA256: Final = (
    "a976754d84f48c01b9ff509c48f7fb7b26c32eabfa0b10d6f221661114a1cdb2"
)
CONSUMER_PREFLIGHT_SIDECAR_SHA256: Final = (
    "aebd6c53788214c8fcf7783f64a60f4ffc97343c040d5422e7ca79a8fd7afbc3"
)
EXPECTED_READ_SET_FILES: Final = 88
EXPECTED_FINAL_FILES: Final = 46
EXPECTED_PAYLOAD_FILES: Final = 23

CHAT_TEMPLATE_POLICY_PATH: Final = (
    "configs/research/gemma3_1b_it_chat_template_policy_v1.json"
)
CHAT_TEMPLATE_POLICY_SHA256: Final = (
    "0ffb2e2597428da4ee5d727697bcb29a116489e220580c7bd1cb5bd8f2e076b6"
)
TOKENIZER_FILES: Final = {
    "config.json": tokenizer_binding.MODEL_FILES["config.json"],
    "tokenizer.model": tokenizer_binding.MODEL_FILES["tokenizer.model"],
    "tokenizer_config.json": tokenizer_binding.MODEL_FILES["tokenizer_config.json"],
    "EXPORT_MANIFEST.json": tokenizer_binding.MODEL_FILES["EXPORT_MANIFEST.json"],
}

_SHA256_RE: Final = re.compile(r"^[0-9a-f]{64}$")
_GIT_OID_RE: Final = re.compile(r"^[0-9a-f]{40}$")
_MAX_JSONL_LINE_BYTES: Final = 2_000_000
_SOURCE_AUTHORITY = object()
_TOKENIZER_AUTHORITY = object()


class RuntimeSourceError(RuntimeError):
    """A source/tokenizer identity or body gate failed closed."""


@dataclass(frozen=True)
class GitBlobPin:
    """One exact path-to-blob identity at the frozen release commit."""

    path: str
    oid: str
    sha256: str
    bytes: int
    kind: str


@dataclass(frozen=True)
class AuthenticatedProducerSource:
    """Capability returned only after the full immutable source gate passes."""

    schema_version: str
    repository: Path
    candidate_commit: str
    release_commit: str
    release_tree: str
    artifact_prefix: str
    consumer_preflight_receipt_sha256: str
    consumer_source_binding_sha256: str
    consumer_binding_physical_sha256: str
    consumer_binding_canonical_sha256: str
    tree_digest_sha256: str
    read_set_digest_sha256: str
    producer_inventory_sha256: str
    read_set_blob_oids_sha256: str
    payload_blob_oids_sha256: str
    payload_pins: Mapping[str, GitBlobPin]
    _read_set_pins: Mapping[str, GitBlobPin]
    _attestation_pins: Mapping[str, GitBlobPin]
    _authority: object

    def public_identity(self) -> dict[str, object]:
        """Return body-free identities suitable for runtime receipts."""

        return {
            "schema_version": self.schema_version,
            "candidate_commit": self.candidate_commit,
            "release_commit": self.release_commit,
            "release_tree": self.release_tree,
            "artifact_prefix": self.artifact_prefix,
            "historical_release_attested": True,
            "current_live_remote_reasserted": False,
            "consumer_preflight_receipt_sha256": (
                self.consumer_preflight_receipt_sha256
            ),
            "consumer_source_binding_sha256": (self.consumer_source_binding_sha256),
            "consumer_binding_physical_sha256": (self.consumer_binding_physical_sha256),
            "consumer_binding_canonical_sha256": (
                self.consumer_binding_canonical_sha256
            ),
            "tree_digest_sha256": self.tree_digest_sha256,
            "read_set_digest_sha256": self.read_set_digest_sha256,
            "producer_inventory_sha256": self.producer_inventory_sha256,
            "read_set_blob_oids_sha256": self.read_set_blob_oids_sha256,
            "payload_blob_oids_sha256": self.payload_blob_oids_sha256,
            "read_set_files": len(self._read_set_pins),
            "final_files": len(self.payload_pins),
        }


@dataclass(frozen=True)
class AuthenticatedTokenizer:
    """Authenticated in-memory SentencePiece processor and policy identity."""

    schema_version: str
    model_root: Path
    processor: Any
    tokenizer_model_sha256: str
    tokenizer_config_sha256: str
    model_config_sha256: str
    export_manifest_sha256: str
    chat_template_policy_sha256: str
    combined_identity_sha256: str
    _authority: object

    def public_identity(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "tokenizer_model_sha256": self.tokenizer_model_sha256,
            "tokenizer_config_sha256": self.tokenizer_config_sha256,
            "model_config_sha256": self.model_config_sha256,
            "export_manifest_sha256": self.export_manifest_sha256,
            "chat_template_policy_sha256": (self.chat_template_policy_sha256),
            "combined_identity_sha256": self.combined_identity_sha256,
            "processor_source": "authenticated_model_proto_bytes",
            "second_model_file_path_read": False,
            "hf_auto_tokenizer_used": False,
            "runtime_special_token_overlay": True,
        }


@dataclass(frozen=True)
class AuthenticatedSerializedExample:
    """Tokenized training example; callers must never persist raw token IDs."""

    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    prompt_tokens: int
    trainable_label_tokens: int
    sequence_tokens: int
    serialization_identity_sha256: str


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _canonical_sha256(value: object) -> str:
    return _sha256(_canonical_json(value))


def _domain_hash(domain: str, value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return _sha256(domain.encode("ascii") + b"\0" + encoded)


def _mapping(value: object, code: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeSourceError(code)
    return value


def _sequence(value: object, code: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise RuntimeSourceError(code)
    return value


def _require_sha(value: object, code: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise RuntimeSourceError(code)
    return value


def _strict_json(raw: bytes, code: str) -> Mapping[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate")
            result[key] = value
        return result

    def reject_constant(_value: str) -> None:
        raise ValueError("nonfinite")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=pairs_hook,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise RuntimeSourceError(code) from None
    if not isinstance(value, Mapping):
        raise RuntimeSourceError(code)
    return value


def _is_reparse(value: os.stat_result) -> bool:
    attributes = getattr(value, "st_file_attributes", 0)
    marker = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(attributes & marker)


def _stable_file_bytes(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int | None = None,
    code: str,
) -> bytes:
    try:
        first = path.lstat()
        if stat.S_ISLNK(first.st_mode) or _is_reparse(first):
            raise RuntimeSourceError(code)
        raw = path.read_bytes()
        second = path.lstat()
    except RuntimeSourceError:
        raise
    except OSError:
        raise RuntimeSourceError(code) from None
    first_identity = (
        first.st_dev,
        first.st_ino,
        first.st_size,
        first.st_mtime_ns,
        first.st_ctime_ns,
    )
    second_identity = (
        second.st_dev,
        second.st_ino,
        second.st_size,
        second.st_mtime_ns,
        second.st_ctime_ns,
    )
    if (
        first_identity != second_identity
        or (expected_bytes is not None and len(raw) != expected_bytes)
        or _sha256(raw) != expected_sha256
    ):
        raise RuntimeSourceError(code)
    return raw


class GitObjectReader:
    """Read exact Git objects with replacement/graft behavior forbidden."""

    def __init__(self, repository: Path) -> None:
        self.repository = repository.resolve()
        if not self.repository.is_dir():
            raise RuntimeSourceError("producer_git_repository_missing")
        self._environment = {
            **os.environ,
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_OPTIONAL_LOCKS": "0",
        }
        common = self._run_text(
            ["rev-parse", "--path-format=absolute", "--git-common-dir"],
            code="producer_git_common_dir_failed",
        ).strip()
        self.common_git_directory = Path(common)
        self.assert_no_replacement_or_graft()

    def _run(
        self,
        arguments: Sequence[str],
        *,
        code: str,
    ) -> bytes:
        try:
            result = subprocess.run(
                ["git", "-C", str(self.repository), *arguments],
                capture_output=True,
                check=False,
                env=self._environment,
                timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired):
            raise RuntimeSourceError(code) from None
        if result.returncode != 0:
            raise RuntimeSourceError(code)
        return result.stdout

    def _run_text(self, arguments: Sequence[str], *, code: str) -> str:
        try:
            return self._run(arguments, code=code).decode("utf-8")
        except UnicodeDecodeError:
            raise RuntimeSourceError(code) from None

    def assert_no_replacement_or_graft(self) -> None:
        replacement_refs = self._run_text(
            ["for-each-ref", "--format=%(refname)", "refs/replace/"],
            code="producer_git_replace_probe_failed",
        )
        grafts = self.common_git_directory / "info" / "grafts"
        try:
            graft_present = (
                grafts.exists() and grafts.is_file() and grafts.stat().st_size > 0
            )
        except OSError:
            raise RuntimeSourceError("producer_git_graft_probe_failed") from None
        if replacement_refs.strip() or graft_present:
            raise RuntimeSourceError("producer_git_replace_or_graft_forbidden")

    def object_type(self, oid: str) -> str:
        if _GIT_OID_RE.fullmatch(oid) is None:
            raise RuntimeSourceError("producer_git_oid_invalid")
        return self._run_text(
            ["cat-file", "-t", oid], code="producer_git_object_missing"
        ).strip()

    def tree_for_commit(self, commit: str) -> str:
        if _GIT_OID_RE.fullmatch(commit) is None:
            raise RuntimeSourceError("producer_git_commit_invalid")
        value = self._run_text(
            ["rev-parse", f"{commit}^{{tree}}"],
            code="producer_release_tree_resolution_failed",
        ).strip()
        if _GIT_OID_RE.fullmatch(value) is None:
            raise RuntimeSourceError("producer_release_tree_invalid")
        return value

    def assert_ancestor(self, ancestor: str, descendant: str) -> None:
        try:
            result = subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.repository),
                    "merge-base",
                    "--is-ancestor",
                    ancestor,
                    descendant,
                ],
                capture_output=True,
                check=False,
                env=self._environment,
                timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired):
            raise RuntimeSourceError("producer_git_ancestry_probe_failed") from None
        if result.returncode != 0:
            raise RuntimeSourceError("producer_candidate_not_release_ancestor")

    def tree_entries(self, commit: str) -> dict[str, tuple[str, str, str]]:
        raw = self._run(
            ["ls-tree", "-r", "-z", commit],
            code="producer_git_tree_inventory_failed",
        )
        result: dict[str, tuple[str, str, str]] = {}
        for item in raw.split(b"\0"):
            if not item:
                continue
            try:
                metadata, raw_path = item.split(b"\t", 1)
                mode, kind, oid = metadata.decode("ascii").split(" ", 2)
                path = raw_path.decode("utf-8")
            except (ValueError, UnicodeDecodeError):
                raise RuntimeSourceError(
                    "producer_git_tree_inventory_invalid"
                ) from None
            if path in result:
                raise RuntimeSourceError("producer_git_tree_path_duplicate")
            result[path] = (mode, kind, oid)
        return result

    def blob_bytes(self, oid: str) -> bytes:
        if _GIT_OID_RE.fullmatch(oid) is None:
            raise RuntimeSourceError("producer_git_blob_oid_invalid")
        return self._run(
            ["cat-file", "blob", oid],
            code="producer_git_blob_read_failed",
        )

    def verify_blob(
        self,
        tree: Mapping[str, tuple[str, str, str]],
        *,
        path: str,
        expected_sha256: str,
        expected_bytes: int,
        kind: str,
    ) -> GitBlobPin:
        entry = tree.get(path)
        if entry is None:
            raise RuntimeSourceError("producer_git_blob_path_missing")
        mode, object_kind, oid = entry
        if mode != "100644" or object_kind != "blob":
            raise RuntimeSourceError("producer_git_blob_mode_invalid")
        raw = self.blob_bytes(oid)
        if len(raw) != expected_bytes or _sha256(raw) != expected_sha256:
            raise RuntimeSourceError("producer_git_blob_identity_drift")
        if kind in {"json", "jsonl", "json_schema", "python", "text"} and (
            b"\r" in raw or (raw and not raw.endswith(b"\n"))
        ):
            raise RuntimeSourceError("producer_git_blob_not_lf")
        return GitBlobPin(
            path=path,
            oid=oid,
            sha256=expected_sha256,
            bytes=expected_bytes,
            kind=kind,
        )


def _blob_oid_inventory_sha256(pins: Mapping[str, GitBlobPin]) -> str:
    return _canonical_sha256(
        [
            {
                "path": path,
                "oid": pins[path].oid,
                "sha256": pins[path].sha256,
                "bytes": pins[path].bytes,
            }
            for path in sorted(pins)
        ]
    )


def _load_checked_in_binding_and_receipt() -> tuple[
    Mapping[str, Any], Mapping[str, Any], str
]:
    root = project_root()
    binding_path = root / CONSUMER_BINDING_PATH
    binding_raw = _stable_file_bytes(
        binding_path,
        expected_sha256=CONSUMER_BINDING_PHYSICAL_SHA256,
        code="consumer_binding_physical_identity_drift",
    )
    binding = consumer.load_binding_document(binding_path)
    canonical_binding_sha = _sha256(_canonical_json(binding))
    if canonical_binding_sha != CONSUMER_BINDING_CANONICAL_SHA256:
        raise RuntimeSourceError("consumer_binding_canonical_identity_drift")

    receipt_path = root / CONSUMER_PREFLIGHT_RECEIPT_PATH
    receipt_raw = _stable_file_bytes(
        receipt_path,
        expected_sha256=CONSUMER_PREFLIGHT_RECEIPT_SHA256,
        code="consumer_preflight_receipt_identity_drift",
    )
    if (
        _canonical_json(
            _strict_json(receipt_raw, "consumer_preflight_receipt_json_invalid")
        )
        != receipt_raw
    ):
        raise RuntimeSourceError("consumer_preflight_receipt_not_canonical")
    sidecar_path = receipt_path.with_name(receipt_path.name + ".sha256")
    sidecar_raw = _stable_file_bytes(
        sidecar_path,
        expected_sha256=CONSUMER_PREFLIGHT_SIDECAR_SHA256,
        code="consumer_preflight_sidecar_identity_drift",
    )
    if sidecar_raw != (
        f"{CONSUMER_PREFLIGHT_RECEIPT_SHA256}  {receipt_path.name}\n".encode("ascii")
    ):
        raise RuntimeSourceError("consumer_preflight_sidecar_content_invalid")
    normalized, observed_receipt_sha, source_binding_sha = (
        contract.load_consumer_preflight_receipt(receipt_path)
    )
    if (
        observed_receipt_sha != CONSUMER_PREFLIGHT_RECEIPT_SHA256
        or normalized["binding_contract_sha256"] != CONSUMER_BINDING_CANONICAL_SHA256
        or not binding_raw
    ):
        raise RuntimeSourceError("consumer_preflight_binding_cross_identity_drift")
    return binding, normalized, source_binding_sha


def authenticate_producer_source(
    producer_repository: str | Path | None = None,
) -> AuthenticatedProducerSource:
    """Authenticate P/R/tree/readset/FINAL before returning a body capability."""

    binding, normalized, source_binding_sha = _load_checked_in_binding_and_receipt()
    if producer_repository is None:
        configured = os.environ.get(PRODUCER_REPOSITORY_ENV)
        producer_repository = configured or (
            project_root() / PRODUCER_REPOSITORY_DEFAULT
        )
    repository = Path(producer_repository).resolve()
    reader = GitObjectReader(repository)
    if (
        reader.object_type(PRODUCER_CANDIDATE_COMMIT) != "commit"
        or reader.object_type(PRODUCER_RELEASE_COMMIT) != "commit"
    ):
        raise RuntimeSourceError("producer_git_commit_object_invalid")
    reader.assert_ancestor(PRODUCER_CANDIDATE_COMMIT, PRODUCER_RELEASE_COMMIT)
    release_tree = reader.tree_for_commit(PRODUCER_RELEASE_COMMIT)
    if release_tree != PRODUCER_RELEASE_TREE:
        raise RuntimeSourceError("producer_release_tree_identity_drift")
    tree = reader.tree_entries(PRODUCER_RELEASE_COMMIT)

    producer_git = _mapping(
        binding.get("producer_git"), "consumer_binding_producer_git_invalid"
    )
    if (
        producer_git.get("candidate_commit") != PRODUCER_CANDIDATE_COMMIT
        or producer_git.get("release_commit") != PRODUCER_RELEASE_COMMIT
        or producer_git.get("release_tree") != PRODUCER_RELEASE_TREE
    ):
        raise RuntimeSourceError("consumer_binding_producer_lineage_drift")
    read_set = _mapping(binding.get("read_set"), "consumer_binding_read_set_invalid")
    read_entries = _sequence(
        read_set.get("files"), "consumer_binding_read_set_files_invalid"
    )
    if (
        len(read_entries) != EXPECTED_READ_SET_FILES
        or read_set.get("count") != EXPECTED_READ_SET_FILES
        or consumer._read_set_digest(read_entries)
        != read_set.get("canonical_digest_sha256")
        or normalized["read_set"]["canonical_digest_sha256"]
        != read_set.get("canonical_digest_sha256")
    ):
        raise RuntimeSourceError("producer_read_set_contract_drift")
    read_set_pins: dict[str, GitBlobPin] = {}
    for raw_entry in read_entries:
        entry = _mapping(raw_entry, "consumer_binding_read_set_entry_invalid")
        path = str(entry["path"])
        if path in read_set_pins:
            raise RuntimeSourceError("consumer_binding_read_set_duplicate")
        read_set_pins[path] = reader.verify_blob(
            tree,
            path=path,
            expected_sha256=_require_sha(
                entry["sha256"], "producer_read_set_sha_invalid"
            ),
            expected_bytes=int(entry["bytes"]),
            kind=str(entry["kind"]),
        )

    payload_pins: dict[str, GitBlobPin] = {}
    tree_inventory: list[dict[str, object]] = []
    payload_entries = _sequence(
        binding.get("payload_files"), "consumer_binding_payload_files_invalid"
    )
    if len(payload_entries) != EXPECTED_PAYLOAD_FILES:
        raise RuntimeSourceError("producer_payload_count_invalid")
    for raw_entry in payload_entries:
        entry = _mapping(raw_entry, "consumer_binding_payload_entry_invalid")
        relative = str(entry["path"])
        if (
            PurePosixPath(relative).is_absolute()
            or ".." in PurePosixPath(relative).parts
            or relative in payload_pins
        ):
            raise RuntimeSourceError("producer_payload_path_invalid")
        full_path = f"{PRODUCER_ARTIFACT_PREFIX}/{relative}"
        payload = reader.verify_blob(
            tree,
            path=full_path,
            expected_sha256=_require_sha(
                entry["sha256"], "producer_payload_sha_invalid"
            ),
            expected_bytes=int(entry["bytes"]),
            kind=str(entry["kind"]),
        )
        sidecar = reader.verify_blob(
            tree,
            path=f"{full_path}.sha256",
            expected_sha256=_require_sha(
                entry["sidecar_sha256"], "producer_payload_sidecar_sha_invalid"
            ),
            expected_bytes=int(entry["sidecar_bytes"]),
            kind="sha256_sidecar",
        )
        sidecar_raw = reader.blob_bytes(sidecar.oid)
        if sidecar_raw != (
            f"{payload.sha256}  {PurePosixPath(relative).name}\n".encode("ascii")
        ):
            raise RuntimeSourceError("producer_payload_sidecar_content_invalid")
        payload_pins[relative] = payload
        payload_pins[f"{relative}.sha256"] = sidecar
        tree_inventory.extend(
            (
                {
                    "path": relative,
                    "sha256": payload.sha256,
                    "bytes": payload.bytes,
                },
                {
                    "path": f"{relative}.sha256",
                    "sha256": sidecar.sha256,
                    "bytes": sidecar.bytes,
                },
            )
        )
    if len(payload_pins) != EXPECTED_FINAL_FILES:
        raise RuntimeSourceError("producer_final_file_count_invalid")
    tree_digest = _domain_hash(
        consumer.PRODUCER_TREE_DIGEST_DOMAIN,
        sorted(tree_inventory, key=lambda item: str(item["path"])),
    )
    if (
        tree_digest != binding.get("tree_digest_sha256")
        or tree_digest != normalized["tree_digest_sha256"]
    ):
        raise RuntimeSourceError("producer_payload_tree_digest_drift")

    release_attestation = _mapping(
        binding.get("release_attestation"),
        "consumer_binding_release_attestation_invalid",
    )
    attestation_pins: dict[str, GitBlobPin] = {}
    for path_key, sha_key, bytes_key, kind in (
        ("path", "sha256", "bytes", "json"),
        (
            "sidecar_path",
            "sidecar_sha256",
            "sidecar_bytes",
            "sha256_sidecar",
        ),
    ):
        attestation_path = str(release_attestation[path_key])
        attestation_pin = reader.verify_blob(
            tree,
            path=attestation_path,
            expected_sha256=_require_sha(
                release_attestation[sha_key],
                "producer_release_attestation_sha_invalid",
            ),
            expected_bytes=int(release_attestation[bytes_key]),
            kind=kind,
        )
        existing = read_set_pins.get(attestation_path)
        if existing is not None and existing != attestation_pin:
            raise RuntimeSourceError("producer_release_attestation_identity_drift")
        attestation_pins[attestation_path] = attestation_pin

    source = AuthenticatedProducerSource(
        schema_version=SOURCE_SCHEMA_VERSION,
        repository=repository,
        candidate_commit=PRODUCER_CANDIDATE_COMMIT,
        release_commit=PRODUCER_RELEASE_COMMIT,
        release_tree=PRODUCER_RELEASE_TREE,
        artifact_prefix=PRODUCER_ARTIFACT_PREFIX,
        consumer_preflight_receipt_sha256=(CONSUMER_PREFLIGHT_RECEIPT_SHA256),
        consumer_source_binding_sha256=source_binding_sha,
        consumer_binding_physical_sha256=(CONSUMER_BINDING_PHYSICAL_SHA256),
        consumer_binding_canonical_sha256=(CONSUMER_BINDING_CANONICAL_SHA256),
        tree_digest_sha256=tree_digest,
        read_set_digest_sha256=str(read_set["canonical_digest_sha256"]),
        producer_inventory_sha256=str(read_set["producer_inventory_sha256"]),
        read_set_blob_oids_sha256=_blob_oid_inventory_sha256(read_set_pins),
        payload_blob_oids_sha256=_blob_oid_inventory_sha256(payload_pins),
        payload_pins=payload_pins,
        _read_set_pins=read_set_pins,
        _attestation_pins=attestation_pins,
        _authority=_SOURCE_AUTHORITY,
    )
    terminal_recheck_source(source)
    return source


def terminal_recheck_source(source: AuthenticatedProducerSource) -> None:
    """Recheck commit/tree/path->OID identities after all body operations."""

    if source._authority is not _SOURCE_AUTHORITY:
        raise RuntimeSourceError("producer_source_capability_invalid")
    reader = GitObjectReader(source.repository)
    reader.assert_no_replacement_or_graft()
    if reader.tree_for_commit(source.release_commit) != source.release_tree:
        raise RuntimeSourceError("producer_release_tree_changed")
    tree = reader.tree_entries(source.release_commit)
    for pin in (
        *source._read_set_pins.values(),
        *source._attestation_pins.values(),
        *source.payload_pins.values(),
    ):
        entry = tree.get(pin.path)
        if entry is None or entry != ("100644", "blob", pin.oid):
            raise RuntimeSourceError("producer_blob_oid_changed")


def read_authenticated_blob(
    source: AuthenticatedProducerSource,
    relative_path: str,
) -> bytes:
    """Read one exact FINAL payload blob by its pinned object ID."""

    if source._authority is not _SOURCE_AUTHORITY:
        raise RuntimeSourceError("producer_body_read_without_authority")
    path = str(PurePosixPath(relative_path))
    if (
        PurePosixPath(path).is_absolute()
        or ".." in PurePosixPath(path).parts
        or path.endswith(".sha256")
    ):
        raise RuntimeSourceError("producer_body_path_invalid")
    pin = source.payload_pins.get(path)
    if pin is None:
        raise RuntimeSourceError("producer_body_path_not_authenticated")
    raw = GitObjectReader(source.repository).blob_bytes(pin.oid)
    if len(raw) != pin.bytes or _sha256(raw) != pin.sha256:
        raise RuntimeSourceError("producer_body_blob_identity_drift")
    return raw


def iter_authenticated_jsonl(
    source: AuthenticatedProducerSource,
    relative_path: str,
    *,
    max_line_bytes: int = _MAX_JSONL_LINE_BYTES,
) -> Iterator[Mapping[str, Any]]:
    """Stream strict JSONL only after source authentication has passed."""

    raw = read_authenticated_blob(source, relative_path)
    for line in raw.splitlines(keepends=True):
        if (
            not line
            or len(line) > max_line_bytes
            or b"\r" in line
            or not line.endswith(b"\n")
        ):
            raise RuntimeSourceError("producer_jsonl_line_invalid")
        yield _strict_json(line[:-1], "producer_jsonl_record_invalid")


def authenticate_tokenizer_snapshot(
    model_root: str | Path,
) -> AuthenticatedTokenizer:
    """Build SentencePiece from one authenticated model-proto byte snapshot."""

    root = Path(model_root).resolve()
    if not root.is_dir():
        raise RuntimeSourceError("tokenizer_model_root_missing")
    snapshots: dict[str, bytes] = {}
    for name, (expected_bytes, expected_sha) in TOKENIZER_FILES.items():
        snapshots[name] = _stable_file_bytes(
            root / name,
            expected_sha256=expected_sha,
            expected_bytes=expected_bytes,
            code=f"tokenizer_snapshot_{name.replace('.', '_')}_drift",
        )
    policy_raw = _stable_file_bytes(
        project_root() / CHAT_TEMPLATE_POLICY_PATH,
        expected_sha256=CHAT_TEMPLATE_POLICY_SHA256,
        code="chat_template_policy_identity_drift",
    )
    policy = _strict_json(policy_raw, "chat_template_policy_json_invalid")
    try:
        tokenizer_binding.validate_policy(policy)
    except Exception as exc:
        raise RuntimeSourceError(
            f"chat_template_policy_invalid:{type(exc).__name__}"
        ) from None

    config = _strict_json(snapshots["config.json"], "model_config_json_invalid")
    tokenizer_config = _strict_json(
        snapshots["tokenizer_config.json"],
        "tokenizer_config_json_invalid",
    )
    manifest = _strict_json(
        snapshots["EXPORT_MANIFEST.json"],
        "export_manifest_json_invalid",
    )
    manifest_files = {
        str(item["path"]): (int(item["bytes"]), str(item["sha256"]))
        for item in (
            _mapping(value, "export_manifest_file_invalid")
            for value in _sequence(
                manifest.get("files"), "export_manifest_files_invalid"
            )
        )
    }
    if (
        config.get("bos_token_id") != 1
        or config.get("eos_token_id") != 2
        or tokenizer_config.get("bos_token") != "<bos>"
        or tokenizer_config.get("eos_token") != "<eos>"
        or "chat_template" in tokenizer_config
        or manifest.get("schema_version") != "anchor.local-gemma3-hf-export-manifest.v1"
        or manifest.get("chat_template_bound") is not False
        or manifest_files
        != {
            key: value
            for key, value in tokenizer_binding.MODEL_FILES.items()
            if key != "EXPORT_MANIFEST.json"
        }
    ):
        raise RuntimeSourceError("tokenizer_export_semantics_drift")
    try:
        import sentencepiece as sentencepiece
    except ImportError:
        raise RuntimeSourceError("sentencepiece_runtime_unavailable") from None
    try:
        processor = sentencepiece.SentencePieceProcessor(
            model_proto=snapshots["tokenizer.model"]
        )
    except Exception:
        raise RuntimeSourceError("sentencepiece_model_proto_invalid") from None
    if processor.vocab_size() != 262_144:
        raise RuntimeSourceError("sentencepiece_vocab_drift")
    for name, token_id in tokenizer_binding.SPECIAL_IDS.items():
        if processor.id_to_piece(token_id) != tokenizer_binding.SPECIAL_PIECES[name]:
            raise RuntimeSourceError("sentencepiece_special_identity_drift")
    if processor.encode("<start_of_turn>", out_type=int) != [
        tokenizer_binding.SPECIAL_IDS["start_of_turn"]
    ] or processor.encode("<end_of_turn>", out_type=int) != [
        tokenizer_binding.SPECIAL_IDS["end_of_turn"]
    ]:
        raise RuntimeSourceError("sentencepiece_turn_marker_drift")
    combined = _canonical_sha256(
        {
            "tokenizer_model_sha256": TOKENIZER_FILES["tokenizer.model"][1],
            "tokenizer_config_sha256": TOKENIZER_FILES["tokenizer_config.json"][1],
            "model_config_sha256": TOKENIZER_FILES["config.json"][1],
            "export_manifest_sha256": TOKENIZER_FILES["EXPORT_MANIFEST.json"][1],
            "chat_template_policy_sha256": CHAT_TEMPLATE_POLICY_SHA256,
            "runtime_overlay": {
                "bos": 2,
                "eos": 1,
                "pad": 0,
                "unk": 3,
                "start_of_turn": 105,
                "end_of_turn": 106,
            },
            "processor_source": "authenticated_model_proto_bytes",
        }
    )
    return AuthenticatedTokenizer(
        schema_version=TOKENIZER_SCHEMA_VERSION,
        model_root=root,
        processor=processor,
        tokenizer_model_sha256=TOKENIZER_FILES["tokenizer.model"][1],
        tokenizer_config_sha256=TOKENIZER_FILES["tokenizer_config.json"][1],
        model_config_sha256=TOKENIZER_FILES["config.json"][1],
        export_manifest_sha256=TOKENIZER_FILES["EXPORT_MANIFEST.json"][1],
        chat_template_policy_sha256=CHAT_TEMPLATE_POLICY_SHA256,
        combined_identity_sha256=combined,
        _authority=_TOKENIZER_AUTHORITY,
    )


def serialize_authenticated_example(
    tokenizer: AuthenticatedTokenizer,
    prompt: str,
    target: str,
    *,
    sequence_length: int = 768,
) -> AuthenticatedSerializedExample:
    """Serialize with the authenticated overlay; reject all truncation."""

    if tokenizer._authority is not _TOKENIZER_AUTHORITY:
        raise RuntimeSourceError("tokenizer_capability_invalid")
    try:
        serialized = tokenizer_binding.serialize_example(
            tokenizer.processor, prompt, target
        )
    except Exception as exc:
        raise RuntimeSourceError(
            f"authenticated_serialization_failed:{type(exc).__name__}"
        ) from None
    if (
        sequence_length != 768
        or len(serialized.input_ids) > sequence_length
        or len(serialized.labels) != len(serialized.input_ids)
    ):
        raise RuntimeSourceError("authenticated_serialization_length_invalid")
    identity = _canonical_sha256(
        {
            "tokenizer_identity_sha256": tokenizer.combined_identity_sha256,
            "prompt_sha256": _sha256(prompt.encode("utf-8")),
            "target_sha256": _sha256(target.encode("utf-8")),
            "sequence_tokens": len(serialized.input_ids),
            "prompt_tokens": serialized.prompt_tokens,
            "trainable_label_tokens": serialized.trainable_label_tokens,
            "truncated": False,
        }
    )
    return AuthenticatedSerializedExample(
        input_ids=serialized.input_ids,
        labels=serialized.labels,
        prompt_tokens=serialized.prompt_tokens,
        trainable_label_tokens=serialized.trainable_label_tokens,
        sequence_tokens=len(serialized.input_ids),
        serialization_identity_sha256=identity,
    )


__all__ = [
    "AuthenticatedProducerSource",
    "AuthenticatedSerializedExample",
    "AuthenticatedTokenizer",
    "GitBlobPin",
    "GitObjectReader",
    "RuntimeSourceError",
    "authenticate_producer_source",
    "authenticate_tokenizer_snapshot",
    "iter_authenticated_jsonl",
    "project_root",
    "read_authenticated_blob",
    "serialize_authenticated_example",
    "terminal_recheck_source",
]
