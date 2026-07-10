from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
from uuid import uuid4


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _within_project(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(PROJECT_ROOT)
    except ValueError as exc:
        raise ValueError(f"path must remain inside the project: {resolved}") from exc
    return resolved


def _emit(progress: Path, phase: str, state: str, **detail: object) -> None:
    event = {"time": _now(), "phase": phase, "state": state, "detail": detail}
    progress.parent.mkdir(parents=True, exist_ok=True)
    temporary = progress.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(event, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(progress)
    print(json.dumps(event, sort_keys=True), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export the frozen Gemma 4 base as a reloadable bitsandbytes NF4 checkpoint"
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_ROOT / "models" / "google-gemma-4-12B-base",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "models" / "google-gemma-4-12B-bnb-nf4",
    )
    parser.add_argument(
        "--processor",
        type=Path,
        default=PROJECT_ROOT / "models" / "google-gemma-4-12B-base",
    )
    parser.add_argument(
        "--progress",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "model-export" / "bnb_nf4_status.json",
    )
    parser.add_argument("--max-shard-size", default="2GB")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = _within_project(args.source)
    output = _within_project(args.output)
    processor_source = _within_project(args.processor)
    progress = _within_project(args.progress)
    if not source.is_dir() or not (source / "model.safetensors").is_file():
        raise FileNotFoundError(f"missing local source checkpoint: {source}")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.parent / f".{output.name}.partial-{uuid4().hex}"
    _within_project(partial)

    _emit(progress, "imports", "started")
    import torch
    from transformers import AutoModelForMultimodalLM, AutoProcessor, BitsAndBytesConfig

    _emit(progress, "imports", "completed", torch=str(torch.__version__))
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    )
    try:
        _emit(progress, "quantized_load", "started")
        model = AutoModelForMultimodalLM.from_pretrained(
            source,
            local_files_only=True,
            dtype=torch.bfloat16,
            device_map={"": 0},
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
            quantization_config=quantization_config,
        )
        if not getattr(model, "is_loaded_in_4bit", False):
            raise RuntimeError("export source did not become a bitsandbytes 4-bit model")
        footprint = int(model.get_memory_footprint())
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        _emit(
            progress,
            "quantized_load",
            "completed",
            model_footprint_bytes=footprint,
            free_vram_mib=int(free_bytes // (1024 * 1024)),
            total_vram_mib=int(total_bytes // (1024 * 1024)),
        )

        _emit(progress, "save", "started", partial=str(partial.relative_to(PROJECT_ROOT)))
        partial.mkdir(parents=True, exist_ok=False)
        model.save_pretrained(
            partial,
            safe_serialization=True,
            max_shard_size=args.max_shard_size,
        )
        processor = AutoProcessor.from_pretrained(processor_source, local_files_only=True)
        processor.save_pretrained(partial)
        weight_files = sorted(partial.glob("*.safetensors"))
        if not weight_files or not (partial / "config.json").is_file():
            raise RuntimeError("serialized NF4 checkpoint is incomplete")
        manifest = {
            "schema_version": "anchor.bnb-nf4-export.v1",
            "created_at": _now(),
            "source": str(source.relative_to(PROJECT_ROOT)),
            "source_weight_sha256": _sha256(source / "model.safetensors"),
            "quantization": {
                "type": "nf4",
                "double_quant": True,
                "compute_dtype": "bfloat16",
                "storage_dtype": "bfloat16",
            },
            "model_footprint_bytes": footprint,
            "weights": [
                {
                    "path": file.name,
                    "bytes": file.stat().st_size,
                    "sha256": _sha256(file),
                }
                for file in weight_files
            ],
        }
        (partial / "anchor_quantization_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        partial.replace(output)
        _emit(progress, "save", "completed", output=str(output.relative_to(PROJECT_ROOT)))
    except BaseException:
        _emit(progress, "export", "failed")
        if partial.exists():
            shutil.rmtree(partial, ignore_errors=True)
        raise
    finally:
        if "model" in locals():
            del model
        gc.collect()
        if "torch" in locals() and torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"export failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
