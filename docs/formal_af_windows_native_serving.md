# Formal A–F Windows native serving / Windows 原生推理服务

This backend keeps the exact registered bitsandbytes NF4 base resident on one
Windows CUDA GPU and permits at most one PEFT adapter at a time. It implements
the same OpenAI and runtime-LoRA endpoints used by the formal vLLM launcher, so
the benchmark DAG and metrics code do not change.

此后端在 Windows CUDA 上常驻同一份已登记的 bitsandbytes NF4 基座，并且任意时刻
最多只加载一个 PEFT LoRA。它复用了正式 vLLM 路线的 OpenAI 与动态 LoRA 接口，
因此 DAG、heldout 门禁和指标实现无需改写。

The server verifies all A–F registries, every indexed adapter file, the processor
and chat template, the NF4 manifest, and every 4-bit shard before importing CUDA.
`/v1/models` intentionally lists only `gemma4-12b-base-q4`; adapter selection is
performed through the localhost-only load/unload endpoints.

启动前会核验 A–F 全部 registry、已索引的适配器文件、训练时 processor/chat template、
NF4 manifest 以及每个 Q4 分片。`/v1/models` 故意只暴露
`gemma4-12b-base-q4`，LoRA 选择只能通过本机 load/unload 接口完成。

```powershell
# Metadata and full-file hashes only; does not import CUDA or open heldout cases.
.\scripts\serve\start_formal_af_serial_transformers.ps1 -PreflightOnly

# Start the single-GPU server. API key is optional and is read only from RAM.
$env:ANCHOR_VLLM_API_KEY = "local-random-secret"
.\scripts\serve\start_formal_af_serial_transformers.ps1 -MaxModelLength 2048
```

The formal benchmark launcher defaults to metadata-only preflight. Live heldout
execution still needs both explicit switches and a new output directory:

正式评测 launcher 默认只做元数据预检。读取 heldout 并执行仍必须显式提供两个开关和
全新的输出目录：

```powershell
.\scripts\benchmark\run_formal_partial_v1_af_windows_native.ps1

.\scripts\benchmark\run_formal_partial_v1_af_windows_native.ps1 `
  -Execute -AuthorizeHeldoutAccess `
  -OutputDir "D:\LLM\anchor-moe-lora\runs\formal-af-windows-001"
```

Useful local-only probes:

```powershell
$Headers = @{ Authorization = "Bearer $env:ANCHOR_VLLM_API_KEY" }
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/admin/probe -Headers $Headers
Invoke-RestMethod http://127.0.0.1:8000/v1/models -Headers $Headers
```

Do not bind this service to a public interface. Runtime LoRA mutation is an
administrative surface, and the launcher fixes the host to `127.0.0.1`.

不要把该服务绑定到公网接口。动态 LoRA 装卸属于管理面，launcher 已固定为
`127.0.0.1`。
