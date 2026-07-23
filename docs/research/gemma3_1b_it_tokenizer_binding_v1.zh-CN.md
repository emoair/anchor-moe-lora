# Gemma 3 1B IT 分词器绑定（diagnostic v1）

这份 receipt 用来消除本地 Gemma 3 1B IT 导出包的模板歧义。它会认证导出包
五个文件、显式修正导出配置中的 BOS/EOS 对调、调用仓库真实的五角色因果
materializer，并对 1000 条记录做完整预检。全过程不加载模型，也不请求 GPU。

它只是 diagnostic 前置条件，**不代表**训练已获授权，也不代表正式结论。

## 一条命令

在仓库根目录执行：

```powershell
& "$env:USERPROFILE\.conda\envs\anchor-mvp\python.exe" scripts\research\build_gemma3_tokenizer_binding_v1.py --publish
```

默认本地模型目录：

```text
D:\LLM\models\google-gemma-3-1b-it-keras-v3\hf-export-keras-hub-0.29.1-bf16
```

如果同一份已认证文件放在其他位置，可只为本次启动设置内存环境变量：

```powershell
$env:ANCHOR_GEMMA3_1B_IT_HF_PATH = "D:\path\to\hf-export"
& "$env:USERPROFILE\.conda\envs\anchor-mvp\python.exe" scripts\research\build_gemma3_tokenizer_binding_v1.py --publish
Remove-Item Env:\ANCHOR_GEMMA3_1B_IT_HF_PATH
```

发布采用原子、禁止覆盖策略。输出已存在时会停下，绝不会覆盖旧证据。复审命令：

```powershell
& "$env:USERPROFILE\.conda\envs\anchor-mvp\python.exe" scripts\research\audit_gemma3_tokenizer_binding_v1.py
```

## 精确序列化

导出的 `config.json` 写成 BOS=1/EOS=2，但已认证的 SentencePiece 和 tokenizer
元数据实际是 BOS=2/EOS=1。我们不修改原文件，只施加运行时叠加层：

```text
[BOS=2]
SP("<start_of_turn>user\n" + prompt + "<end_of_turn>")
[EOS=1]
SP("\n<start_of_turn>model\n" + target + "<end_of_turn>")
[EOS=1]
SP("\n")
```

字面量 `<bos>`、`<eos>` 绝不会送入 SentencePiece。prompt 和 assistant 前缀的
label 都是 `-100`；target、`<end_of_turn>`、EOS 参与训练；末尾分隔换行继续
mask。每一条结构化序列还必须与本地 HF tokenizer 在
`fix_mistral_regex=false` 时的可见模板编码逐 token 相等。

文本结构绑定 Google 官方文档：

- [Gemma prompt structure](https://ai.google.dev/gemma/docs/core/prompt-structure)
- [Gemma PyTorch guide](https://ai.google.dev/gemma/docs/core/pytorch_gemma)

此契约不提供独立 system 角色；system 指令放进首个 user prompt。

## 为什么冻结 768，而不是 512

这批数据无法无损塞进 512：1000 条里有 514 条超过 512，security 角色最大
达到 665 token。因此 Gemma diagnostic 配置在任何 GPU 运行前就冻结为 768，
并明确禁止截断；这不是训练过程中临时加长。

manifest 只写每角色、每分区的汇总统计和有序序列摘要，不写 prompt、target、
record ID 或原始 token ID 数组。

## 固定文件

- 配置：`configs/research/gemma3_1b_it_tokenizer_binding_v1.yaml`
- 模板策略：`configs/research/gemma3_1b_it_chat_template_policy_v1.json`
- 实现：`src/anchor_mvp/training/gemma3_tokenizer_binding_v1.py`
- 输出：`fixtures/research/gemma3_1b_it_tokenizer_binding_v1`

构建过程不读取 Gold、heldout、provider 或网络来源。读取权重文件只是做字节
哈希认证，不等于加载模型。
