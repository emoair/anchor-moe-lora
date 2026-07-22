# 自然语言脚手架消费端

状态：仅合约 MVP。它不会加载模型、请求 Provider、占用 GPU、执行梯度更新，
也不会声称已经实现物理 KV 复用。

## 数据流

1. 每个文件只读取一次字节快照，并认证冻结的 Producer 配置、三个 Schema、
   Smoke 合约、Manifest sidecar、Manifest 与四个 JSONL 分区。
2. 每条记录必须通过闭合 Schema。未知字段、不安全路径、符号链接、SHA 漂移、
   数量漂移或读取后的换档都会 fail-closed。消费端会用 `jsonschema` 的 Draft 2020-12
   validator 检查全部 20 条发布记录；本地结构校验只是纵深防御，不能替代发布 Schema。
3. 用 Projector Manifest SHA、源分区 SHA、规范化源行 SHA、任务 bundle、任务 ID、
   源 Gold、阶段、角色、语言、segment plan 和目标哈希，把每条 scaffold 与已认证
   TaskBoard 行交叉绑定。
4. 用户输入只能由 `build_training_view()` 生成。该硬过滤路径会移除 forbidden、
   current 与 future 正文。`scaffold_text` 是 assistant target，原阶段答案不会被复制
   到 prompt。
5. `json_only` 与 `concise_rationale_plus_json` 是两个物理、逻辑均隔离的消融组。
   同一 pair 的输入必须相同，target 随 variant 变化；未来训练启动器必须只选一个组。

## 两请求状态机

请求 1 使用冻结基座且 adapter 关闭。它可以输出简洁、可审计的理由摘要、严格路由
JSON、工具轨迹与专家 trigger 候选。严格验证并显式 commit 后，只晋升文本；Planner
私有 KV 永远不会转交给专家。

已 commit 的 scaffold 必须由 adapter-off 的冻结基座重新编码，产生新的不可变 lineage。
只有请求 2 才能激活所选专家，而且必须先绑定 tokenizer 身份与具体 adapter attestation。
同请求切换和根据生成中的 trigger 中途切换仍被禁止。

当前 synthetic fixture 的 tokenizer/cache identity 未绑定，没有 adapter 产物、重新编码
证明，并且 `execution_authorized=false`。因此它只能物化请求 1 的合约视图；请求 2 与
梯度训练继续 fail-closed。

## 复现低内存预检

```powershell
$env:PYTHONPATH="$PWD\src"
python scripts\research\preflight_natural_language_scaffold_consumer.py `
  --expected-consumer-config-sha256 `
  79cf993e4f4496b57786602bcbec3ac9048d4ad2a9fd6d5033bff64ab65c0640
```

命令只输出不含正文的计数与哈希，不会输出样本正文或 heldout 数据。预期配置哈希是
必填参数；若消费端配置发生变化但启动锁没有被明确更新，预检会 fail-closed。

## 训练烟测前仍缺少的门

- 冻结 formal-v3 release lock；
- tokenizer、chat template、trigger text 与 ordered tokens 的精确身份；
- 真实 adapter 文件与 tensor inventory attestation；
- committed scaffold 重新编码收据与不可变 lineage；
- 显式单一消融组训练配置；
- 正确性、内存、吞吐和质量评测。

`q_only`、`q_plus_o`、`wide_lora` 只是实验标签，不证明真实训练 tensor，
也不构成执行授权。
