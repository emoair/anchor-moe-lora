# Planner-first Coding-specialist 22,200 v1

这是一个加法式迁移合同，不是 provider 启动或训练授权。

22,200 固定为问题对象总数：19,000 个旧 coding 候选对象，加上 3,200 个单独物化的规划师 bootstrap 对象；它不等同于 provider 请求数、专家投影行数或训练行数。

严格顺序为：

```text
源资产登记 -> 规划师训练 -> 规划师冻结（Q-only 主臂）
-> 冻结规划师的路由/问题提交 -> 专家数据投影 -> 专家叶子发布
```

旧 SWE-bench 的 19,008 条公共导出仅是候选源。未来的选择清单必须明确选出 19,000 条、记录 8 条排除 ID、带独立的来源条款接受证据，并保持 body-free。现有 320 条或 1,000 条 fixture 不能冒充 3,200 条规划师 bootstrap 资产。

未来 source gate 还必须有本地化/翻译证据与真实工具轨迹；候选计数或旧的 launch-ready 标记都不能替代这些 provenance 条件。

最终发布遵循单向内容寻址：

```text
Teacher FINAL -> Producer attestation -> Consumer acceptance -> Training-input authority
```

Teacher FINAL 可绑定规范化 semantic root，但不能反向包含后续 attestation 的物理 SHA，避免 hash-DAG 循环。

在两个 source manifest 和规划师训练/冻结证据都完成认证前，该模块保持非 live：不产生 provider 请求、模型加载、GPU 运行或训练授权。
