# Gemma 3 既有蒸馏 v14 验证资产 v1

该 additive 资产只把既有 v14 路由数据冻结为验证资产，不授权再次训练、部署、formal 评测或生产使用。

构建器逐字节认证 16,632 行源数据及 manifest、训练阶段 PASS 回执、整体尝试 FAIL 回执和冻结 source-snapshot manifest。训练阶段通过与 provenance 门失败被明确分开记录；`attempt_consumed` 保持 false。

每行只持久化 `sample_id`、`split`、`source_group_sha256`、`leakage_family_sha256`、`prompt_sha256` 和源 ordinal。prompt 只在流式扫描时用于复算其声明 digest，绝不写入资产。labels、target、messages、raw token IDs 与其他正文均禁止出现。

三分区策略固定为：

- `train`：已见训练集回归验证；
- `calibration`：当前验证；
- `holdout`：密封 single-blind 最终验证。

三者全部 `consumable_for_training=false`。构建器证明 13,131/1,923/1,578 行、全局 sample ID 唯一、按 sample ID 确定性排序，并证明 source-group、leakage-family、prompt 三个 digest 域跨 split 交集均为 0。

发布采用认证 staging、文件独占创建、强制 SHA-256 sidecar、rename 前 inventory/identity 末次重验、no-replace rename 和发布后复验。失败时保留 staging 诊断，不递归删除未绑定路径。
