# GLM-5.2 按服务渠道计价

[English](glm52_provider_pricing.md) | [简体中文](glm52_provider_pricing.zh-CN.md)

GLM-5.2 并不存在一份可跨渠道套用的全局 token 单价。因此，面板先用
`base_url + protocol` 精确确定服务渠道，再解析该渠道明确列出的请求模型别名；绝不只凭
模型名称选择价格。

2026-07-14 审计的元数据记录如下：

- 智谱按量 API：输入 8 元/百万 tokens、输出 28 元/百万 tokens、缓存命中
  2 元/百万 tokens。官方缓存存储优惠的单位是“元/百万 tokens/小时”，所以单独记录为
  当日限时免费，**不会**伪装成 `cache_write` token 单价 0。只要样本出现正数
  `cache_write` 用量，总成本就保持不可用。
- 火山方舟 Coding Plan：计费模式为 `subscription_quota`。尚未配置项目级套餐分摊规则，
  因此边际 token 成本保持不可用；不得把智谱按量单价套在 Ark 用量上。

Ark 只把已观测到的请求 ID `glm-5-2-260617` 映射到规范模型 `glm-5.2`。在没有服务商
证据前，不把规范名称宣称为 Ark 可请求别名。

官方来源：[智谱价格页](https://bigmodel.cn/pricing)；
[火山方舟 Coding Plan](https://www.volcengine.com/activity/codingplan)。

机器可读元数据位于
`src/anchor_mvp/integrations/ccswitch_metadata/fixtures/provider_pricing_2026_07_14.json`。
它是本项目自有的审计叠加层，不会伪称这些数据来自固定的 CC Switch v3.16.5 快照。
