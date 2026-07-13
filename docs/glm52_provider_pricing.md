# GLM-5.2 provider-scoped pricing

[English](glm52_provider_pricing.md) | [简体中文](glm52_provider_pricing.zh-CN.md)

GLM-5.2 does not have one globally interchangeable token price. The dashboard
therefore resolves an exact provider channel from `base_url + protocol` before it
resolves an exact request-model alias. It never selects a price from the model
name alone.

The audited metadata retrieved on 2026-07-14 records:

- Zhipu pay-as-you-go API: CNY 8 / million input tokens, CNY 28 / million
  output tokens, and CNY 2 / million cache-hit tokens. The official cache-storage
  promotion uses CNY / million tokens / hour, so it is recorded separately as
  promotional zero on that date. It is **not** converted into a zero
  `cache_write` token price. A sample with positive `cache_write` usage therefore
  has unavailable total cost.
- Volcengine Ark Coding Plan: `subscription_quota`. The dashboard keeps marginal
  token cost unavailable because no project-specific subscription allocation has
  been configured. It must not apply Zhipu's pay-as-you-go rates to Ark usage.

Only the observed Ark request ID `glm-5-2-260617` maps to canonical `glm-5.2`.
The canonical name is not advertised as an Ark request alias without provider
evidence.

Sources: [Zhipu pricing](https://bigmodel.cn/pricing) and
[Volcengine Coding Plan](https://www.volcengine.com/activity/codingplan).

The machine-readable metadata is
`src/anchor_mvp/integrations/ccswitch_metadata/fixtures/provider_pricing_2026_07_14.json`.
It is a project-owned audited overlay; it does not claim that these rows came from
the pinned CC Switch v3.16.5 snapshot.
