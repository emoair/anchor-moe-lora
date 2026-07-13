# Teacher transport retry contract

The teacher client owns the only automatic wire retry loop. A logical seed or
task keeps the same system prompt, user prompt, task-card alignment, and output
identity across every wire attempt. A successful record is appended once by its
stable ID; a retry never creates a replacement question.

## Retry matrix

| Condition | Automatic retry | Rationale |
| --- | ---: | --- |
| `IncompleteRead`, remote disconnect, URL/connection interruption, transport timeout | At most 2 | The request may have ended before a complete response was available. |
| SSE EOF before `[DONE]` or Responses EOF before `response.completed` | At most 2 | Partial stream text is discarded and is never accepted as gold. |
| HTTP 408 or 499 | At most 2 | The request was interrupted or timed out in transit. |
| HTTP 500, 502, 503, 504, 520–524 | At most 2 | These statuses represent the explicitly supported transient server/proxy set. |
| HTTP 400, 409, 429, 501 and other non-transient statuses | No | Replaying a bad schema, conflict, rate/quota signal, or unsupported operation does not repair it. |
| Provider quota exhaustion, JSON/schema/safety validation failure | No | These require a new quota window or a pipeline/prompt correction, not request replay. |

Backoff is bounded exponential delay (nominally 1 then 2 seconds, with small
jitter and an 8-second cap). Successful record provenance includes
`wire_attempts`, `retry_count`, `max_retries`, and redacted `retry_reasons` from
the same request-local context.

## Recommended next collection profile

Use
`configs/data/automation.full_v3.ark_glm52.max384.c10.retry2.yaml` for the next
controlled collection. One process owns concurrency 10. When several shards
are scheduled together, their summed active concurrency must remain at or below
30. The 6,912 request budget is the strict worst-case envelope for 2,304 logical
calls with two retries each; it is not a target to consume.

The profile deliberately keeps OpenAI Responses streaming enabled. With a 128K
output ceiling, non-streaming would buffer the entire result and can move the
failure point to an intermediary timeout. Streaming has now been hardened to
require the provider's terminal event. Do not switch this production profile to
non-streaming until a controlled, credential-safe A/B probe verifies complete
terminal behavior, output equivalence, latency, and usage accounting against the
actual provider endpoint.
