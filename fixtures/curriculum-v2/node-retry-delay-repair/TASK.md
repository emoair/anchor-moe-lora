# Repair retry delay calculation

Repair `retryDelay`. Attempt zero returns `baseMs`; each later attempt doubles the prior
delay, capped at `maxMs`. Inputs must be non-negative safe integers, `baseMs <= maxMs`,
and unsafe multiplication must cap rather than overflow. Invalid inputs throw `RangeError`.
