# Typed duration parser

Complete `parseDuration`. Accept a non-negative integer followed immediately by `ms`,
`s`, `m`, or `h`; return milliseconds. Reject decimals, signs, whitespace, unknown units,
and values whose millisecond result exceeds `Number.MAX_SAFE_INTEGER` with `TypeError`.
