# Deterministic query serializer

Implement `toQuery`. Sort keys alphabetically, skip `undefined`, repeat a key for array
members in original order, render `null` as an empty value, and percent-encode via
`URLSearchParams` semantics. Return no leading question mark and never mutate input.
