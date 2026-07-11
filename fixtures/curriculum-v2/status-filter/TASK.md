# Stable status filter

Implement `filterRows`. Match status case-insensitively, preserve source order, never
mutate the input, and return `{ rows, count, query }` where query is trimmed and
lowercase. An empty query returns all rows.
