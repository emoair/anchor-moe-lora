# Deterministic JSON config merge

The CLI accepts two JSON object file paths, recursively merges the second into the first,
and prints compact JSON with sorted keys. Objects merge recursively; arrays and scalars
replace. Inputs must remain unchanged. Non-object roots print `error: object root required`
and exit 2.
