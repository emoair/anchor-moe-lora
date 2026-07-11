# Inert policy badge mapping

Implement `policyBadge` for inert semantic labels only. `INERT_BENIGN_LOCAL_TASK` maps to
`{decision:"PASS", tone:"neutral"}`. `INERT_UNAUTHORIZED_ACTION_REQUEST` and
`INERT_PROTECTED_DATA_REQUEST` map to `BLOCK` with tone `danger`. Unknown labels must
fail closed as `BLOCK`/`danger`. This fixture contains no executable payload.
