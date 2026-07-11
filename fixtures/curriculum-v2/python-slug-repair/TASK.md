# Repair Unicode slug generation

Repair `slugify`. Normalize Unicode with NFKD, remove combining marks, use lowercase
ASCII letters and digits, collapse every other run to one hyphen, and trim hyphens.
Return `item` when no ASCII alphanumeric characters remain.
