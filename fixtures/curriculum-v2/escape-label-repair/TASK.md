# Repair safe label rendering

Repair `renderLabel` so ordinary text is returned inside a `span` and the five HTML
metacharacters are escaped. The function must coerce numbers to strings. It must not
interpret, execute, or fetch anything from the label.
