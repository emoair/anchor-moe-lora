# Stable status-list sorting

Repair `src/status-list.js` so `sortStatusRows(rows, direction)` returns a new array sorted by numeric priority.

Constraints:

- `direction` is exactly `"asc"` or `"desc"`; reject other values with `TypeError`.
- Equal-priority rows retain their original relative order in both directions.
- Do not mutate the input array or row objects.
- Keep the exported function name and module format unchanged.
- Make the smallest maintainable change and run build, test, and lint.
