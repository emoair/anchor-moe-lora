export function filterRows(rows, query) {
  return { rows, count: rows.length, query };
}
