export function sortStatusRows(rows, direction) {
  return rows.sort((left, right) => {
    const delta = String(left.priority).localeCompare(String(right.priority));
    return direction === "desc" ? -delta : delta;
  });
}
