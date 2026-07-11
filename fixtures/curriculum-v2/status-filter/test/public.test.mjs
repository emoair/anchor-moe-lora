import test from "node:test";
import assert from "node:assert/strict";
import { filterRows } from "../src/status-filter.mjs";

test("filters case-insensitively with stable metadata", () => {
  const rows = [{ id: 1, status: "Ready" }, { id: 2, status: "held" }, { id: 3, status: "READY" }];
  const snapshot = structuredClone(rows);
  assert.deepEqual(filterRows(rows, "  ready "), {
    rows: [rows[0], rows[2]], count: 2, query: "ready",
  });
  assert.deepEqual(rows, snapshot);
});

test("empty query returns a new row array", () => {
  const rows = [{ id: 1, status: "Ready" }];
  const result = filterRows(rows, " ");
  assert.deepEqual(result, { rows, count: 1, query: "" });
  assert.notEqual(result.rows, rows);
});
