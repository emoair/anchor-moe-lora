import assert from "node:assert/strict";
import test from "node:test";

import { sortStatusRows } from "../src/status-list.js";

const rows = [
  { id: "a", priority: 10 },
  { id: "b", priority: 2 },
  { id: "c", priority: 10 },
  { id: "d", priority: 1 },
];

test("sorts numeric priority ascending without mutation", () => {
  const input = rows.map((row) => ({ ...row }));
  const before = structuredClone(input);
  const result = sortStatusRows(input, "asc");

  assert.deepEqual(result.map((row) => row.id), ["d", "b", "a", "c"]);
  assert.deepEqual(input, before);
  assert.notStrictEqual(result, input);
  assert.strictEqual(result[2], input[0]);
});

test("keeps equal priorities stable when descending", () => {
  assert.deepEqual(sortStatusRows(rows, "desc").map((row) => row.id), ["a", "c", "b", "d"]);
});

test("rejects an unsupported direction", () => {
  assert.throws(() => sortStatusRows(rows, "sideways"), TypeError);
});
