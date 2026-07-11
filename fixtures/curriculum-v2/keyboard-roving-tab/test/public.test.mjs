import test from "node:test";
import assert from "node:assert/strict";
import { moveFocus } from "../src/roving-tab.mjs";

const base = [
  { id: "a", disabled: false, tabIndex: 0 },
  { id: "b", disabled: true, tabIndex: -1 },
  { id: "c", disabled: false, tabIndex: -1 },
];

test("moves, wraps, and skips disabled items", () => {
  assert.deepEqual(moveFocus(base, "ArrowRight").map((x) => x.tabIndex), [-1, -1, 0]);
  assert.deepEqual(moveFocus(base, "ArrowLeft").map((x) => x.tabIndex), [-1, -1, 0]);
  assert.deepEqual(moveFocus(base, "End").map((x) => x.tabIndex), [-1, -1, 0]);
});

test("unsupported key leaves values unchanged but returns a new array", () => {
  const result = moveFocus(base, "Enter");
  assert.deepEqual(result, base);
  assert.notEqual(result, base);
});
