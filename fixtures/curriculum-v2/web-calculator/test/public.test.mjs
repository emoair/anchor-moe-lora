import test from "node:test";
import assert from "node:assert/strict";
import { calculate } from "../src/calculator.mjs";

test("supports the four declared operations", () => {
  assert.equal(calculate("1.5", "+", "2.25"), 3.75);
  assert.equal(calculate(7, "-", 2), 5);
  assert.equal(calculate(3, "*", 4), 12);
  assert.equal(calculate(8, "/", 2), 4);
});

test("returns the frozen division error", () => {
  assert.equal(calculate(8, "/", 0), "Cannot divide by zero");
});
