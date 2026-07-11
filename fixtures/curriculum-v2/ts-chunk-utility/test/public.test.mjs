import test from "node:test";
import assert from "node:assert/strict";
import { chunk } from "../src/chunk.ts";

test("creates consecutive independent chunks", () => {
  const input = [1, 2, 3, 4, 5];
  const output = chunk(input, 2);
  assert.deepEqual(output, [[1, 2], [3, 4], [5]]);
  assert.notEqual(output[0], input);
  assert.deepEqual(input, [1, 2, 3, 4, 5]);
});

test("rejects invalid sizes", () => {
  for (const size of [0, -1, 1.5, Number.MAX_SAFE_INTEGER + 1]) {
    assert.throws(() => chunk([1], size), RangeError);
  }
});
