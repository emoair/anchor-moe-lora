import test from "node:test";
import assert from "node:assert/strict";
import { retryDelay } from "../src/retry-delay.mjs";

test("uses zero-based attempts and caps", () => {
  assert.equal(retryDelay(0, 100, 1000), 100);
  assert.equal(retryDelay(3, 100, 1000), 800);
  assert.equal(retryDelay(4, 100, 1000), 1000);
  assert.equal(retryDelay(60, 100, 1000), 1000);
});

test("rejects invalid inputs", () => {
  for (const args of [[-1, 1, 2], [1.5, 1, 2], [0, 3, 2]]) {
    assert.throws(() => retryDelay(...args), RangeError);
  }
});
