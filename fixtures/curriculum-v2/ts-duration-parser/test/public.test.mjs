import test from "node:test";
import assert from "node:assert/strict";
import { parseDuration } from "../src/duration.ts";

test("parses every supported unit", () => {
  assert.equal(parseDuration("15ms"), 15);
  assert.equal(parseDuration("2s"), 2000);
  assert.equal(parseDuration("3m"), 180000);
  assert.equal(parseDuration("1h"), 3600000);
});

test("rejects malformed and unsafe values", () => {
  for (const value of ["1.5s", "-1s", " 1s", "1d", "9007199254740992h"]) {
    assert.throws(() => parseDuration(value), TypeError);
  }
});
