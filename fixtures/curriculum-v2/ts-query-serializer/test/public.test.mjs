import test from "node:test";
import assert from "node:assert/strict";
import { toQuery } from "../src/query.ts";

test("sorts keys, repeats arrays, and encodes values", () => {
  const input = { z: "a b", a: [2, 1], skip: undefined, empty: null };
  assert.equal(toQuery(input), "a=2&a=1&empty=&z=a+b");
  assert.deepEqual(input.a, [2, 1]);
});

test("serializes booleans and empty arrays", () => {
  assert.equal(toQuery({ enabled: false, none: [] }), "enabled=false");
});
