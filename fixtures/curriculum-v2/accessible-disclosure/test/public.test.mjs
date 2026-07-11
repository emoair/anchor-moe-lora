import test from "node:test";
import assert from "node:assert/strict";
import { toggleDisclosure } from "../src/disclosure.mjs";

test("toggles one known disclosure without mutation", () => {
  const input = [{ id: "profile", open: false }, { id: "alerts", open: true }];
  const output = toggleDisclosure(input, "profile");
  assert.deepEqual(output, [
    { id: "profile", open: true, ariaExpanded: "true" },
    { id: "alerts", open: true, ariaExpanded: "true" },
  ]);
  assert.deepEqual(input, [{ id: "profile", open: false }, { id: "alerts", open: true }]);
  assert.notEqual(output, input);
});

test("unknown id normalizes aria state without toggling", () => {
  assert.deepEqual(toggleDisclosure([{ id: "x", open: false }], "missing"), [
    { id: "x", open: false, ariaExpanded: "false" },
  ]);
});
