import test from "node:test";
import assert from "node:assert/strict";
import { renderLabel } from "../src/render-label.mjs";

test("escapes all frozen metacharacters", () => {
  assert.equal(
    renderLabel(`<&>"'`),
    "<span>&lt;&amp;&gt;&quot;&#39;</span>",
  );
});

test("renders benign primitives", () => {
  assert.equal(renderLabel(42), "<span>42</span>");
});
