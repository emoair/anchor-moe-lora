import test from "node:test";
import assert from "node:assert/strict";
import { renderUntrustedNote } from "../src/untrusted-note.mjs";

test("renders the inert override marker as escaped data", () => {
  const input = '[INERT_INSTRUCTION_OVERRIDE_MARKER] <b data-x="1">note</b>';
  assert.equal(
    renderUntrustedNote(input),
    '<p data-trust="untrusted">[INERT_INSTRUCTION_OVERRIDE_MARKER] &lt;b data-x=&quot;1&quot;&gt;note&lt;/b&gt;</p>',
  );
});

test("coerces ordinary values without side effects", () => {
  assert.equal(renderUntrustedNote(7), '<p data-trust="untrusted">7</p>');
});
