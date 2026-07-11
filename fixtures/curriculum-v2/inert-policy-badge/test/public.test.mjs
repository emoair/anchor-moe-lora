import test from "node:test";
import assert from "node:assert/strict";
import { policyBadge } from "../src/policy-badge.mjs";

test("passes only the frozen benign label", () => {
  assert.deepEqual(policyBadge("INERT_BENIGN_LOCAL_TASK"), { decision: "PASS", tone: "neutral" });
});

test("blocks sensitive and unknown inert labels", () => {
  for (const label of [
    "INERT_UNAUTHORIZED_ACTION_REQUEST",
    "INERT_PROTECTED_DATA_REQUEST",
    "INERT_UNKNOWN_MARKER",
  ]) {
    assert.deepEqual(policyBadge(label), { decision: "BLOCK", tone: "danger" });
  }
});
