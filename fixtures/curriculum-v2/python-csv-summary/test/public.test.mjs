import test from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";

const python = process.platform === "win32" ? ["py", "-3"] : ["python3"];
const run = (input) => spawnSync(python[0], [...python.slice(1), "src/summarize.py"], { input, encoding: "utf8" });

test("summarizes normalized statuses deterministically", () => {
  const result = run("id,status\n1, Ready \n2,held\n3,READY\n4,\n");
  assert.equal(result.status, 0);
  assert.equal(result.stdout, "held=1\nready=2\nunknown=1\n");
});

test("reports a missing status column", () => {
  const result = run("id,name\n1,A\n");
  assert.equal(result.status, 2);
  assert.equal(result.stderr.trim(), "error: missing status column");
});
