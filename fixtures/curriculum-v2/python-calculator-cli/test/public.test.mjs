import test from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";

const python = process.platform === "win32" ? ["py", "-3"] : ["python3"];
const run = (...args) => spawnSync(python[0], [...python.slice(1), "src/calculator.py", ...args], { encoding: "utf8" });

test("calculates all operators with compact output", () => {
  assert.equal(run("7", "-", "2").stdout.trim(), "5");
  assert.equal(run("3", "*", "4").stdout.trim(), "12");
  assert.equal(run("8", "/", "2").stdout.trim(), "4");
});

test("rejects division by zero without traceback", () => {
  const result = run("1", "/", "0");
  assert.equal(result.status, 2);
  assert.equal(result.stderr.trim(), "error: division by zero");
  assert.doesNotMatch(result.stderr, /Traceback/);
});
