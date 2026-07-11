import test from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtempSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";

test("recursively merges objects and replaces arrays", () => {
  const dir = mkdtempSync(join(tmpdir(), "cv2-merge-"));
  const left = join(dir, "left.json");
  const right = join(dir, "right.json");
  writeFileSync(left, JSON.stringify({ z: 1, nested: { a: 1, list: [1] } }));
  writeFileSync(right, JSON.stringify({ nested: { b: 2, list: [9] }, x: true }));
  const python = process.platform === "win32" ? ["py", "-3"] : ["python3"];
  const result = spawnSync(python[0], [...python.slice(1), "src/merge_config.py", left, right], { encoding: "utf8" });
  assert.equal(result.status, 0);
  assert.equal(result.stdout.trim(), '{"nested":{"a":1,"b":2,"list":[9]},"x":true,"z":1}');
});

test("rejects non-object roots", () => {
  const dir = mkdtempSync(join(tmpdir(), "cv2-merge-"));
  const left = join(dir, "left.json");
  const right = join(dir, "right.json");
  writeFileSync(left, "[]");
  writeFileSync(right, "{}");
  const python = process.platform === "win32" ? ["py", "-3"] : ["python3"];
  const result = spawnSync(python[0], [...python.slice(1), "src/merge_config.py", left, right], { encoding: "utf8" });
  assert.equal(result.status, 2);
  assert.equal(result.stderr.trim(), "error: object root required");
});
