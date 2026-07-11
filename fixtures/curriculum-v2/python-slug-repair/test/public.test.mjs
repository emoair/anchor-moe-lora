import test from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";

const slug = (value) => {
  const code = "import sys; from src.slug import slugify; print(slugify(sys.argv[1]))";
  const python = process.platform === "win32" ? ["py", "-3"] : ["python3"];
  return spawnSync(python[0], [...python.slice(1), "-c", code, value], { encoding: "utf8" }).stdout.trim();
};

test("normalizes accents and separators", () => {
  assert.equal(slug("  Café + Déjà Vu  "), "cafe-deja-vu");
  assert.equal(slug("A___B...C"), "a-b-c");
});

test("uses a deterministic empty fallback", () => {
  assert.equal(slug("東京"), "item");
});
