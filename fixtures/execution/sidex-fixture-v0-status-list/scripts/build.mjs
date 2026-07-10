import { readFile } from "node:fs/promises";

const source = await readFile(new URL("../src/status-list.js", import.meta.url), "utf8");
if (!source.includes("export function sortStatusRows")) {
  throw new Error("missing public export");
}
