import { readFile } from "node:fs/promises";

const source = await readFile(new URL("../src/status-list.js", import.meta.url), "utf8");
if (source.includes("eval(") || source.includes("var ")) {
  throw new Error("disallowed source pattern");
}
if (!source.endsWith("\n")) {
  throw new Error("source must end with a newline");
}
