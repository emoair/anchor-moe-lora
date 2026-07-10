import { readFileSync } from "node:fs";

const mode = process.argv[2];
const html = readFileSync(new URL("../submission.html", import.meta.url), "utf8");
const expectation = JSON.parse(
  readFileSync(new URL("../expectation.json", import.meta.url), "utf8"),
);

function fail(message) {
  process.stderr.write(`${message}\n`);
  process.exit(1);
}

if (mode === "build") {
  if (html.length < 120 || html.length > 2_000_000) fail("invalid artifact size");
  if (!/<!doctype\s+html/i.test(html)) fail("doctype missing");
  if (!/<html\b/i.test(html) || !/<\/html>/i.test(html)) fail("html document incomplete");
  if (!/<main\b/i.test(html) || !/<\/main>/i.test(html)) fail("main landmark incomplete");
  if (/```/.test(html)) fail("markdown fence leaked into artifact");
} else if (mode === "test") {
  if (!html.includes(expectation.required_marker)) fail("known benign mutation not repaired");
  for (const marker of expectation.required_substrings) {
    if (!html.includes(marker)) fail("required held-out behavior missing");
  }
} else {
  fail("unknown validation mode");
}
