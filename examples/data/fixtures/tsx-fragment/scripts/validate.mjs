import { readFileSync } from "node:fs";

const mode = process.argv[2];
const source = readFileSync(new URL("../submission.tsx", import.meta.url), "utf8");

function fail(message) {
  process.stderr.write(`${message}\n`);
  process.exit(1);
}

function balancedDelimiters(value) {
  const pairs = new Map([[")", "("], ["]", "["], ["}", "{"]]);
  const stack = [];
  let quote = null;
  let escaped = false;
  let lineComment = false;
  let blockComment = false;
  for (let index = 0; index < value.length; index += 1) {
    const char = value[index];
    const next = value[index + 1];
    if (lineComment) {
      if (char === "\n") lineComment = false;
      continue;
    }
    if (blockComment) {
      if (char === "*" && next === "/") {
        blockComment = false;
        index += 1;
      }
      continue;
    }
    if (quote) {
      if (escaped) {
        escaped = false;
      } else if (char === "\\") {
        escaped = true;
      } else if (char === quote) {
        quote = null;
      }
      continue;
    }
    if (char === "/" && next === "/") {
      lineComment = true;
      index += 1;
      continue;
    }
    if (char === "/" && next === "*") {
      blockComment = true;
      index += 1;
      continue;
    }
    if (["'", "\"", "`"].includes(char)) {
      quote = char;
      continue;
    }
    if (["(", "[", "{"].includes(char)) stack.push(char);
    if (pairs.has(char) && stack.pop() !== pairs.get(char)) return false;
  }
  return !quote && !lineComment && !blockComment && stack.length === 0;
}

function balancedJsxTags(value) {
  const stack = [];
  const tags = value.matchAll(/<\/?([A-Za-z][A-Za-z0-9.-]*)(?:\s[^<>]*)?\s*\/?>/g);
  for (const match of tags) {
    const full = match[0];
    const name = match[1];
    if (full.startsWith("</")) {
      if (stack.pop() !== name) return false;
    } else if (!full.endsWith("/>") && !["input", "img", "br", "hr", "meta", "link"].includes(name)) {
      stack.push(name);
    }
  }
  return stack.length === 0;
}

function assertFragment() {
  if (source.length < 40 || source.length > 12000) fail("invalid artifact size");
  if (/```/.test(source)) fail("markdown fence leaked into artifact");
  if (/<\s*script\b|javascript\s*:|\beval\s*\(|\bnew\s+Function\b/i.test(source)) {
    fail("active execution form is forbidden");
  }
  if (!/\b(?:export\s+)?(?:default\s+)?function\s+[A-Za-z_$]/.test(source)) {
    fail("component function missing");
  }
  if (!/\breturn\s*(?:\(|<)/.test(source)) fail("component JSX return missing");
  if (!balancedDelimiters(source)) fail("unbalanced JavaScript delimiters");
  if (!balancedJsxTags(source)) fail("unbalanced JSX tags");
}

if (mode === "build") {
  assertFragment();
} else if (mode === "test") {
  assertFragment();
  if (!/<[A-Za-z][A-Za-z0-9.-]*/.test(source)) fail("JSX element missing");
} else {
  fail("unknown validation mode");
}
