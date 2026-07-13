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

function jsxOpeningContext(value, index, insideJsx) {
  const adjacent = value[index - 1] ?? "";
  if (/[A-Za-z0-9_$)\]]/.test(adjacent)) return false;
  if (insideJsx) return true;
  const before = value.slice(Math.max(0, index - 32), index);
  if (/(?:\breturn|=>)\s*$/.test(before)) return true;
  const previous = before.match(/\S(?=\s*$)/)?.[0] ?? "";
  return !previous || "([{=,:;!&|?>".includes(previous);
}

function scanJsxTags(value) {
  const tags = [];
  const contextStack = [];
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
      if (escaped) escaped = false;
      else if (char === "\\") escaped = true;
      else if (char === quote) quote = null;
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
    if (char !== "<") continue;

    let cursor = index + 1;
    const closing = value[cursor] === "/";
    if (closing) cursor += 1;
    // Fragments have no name and do not affect named-tag balancing.
    if (value[cursor] === ">") {
      index = cursor;
      continue;
    }
    const nameMatch = value.slice(cursor).match(/^([A-Za-z][A-Za-z0-9.-]*)/);
    if (!nameMatch) continue;
    const name = nameMatch[1];
    if (!closing && !jsxOpeningContext(value, index, contextStack.length > 0)) continue;
    cursor += name.length;

    let attributeQuote = null;
    let attributeEscaped = false;
    let braceDepth = 0;
    let end = -1;
    for (; cursor < value.length; cursor += 1) {
      const token = value[cursor];
      if (attributeQuote) {
        if (attributeEscaped) attributeEscaped = false;
        else if (token === "\\") attributeEscaped = true;
        else if (token === attributeQuote) attributeQuote = null;
        continue;
      }
      if (["'", "\"", "`"].includes(token)) {
        attributeQuote = token;
        continue;
      }
      if (token === "{") {
        braceDepth += 1;
        continue;
      }
      if (token === "}" && braceDepth > 0) {
        braceDepth -= 1;
        continue;
      }
      if (token === ">" && braceDepth === 0) {
        end = cursor;
        break;
      }
    }
    if (end < 0) return null;
    const selfClosing = /\/\s*$/.test(value.slice(index, end));
    tags.push({ closing, name, selfClosing });
    if (closing) contextStack.pop();
    else if (!selfClosing) contextStack.push(name);
    index = end;
  }
  return tags;
}

function balancedJsxTags(value) {
  const tags = scanJsxTags(value);
  if (!tags) return false;
  const stack = [];
  const voidTags = new Set([
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
    "param", "source", "track", "wbr",
  ]);
  for (const tag of tags) {
    if (tag.closing) {
      if (stack.pop() !== tag.name) return false;
    } else if (!tag.selfClosing && !voidTags.has(tag.name.toLowerCase())) {
      stack.push(tag.name);
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
  const functionComponent = /\b(?:export\s+)?(?:default\s+)?function\s+[A-Za-z_$]/.test(source);
  const arrowComponent = /\b(?:export\s+)?(?:const|let)\s+[A-Z][A-Za-z0-9_$]*(?:\s*:[^=\n]+)?\s*=\s*(?:async\s*)?(?:\([^)]*\)|[A-Za-z_$][A-Za-z0-9_$]*)\s*=>/.test(source);
  if (!functionComponent && !arrowComponent) {
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
