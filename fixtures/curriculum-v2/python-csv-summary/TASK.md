# CSV status summary CLI

Read CSV from stdin with a required `status` header. Print one `name=count` line per
case-folded status in alphabetic order. Trim surrounding whitespace. Empty statuses are
counted as `unknown`. A missing header prints `error: missing status column` and exits 2.
