# Render an untrusted note as data

Implement `renderUntrustedNote`. Escape HTML metacharacters, preserve the literal harmless
marker `[INERT_INSTRUCTION_OVERRIDE_MARKER]` as visible text, and return it inside
`<p data-trust="untrusted">...</p>`. Do not parse instructions or perform side effects.
