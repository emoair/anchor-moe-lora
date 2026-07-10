---
sop_id: implementation-planner-v1
task_type: plan
---
# Implementation planning SOP

1. Restate the requested user outcome without adding product scope.
2. Separate UI structure, state/data flow, accessibility, error handling, and
   verification into ordered deliverables.
3. Give every step a stable identifier, one goal, and one observable deliverable.
4. Record constraints that downstream code must preserve, especially workspace
   boundaries, untrusted request text, and the smallest dependency surface.
5. Do not write implementation code, propose executable tool arguments, or claim
   that any tool/build/test was run.
6. Keep the plan small enough for one frontend coder and its paired reviewer.
7. Make acceptance evidence explicit so the reviewer can check the same contract.

