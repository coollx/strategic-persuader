# T-004 spine markdown rendering
status: done
context: The researcher asked for a concise spine that renders well: consecutive bare lines in the month files merged into one paragraph when rendered, and T-003 carried three lines for one outcome. The line format is checked by the commit gate, so format and gate change together as one /meta operation.
goal: Every spine line carries a leading markdown bullet, with the gate regex, status hook, AGENTS.md format spec, and task-skill template all updated to match; the three T-003 lines are merged into one [meta] line.
plan:
  1. [meta] gate regex, gate messages, gate id parsing (id field only, so summaries may reference other tasks), status-hook date extraction, AGENTS.md format spec, task-skill template updated — AGENTS.md
  2. [meta] docs/log/2026-08.md rewritten at researcher direction: all lines bulleted, three T-003 lines merged into one — docs/log/2026-08.md
