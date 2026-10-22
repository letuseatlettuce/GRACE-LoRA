# AGENTS.md

This repository is an **agent skill**: it adapts the GRACE foundation interatomic potential to a
new domain with a LoRA adapter on a laptop CPU, then validates the result and writes a report.
The workflow is plain Markdown instructions plus Python command-line tools, so **any coding agent that can read
files and run shell commands can follow it**: Claude Code, Codex, Cursor, Gemini CLI, Aider,
OpenHands, local models with tool use, and so on.

## Where the instructions are
- **Main workflow:** [`.claude/skills/grace-lora/SKILL.md`](.claude/skills/grace-lora/SKILL.md).
  Read it fully before acting, and follow it step by step, starting with **Step 0: settings
  review**: show the user `python3 pipeline.py settings` and wait for their confirmation.
- **References**, read when needed: `.claude/skills/grace-lora/references/`:
  - `settings.md`: every setting explained;
  - `commands.md`: the CLI reference;
  - `choosing-a-teacher.md`, `method.md`, `interpreting-results.md`, `agent-options.md`,
    `troubleshooting.md`.
- **Entry point:** `python3 pipeline.py` (standard library only; it creates the `.venv` itself).

## Rules that apply to every agent
- **Tested scope:** only GRACE models as student and teacher (student
  `GRACE-2L-OMAT-medium-ft-AM`, teacher `GRACE-OFF-2L-medium`, CPU-only laptop). Other GRACE
  models are untested; non-GRACE models (MACE, PyTorch) are not supported.
- **Ask before** multi-GB downloads, environment creation, or any run longer than ~30 minutes.
- **Never edit the scripts to get past an error.** Use `references/troubleshooting.md`, or report
  the error to the user.
- **Stop long runs by killing the process, never by pausing or suspending it.** Experiments are
  resumable from the ledger.
- **No subagents or parallel agents** unless the user enabled `agent_options.subagents`
  (`python3 pipeline.py options`). The default workflow needs only one agent.

## Claude-Code-specific parts (safe to skip in other agents)
- **Skill discovery:** Claude Code loads `.claude/skills/grace-lora/` automatically. Other agents
  start here.
- **The optional `critic` feature** launches a reviewer subagent. If your agent cannot launch
  subagents, leave `critic` off (the default). Everything else (`self_heal`, `predictions`,
  `cockpit`) is plain CLI and works in any agent.
