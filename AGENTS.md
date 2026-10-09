# Agent entry point

1. **Read `docs/agent_execution_guide.md` first.** It is the single source of what is approved, in what order, and how each item is validated. Do not start work that is not in its queue.
2. **What the project is and why:** `docs/00_thesis.md`. The contracts the guide points at are in `docs/02_data_sources.md` and `docs/03_eval_harness.md`. Read the sections an item names before writing code.
3. **Findings, open decisions, maintainer actions and deferred work** live in `docs/ongoing_general_errors.md`. **Never fill in a `Your selection: _____` line.** It belongs to the maintainer.
4. **Commits:** one item, one Conventional Commit on `main` (no branches), scope = item id (`feat(a3): …`), with the WHY and the red/green runs in the body. Push with `/usr/bin/git push origin main`.
5. **Read exit codes bare, falsify every gate, open every artefact you produce, and never loosen a bar to pass it.**
6. **The v0 pipeline is archived at tag `v0-saf-final`.** Read it for reference; never restore it. Its lessons are in `docs/LESSONS_v0.md`.
