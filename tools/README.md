# tools/

Operational helpers for long unattended runs on the Mac Studio. Neither is
research logic; they wrap any resumable runner.

| tool | what it does | usage |
|---|---|---|
| `daemonize.py` | Detach a command into its own session (double-fork + `setsid`, reparented to launchd, PPID 1) so it survives the agent harness reaping background tasks after ~1–2 h. Appends stdout/stderr to a log. | `./venv/bin/python tools/daemonize.py <logfile> <cmd> [args...]` |
| `run_supervised.sh` | Relaunch a resumable runner until it exits 0, under `caffeinate` + `PYTHONFAULTHANDLER`; aborts after 2 relaunches that add no records (poison-item guard). | `tools/run_supervised.sh <result_json> <runner command...>` |

Multi-hour job = both, nested: `daemonize.py` → `run_supervised.sh` → runner.
Verify detachment with `ps -Ao pid,ppid,command | grep run_supervised` (PPID
must be 1). Background on why: `docs/LESSONS_v0.md`, "Operations".
