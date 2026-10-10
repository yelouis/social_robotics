"""Detach a long-running command from the caller so it survives the harness
reaping its background tasks. Double-fork + setsid -> new session, reparented to
launchd (PPID 1); stdout/stderr appended to a log. Usage:
    daemonize.py <logfile> <cmd> [args...]
"""
import os
import sys

logfile = sys.argv[1]
cmd = sys.argv[2:]
if not cmd:
    sys.exit("daemonize.py: no command given")

if os.fork() > 0:            # parent returns immediately to the caller
    os._exit(0)
os.setsid()                  # new session: escapes the caller's process group
if os.fork() > 0:            # first child exits; grandchild is the daemon
    os._exit(0)

fd = os.open(logfile, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
os.dup2(fd, 1)
os.dup2(fd, 2)
os.dup2(os.open(os.devnull, os.O_RDONLY), 0)
os.execvp(cmd[0], cmd)
