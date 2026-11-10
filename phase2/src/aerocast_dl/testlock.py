"""Test lock (rule G5): Nov-Dec labels are read only by the final stage.

``require_final`` passes only when the final stage was requested explicitly
(``run_phase2.py --final`` sets ``AEROCAST_FINAL=1`` for its own process) and
the protocol files have no uncommitted changes, so the rules in force at the
test run are the committed ones.
"""
from __future__ import annotations

import os
import subprocess

from . import config, guards

PROTOCOL_FILES = ["docs/development_protocol.md", "docs/leakage_safeguards.md"]
ENV = "AEROCAST_FINAL"


def protocol_is_committed() -> bool:
    res = subprocess.run(["git", "status", "--porcelain", "--", *PROTOCOL_FILES],
                         cwd=config.ROOT, capture_output=True, text=True, check=True)
    return res.stdout.strip() == ""


def require_final() -> None:
    if os.environ.get(ENV) != "1":
        raise guards.LeakageError("test labels requested outside the final stage (--final)")
    if not protocol_is_committed():
        raise guards.LeakageError("protocol files have uncommitted changes; commit them first")
