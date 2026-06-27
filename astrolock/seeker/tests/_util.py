"""Shared helpers for Seeker tests."""

import os
import pathlib
import shutil

# repo root: tests -> seeker -> astrolock -> <root>
REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
TEST_OUT_ROOT = REPO_ROOT / 'sessions' / '_tests'


def fresh_dir(name):
    """A clean, project-local output dir under sessions/_tests/ (gitignored, inspectable)."""
    d = TEST_OUT_ROOT / name
    shutil.rmtree(d, ignore_errors=True)
    os.makedirs(d, exist_ok=True)
    return str(d)
