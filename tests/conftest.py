"""Shared pytest configuration."""

import os

# Keep the heavy-artifact root out of the user's real DATA_ROOT during tests unless a
# test sets it explicitly; individual tests use tmp_path through Paths(...).
os.environ.setdefault("PANDA_IL_TEST", "1")
