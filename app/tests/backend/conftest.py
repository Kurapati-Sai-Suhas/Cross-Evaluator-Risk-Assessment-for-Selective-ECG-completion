"""Shared fixtures for the app backend tests."""
import builtins
import os
import pathlib

import pandas as pd
import pytest


@pytest.fixture
def no_writes(monkeypatch):
    """Fail the test if anything tries to write a file while it is active (the app must not persist uploads)."""
    real_open = builtins.open

    def guarded_open(file, mode="r", *args, **kwargs):
        if any(c in mode for c in "wax+"):
            raise AssertionError(f"file write attempted: {file!r} ({mode})")
        return real_open(file, mode, *args, **kwargs)

    def refuse(*a, **k):
        raise AssertionError("file write attempted")

    monkeypatch.setattr(builtins, "open", guarded_open)
    monkeypatch.setattr(os, "replace", refuse)
    monkeypatch.setattr(pathlib.Path, "write_text", refuse)
    monkeypatch.setattr(pathlib.Path, "write_bytes", refuse)
    monkeypatch.setattr(pd.DataFrame, "to_csv", refuse)
    yield
