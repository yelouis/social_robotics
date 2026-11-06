from __future__ import annotations

import pytest
from shared.memguard import MemoryDeferred


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    outcome = yield
    _ = outcome.get_result()
    if call.excinfo is not None and call.excinfo.errisinstance(MemoryDeferred):
        pytest.exit(f"memory guard: deferred: {call.excinfo.value}", returncode=75)


@pytest.fixture(autouse=True)
def _memguard_log_in_tmp(tmp_path_factory, monkeypatch):
    import shared.memguard as memguard
    monkeypatch.setattr(memguard, "DATA_ROOT", tmp_path_factory.mktemp("memguard_data_root"))

