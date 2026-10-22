"""Shared fixtures: settings that point at a temporary data root, and small simulator contexts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from reactive_lob_sim.settings import Settings, load_settings
from reactive_lob_sim.simctx import SimContext


def with_updates(settings: Settings, **sections: dict[str, Any]) -> Settings:
    """Copy ``settings`` with field updates inside the named sections."""
    update = {}
    for name, fields in sections.items():
        section = getattr(settings, name)
        update[name] = section.model_copy(update=fields) if hasattr(section, "model_copy") else fields
    return settings.model_copy(update=update)


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("TRADEQ_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.chdir(tmp_path)  # no .env from the repository is picked up
    return load_settings()


@pytest.fixture
def ctx(settings: Settings) -> SimContext:
    return SimContext.from_settings(settings)
