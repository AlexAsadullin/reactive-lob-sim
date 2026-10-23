"""The scenario YAML files are a contract (6.5) that the Shadow Exchange (I6) parses strictly."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from reactive_lob_sim.scenarios.library import (
    default_scenarios,
    load_scenario,
    scenario_error,
    seed_library,
    validate_library,
)
from reactive_lob_sim.scenarios.models import DOCUMENT_KEYS, REQUIRED_PARAMS, Scenario, ShockSchedule

SHOCK_KEYS = {"kind", "start_s", "duration_s", "params"}

# A liquidation-cascade file as written by the schema before ``layers`` and ``layer_gap_bps`` existed.
OLD_CASCADE = """\
schema_version: '1'
id: liquidation-cascade
description: old
default_duration_s: 600.0
shocks:
- kind: liquidation_cascade
  start_s: 150.0
  duration_s: 120.0
  params: {oi_multiplier: 2.0, long_share: 0.9, trigger_depth_fraction: 0.6}
abm_overrides: {}
"""


def test_seeded_files_carry_every_required_field(tmp_path: Path) -> None:
    seed_library(tmp_path)
    paths = sorted(tmp_path.glob("*.yaml"))
    assert len(paths) == len(default_scenarios())
    for path in paths:
        doc = yaml.safe_load(path.read_text())
        assert set(doc) == DOCUMENT_KEYS, path.name  # default timing keys are not written
        for shock in doc["shocks"]:
            assert set(shock) == SHOCK_KEYS
            missing = REQUIRED_PARAMS[shock["kind"]] - set(shock["params"])
            assert not missing, f"{path.name} misses {sorted(missing)}"
    assert validate_library(tmp_path) == {}


def test_old_file_fails_validation_and_refresh_rewrites_it(tmp_path: Path) -> None:
    seed_library(tmp_path)
    old = tmp_path / "liquidation-cascade.yaml"
    old.write_text(OLD_CASCADE)
    custom = tmp_path / "my-own.yaml"
    custom.write_text(OLD_CASCADE.replace("id: liquidation-cascade", "id: my-own"))
    error = scenario_error(old)
    assert error is not None
    assert "layer_gap_bps" in error
    assert seed_library(tmp_path) == []  # without refresh nothing is overwritten
    assert set(validate_library(tmp_path)) == {old, custom}
    assert seed_library(tmp_path, refresh=True) == [old]
    assert set(validate_library(tmp_path)) == {custom}  # files with other ids are never touched
    assert seed_library(tmp_path, refresh=True) == []
    assert load_scenario(tmp_path, "liquidation-cascade").shocks[0].params["layers"] == 8


def test_time_scale_stretches_shocks_and_round_trips(tmp_path: Path) -> None:
    base = next(s for s in default_scenarios() if s.id == "funding-shock")
    assert base.scaled_shocks() == base.shocks
    assert "8-hour" in base.description
    long = Scenario.model_validate(
        {**base.to_document(), "id": "funding-shock-24h", "time_scale": 144.0, "time_offset_s": 60.0}
    )
    shock = long.scaled_shocks()[0]
    assert shock.start_s == pytest.approx(60.0 + 144.0 * 120.0)
    assert shock.duration_s == pytest.approx(144.0 * 300.0)
    assert long.effective_duration_s == pytest.approx(60.0 + 144.0 * 600.0)
    doc = long.to_document()
    assert doc["time_scale"] == 144.0
    assert doc["time_offset_s"] == 60.0
    path = tmp_path / "funding-shock-24h.yaml"
    path.write_text(yaml.safe_dump(doc))
    assert load_scenario(tmp_path, "funding-shock-24h") == long
    sched = ShockSchedule(long.scaled_shocks())
    assert sched.funding_override(60.0 + 144.0 * 200.0) == pytest.approx(0.003)
    assert sched.funding_override(200.0) is None
    with pytest.raises(ValueError, match="greater than 0"):
        Scenario.model_validate({**base.to_document(), "time_scale": 0.0})
