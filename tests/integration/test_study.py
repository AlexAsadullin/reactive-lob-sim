"""End to end: a shrunken report study (calibration, simulators, gap score, tables) and the sim-service."""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from reactive_lob_sim.experiments import run_report
from reactive_lob_sim.reproduce import SIMULATORS
from reactive_lob_sim.service.app import create_app
from reactive_lob_sim.settings import Settings
from tests.conftest import with_updates


def small(settings: Settings) -> Settings:
    cal = settings.calibration
    s = with_updates(
        settings,
        reproduce={
            "calibration_duration_s": 900.0,
            "episode_duration_s": 60.0,
            "n_episodes": 1,
            "abm_seeds": 2,
            "generative_seeds": 2,
        },
        generative={"epochs": 1},
        gap={"bootstrap_samples": 50},
        report={
            "kappa_grid": [0.0, 1.0],
            "kappa_seeds": 2,
            "kappa_duration_s": 60.0,
            "speed_events": [2000],
            "speed_repeats": 2,
        },
    )
    impact = cal.impact.model_copy(update={"kappa_grid": [0.0, 1.0], "probe_seeds": 1})
    return s.model_copy(update={"calibration": cal.model_copy(update={"impact": impact})})


def test_report_pipeline_writes_ranked_tables(settings: Settings) -> None:
    s = small(settings)
    out = run_report(s)
    assert out == s.reports_dir
    tables = (out / "tables.md").read_text()
    for heading in ("E1 gap score", "E2 coupling kappa", "E3 matching-engine speed"):
        assert heading in tables
    study = json.loads((out / "study.json").read_text())
    ranks = sorted(g["rank"] for g in study["gap_scores"])
    assert ranks == [1, 2, 3, 4]
    assert {g["simulator"] for g in study["gap_scores"]} == set(SIMULATORS)
    assert study["calibration"]["coupling"]["grid"] == [0.0, 1.0]
    speed = json.loads((out / "engine_speed.json").read_text())
    assert speed["rows"][0]["max_abs_difference"] < 1e-9
    sweep = json.loads((out / "kappa_sweep.json").read_text())
    assert [r["kappa"] for r in sweep["rows"]] == [0.0, 1.0]
    calib = list(s.calibration_dir.rglob("impact.json"))
    assert len(calib) == 1
    assert json.loads(calib[0].read_text())["params"]["kappa"] >= 0.0


def test_sim_service_round_trip(settings: Settings) -> None:
    client = TestClient(create_app(settings))
    assert client.get("/healthz").json() == {"status": "ok"}
    ids = {sc["id"] for sc in client.get("/v1/scenarios").json()["scenarios"]}
    assert "liquidation-cascade" in ids
    body = {
        "scenario": "baseline",
        "seed": 1,
        "duration_s": 30.0,
        "venue": "synth",
        "symbol": "SYN-USDT-PERP",
    }
    resp = client.post("/v1/simulations", json=body)
    assert resp.status_code == 200
    sim_id = resp.json()["simulation_id"]
    assert client.get(f"/v1/simulations/{sim_id}").status_code == 200
    assert client.get("/v1/simulations/does-not-exist").status_code == 404
    assert client.post("/v1/simulations", json={**body, "scenario": "nope"}).status_code == 404
    too_long = {**body, "duration_s": settings.service.max_duration_s + 1}
    assert client.post("/v1/simulations", json=too_long).status_code == 422
