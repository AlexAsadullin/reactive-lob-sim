"""sim-service: FastAPI application of contract 6.5 (simulation service)."""

from __future__ import annotations

import json
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from reactive_lob_sim.scenarios.library import list_scenarios, seed_library
from reactive_lob_sim.scenarios.models import Scenario
from reactive_lob_sim.service.simulate import run_simulation
from reactive_lob_sim.settings import Settings


class ScenarioList(BaseModel):
    schema_version: Literal["1"] = "1"
    scenarios: list[Scenario]


class SimulationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["1"] = "1"
    scenario: str
    seed: int
    duration_s: float = Field(gt=0)
    venue: str
    symbol: str


class SimulationResponse(BaseModel):
    schema_version: Literal["1"] = "1"
    simulation_id: str
    tape_path: str


def create_app(settings: Settings) -> FastAPI:
    app = FastAPI(title="reactive-lob-sim sim-service", version="0.1.0")
    seed_library(settings.scenarios_dir)

    @app.get("/healthz")
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/scenarios")
    def scenarios() -> ScenarioList:
        return ScenarioList(scenarios=list_scenarios(settings.scenarios_dir))

    @app.post("/v1/simulations")
    def simulate(req: SimulationRequest) -> SimulationResponse:
        if req.duration_s > settings.service.max_duration_s:
            raise HTTPException(422, f"duration_s exceeds the limit of {settings.service.max_duration_s} s")
        try:
            result = run_simulation(settings, req.scenario, req.seed, req.duration_s, req.venue, req.symbol)
        except KeyError as exc:
            raise HTTPException(404, f"unknown scenario {req.scenario!r}") from exc
        return SimulationResponse(simulation_id=result.simulation_id, tape_path=str(result.tape_path))

    @app.get("/v1/simulations/{simulation_id}")
    def simulation(simulation_id: str) -> dict[str, object]:
        if not simulation_id.isalnum():
            raise HTTPException(404, "unknown simulation")
        path = settings.simulations_dir / simulation_id / "simulation.json"
        if not path.exists():
            raise HTTPException(404, "unknown simulation")
        data: dict[str, object] = json.loads(path.read_text())
        return data

    return app
