"""Calibration files (contract 6.5).

``calibration/venue=<v>/symbol=<s>/impact.json`` carries ``{model, params, fitted_at, sample_start,
sample_end}`` and is read by the Shadow Exchange (I6). ``abm.json`` next to it is this project's own full
simulator calibration (an optional extension that other projects may ignore).
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from reactive_lob_sim.params import ABMParams, ImpactParams

IMPACT_MODEL = "propagator_sqrt_v1"


class ImpactCalibration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1"
    model: str = IMPACT_MODEL
    params: dict[str, float | str]
    fitted_at: str
    sample_start: str
    sample_end: str


def calibration_dir(root: Path, venue: str, symbol: str) -> Path:
    return root / f"venue={venue}" / f"symbol={symbol}"


def impact_document(
    params: ImpactParams, fitted_at: str, sample_start: str, sample_end: str
) -> ImpactCalibration:
    return ImpactCalibration(
        params={k: v for k, v in params.model_dump().items() if isinstance(v, (int, float, str))},
        fitted_at=fitted_at,
        sample_start=sample_start,
        sample_end=sample_end,
    )


def write_calibration(
    root: Path, venue: str, symbol: str, impact: ImpactCalibration, abm: ABMParams | None = None
) -> Path:
    directory = calibration_dir(root, venue, symbol)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "impact.json"
    path.write_text(impact.model_dump_json(indent=2))
    if abm is not None:
        (directory / "abm.json").write_text(abm.model_dump_json(indent=2))
    return path


def read_impact(root: Path, venue: str, symbol: str) -> ImpactCalibration | None:
    path = calibration_dir(root, venue, symbol) / "impact.json"
    if not path.exists():
        return None
    return ImpactCalibration.model_validate(json.loads(path.read_text()))


def impact_params(doc: ImpactCalibration) -> ImpactParams:
    return ImpactParams.model_validate(doc.params)


def read_abm(root: Path, venue: str, symbol: str) -> ABMParams | None:
    path = calibration_dir(root, venue, symbol) / "abm.json"
    if not path.exists():
        return None
    return ABMParams.model_validate_json(path.read_text())
