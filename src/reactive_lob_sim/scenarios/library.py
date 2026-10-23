"""The counterfactual stress library under ``$TRADEQ_DATA_DIR/reactive-lob-sim/scenarios/*.yaml``.

The package ships the default scenarios; :func:`seed_library` copies any that are missing into the data root,
where users (and the Shadow Exchange) read and extend them.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from reactive_lob_sim.params import ABMParams
from reactive_lob_sim.scenarios.models import Scenario


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def apply_overrides(params: ABMParams, overrides: dict[str, Any]) -> ABMParams:
    """Deep-merge a partial ABM parameter document into ``params`` and validate the result."""
    if not overrides:
        return params
    return ABMParams.model_validate(_deep_merge(params.model_dump(), overrides))


def default_scenarios() -> list[Scenario]:
    root = resources.files("reactive_lob_sim.scenarios") / "defaults"
    out = []
    for entry in sorted(root.iterdir(), key=lambda e: e.name):
        if entry.name.endswith(".yaml"):
            out.append(Scenario.model_validate(yaml.safe_load(entry.read_text())))
    return out


def _dump(sc: Scenario) -> str:
    return yaml.safe_dump(sc.to_document(), sort_keys=False)


def scenario_error(path: Path) -> str | None:
    """Why ``path`` is not a valid scenario under the current schema, or ``None`` if it is."""
    try:
        Scenario.model_validate(yaml.safe_load(path.read_text()))
    except (ValueError, yaml.YAMLError) as exc:
        return str(exc)
    return None


def seed_library(directory: Path, refresh: bool = False) -> list[Path]:
    """Write the default scenarios into ``directory``.

    Without ``refresh`` only missing files are written. With ``refresh`` a default scenario's file is also
    rewritten when it differs from the shipped default (for example a file written by an older schema);
    files with other ids are never touched.
    """
    directory.mkdir(parents=True, exist_ok=True)
    written = []
    for sc in default_scenarios():
        path = directory / f"{sc.id}.yaml"
        text = _dump(sc)
        stale = refresh and path.exists() and path.read_text() != text
        if not path.exists() or stale:
            path.write_text(text)
            written.append(path)
    return written


def validate_library(directory: Path) -> dict[Path, str]:
    """Validation errors of every ``*.yaml`` in ``directory`` (empty when all files are valid)."""
    errors: dict[Path, str] = {}
    for path in sorted(directory.glob("*.yaml")):
        error = scenario_error(path)
        if error is not None:
            errors[path] = error
    return errors


def list_scenarios(directory: Path) -> list[Scenario]:
    if not directory.exists():
        return []
    return [Scenario.model_validate(yaml.safe_load(p.read_text())) for p in sorted(directory.glob("*.yaml"))]


def load_scenario(directory: Path, scenario_id: str) -> Scenario:
    """Load one scenario; ``<id>.yaml`` is read directly, so an invalid unrelated file does not block it."""
    path = directory / f"{scenario_id}.yaml"
    if path.exists():
        sc = Scenario.model_validate(yaml.safe_load(path.read_text()))
        if sc.id == scenario_id:
            return sc
    for sc in list_scenarios(directory):
        if sc.id == scenario_id:
            return sc
    raise KeyError(scenario_id)
