"""Command-line interface: ``sim calibrate | seed-scenarios | simulate | serve | reproduce | report``."""

from __future__ import annotations

from typing import Annotated

import typer

from reactive_lob_sim.logging_setup import configure_logging
from reactive_lob_sim.settings import Settings, load_settings

app = typer.Typer(add_completion=False, help="Reactive crypto LOB simulator and sim-to-real gap score (Tradeq S3).")


def _settings() -> Settings:
    settings = load_settings()
    configure_logging(settings.logging)
    return settings


@app.command("seed-scenarios")
def seed_scenarios() -> None:
    """Copy the default stress scenarios into the data root (existing files are kept)."""
    from reactive_lob_sim.scenarios.library import seed_library

    settings = _settings()
    written = seed_library(settings.scenarios_dir)
    typer.echo(f"{len(written)} scenario file(s) written to {settings.scenarios_dir}")


@app.command()
def calibrate(
    venue: Annotated[str, typer.Option(help="Venue of the I1 tape")],
    symbol: Annotated[str, typer.Option(help="Normalised symbol, e.g. BTC-USDT-PERP")],
    start: Annotated[str, typer.Option(help="Sample start, ISO-8601 UTC")],
    end: Annotated[str, typer.Option(help="Sample end, ISO-8601 UTC")],
    seed: Annotated[int, typer.Option(help="Seed of the coupling probe runs")] = 0,
) -> None:
    """Calibrate the simulator on an I1 tape window and write the contract 6.5 calibration files."""
    from reactive_lob_sim.calibration.pipeline import calibrate as run_calibration
    from reactive_lob_sim.calibration.pipeline import save_calibration
    from reactive_lob_sim.contracts.pit_client import PitClient, PitConditioningSource, TapeConditioningSource
    from reactive_lob_sim.simctx import SimContext
    from reactive_lob_sim.tape.io import read_tape
    from reactive_lob_sim.timeutil import parse_iso, to_ns

    settings = _settings()
    ctx = SimContext.from_settings(settings).with_symbol(venue, symbol)
    tape = read_tape(settings.tape_root, venue, symbol, to_ns(parse_iso(start)), to_ns(parse_iso(end)))
    c = settings.contracts
    if c.pit_enabled:
        client = PitClient(c.pit_base_url, c.pit_timeout_s)
        source: PitConditioningSource | TapeConditioningSource = PitConditioningSource(
            client, c.pit_funding_dataset, c.pit_open_interest_dataset, venue, symbol
        )
    else:
        source = TapeConditioningSource(tape)
    result = run_calibration(tape, settings.abm, settings.calibration, ctx, source, seed)
    path = save_calibration(result, settings.calibration_dir, venue, symbol)
    typer.echo(f"kappa={result.coupling.kappa:.3f} Y={result.abm.impact.Y:.3f} delta={result.abm.impact.delta:.3f}")
    typer.echo(f"written {path}")


@app.command()
def simulate(
    scenario: Annotated[str, typer.Option(help="Scenario id from the library")] = "baseline",
    seed: Annotated[int, typer.Option()] = 0,
    duration_s: Annotated[float, typer.Option()] = 600.0,
    venue: Annotated[str | None, typer.Option()] = None,
    symbol: Annotated[str | None, typer.Option()] = None,
) -> None:
    """Run one scenario and write its tape in the I1 layout."""
    from reactive_lob_sim.scenarios.library import seed_library
    from reactive_lob_sim.service.simulate import run_simulation

    settings = _settings()
    seed_library(settings.scenarios_dir)
    result = run_simulation(
        settings, scenario, seed, duration_s, venue or settings.market.venue, symbol or settings.market.symbol
    )
    typer.echo(f"simulation_id={result.simulation_id} trades={result.n_trades} liquidations={result.n_liquidations}")
    typer.echo(f"tape_path={result.tape_path}")


@app.command()
def serve() -> None:
    """Run the sim-service (contract 6.5)."""
    import uvicorn

    from reactive_lob_sim.service.app import create_app

    settings = _settings()
    uvicorn.run(create_app(settings), host=settings.service.host, port=settings.service.port)


@app.command()
def reproduce() -> None:
    """Offline end-to-end study: ground truth, calibration, four simulators, realism and gap score."""
    from reactive_lob_sim.reproduce import run_study

    settings = _settings()
    result = run_study(settings)
    typer.echo((result.out_dir / "summary.md").read_text())
    typer.echo(f"outputs in {result.out_dir}")


@app.command()
def report() -> None:
    """Run the research-report experiments and write the metric tables under reports/."""
    from reactive_lob_sim.experiments import run_report

    settings = _settings()
    out = run_report(settings)
    typer.echo((out / "tables.md").read_text())
    typer.echo(f"tables in {out}")
