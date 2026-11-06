"""Agent-based market, stress scenarios, liquidation agent and the reference strategies."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reactive_lob_sim.calibration.flow import zero_intelligence
from reactive_lob_sim.crypto.liquidation import LiquidationBook
from reactive_lob_sim.engine.l3market import IdPool
from reactive_lob_sim.engine.types import OrderStatus, OrderType, Side
from reactive_lob_sim.gap.outcomes import outcome_from_run
from reactive_lob_sim.params import EVENT_TYPES
from reactive_lob_sim.scenarios.library import (
    apply_overrides,
    default_scenarios,
    list_scenarios,
    load_scenario,
    seed_library,
)
from reactive_lob_sim.scenarios.models import Scenario, Shock, ShockSchedule
from reactive_lob_sim.settings import Settings, StrategySpec
from reactive_lob_sim.simctx import SimContext
from reactive_lob_sim.strategies.base import run_strategy
from reactive_lob_sim.strategies.simple import build_strategy
from reactive_lob_sim.tape.l2 import book_series


def scenario(kind: str, **params: float) -> Scenario:
    return Scenario(
        id="t",
        description="t",
        default_duration_s=120,
        shocks=[Shock(kind=kind, start_s=30, duration_s=60, params=params)],
    )  # type: ignore[arg-type]


def test_abm_is_deterministic_and_records_tape(settings: Settings, ctx: SimContext) -> None:
    def run() -> tuple[float | None, int]:
        rec = ctx.recorder("a")
        m = ctx.abm(settings.abm, 120.0, 7, rec)
        m.advance(120.0)
        return m.mid_price(), rec.to_tape()["trades"].num_rows

    assert run() == run()
    rec = ctx.recorder("b")
    m = ctx.abm(settings.abm, 120.0, 7, rec)
    m.advance(120.0)
    tape = rec.to_tape()
    assert tape["book_snapshot"].num_rows > 0
    assert "funding" not in tape.tables or tape["funding"].num_rows == 0  # MOEX: no funding fixes
    bs = book_series(tape, ctx.tick_size, 5)
    assert np.all(bs.ask > bs.bid)
    assert m.n_scheduled_events > 0
    assert m.regime(10.0) == 0


def test_abm_two_regimes(settings: Settings, ctx: SimContext) -> None:
    params = apply_overrides(
        settings.abm,
        {
            "regimes": [
                {"vol_multiplier": 1.0, "rate_multiplier": 1.0, "mean_duration_s": 20.0},
                {"vol_multiplier": 3.0, "rate_multiplier": 2.0, "mean_duration_s": 20.0},
            ]
        },
    )
    m = ctx.abm(params, 200.0, 3)
    m.advance(200.0)
    assert {m.regime(t) for t in np.linspace(0, 199, 200)} == {0, 1}


def _mean_spread(ctx: SimContext, settings: Settings, sc: Scenario | None, t0: float, t1: float) -> float:
    rec = ctx.recorder("s")
    m = ctx.abm(settings.abm, 120.0, 11, rec, sc)
    m.advance(120.0)
    bs = book_series(rec.to_tape(), ctx.tick_size, 5)
    t = (bs.ts - bs.ts[0]) / 1e9
    sel = (t >= t0) & (t < t1)
    return float(bs.spread[sel].mean())


def test_spread_widening_and_withdrawal_shocks(settings: Settings, ctx: SimContext) -> None:
    base = _mean_spread(ctx, settings, None, 35, 90)
    wide = _mean_spread(ctx, settings, scenario("spread_widening", extra_ticks=6), 35, 90)
    assert wide > base + 2
    rec = ctx.recorder("w")
    m = ctx.abm(
        settings.abm, 120.0, 11, rec, scenario("liquidity_withdrawal", rate_factor=0.1, cancel_fraction=0.9)
    )
    m.advance(31.0)
    assert m.book.n_orders() < 40


def test_liquidation_cascade_and_funding_shock(crypto_settings: Settings, crypto_ctx: SimContext) -> None:
    settings, ctx = crypto_settings, crypto_ctx
    assert settings.abm.crypto is not None
    rec = ctx.recorder("c")
    m = ctx.abm(settings.abm, 120.0, 7, rec)
    m.advance(120.0)
    # fixes happen strictly inside the horizon
    assert rec.to_tape()["funding"].num_rows == int(
        (120 - 1e-9) // settings.abm.crypto.macro.funding_interval_s
    )
    calm = ctx.abm(settings.abm, 120.0, 5)
    calm.advance(120.0)
    shock = ctx.abm(
        settings.abm,
        120.0,
        5,
        None,
        scenario(
            "liquidation_cascade",
            oi_multiplier=2.0,
            long_share=0.9,
            trigger_depth_fraction=0.6,
            layers=8,
            layer_gap_bps=1.5,
        ),
    )
    shock.advance(120.0)
    assert shock.n_liquidations > calm.n_liquidations
    fs = ctx.abm(settings.abm, 120.0, 5, None, scenario("funding_shock", rate=0.003))
    fs.advance(70.0)
    assert fs.funding_rate == pytest.approx(0.003)


def test_liquidation_book_triggers() -> None:
    from reactive_lob_sim.params import LiquidationParams

    p = LiquidationParams(
        enabled=True,
        leverages=[50.0],
        maintenance_margin=0.0,
        crowding_sensitivity=0.3,
        funding_scale=0.001,
        oi_fraction_at_risk=0.1,
        refresh_interval_s=1.0,
        max_positions=2,
    )
    lb = LiquidationBook(p, np.random.default_rng(0))
    assert lb.long_share(0.0) == 0.5
    assert lb.long_share(1.0) == pytest.approx(0.8, abs=1e-3)
    lb.refresh(100.0, 0.001, 1000.0)
    assert len(lb.positions) == 2
    forced = lb.triggered(97.9)  # long trigger at 98
    assert [s for s, _ in forced] == [Side.SELL]
    assert lb.triggered(103.0)[0][0] is Side.BUY
    lb.add_crowding(100.0, 5.0, layers=2, gap=0.01)
    assert lb.triggered(99.0) == [(Side.SELL, 2.5)]
    assert lb.triggered(98.0) == [(Side.SELL, 2.5)]


def test_crypto_scenarios_need_the_crypto_module(settings: Settings, ctx: SimContext) -> None:
    with pytest.raises(ValueError, match="crypto module"):
        ctx.abm(settings.abm, 120.0, 5, None, scenario("funding_shock", rate=0.003))


def test_forced_order_after_emptied_side_still_executes(
    crypto_settings: Settings, crypto_ctx: SimContext
) -> None:
    """A batch whose first forced sell sweeps every bid must not drop the second forced sell.

    ``LiquidationBook.triggered`` removes the positions before they execute, so skipping an order on an
    empty side would lose the liquidation and leave open interest too high.
    """
    from reactive_lob_sim.crypto.liquidation import Position

    settings, ctx = crypto_settings, crypto_ctx
    m = ctx.abm(settings.abm, 120.0, 5)
    assert m.liquidations is not None
    assert m.macro is not None
    mid = m.mid_price()
    assert mid is not None
    bid_depth = sum(q for _, q in m.book.levels(Side.BUY, 10_000))
    trigger = mid * 2.0  # far above the mid, so both longs are hit at once
    m.liquidations.positions = [
        Position(True, mid, 50.0, 3.0 * bid_depth, trigger),
        Position(True, mid, 50.0, 1.0, trigger),
    ]
    oi_before = m.macro.open_interest
    m._check_liquidations()
    assert m.n_liquidations == 2
    assert m.liquidations.positions == []
    # the first order fills at most the whole bid depth; the second must add a positive fill on top
    assert oi_before - m.macro.open_interest > bid_depth
    assert m.book.best(Side.BUY) is not None


def test_scenario_validation_schedule_and_library(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing"):
        Shock(kind="funding_shock", start_s=0, duration_s=1, params={})
    sched = ShockSchedule(
        [
            Shock(kind="spread_widening", start_s=10, duration_s=5, params={"extra_ticks": 3}),
            Shock(
                kind="liquidity_withdrawal",
                start_s=12,
                duration_s=5,
                params={"rate_factor": 0.5, "cancel_fraction": 0.1},
            ),
        ]
    )
    assert sched.extra_offset_ticks(11) == 3
    assert sched.extra_offset_ticks(16) == 0
    assert sched.limit_rate_factor(13) == 0.5
    assert sched.funding_override(13) is None
    assert sched.long_share_override(13) is None
    assert [s.kind for s in sched.due(12)] == ["spread_widening", "liquidity_withdrawal"]
    assert sched.due(20) == []
    ids = {s.id for s in default_scenarios()}
    assert {
        "baseline",
        "spread-widening",
        "liquidity-withdrawal",
        "liquidation-cascade",
        "funding-shock",
    } <= ids
    assert len(seed_library(tmp_path)) == len(ids)
    assert seed_library(tmp_path) == []
    assert {s.id for s in list_scenarios(tmp_path)} == ids
    assert load_scenario(tmp_path, "baseline").shocks == []
    with pytest.raises(KeyError):
        load_scenario(tmp_path, "nope")
    assert list_scenarios(tmp_path / "missing") == []


def test_zero_intelligence_baseline_switches_liquidations_off(crypto_settings: Settings) -> None:
    zi = zero_intelligence(crypto_settings.abm)
    assert zi.crypto is not None
    assert not zi.crypto.liquidation.enabled


def test_zero_intelligence_baseline(settings: Settings) -> None:
    zi = zero_intelligence(settings.abm)
    assert zi.hawkes.spectral_radius() == 0.0
    np.testing.assert_allclose(zi.hawkes.mu, settings.abm.hawkes.stationary_rates())
    assert zi.impact.kappa == 0.0
    assert zi.crypto is None  # MOEX: no liquidation book at all
    assert len(EVENT_TYPES) == zi.hawkes.dim


def test_id_pool() -> None:
    pool = IdPool()
    for i in range(5):
        pool.add(i)
    pool.add(3)
    pool.discard(1)
    pool.discard(42)
    assert sorted(pool.items()) == [0, 2, 3, 4]
    assert 3 in pool
    assert 1 not in pool
    assert pool.sample(np.random.default_rng(0)) in {0, 2, 3, 4}


def test_strategies_run_in_abm(settings: Settings, ctx: SimContext) -> None:
    results = {}
    for spec in settings.strategies:
        run = run_strategy(
            ctx.abm(settings.abm, 150.0, 9), build_strategy(spec), 149.0, settings.runner.decision_interval_s
        )
        results[spec.id] = (run, outcome_from_run(run))
    lot = ctx.lot_size
    _, twap = results["twap-buy-market"]
    assert twap.filled_qty == pytest.approx(30 * lot, abs=0.01)
    assert twap.fill_rate == pytest.approx(1.0)
    mm_run, _ = results["mm-touch"]
    pos = 0.0
    for f in mm_run.fills:
        pos += f.side.sign * f.qty
        assert abs(pos) <= (5 + 1) * lot + 1e-9
    assert any(f.liquidity.value == "maker" for f in mm_run.fills)
    assert results["mom-fast"][1].n_fills > 0
    passive = results["twap-sell-passive"][1]
    assert passive.filled_qty == pytest.approx(30 * lot, abs=0.01)


def test_market_submit_and_cancel_paths(settings: Settings, ctx: SimContext) -> None:
    m = ctx.abm(settings.abm, 60.0, 1)
    m.advance(1.0)
    view = m.view()
    assert view.mid is not None
    assert view.spread is not None
    lot = ctx.lot_size
    assert m.submit(Side.BUY, OrderType.LIMIT, lot / 2, view.best_bid).status is OrderStatus.REJECTED
    ack = m.submit(Side.BUY, OrderType.LIMIT, lot, view.best_bid - 0.05)  # type: ignore[operator]
    assert ack.status is OrderStatus.ACCEPTED
    assert m.open_orders() == [ack.order_id]
    assert m.cancel(ack.order_id)
    assert not m.cancel(ack.order_id)
    assert m.submit(Side.SELL, OrderType.LIMIT, lot, None).status is OrderStatus.REJECTED
    assert m.tape() is None


def test_unknown_twap_style() -> None:
    with pytest.raises(ValueError, match="style"):
        build_strategy(
            StrategySpec(
                id="x",
                kind="twap",
                params={"side": "buy", "total_qty": 1.0, "style": "odd", "catchup_fraction": 0.1},
            )
        )
