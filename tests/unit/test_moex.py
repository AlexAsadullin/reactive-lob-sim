"""MOEX microstructure: lots, ticks, price bands, auctions, halts, sessions, instruments and fees."""

from __future__ import annotations

import json
from datetime import UTC, datetime, time
from decimal import Decimal
from pathlib import Path

import pytest

from reactive_lob_sim.engine.auction import equilibrium
from reactive_lob_sim.engine.rules import TradingRules
from reactive_lob_sim.engine.types import OrderStatus, OrderType, RejectReason, Side
from reactive_lob_sim.moex.instruments import build_registry, load_universe, load_zero_fee
from reactive_lob_sim.moex.sessions import Phase, SessionSchedule, Window
from reactive_lob_sim.scenarios.library import default_scenarios
from reactive_lob_sim.scenarios.models import Scenario, Shock
from reactive_lob_sim.settings import Settings
from reactive_lob_sim.simctx import SimContext, session_schedule
from reactive_lob_sim.timeutil import to_ns


def test_rules_lots_ticks_and_band() -> None:
    r = TradingRules.with_band(0.01, 10.0, 234.164, 351.236)
    assert r.check(10.0, 292.70) is None
    assert r.check(15.0, 292.70) is RejectReason.INVALID_QTY  # 1.5 lots
    assert r.check(0.0, None) is RejectReason.INVALID_QTY
    assert r.check(10.0, 292.705) is RejectReason.INVALID_PRICE  # off the 1-kopek grid
    assert r.check(10.0, 400.0) is RejectReason.PRICE_OUT_OF_BAND
    assert r.band_low_tick == 23417  # low edge rounded up onto the grid
    assert r.band_high_tick == 35123  # high edge rounded down
    assert r.clamp(99_999) == 35123
    assert r.protection_tick(True) == 35123
    assert r.round_qty(3.0) == 10.0  # background flow: at least one lot
    assert r.floor_qty(29.0) == 20.0


def test_auction_equilibrium_maximises_volume_then_minimises_surplus() -> None:
    bids = [(102, 5.0), (101, 5.0), (100, 10.0)]
    asks = [(99, 4.0), (100, 6.0), (101, 10.0)]
    eq = equilibrium(bids, asks, reference_tick=100.0)
    assert eq is not None
    assert (
        eq.price_tick == 100
    )  # D(100)=20, S(100)=10 -> 10; D(101)=10, S(101)=20 -> 10; surplus ties, closer to ref
    assert eq.volume == 10.0
    assert equilibrium([(99, 1.0)], [(100, 1.0)], None) is None  # no cross: nothing to uncross


def _shock(kind: str, start: float, duration: float, **params: float) -> Scenario:
    return Scenario(
        id="t",
        description="t",
        default_duration_s=start + duration + 60,
        shocks=[Shock(kind=kind, start_s=start, duration_s=duration, params=params)],
    )  # type: ignore[arg-type]


def test_opening_auction_collects_then_uncrosses(settings: Settings, ctx: SimContext) -> None:
    sc = _shock("opening_auction_imbalance", 30, 30, imbalance=0.8, depth_multiple=2.0, price_range_bps=40.0)
    m = ctx.abm(settings.abm, sc.default_duration_s, 3, None, sc)
    m.advance(35.0)
    assert m.phase is Phase.AUCTION or m.phase.is_auction
    view = m.view()
    ack = m.submit(
        Side.BUY, OrderType.MARKET, ctx.lot_size
    )  # a market order becomes a limit at the band edge
    assert ack.status is OrderStatus.ACCEPTED
    assert view is not None
    m.advance(sc.default_duration_s - 1)
    assert m.phase is Phase.CONTINUOUS
    assert m.n_uncrosses >= 1
    assert m.auction_volume > 0


def test_halt_without_reopen_auction_rejects_orders(settings: Settings, ctx: SimContext) -> None:
    sc = _shock("trading_halt", 30, 30, reopen_auction=0.0, jump_bps=-100.0)
    m = ctx.abm(settings.abm, sc.default_duration_s, 4, None, sc)
    m.advance(40.0)
    assert m.phase is Phase.HALTED
    mid = m.mid_price() or settings.abm.initial_price
    ack = m.submit(Side.BUY, OrderType.LIMIT, ctx.lot_size, round(mid * 0.99, 2))
    assert ack.status is OrderStatus.REJECTED
    assert ack.reject_reason is RejectReason.TRADING_HALTED
    m.advance(sc.default_duration_s - 1)
    assert m.phase is Phase.CONTINUOUS


def test_dividend_gap_moves_the_price_down(settings: Settings, ctx: SimContext) -> None:
    base = ctx.abm(settings.abm, 200.0, 5)
    base.advance(199.0)
    sc = _shock("dividend_gap", 30, 30, dividend_bps=500.0)
    gap = ctx.abm(settings.abm, 200.0, 5, None, sc)
    gap.advance(199.0)
    assert base.mid_price() is not None
    assert gap.mid_price() is not None
    assert gap.mid_price() < base.mid_price() * 0.98  # type: ignore[operator]


def test_moex_scenarios_ship_and_run_without_the_crypto_module(settings: Settings, ctx: SimContext) -> None:
    moex = [s for s in default_scenarios() if not s.needs_crypto]
    ids = {s.id for s in moex}
    assert {"opening-auction-imbalance", "dividend-gap", "key-rate-surprise", "trading-halt"} <= ids
    for sc in moex:
        m = ctx.abm(settings.abm, 120.0, 1, None, sc)
        m.advance(119.0)
        assert m.mid_price() is not None, sc.id


def test_session_schedule_phases_in_moscow_time(settings: Settings) -> None:
    sched = session_schedule(settings)
    msk = 3 * 3600 * 10**9

    def at(y: int, mo: int, d: int, hh: int, mm: int) -> int:
        return to_ns(datetime(y, mo, d, hh, mm, tzinfo=UTC)) - msk

    assert sched.phase_at(at(2026, 11, 5, 9, 55)) == ("main", Phase.OPENING_AUCTION)  # Thursday
    assert sched.phase_at(at(2026, 11, 5, 12, 0)) == ("main", Phase.CONTINUOUS)
    assert sched.phase_at(at(2026, 11, 5, 18, 45)) == ("main", Phase.CLOSING_AUCTION)
    assert sched.phase_at(at(2026, 11, 5, 23, 55))[1] is Phase.CLOSED
    assert sched.phase_at(at(2026, 11, 7, 12, 0)) == ("weekend", Phase.CONTINUOUS)  # Saturday
    segs = sched.segments(at(2026, 11, 5, 9, 50), 20 * 60.0)
    assert [s.phase for s in segs] == [Phase.OPENING_AUCTION, Phase.CONTINUOUS]
    assert segs[0].t1 == pytest.approx(600.0)


def test_overlapping_session_windows_are_rejected() -> None:
    w = [
        Window("main", Phase.CONTINUOUS, time(10, 0), time(12, 0)),
        Window("main", Phase.CLOSING_AUCTION, time(11, 0), time(13, 0)),
    ]
    with pytest.raises(ValueError, match="overlap"):
        SessionSchedule(w, [], False, 3.0)


def test_registry_lot_tick_from_universe_and_zero_fee_funds(tmp_path: Path) -> None:
    universe = tmp_path / "universe.json"
    universe.write_text(
        json.dumps(
            {
                "instruments": [
                    {"ticker": "SBER", "class_code": "TQBR", "lot": 10, "min_price_increment": 0.01},
                    {"ticker": "TMOS", "class_code": "TQTF", "lot": 1, "min_price_increment": 0.002},
                ]
            }
        )
    )
    zero = tmp_path / "zero.json"
    zero.write_text(
        json.dumps({"funds": [{"ticker": "TMOS", "class_code": "TQTF"}], "base_tickers": ["TMOS"]})
    )
    configured = {
        "SBER-RUB-TQBR": {
            "secid": "SBER",
            "board": "TQBR",
            "lot_size": 1,
            "tick": 0.1,
            "reference_price": 290,
        },
        "TMOS-RUB-TQTF": {
            "secid": "TMOS",
            "board": "TQTF",
            "lot_size": 1,
            "tick": 0.01,
            "reference_price": 6.5,
        },
    }
    defaults = {"currency": "RUB", "price_band_fraction": 0.2}
    reg = build_registry(configured, defaults, load_universe(universe), load_zero_fee(zero), 25.0, 0.0)
    sber, tmos = reg.get("SBER-RUB-TQBR"), reg.get("TMOS-RUB-TQTF")
    assert (sber.lot_size, sber.tick) == (10.0, 0.01)  # the universe file wins over the settings
    assert tmos.tick == 0.002
    assert not sber.zero_fee
    assert tmos.zero_fee
    assert reg.fees("SBER-RUB-TQBR").taker_bps == Decimal("25.0")
    assert reg.fees("TMOS-RUB-TQTF").taker_bps == Decimal("0.0")
    assert load_universe(tmp_path / "missing.json") == {}


def test_context_uses_instrument_rules_and_fees(settings: Settings) -> None:
    ctx = SimContext.from_settings(settings)
    assert ctx.symbol == "SBER-RUB-TQBR"
    assert ctx.lot_size == 10.0
    assert ctx.fees.taker_bps == Decimal("25.0")
    fund = ctx.with_symbol("synth", "TMOS-RUB-TQTF")
    assert fund.lot_size == 1.0
    lk = ctx.with_symbol("synth", "LKOH-RUB-TQBR")
    assert lk.tick_size == 0.5
    band = ctx.rules(292.7)
    assert band.band_low_tick is not None
    assert band.band_high_tick is not None
    assert band.band_low_tick * 0.01 == pytest.approx(292.7 * 0.8, abs=0.01)
