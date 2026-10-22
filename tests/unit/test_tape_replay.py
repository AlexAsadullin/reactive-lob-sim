"""Tape round-trips (contract 6.2), event extraction and replay queue tracking."""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pytest

from reactive_lob_sim.calibration.events import extract_events
from reactive_lob_sim.engine.accounting import FeeSchedule
from reactive_lob_sim.engine.types import Liquidity, OrderStatus, OrderType, RejectReason, Side, TimeInForce
from reactive_lob_sim.params import CXL_BID, LIM_ASK, MKT_SELL
from reactive_lob_sim.replay.market import ReplayMarket
from reactive_lob_sim.tape.io import read_tape, write_tape
from reactive_lob_sim.tape.l2 import book_series, channel_values, trade_series
from reactive_lob_sim.tape.schema import SCHEMAS, conform
from reactive_lob_sim.tape.tape import Tape, TapeRecorder
from reactive_lob_sim.timeutil import NS_PER_S, iso, parse_iso, to_ns

TICK = 0.01
EPOCH = to_ns(parse_iso("2026-01-05T23:59:58Z"))


def base_recorder() -> TapeRecorder:
    rec = TapeRecorder("synth", "SYN-USDT-PERP", EPOCH, TICK, "sim", "test")
    rec.snapshot(0.0, [(10000, 5.0), (9999, 4.0)], [(10001, 3.0), (10002, 6.0)])
    return rec


def fees() -> FeeSchedule:
    return FeeSchedule.from_bps(1.0, 4.0)


def replay(tape: Tape, model: str = "proportional") -> ReplayMarket:
    return ReplayMarket(tape, TICK, 0.001, 1e-9, fees(), model, EPOCH)  # type: ignore[arg-type]


def test_tape_roundtrip_splits_by_date(tmp_path: Path) -> None:
    rec = base_recorder()
    rec.trade(1.0, 10001, 1.0, Side.BUY)
    rec.book_delta(1.0, Side.SELL, 10001, 2.0)
    rec.funding(1.5, 0.0001, 100.0, 100.0, 61.0)
    rec.open_interest(1.5, 5000.0, 100.0)
    rec.liquidation(3.0, Side.SELL, 99.99, 1.0)
    rec.trade(3.0, 10000, 0.5, Side.SELL)
    tape = rec.to_tape()
    paths = write_tape(tape, tmp_path)
    assert any("date=2026-01-06" in str(p) for p in paths)
    assert all("venue=synth/channel=" in str(p) for p in paths)
    back = read_tape(tmp_path, "synth", "SYN-USDT-PERP")
    assert back["trades"].num_rows == 2
    assert back["trades"].schema == SCHEMAS["trades"]
    window = read_tape(tmp_path, "synth", "SYN-USDT-PERP", EPOCH, EPOCH + 2 * NS_PER_S)
    assert window["trades"].num_rows == 1
    _, rate = channel_values(back, "funding", "rate")
    assert rate.tolist() == [0.0001]
    with pytest.raises(ValueError, match="scalar"):
        channel_values(back, "trades", "price")
    assert back.time_range()[0] == EPOCH
    assert iso(EPOCH).endswith("Z")


def test_conform_fills_missing_nullable_columns() -> None:
    table = pa.table(
        {
            "venue": ["v"],
            "symbol": ["s"],
            "ts_local": [1],
            "price": [1.0],
            "size": [2.0],
            "side": ["buy"],
            "trade_id": ["1"],
        }
    )
    out = conform("trades", table)
    assert out.schema == SCHEMAS["trades"]
    assert out.column("ts_exchange").null_count == 1


def test_book_series_and_trades() -> None:
    rec = base_recorder()
    rec.book_delta(1.0, Side.BUY, 10000, 0.0)
    tape = rec.to_tape()
    bs = book_series(tape, TICK, 2)
    assert bs.bid.tolist() == [10000, 9999]
    assert bs.ask_depth[0] == pytest.approx(9.0)
    assert bs.spread[-1] == 2
    assert trade_series(tape).ts.size == 0
    with pytest.raises(ValueError, match="two-sided"):
        book_series(Tape.from_tables("v", "s", {}), TICK, 1)


def test_event_extraction_separates_trades_and_cancels() -> None:
    rec = base_recorder()
    rec.trade(1.0, 10000, 2.0, Side.SELL)
    rec.book_delta(1.0, Side.BUY, 10000, 3.0)  # explained by the trade
    rec.book_delta(2.0, Side.BUY, 10000, 1.0)  # cancel of 2
    rec.book_delta(3.0, Side.SELL, 10003, 1.5)  # new ask level
    ev = extract_events(rec.to_tape(), TICK, 1e-9)
    assert ev.kind.tolist() == [MKT_SELL, CXL_BID, LIM_ASK]
    assert ev.size.tolist() == pytest.approx([2.0, 2.0, 1.5])
    assert ev.offset[2] == 2.0  # ceil(10000.5 + 0.5) = 10001 -> 10003 is two ticks out


def test_replay_queue_position_proportional_and_risk_averse() -> None:
    rec = base_recorder()
    rec.book_delta(1.0, Side.BUY, 10000, 8.0)  # 3 join behind the agent
    rec.book_delta(2.0, Side.BUY, 10000, 6.0)  # cancel of 2 out of 8
    rec.trade(3.0, 10000, 4.5, Side.SELL)
    tape = rec.to_tape()
    # proportional: ahead 5 - 2 * 5/8 = 3.75, so a 4.5 trade fills 0.75; risk averse: ahead stays 5, no fill
    for model, expected in (("proportional", 0.75), ("risk_averse", 0.0)):
        mk = replay(tape, model)
        mk.advance(0.5)
        ack = mk.submit(Side.BUY, OrderType.LIMIT, 1.0, 100.00)
        assert ack.status is OrderStatus.ACCEPTED
        mk.advance(2.5)
        assert mk.drain_fills() == []
        mk.advance(10.0)
        fills = mk.drain_fills()
        assert sum(f.qty for f in fills) == pytest.approx(expected)
        if fills:
            assert fills[0].liquidity is Liquidity.MAKER
            assert fills[0].queue_ahead_at_entry == pytest.approx(5.0)


def test_replay_trade_explains_decrease() -> None:
    rec = base_recorder()
    rec.trade(1.0, 10000, 2.0, Side.SELL)
    rec.book_delta(1.0, Side.BUY, 10000, 3.0)
    rec.trade(2.0, 10000, 3.5, Side.SELL)
    mk = replay(rec.to_tape())
    mk.advance(0.5)
    mk.submit(Side.BUY, OrderType.LIMIT, 1.0, 100.00)
    mk.advance(5.0)
    assert sum(f.qty for f in mk.drain_fills()) == pytest.approx(0.5)


def test_replay_market_orders_do_not_consume_twice() -> None:
    mk = replay(base_recorder().to_tape())
    mk.advance(0.0)
    first = mk.submit(Side.BUY, OrderType.MARKET, 4.0)
    assert first.filled_qty == pytest.approx(4.0)
    fills = mk.drain_fills()
    assert [f.price for f in fills] == pytest.approx([100.01, 100.02])
    view = mk.view()
    assert view.best_ask == pytest.approx(100.02)
    assert view.ask_size == pytest.approx(5.0)
    assert (
        mk.submit(Side.BUY, OrderType.LIMIT, 1.0, 100.05, TimeInForce.POST_ONLY).reject_reason
        is RejectReason.POST_ONLY_WOULD_CROSS
    )
    assert (
        mk.submit(Side.SELL, OrderType.LIMIT, 50.0, 99.0, TimeInForce.FOK).reject_reason
        is RejectReason.FOK_INSUFFICIENT_LIQUIDITY
    )
    assert mk.submit(Side.SELL, OrderType.LIMIT, 1.0, None).reject_reason is RejectReason.INVALID_PRICE
    ioc = mk.submit(Side.SELL, OrderType.LIMIT, 1.0, 101.0, TimeInForce.IOC)
    assert ioc.status is OrderStatus.CANCELLED
    rest = mk.submit(Side.SELL, OrderType.LIMIT, 1.0, 101.0)
    assert mk.open_orders() == [rest.order_id]
    assert mk.cancel(rest.order_id)
    assert not mk.cancel(rest.order_id)
    assert mk.tape() is None


def test_replay_fills_when_touch_moves_through_order() -> None:
    rec = base_recorder()
    rec.book_delta(2.0, Side.SELL, 9999, 1.0)
    mk = replay(rec.to_tape())
    mk.advance(0.0)
    mk.submit(Side.BUY, OrderType.LIMIT, 1.0, 99.99)
    mk.advance(3.0)
    assert sum(f.qty for f in mk.drain_fills()) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="empty"):
        replay(Tape.from_tables("v", "s", {}))
