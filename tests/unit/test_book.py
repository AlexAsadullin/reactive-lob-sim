from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from reactive_lob_sim.engine.book import OrderBook
from reactive_lob_sim.engine.fastbook import (
    EventBatch,
    run_events,
    run_events_reference,
    zero_intelligence_stream,
)
from reactive_lob_sim.engine.types import (
    OrderRequest,
    OrderStatus,
    OrderType,
    RejectReason,
    Side,
    TimeInForce,
)

EPS = 1e-9


def limit(
    side: Side, qty: float, tick: int, owner: int = 2, tif: TimeInForce = TimeInForce.GTC
) -> OrderRequest:
    return OrderRequest(owner, side, OrderType.LIMIT, qty, tick, tif)


def market(side: Side, qty: float, owner: int = 1) -> OrderRequest:
    return OrderRequest(owner, side, OrderType.MARKET, qty)


@pytest.fixture
def book() -> OrderBook:
    b = OrderBook(EPS)
    b.submit(limit(Side.SELL, 1.0, 101))
    b.submit(limit(Side.SELL, 2.0, 101))
    b.submit(limit(Side.SELL, 3.0, 103))
    b.submit(limit(Side.BUY, 1.5, 99))
    return b


def test_price_time_priority(book: OrderBook) -> None:
    res = book.submit(market(Side.BUY, 2.5))
    assert res.status is OrderStatus.FILLED
    assert [(f.maker_order_id, f.price_tick, f.qty) for f in res.fills] == [(1, 101, 1.0), (2, 101, 1.5)]
    assert book.levels(Side.SELL) == [(101, 0.5), (103, 3.0)]


def test_market_walks_levels_and_partial(book: OrderBook) -> None:
    res = book.submit(market(Side.BUY, 10.0))
    assert res.status is OrderStatus.PARTIALLY_FILLED
    assert res.filled_qty == pytest.approx(6.0)
    assert book.best_ask() is None
    assert book.submit(market(Side.BUY, 1.0)).reject_reason is RejectReason.NO_LIQUIDITY


def test_limit_crosses_then_rests(book: OrderBook) -> None:
    res = book.submit(limit(Side.BUY, 4.0, 101))
    assert res.filled_qty == pytest.approx(3.0)
    assert res.status is OrderStatus.PARTIALLY_FILLED
    assert book.best_bid() == 101
    assert book.level_qty(Side.BUY, 101) == pytest.approx(1.0)


def test_post_only_ioc_fok(book: OrderBook) -> None:
    assert (
        book.submit(limit(Side.BUY, 1.0, 101, tif=TimeInForce.POST_ONLY)).reject_reason
        is RejectReason.POST_ONLY_WOULD_CROSS
    )
    ok = book.submit(limit(Side.BUY, 1.0, 100, tif=TimeInForce.POST_ONLY))
    assert ok.status is OrderStatus.ACCEPTED
    assert ok.queue_ahead == 0.0
    fok = book.submit(limit(Side.BUY, 4.0, 101, tif=TimeInForce.FOK))
    assert fok.reject_reason is RejectReason.FOK_INSUFFICIENT_LIQUIDITY
    assert book.level_qty(Side.SELL, 101) == pytest.approx(3.0)
    ioc = book.submit(limit(Side.BUY, 4.0, 101, tif=TimeInForce.IOC))
    assert ioc.filled_qty == pytest.approx(3.0)
    assert ioc.resting_qty == 0.0
    assert book.best_bid() == 100
    none = book.submit(limit(Side.BUY, 1.0, 90, tif=TimeInForce.IOC))
    assert none.status is OrderStatus.CANCELLED


def test_invalid_orders(book: OrderBook) -> None:
    assert book.submit(limit(Side.BUY, 0.0, 99)).reject_reason is RejectReason.INVALID_QTY
    assert (
        book.submit(OrderRequest(1, Side.BUY, OrderType.LIMIT, 1.0)).reject_reason
        is RejectReason.INVALID_PRICE
    )
    assert book.modify(999, 1.0).reject_reason is RejectReason.UNKNOWN_ORDER
    assert not book.cancel(999).cancelled


def test_cancel_and_modify_priority(book: OrderBook) -> None:
    assert book.queue_ahead(2) == pytest.approx(1.0)
    reduced = book.modify(1, new_qty=0.5)
    assert reduced.status is OrderStatus.ACCEPTED
    assert book.queue_ahead(2) == pytest.approx(0.5)  # order 1 kept its place
    book.modify(1, new_qty=2.0)  # size increase loses priority
    assert book.queue_ahead(1) == pytest.approx(2.0)
    moved = book.modify(4, new_price_tick=101)  # bid moves through the ask and trades
    assert moved.filled_qty == pytest.approx(1.5)
    assert book.modify(2, new_qty=0.0).reject_reason is RejectReason.INVALID_QTY
    res = book.cancel(1)
    assert res.cancelled
    assert res.qty == pytest.approx(2.0)
    assert book.order(1) is None


ops = st.lists(
    st.tuples(
        st.sampled_from(["limit", "market", "cancel", "modify"]),
        st.sampled_from([Side.BUY, Side.SELL]),
        st.integers(95, 105),
        st.integers(1, 5),
        st.sampled_from(list(TimeInForce)),
        st.integers(0, 40),
    ),
    min_size=1,
    max_size=60,
)


@settings(max_examples=150, deadline=None)
@given(ops)
def test_book_invariants_and_conservation(
    sequence: list[tuple[str, Side, int, int, TimeInForce, int]],
) -> None:
    book = OrderBook(EPS)
    ids: list[int] = []
    for op, side, tick, qty, tif, pick in sequence:
        before = book.total_qty()
        if op == "limit":
            res = book.submit(limit(side, float(qty), tick, tif=tif))
            ids.append(res.order_id)
            added = res.resting_qty
            maker_fills = res.filled_qty
            assert res.filled_qty + res.resting_qty <= qty + EPS
            prices = [f.price_tick for f in res.fills]
            assert prices == sorted(prices, reverse=side is Side.SELL)
            assert all((p <= tick) if side is Side.BUY else (p >= tick) for p in prices)
            assert book.total_qty() == pytest.approx(before + added - maker_fills, abs=1e-6)
        elif op == "market":
            res = book.submit(market(side, float(qty)))
            assert book.total_qty() == pytest.approx(before - res.filled_qty, abs=1e-6)
        elif op == "cancel" and ids:
            res_c = book.cancel(ids[pick % len(ids)])
            assert book.total_qty() == pytest.approx(before - res_c.qty, abs=1e-6)
        elif op == "modify" and ids:
            oid = ids[pick % len(ids)]
            book.modify(oid, new_qty=float(qty), new_price_tick=tick)
        assert not book.is_crossed()
        for s in (Side.BUY, Side.SELL):
            for t, q in book.levels(s):
                orders = book.level_orders(s, t)
                assert orders
                assert q == pytest.approx(sum(o.qty for o in orders), abs=1e-6)
                assert all(o.qty > EPS for o in orders)
                arrivals = [o.arrival for o in orders]
                assert arrivals == sorted(arrivals)
        assert book.n_orders() == book.n_orders(Side.BUY) + book.n_orders(Side.SELL)


@settings(max_examples=40, deadline=None)
@given(st.integers(0, 10_000), st.integers(50, 600))
def test_numba_kernel_matches_object_book(seed: int, n: int) -> None:
    batch = zero_intelligence_stream(n, 60, 0.5, 0.2, 0.4, 0.0, 0.5, 0.001, seed)
    fast = run_events(batch, 60, EPS)
    slow = run_events_reference(batch, 60, EPS)
    np.testing.assert_allclose(fast.filled, slow.filled, atol=1e-9)
    np.testing.assert_allclose(fast.bid_qty, slow.bid_qty, atol=1e-9)
    np.testing.assert_allclose(fast.ask_qty, slow.ask_qty, atol=1e-9)


def test_numba_kernel_rejects_off_grid() -> None:
    one = np.ones(1, dtype=np.int64)
    batch = EventBatch(np.zeros(1, dtype=np.int64), np.zeros(1, dtype=np.int64), one * 99, np.ones(1), -one)
    with pytest.raises(ValueError, match="grid"):
        run_events(batch, 10, EPS)
