"""MOEX instrument parameters (lot, tick, price band) and the broker commission per instrument.

Instruments come from ``instruments.symbols`` in the settings. Two optional reference files under the I1
data subtree refine them, and both may be absent:

``universe_moex.json`` (lot and tick per security, produced by I1's reference job). The loader accepts a list
of records or an object holding one under ``instruments``, ``securities`` or ``shares``, with the T-Bank
(``ticker``, ``class_code``, ``lot``, ``min_price_increment``) or ISS (``SECID``, ``BOARDID``, ``LOTSIZE``,
``MINSTEP``) field names; a T-Bank ``Quotation`` (``{units, nano}``) is accepted for the tick. A value in the
universe file wins over the settings, because the file is read from the exchange.

``tbank_zero_fee_funds.json`` (T-Capital funds that T-Invest trades without commission): key ``funds``
(records with ``ticker`` and ``class_code``) and key ``base_tickers``. An instrument whose SECID is a base
ticker, or whose (SECID, board) is a listed fund, pays ``zero_fee_commission_bps``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from reactive_lob_sim.engine.accounting import FeeSchedule

log = structlog.get_logger(__name__)

_LIST_KEYS = ("instruments", "securities", "shares", "data")
_TICKER_KEYS = ("ticker", "secid", "SECID")
_BOARD_KEYS = ("class_code", "classCode", "board", "boardid", "BOARDID")
_LOT_KEYS = ("lot", "lot_size", "LOTSIZE", "contract_size")
_TICK_KEYS = ("min_price_increment", "minPriceIncrement", "tick", "MINSTEP", "price_step")


@dataclass(frozen=True, slots=True)
class Instrument:
    """One MOEX security on one board. ``lot_size`` is shares per lot; prices are roubles per share."""

    symbol: str
    secid: str
    board: str
    lot_size: float
    tick: float
    currency: str
    price_band_fraction: float
    reference_price: float
    zero_fee: bool

    def band(self, reference: float) -> tuple[float, float]:
        """Daily price band around ``reference``: ``reference * (1 -/+ price_band_fraction)``."""
        return reference * (1.0 - self.price_band_fraction), reference * (1.0 + self.price_band_fraction)


def symbol_of(secid: str, board: str, currency: str = "RUB") -> str:
    """Normalised symbol of contract 6.2: ``<SECID>-RUB-<BOARD>``."""
    return f"{secid}-{currency.upper()}-{board}"


def _first(record: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        if record.get(key) is not None:
            return record[key]
    return None


def _number(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, dict) and "units" in value:
        return float(value.get("units", 0)) + float(value.get("nano", 0)) / 1e9
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _records(doc: Any) -> list[dict[str, Any]]:
    if isinstance(doc, list):
        return [r for r in doc if isinstance(r, dict)]
    if isinstance(doc, dict):
        for key in _LIST_KEYS:
            if isinstance(doc.get(key), list):
                return _records(doc[key])
        if all(isinstance(v, dict) for v in doc.values()):
            return [{"symbol": k, **v} for k, v in doc.items()]
    return []


def load_universe(path: Path | None) -> dict[str, dict[str, float]]:
    """``{symbol: {"lot_size": .., "tick": ..}}`` from a universe file; empty when the file is missing or
    unreadable (the file is produced by another project and is optional)."""
    if path is None or not path.exists():
        return {}
    try:
        doc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("universe_unreadable", path=str(path), error=str(exc))
        return {}
    out: dict[str, dict[str, float]] = {}
    for rec in _records(doc):
        ticker, board = _first(rec, _TICKER_KEYS), _first(rec, _BOARD_KEYS)
        symbol = rec.get("symbol")
        if not isinstance(symbol, str) or symbol.count("-") != 2:
            if ticker is None or board is None:
                continue
            symbol = symbol_of(str(ticker), str(board), str(rec.get("currency") or "RUB"))
        values = {"lot_size": _number(_first(rec, _LOT_KEYS)), "tick": _number(_first(rec, _TICK_KEYS))}
        clean = {k: v for k, v in values.items() if v is not None and v > 0}
        if clean:
            out[symbol] = clean
    return out


@dataclass(frozen=True, slots=True)
class ZeroFeeList:
    base_tickers: frozenset[str]
    listings: frozenset[tuple[str, str]]

    def contains(self, secid: str, board: str) -> bool:
        return secid in self.base_tickers or (secid, board) in self.listings


def load_zero_fee(path: Path | None) -> ZeroFeeList:
    """The T-Capital zero-commission list; empty when the file is missing or unreadable."""
    empty = ZeroFeeList(frozenset(), frozenset())
    if path is None or not path.exists():
        return empty
    try:
        doc = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        log.warning("zero_fee_list_unreadable", path=str(path), error=str(exc))
        return empty
    funds = doc.get("funds", []) if isinstance(doc, dict) else []
    listings = frozenset(
        (str(f["ticker"]), str(f["class_code"]))
        for f in funds
        if isinstance(f, dict) and f.get("ticker") and f.get("class_code")
    )
    bases = doc.get("base_tickers", []) if isinstance(doc, dict) else []
    return ZeroFeeList(frozenset(str(b) for b in bases), listings)


@dataclass(frozen=True)
class InstrumentRegistry:
    instruments: dict[str, Instrument]
    commission_bps: float
    zero_fee_commission_bps: float

    def get(self, symbol: str) -> Instrument:
        try:
            return self.instruments[symbol]
        except KeyError as exc:
            raise KeyError(f"unknown instrument {symbol!r}; add it under instruments.symbols") from exc

    def fees(self, symbol: str) -> FeeSchedule:
        """Broker commission per fill (maker and taker alike on T-Invest)."""
        bps = self.zero_fee_commission_bps if self.get(symbol).zero_fee else self.commission_bps
        return FeeSchedule.from_bps(bps, bps)


def build_registry(
    configured: dict[str, dict[str, Any]],
    defaults: dict[str, Any],
    universe: dict[str, dict[str, float]],
    zero_fee: ZeroFeeList,
    commission_bps: float,
    zero_fee_commission_bps: float,
) -> InstrumentRegistry:
    """Merge settings, universe file and zero-fee list into one registry."""
    out: dict[str, Instrument] = {}
    for symbol, cfg in configured.items():
        merged = {**defaults, **cfg, **universe.get(symbol, {})}
        secid, board = str(merged["secid"]), str(merged["board"])
        out[symbol] = Instrument(
            symbol=symbol,
            secid=secid,
            board=board,
            lot_size=float(merged["lot_size"]),
            tick=float(merged["tick"]),
            currency=str(merged["currency"]),
            price_band_fraction=float(merged["price_band_fraction"]),
            reference_price=float(merged["reference_price"]),
            zero_fee=bool(merged.get("zero_fee", False)) or zero_fee.contains(secid, board),
        )
    return InstrumentRegistry(out, commission_bps, zero_fee_commission_bps)
