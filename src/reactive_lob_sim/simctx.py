"""Construction of simulated markets from settings (one place that knows every constructor argument)."""

from __future__ import annotations

from dataclasses import dataclass

from reactive_lob_sim.abm.market import ABMMarket
from reactive_lob_sim.engine.accounting import FeeSchedule
from reactive_lob_sim.engine.rules import TradingRules
from reactive_lob_sim.moex.instruments import (
    InstrumentRegistry,
    build_registry,
    load_universe,
    load_zero_fee,
)
from reactive_lob_sim.moex.sessions import Phase, Segment, SessionSchedule, Window
from reactive_lob_sim.params import ABMParams
from reactive_lob_sim.replay.market import QueueModel, ReplayMarket
from reactive_lob_sim.scenarios.models import Scenario
from reactive_lob_sim.settings import Settings
from reactive_lob_sim.tape.tape import Tape, TapeRecorder
from reactive_lob_sim.timeutil import to_ns


@dataclass(frozen=True)
class SimContext:
    venue: str
    symbol: str
    tick_size: float
    lot_size: float
    qty_tolerance: float
    epoch_ns: int
    capture_region: str
    snapshot_depth: int
    fees: FeeSchedule
    queue_model: QueueModel
    price_band_fraction: float | None = None
    registry: InstrumentRegistry | None = None
    schedule: SessionSchedule | None = None

    @classmethod
    def from_settings(cls, s: Settings) -> SimContext:
        m = s.market
        registry = instrument_registry(s)
        base = cls(
            m.venue,
            m.symbol,
            m.tick_size,
            m.lot_size,
            m.qty_tolerance,
            to_ns(m.epoch),
            m.capture_region,
            m.snapshot_depth,
            FeeSchedule.from_bps(s.fees.maker_bps, s.fees.taker_bps),
            s.replay.queue_model,
            None,
            registry,
            session_schedule(s) if s.sessions.apply else None,
        )
        return base.with_symbol(m.venue, m.symbol)

    def with_symbol(self, venue: str, symbol: str) -> SimContext:
        """The context for ``symbol``: lot, tick, band and commission from the instrument registry when the
        symbol is listed there, else the market defaults."""
        tick, lot, fees, band = self.tick_size, self.lot_size, self.fees, self.price_band_fraction
        if self.registry is not None and symbol in self.registry.instruments:
            inst = self.registry.get(symbol)
            tick, lot, fees, band = (
                inst.tick,
                inst.lot_size,
                self.registry.fees(symbol),
                inst.price_band_fraction,
            )
        return SimContext(
            venue,
            symbol,
            tick,
            lot,
            self.qty_tolerance,
            self.epoch_ns,
            self.capture_region,
            self.snapshot_depth,
            fees,
            self.queue_model,
            band,
            self.registry,
            self.schedule,
        )

    def rules(self, reference_price: float | None = None) -> TradingRules:
        """Lot and tick; the daily price band around ``reference_price`` when a band fraction is known."""
        if reference_price is None or not self.price_band_fraction:
            return TradingRules(self.tick_size, self.lot_size)
        f = self.price_band_fraction
        return TradingRules.with_band(
            self.tick_size, self.lot_size, reference_price * (1.0 - f), reference_price * (1.0 + f)
        )

    def segments(self, horizon_s: float) -> list[Segment] | None:
        return None if self.schedule is None else self.schedule.segments(self.epoch_ns, horizon_s)

    def recorder(self, connection_id: str, epoch_ns: int | None = None) -> TapeRecorder:
        return TapeRecorder(
            self.venue,
            self.symbol,
            self.epoch_ns if epoch_ns is None else epoch_ns,
            self.tick_size,
            self.capture_region,
            connection_id,
        )

    def abm(
        self,
        params: ABMParams,
        horizon: float,
        seed: int,
        recorder: TapeRecorder | None = None,
        scenario: Scenario | None = None,
    ) -> ABMMarket:
        return ABMMarket(
            params,
            horizon,
            seed,
            self.rules(params.initial_price),
            self.qty_tolerance,
            self.fees,
            recorder,
            scenario,
            self.snapshot_depth,
            self.segments(horizon),
        )

    def replay(self, tape: Tape, start_ns: int | None = None) -> ReplayMarket:
        return ReplayMarket(
            tape, self.tick_size, self.lot_size, self.qty_tolerance, self.fees, self.queue_model, start_ns
        )


def instrument_registry(s: Settings) -> InstrumentRegistry:
    cfg = s.instruments
    universe = load_universe(s.data_dir / cfg.universe_file if cfg.universe_file else None)
    zero_fee = load_zero_fee(s.data_dir / cfg.zero_fee_file if cfg.zero_fee_file else None)
    return build_registry(
        cfg.symbols, cfg.defaults, universe, zero_fee, cfg.commission_bps, cfg.zero_fee_commission_bps
    )


def session_schedule(s: Settings) -> SessionSchedule:
    from datetime import time

    def windows(items: list) -> list[Window]:  # type: ignore[type-arg]
        return [
            Window(w.session, Phase(w.phase), time.fromisoformat(w.start), time.fromisoformat(w.end))
            for w in items
        ]

    cfg = s.sessions
    return SessionSchedule(windows(cfg.weekday), windows(cfg.weekend), cfg.weekend_enabled, cfg.utc_offset_h)
