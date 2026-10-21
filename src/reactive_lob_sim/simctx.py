"""Construction of simulated markets from settings (one place that knows every constructor argument)."""

from __future__ import annotations

from dataclasses import dataclass

from reactive_lob_sim.abm.market import ABMMarket
from reactive_lob_sim.engine.accounting import FeeSchedule
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

    @classmethod
    def from_settings(cls, s: Settings) -> SimContext:
        m = s.market
        return cls(
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
        )

    def with_symbol(self, venue: str, symbol: str) -> SimContext:
        return SimContext(
            venue,
            symbol,
            self.tick_size,
            self.lot_size,
            self.qty_tolerance,
            self.epoch_ns,
            self.capture_region,
            self.snapshot_depth,
            self.fees,
            self.queue_model,
        )

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
            self.tick_size,
            self.lot_size,
            self.qty_tolerance,
            self.fees,
            recorder,
            scenario,
            self.snapshot_depth,
        )

    def replay(self, tape: Tape, start_ns: int | None = None) -> ReplayMarket:
        return ReplayMarket(
            tape, self.tick_size, self.lot_size, self.qty_tolerance, self.fees, self.queue_model, start_ns
        )
