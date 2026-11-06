"""MOEX trading-session schedule: which trading phase holds at a given UTC time.

The schedule is configuration (``sessions`` in the settings), written in Moscow time (UTC+3, no daylight
saving). A trading day is a list of windows ``(session, phase, start, end)``; times outside every window are
closed. Weekdays and weekend days have separate lists, and ``weekend_enabled`` switches the weekend session
(ISS ``tradingsession`` 5) on or off. Exchange holidays (ISS ``off_days``) are not modelled.

Phases: ``opening_auction`` and ``closing_auction`` collect orders without matching and end in an uncross;
``continuous`` is price-time matching; ``break`` and ``closed`` accept no orders. A trading halt is the
phase ``halted`` (set only by scenarios, never by the schedule).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from enum import StrEnum

from reactive_lob_sim.timeutil import NS_PER_S, from_ns, to_ns


class Phase(StrEnum):
    CLOSED = "closed"
    OPENING_AUCTION = "opening_auction"
    CONTINUOUS = "continuous"
    CLOSING_AUCTION = "closing_auction"
    BREAK = "break"
    HALTED = "halted"
    AUCTION = "auction"  # a discrete auction forced by a scenario (halt reopening, opening imbalance)

    @property
    def is_auction(self) -> bool:
        return self in (Phase.OPENING_AUCTION, Phase.CLOSING_AUCTION, Phase.AUCTION)

    @property
    def accepts_orders(self) -> bool:
        return self is Phase.CONTINUOUS or self.is_auction


# MXSE trading-status codes (docs/moex-sources.md section 2.3) written to the simulated tape.
STATUS_CODES: dict[Phase, str] = {
    Phase.CLOSED: "N",
    Phase.OPENING_AUCTION: "S",
    Phase.CONTINUOUS: "T",
    Phase.CLOSING_AUCTION: "L",
    Phase.BREAK: "B",
    Phase.HALTED: "B",
    Phase.AUCTION: "I",
}
# ISS ``tradingsession`` codes of the trades channel (contract 6.2 revision 2; 5 is the weekend session).
SESSION_CODES: dict[str, str] = {"morning": "0", "main": "1", "evening": "2", "weekend": "5"}


@dataclass(frozen=True, slots=True)
class Window:
    session: str
    phase: Phase
    start: time
    end: time


@dataclass(frozen=True, slots=True)
class Segment:
    """A constant-phase interval ``[t0, t1)`` in seconds from the simulation start."""

    t0: float
    t1: float
    session: str | None
    phase: Phase


class SessionSchedule:
    def __init__(
        self,
        weekday: list[Window],
        weekend: list[Window],
        weekend_enabled: bool,
        utc_offset_h: float,
    ) -> None:
        for windows in (weekday, weekend):
            ordered = sorted(windows, key=lambda w: w.start)
            for a, b in itertools.pairwise(ordered):
                if b.start < a.end:
                    raise ValueError(f"session windows overlap: {a} and {b}")
            for w in ordered:
                if w.end <= w.start:
                    raise ValueError(f"session window ends before it starts: {w}")
        self._weekday = sorted(weekday, key=lambda w: w.start)
        self._weekend = sorted(weekend, key=lambda w: w.start) if weekend_enabled else []
        self._offset = timedelta(hours=utc_offset_h)

    def _local(self, ns: int) -> datetime:
        return from_ns(ns) + self._offset

    def windows_for(self, day: datetime) -> list[Window]:
        return self._weekend if day.weekday() >= 5 else self._weekday

    def phase_at(self, ns: int) -> tuple[str | None, Phase]:
        local = self._local(ns)
        clock = local.time()
        for w in self.windows_for(local):
            if w.start <= clock < w.end:
                return w.session, w.phase
        return None, Phase.CLOSED

    def segments(self, start_ns: int, horizon_s: float) -> list[Segment]:
        """Constant-phase segments covering ``[0, horizon_s)`` from ``start_ns``."""
        end_ns = start_ns + round(horizon_s * NS_PER_S)
        cuts = {start_ns, end_ns}
        day = (self._local(start_ns)).replace(hour=0, minute=0, second=0, microsecond=0)
        while to_ns((day - self._offset).replace(tzinfo=UTC)) < end_ns:
            for w in self.windows_for(day):
                for clock in (w.start, w.end):
                    ns = to_ns(datetime.combine(day.date(), clock, tzinfo=UTC) - self._offset)
                    if start_ns < ns < end_ns:
                        cuts.add(ns)
            day += timedelta(days=1)
        ordered = sorted(cuts)
        out: list[Segment] = []
        for a, b in itertools.pairwise(ordered):
            session, phase = self.phase_at(a)
            t0, t1 = (a - start_ns) / NS_PER_S, (b - start_ns) / NS_PER_S
            if out and out[-1].phase is phase and out[-1].session == session:
                out[-1] = Segment(out[-1].t0, t1, session, phase)
            else:
                out.append(Segment(t0, t1, session, phase))
        return out


def always_continuous(horizon_s: float, session: str = "main") -> list[Segment]:
    """The schedule used when sessions are disabled: one continuous segment."""
    return [Segment(0.0, horizon_s, session, Phase.CONTINUOUS)]
