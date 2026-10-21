"""Client of the Point-in-time Store (contract 6.3) and the conditioning-series sources built on it.

The simulator conditions its flow on funding and open interest. In research use these series must be read
"as known at" the end of the calibration sample, so a calibration never sees revisions published later.
:class:`PitConditioningSource` does this through I2; :class:`TapeConditioningSource` is the offline fallback
that reads the I1 tape directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

import httpx
import numpy as np

from reactive_lob_sim.tape.l2 import channel_values
from reactive_lob_sim.tape.schema import FUNDING, OPEN_INTEREST
from reactive_lob_sim.tape.tape import Tape
from reactive_lob_sim.timeutil import iso


class LeakageBlockedError(RuntimeError):
    """I2 refused a research query that would read data not yet known (HTTP 409)."""

    def __init__(self, event: dict[str, Any]) -> None:
        super().__init__(f"point-in-time store blocked the query: {event.get('kind', 'leakage')}")
        self.event = event


@dataclass(frozen=True)
class PitResult:
    rows: list[dict[str, Any]]
    fingerprint: str | None


class PitClient:
    def __init__(self, base_url: str, timeout_s: float, transport: httpx.BaseTransport | None = None) -> None:
        self._client = httpx.Client(base_url=base_url, timeout=timeout_s, transport=transport)

    def close(self) -> None:
        self._client.close()

    def query(
        self,
        dataset: str,
        start_ns: int,
        end_ns: int,
        known_at_ns: int,
        columns: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        mode: Literal["research", "live"] = "research",
    ) -> PitResult:
        body: dict[str, Any] = {
            "schema_version": "1",
            "dataset": dataset,
            "start": iso(start_ns),
            "end": iso(end_ns),
            "known_at": iso(known_at_ns),
            "mode": mode,
        }
        if columns is not None:
            body["columns"] = columns
        if filters is not None:
            body["filters"] = filters
        resp = self._client.post("/v1/query", params={"format": "json"}, json=body)
        if resp.status_code == httpx.codes.CONFLICT:
            raise LeakageBlockedError(resp.json())
        resp.raise_for_status()
        payload = resp.json()
        rows = payload["rows"] if isinstance(payload, dict) else payload
        return PitResult(list(rows), resp.headers.get("X-Dataset-Fingerprint"))


@dataclass(frozen=True)
class ConditioningSeries:
    """Funding and open interest series (int64 ns timestamps)."""

    funding_ts: np.ndarray
    funding: np.ndarray
    oi_ts: np.ndarray
    oi: np.ndarray
    fingerprint: str | None = None


class ConditioningSource(Protocol):
    def load(self, start_ns: int, end_ns: int, known_at_ns: int) -> ConditioningSeries: ...


class TapeConditioningSource:
    def __init__(self, tape: Tape) -> None:
        self._tape = tape

    def load(self, start_ns: int, end_ns: int, known_at_ns: int) -> ConditioningSeries:
        cut = min(end_ns, known_at_ns + 1)
        sl = self._tape.slice(start_ns, cut)
        f_ts, f = channel_values(sl, FUNDING, "rate")
        o_ts, o = channel_values(sl, OPEN_INTEREST, "open_interest")
        return ConditioningSeries(f_ts, f, o_ts, o)


def _ts_of(row: dict[str, Any]) -> int:
    for key in ("valid_time", "ts_exchange", "ts_local"):
        value = row.get(key)
        if value is not None:
            return int(value)
    raise KeyError("row has no valid_time/ts column")


class PitConditioningSource:
    def __init__(
        self, client: PitClient, funding_dataset: str, oi_dataset: str, venue: str, symbol: str
    ) -> None:
        self._client = client
        self._datasets = (funding_dataset, oi_dataset)
        self._filters = {"venue": venue, "symbol": symbol}

    def load(self, start_ns: int, end_ns: int, known_at_ns: int) -> ConditioningSeries:
        funding = self._client.query(self._datasets[0], start_ns, end_ns, known_at_ns, filters=self._filters)
        oi = self._client.query(self._datasets[1], start_ns, end_ns, known_at_ns, filters=self._filters)
        f_rows = sorted(funding.rows, key=_ts_of)
        o_rows = sorted(oi.rows, key=_ts_of)
        return ConditioningSeries(
            np.asarray([_ts_of(r) for r in f_rows], dtype=np.int64),
            np.asarray([float(r["rate"]) for r in f_rows]),
            np.asarray([_ts_of(r) for r in o_rows], dtype=np.int64),
            np.asarray([float(r["open_interest"]) for r in o_rows]),
            funding.fingerprint,
        )
