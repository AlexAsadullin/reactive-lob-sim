"""Time conversions. Storage is int64 nanoseconds since the Unix epoch (UTC); the wire format is ISO-8601."""

from __future__ import annotations

from datetime import UTC, datetime

NS_PER_S = 1_000_000_000


def to_ns(dt: datetime) -> int:
    if dt.tzinfo is None:
        raise ValueError("naive datetimes are not allowed; use timezone-aware UTC")
    delta = dt.astimezone(UTC) - datetime(1970, 1, 1, tzinfo=UTC)
    return (delta.days * 86_400 + delta.seconds) * NS_PER_S + delta.microseconds * 1_000


def from_ns(ns: int) -> datetime:
    seconds, rem = divmod(int(ns), NS_PER_S)
    return datetime.fromtimestamp(seconds, tz=UTC).replace(microsecond=rem // 1_000)


def iso(ns: int) -> str:
    return from_ns(ns).isoformat().replace("+00:00", "Z")


def parse_iso(text: str) -> datetime:
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError(f"timestamp {text!r} has no timezone")
    return dt.astimezone(UTC)


def seconds_to_ns(t_s: float) -> int:
    return round(t_s * NS_PER_S)


def utc_now() -> datetime:
    return datetime.now(UTC)
