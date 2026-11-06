"""Call-auction price determination (opening, closing and discrete auctions).

Orders collected during an auction may cross. At the uncross one price ``p*`` is chosen and every bid at or
above it trades with every ask at or below it, at ``p*``, in price-time priority. The rule is the usual
equilibrium rule of order-driven exchanges (MOEX, Xetra, Euronext; see Madhavan 1992 and the MOEX trading
rules on auctions): among candidate prices (every limit price in the book),

1. maximise the executable volume ``min(D(p), S(p))``, with ``D(p)`` the bid quantity at or above ``p`` and
   ``S(p)`` the ask quantity at or below ``p``;
2. then minimise the surplus ``|D(p) - S(p)|``;
3. then take the price closest to the reference price (the last trade or the previous close);
4. then the lower price.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Equilibrium:
    price_tick: int
    volume: float
    surplus: float


def equilibrium(
    bids: list[tuple[int, float]], asks: list[tuple[int, float]], reference_tick: float | None
) -> Equilibrium | None:
    """Uncross price of a crossed book, or ``None`` when no bid meets an ask.

    ``bids`` and ``asks`` are ``(tick, qty)`` levels in any order.
    """
    if not bids or not asks or max(t for t, _ in bids) < min(t for t, _ in asks):
        return None
    candidates = sorted({t for t, _ in bids} | {t for t, _ in asks})
    best: tuple[float, float, float, int] | None = None
    best_eq: Equilibrium | None = None
    for p in candidates:
        demand = sum(q for t, q in bids if t >= p)
        supply = sum(q for t, q in asks if t <= p)
        volume = min(demand, supply)
        if volume <= 0:
            continue
        surplus = abs(demand - supply)
        distance = abs(p - reference_tick) if reference_tick is not None else 0.0
        key = (-volume, surplus, distance, p)
        if best is None or key < best:
            best = key
            best_eq = Equilibrium(p, volume, surplus)
    return best_eq
