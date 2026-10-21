"""Enumerate a small dataset universe and all its add/remove neighbour pairs.

A dataset is a multiset of values from a finite domain, represented as a
sorted tuple. (D, D') is a neighbour pair when D' = D plus one record; the
relation is symmetric, so each unordered pair is listed once.
"""

from __future__ import annotations

from itertools import combinations_with_replacement


def enumerate_universe(domain: tuple, max_size: int) -> list[tuple]:
    """Every multiset of size 0..max_size over the domain."""
    universe: list[tuple] = []
    for size in range(max_size + 1):
        universe.extend(combinations_with_replacement(sorted(domain), size))
    return universe


def neighbour_pairs(domain: tuple, max_size: int) -> list[tuple[tuple, tuple]]:
    """All (D, D + {v}) with |D| < max_size and v in the domain."""
    pairs = []
    for d in enumerate_universe(domain, max_size - 1):
        for v in sorted(set(domain)):
            pairs.append((d, tuple(sorted(d + (v,)))))
    return pairs


def is_add_remove_neighbour(a: tuple, b: tuple) -> bool:
    """True iff one multiset equals the other with exactly one record added."""
    small, big = (a, b) if len(a) < len(b) else (b, a)
    if len(big) != len(small) + 1:
        return False
    remaining = list(big)
    for x in small:
        if x not in remaining:
            return False
        remaining.remove(x)
    return len(remaining) == 1
