"""Privacy-budget ledger for releases made with the hand-written mechanisms.

Composition rule: basic sequential composition. Running mechanisms with
(ε_i, δ_i) on the same data is (Σε_i, Σδ_i)-DP. It is loose but needs no
assumptions, which suits a handful of summary statistics. DP-SGD is NOT
tracked here: it has its own accountant inside Opacus (RDP), and the two are
reported separately because they are separate releases.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from .validation import check_epsilon


class BudgetExceededError(RuntimeError):
    pass


@dataclass
class LedgerEntry:
    name: str
    epsilon: float
    delta: float


@dataclass
class BudgetLedger:
    total_epsilon: float
    total_delta: float = 0.0
    entries: list[LedgerEntry] = field(default_factory=list)

    def __post_init__(self) -> None:
        check_epsilon(self.total_epsilon)
        if not 0 <= self.total_delta < 1:
            raise ValueError("total_delta must be in [0, 1)")

    @property
    def spent_epsilon(self) -> float:
        return sum(e.epsilon for e in self.entries)

    @property
    def spent_delta(self) -> float:
        return sum(e.delta for e in self.entries)

    def spend(self, name: str, epsilon: float, delta: float = 0.0) -> None:
        """Record a release, refusing it if the total would be exceeded.
        A small relative tolerance absorbs floating-point error from splitting
        a budget into fractions that should sum exactly to the total."""
        check_epsilon(epsilon)
        if delta < 0:
            raise ValueError("delta must be >= 0")
        tol = 1e-9
        if self.spent_epsilon + epsilon > self.total_epsilon * (1 + tol):
            raise BudgetExceededError(
                f"{name}: ε {epsilon:g} would exceed the budget "
                f"({self.spent_epsilon:g} of {self.total_epsilon:g} spent)")
        if self.spent_delta + delta > self.total_delta * (1 + tol) + 1e-300:
            raise BudgetExceededError(
                f"{name}: δ {delta:g} would exceed the budget "
                f"({self.spent_delta:g} of {self.total_delta:g} spent)")
        self.entries.append(LedgerEntry(name, epsilon, delta))

    def to_dict(self) -> dict:
        return {"composition": "basic sequential",
                "total_epsilon": self.total_epsilon, "total_delta": self.total_delta,
                "spent_epsilon": self.spent_epsilon, "spent_delta": self.spent_delta,
                "entries": [asdict(e) for e in self.entries]}
