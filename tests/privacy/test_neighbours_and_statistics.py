"""Exhaustive toy-universe checks, sensitivity regressions and budget ledger."""

import numpy as np
import pytest

from dp_ecg.privacy import sensitivity as S
from dp_ecg.privacy.accounting import BudgetExceededError, BudgetLedger
from dp_ecg.privacy.neighbours import enumerate_universe, is_add_remove_neighbour, neighbour_pairs
from dp_ecg.privacy.statistics import dp_count, dp_histogram, dp_mean, dp_sum
from dp_ecg.privacy.verification import verify_all


def test_universe_and_pair_counts():
    # Multisets of size <= 2 over 3 values: 1 + 3 + 6 = 10.
    assert len(enumerate_universe((0, 1, 2), 2)) == 10
    pairs = neighbour_pairs((0, 1, 2), 2)
    # Every dataset of size <= 1 (4 of them) extended by each of 3 values.
    assert len(pairs) == 12
    assert all(is_add_remove_neighbour(a, b) for a, b in pairs)


def test_neighbour_relation_rejects_non_neighbours():
    assert not is_add_remove_neighbour((0, 1), (0, 2))        # replace, not add/remove
    assert not is_add_remove_neighbour((0,), (0, 1, 1))       # two records added
    assert is_add_remove_neighbour((1, 1), (1,))


@pytest.mark.parametrize("eps,delta", [(0.5, 1e-5), (1.0, 1e-6), (4.0, 1e-5)])
def test_exhaustive_verification_passes(eps, delta):
    report = verify_all(epsilon=eps, delta=delta)
    failed = [c["check"] for c in report["checks"] if not c["passed"]]
    assert not failed, failed
    # The sensitivity bounds are tight on this universe, not just valid.
    for c in report["checks"]:
        if c["check"].startswith("sensitivity"):
            assert c["bound_attained"], c


def _max_sum_change(domain, lo, hi):
    return max(abs(np.clip(a, lo, hi).sum() - np.clip(b, lo, hi).sum())
               for a, b in neighbour_pairs(domain, 3))


def test_wrong_sensitivity_would_be_caught():
    """Regression for two classic bugs, using the toy universe as the judge:
    (1) max(lo, hi) without abs() when lo < 0, and
    (2) using the centred half-range for an UNcentred sum."""
    domain, lo, hi = (-5.0, 0.0, 1.0), -5.0, 1.0
    observed = _max_sum_change(domain, lo, hi)
    assert observed <= S.bounded_sum_sensitivity(lo, hi)      # correct bound holds
    assert observed > max(lo, hi)                             # bug (1) is violated
    assert observed > S.centered_sum_sensitivity(lo, hi)      # bug (2) is violated


def test_invalid_bounds_rejected():
    with pytest.raises(ValueError):
        S.bounded_sum_sensitivity(3.0, 1.0)
    with pytest.raises(ValueError):
        S.vector_sum_sensitivities([0, 0], [1, -1])


def test_ledger_enforces_total():
    led = BudgetLedger(total_epsilon=1.0, total_delta=1e-5)
    led.spend("a", 0.6)
    with pytest.raises(BudgetExceededError):
        led.spend("b", 0.5)
    led.spend("c", 0.4, 1e-5)
    with pytest.raises(BudgetExceededError):
        led.spend("d", 1e-9, 1e-6)
    assert led.spent_epsilon == pytest.approx(1.0)


def test_mean_spends_exactly_its_budget_and_is_accurate():
    rng = np.random.default_rng(1)
    led = BudgetLedger(1.0, 1e-6)
    x = rng.uniform(40, 120, size=20_000)
    r = dp_mean("hr", x, 20, 250, led, 1.0, rng)
    assert led.spent_epsilon == pytest.approx(1.0) and len(led.entries) == 2
    assert r.value == pytest.approx(x.mean(), abs=0.5)
    g = dp_mean("hr", x, 20, 250, BudgetLedger(1.0, 1e-6), 1.0, rng, mechanism="gaussian", delta=1e-6)
    assert g.delta == 1e-6 and g.sensitivity_norm == "L2"


def test_mean_of_empty_input_is_defined():
    r = dp_mean("empty", [], 0, 10, BudgetLedger(1.0), 1.0, np.random.default_rng(0))
    assert 0 <= r.value <= 10


def test_nan_input_rejected():
    with pytest.raises(ValueError):
        dp_sum("s", [1.0, np.nan], 0, 2, BudgetLedger(1.0), 1.0, np.random.default_rng(0))


def test_count_and_histogram_release_metadata():
    rng = np.random.default_rng(0)
    led = BudgetLedger(2.0)
    c = dp_count("n", 100, led, 1.0, rng)
    h = dp_histogram("h", [1, 2, 2, 9], [0, 5, 10], led, 1.0, rng)
    assert c.noise_scale == 1.0 and c.value >= 0
    assert len(h.value) == 2 and h.sensitivity == 1.0
    assert "true" not in c.to_dict()  # releases never carry exact values
