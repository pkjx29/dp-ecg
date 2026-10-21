"""Exhaustive checks on a small dataset universe.

What these checks do and do not establish:

* Sensitivity: for EVERY neighbour pair in the toy universe, the query moves
  by no more than the claimed bound (and the bound is attained). This tests
  the implementation of the bound on that universe; the bound itself holds
  for all datasets by the proofs in KNOWLEDGE.md.
* Geometric mechanism: the output space is the integers, and outside a
  window around the two true answers the pmf ratio is constant, so checking
  that window checks EVERY output exactly. This is a complete check of the
  ε-DP inequality for each pair in the universe.
* Laplace: the supremum of the log density ratio has a closed form,
  |f(D) − f(D')| / b; we check it is ≤ ε for every pair. A grid evaluation of
  the densities is reported as a supplementary numerical check (finite grid,
  so tails are not covered by it).
* Gaussian: for every pair the exact δ(ε) at the pair's actual shift is
  computed in closed form and checked against the target δ.

None of this is a proof of DP for arbitrary data. The proof is the
mechanism theorem plus the sensitivity bound; these checks guard against
implementation mistakes in either.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import laplace as laplace_dist

from . import sensitivity as S
from .discrete import geometric_alpha, geometric_log_pmf
from .gaussian import gaussian_exact_delta, gaussian_sigma
from .laplace import laplace_scale
from .neighbours import is_add_remove_neighbour, neighbour_pairs

DEFAULT_DOMAIN = (-2, 0, 1, 3)
DEFAULT_MAX_SIZE = 4
LOWER, UPPER = -2.0, 3.0


def _queries(lower: float, upper: float, bin_edges: np.ndarray) -> dict:
    c = (lower + upper) / 2
    return {
        "count": (lambda d: np.array([len(d)], float), S.count_sensitivity(), S.count_sensitivity()),
        "clipped sum": (lambda d: np.array([np.clip(d, lower, upper).sum()]),
                        S.bounded_sum_sensitivity(lower, upper), S.bounded_sum_sensitivity(lower, upper)),
        "centered clipped sum": (lambda d: np.array([(np.clip(d, lower, upper) - c).sum()]),
                                 S.centered_sum_sensitivity(lower, upper),
                                 S.centered_sum_sensitivity(lower, upper)),
        "histogram": (lambda d: np.histogram(np.asarray(d, float), bins=bin_edges)[0].astype(float),
                      S.histogram_l1_sensitivity(), S.histogram_l1_sensitivity()),
    }


def verify_all(epsilon: float = 1.0, delta: float = 1e-5, domain: tuple = DEFAULT_DOMAIN,
               max_size: int = DEFAULT_MAX_SIZE, lower: float = LOWER, upper: float = UPPER) -> dict:
    pairs = neighbour_pairs(domain, max_size)
    bin_edges = np.array([-2.5, -0.5, 0.5, 2.0, 3.5])
    report: dict = {"adjacency": "add/remove one record", "domain": list(domain),
                    "max_size": max_size, "n_pairs": len(pairs), "epsilon": epsilon,
                    "delta": delta, "checks": []}

    def add(name, passed, detail):
        report["checks"].append({"check": name, "passed": bool(passed), **detail})

    add("neighbour generation", all(is_add_remove_neighbour(a, b) for a, b in pairs),
        {"what": "every generated pair differs by exactly one added record"})

    for qname, (f, l1, l2) in _queries(lower, upper, bin_edges).items():
        diffs = np.array([f(a) - f(b) for a, b in pairs])
        obs_l1 = float(np.abs(diffs).sum(axis=1).max())
        obs_l2 = float(np.linalg.norm(diffs, axis=1).max())
        add(f"sensitivity: {qname}", obs_l1 <= l1 + 1e-12 and obs_l2 <= l2 + 1e-12,
            {"claimed_l1": l1, "observed_max_l1": obs_l1, "claimed_l2": l2,
             "observed_max_l2": obs_l2, "bound_attained": bool(np.isclose(obs_l1, l1))})

        # Laplace: closed-form worst-case log ratio per pair.
        b = laplace_scale(l1, epsilon)
        worst = float(np.abs(diffs).sum(axis=1).max() / b)
        add(f"laplace ε-DP (closed form): {qname}", worst <= epsilon + 1e-12,
            {"max_log_ratio": worst, "what": "sup_y log p_D(y)/p_D'(y) = ||Δf||_1 / b"})

        # Gaussian: exact δ at each pair's actual L2 shift.
        sigma = gaussian_sigma(l2, epsilon, delta, "analytic")
        shifts = np.linalg.norm(diffs, axis=1)
        worst_delta = max(gaussian_exact_delta(float(s), sigma, epsilon) for s in shifts if s > 0)
        add(f"gaussian (ε,δ)-DP (exact δ): {qname}", worst_delta <= delta,
            {"sigma": sigma, "max_exact_delta": worst_delta})

    # Laplace, supplementary numerical grid for the scalar sum.
    f, l1, _ = _queries(lower, upper, bin_edges)["clipped sum"]
    b = laplace_scale(l1, epsilon)
    grid = np.linspace(-40, 40, 80_001)
    grid_worst = max(float(np.abs(laplace_dist.logpdf(grid, f(a)[0], b)
                                  - laplace_dist.logpdf(grid, f(bb)[0], b)).max()) for a, bb in pairs)
    add("laplace ε-DP (numerical grid, supplementary): clipped sum", grid_worst <= epsilon + 1e-9,
        {"max_log_ratio_on_grid": grid_worst, "grid": "[-40, 40], step 0.001",
         "limitation": "finite grid; tails covered by the closed-form check, not this one"})

    # Geometric: exact check on every integer output.
    f = _queries(lower, upper, bin_edges)["count"][0]
    alpha = geometric_alpha(S.count_sensitivity(), epsilon)
    worst_geo = 0.0
    for a, bb in pairs:
        ca, cb = int(f(a)[0]), int(f(bb)[0])
        # Outside [min-1, max+1] the ratio is constant, so this window is exhaustive.
        ys = np.arange(min(ca, cb) - 1, max(ca, cb) + 2)
        lr = np.abs(geometric_log_pmf(ys - ca, alpha) - geometric_log_pmf(ys - cb, alpha))
        worst_geo = max(worst_geo, float(lr.max()))
    add("geometric ε-DP (exact, all integer outputs): count", worst_geo <= epsilon + 1e-12,
        {"max_log_ratio": worst_geo})

    report["all_passed"] = all(c["passed"] for c in report["checks"])
    return report
