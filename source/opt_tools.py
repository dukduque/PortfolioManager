"""Portfolio optimisation via a mean–CVaR mixed-integer programme.

Mathematical formulation
------------------------
Given *n* historical scenarios and *m* assets, let:

    r_{ij}  gross return (price ratio) of asset j in scenario i,
              r_{ij} = price_{j,t+1} / price_{j,t}  (> 0)
    r̄_j     sample mean gross return: r̄_j = (1/n) Σ_i r_{ij}
    p_j     current price of asset j
    B       total portfolio budget = current equity + available cash
    α ∈ (0,1)  CVaR confidence level (e.g. 0.90 → worst 10 % of scenarios)
    β ∈ [0,1]  convex weight on expected return vs CVaR (see CvarParameters)

Decision variables:

    x_j ≥ 0       shares held in asset j  (continuous, or integer when
                   fractional=False and the position is not already fractional)
    cash ≥ 0      uninvested cash; earns 0 net return
    η ∈ ℝ         Value at Risk threshold (negative when portfolio typically gains)
    z_i ≥ 0       per-scenario shortfall above η  (CVaR auxiliary)

Portfolio gross return in scenario i:

    ρ_i = ( Σ_j r_{ij} · p_j · x_j  +  cash ) / B

Portfolio gross loss in scenario i  (negative of gross return):

    L_i = −ρ_i = −Σ_j r_{ij} · p_j · x_j / B  −  cash / B

The CVaR at level α is expressed via the Rockafellar–Uryasev (2000)
linearisation:

    CVaR_α(L) = min_η { η + 1 / [n(1−α)] · Σ_i (L_i − η)⁺ }

Substituting z_i ≥ L_i − η, z_i ≥ 0:

    CVaR = η + 1 / [n(1−α)] · Σ_i z_i

Full programme
~~~~~~~~~~~~~~
    max   β · Σ_j r̄_j · p_j · x_j / B
        − (1−β) · [ η + 1/(n(1−α)) · Σ_i z_i ]            (P)

    s.t.  Σ_j p_j · x_j + cash = B                         budget
          z_i ≥ −Σ_j r_{ij} · p_j · x_j / B
                − cash / B − η        ∀ i                   CVaR linearisation
          z_i ≥ 0                     ∀ i
          η  ∈ ℝ
          cash ∈ [0, B]
          x_j ∈ [0, B/p_j]  or  ℤ≥0  (per asset)

Optional turnover constraint (active when portfolio_delta is finite):

    Let pos_j = current shares held in asset j.
    Introduce δ_j ≥ 0 with δ_j ≥ pos_j − x_j  (sell amount for asset j).
    Then: Σ_j δ_j ≤ Δ  limits total shares reduced across the portfolio.

Objective interpretation
~~~~~~~~~~~~~~~~~~~~~~~~
    β = 0   → pure CVaR minimisation (most conservative)
    β = 1   → maximise expected gross return (ignores risk)
    β = 0.95 (default) → high weight on return, CVaR acts as a soft penalty

Reported statistics
~~~~~~~~~~~~~~~~~~~
    VaR  = −η              (positive when the α-quantile loss is a gain)
    CVaR = −CVaR_value     (positive value = expected gross loss in worst scenarios)
    mean = r̄ · w           (expected portfolio gross return; w = allocation weights)
    std  = √(w᷊ Σ w)        (portfolio gross-return standard deviation)

References
----------
Rockafellar, R.T. & Uryasev, S. (2000). Optimization of Conditional
Value-at-Risk. Journal of Risk, 2(3), 21–41.
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

log = logging.getLogger(__name__)

# Solver status codes returned by OR-Tools CBC.
_STATUS_NAMES = {
    pywraplp.Solver.OPTIMAL:    "OPTIMAL",
    pywraplp.Solver.FEASIBLE:   "FEASIBLE",
    pywraplp.Solver.INFEASIBLE: "INFEASIBLE",
    pywraplp.Solver.ABNORMAL:   "ABNORMAL",
    pywraplp.Solver.NOT_SOLVED: "NOT_SOLVED",
}


class AbstractModel(ABC):
    """Abstract representation of an asset-allocation optimisation model."""

    @abstractmethod
    def __init__(self):
        self.m = None

    @abstractmethod
    def optimize(self):
        pass


@dataclass
class CvarParameters:
    """Parameters controlling the mean–CVaR objective (P).

    Attributes:
        alpha: CVaR confidence level α ∈ (0, 1).  The CVaR is computed over
               the worst (1−α) fraction of scenarios.  Typical value: 0.90
               (CVaR covers the worst 10 % of return scenarios).
        beta:  Convex weight β ∈ [0, 1] on the expected-return term of (P).
               β = 0 → minimise CVaR only; β = 1 → maximise expected return
               only; β = 0.95 (default) → strong return focus with CVaR
               penalty.
    """
    alpha: float
    beta: float


def default_cvar_parameters() -> CvarParameters:
    """Return α = 0.90, β = 0.95 (return-focused with tail-risk penalty)."""
    return CvarParameters(alpha=0.90, beta=0.95)


class cvar_model_ortools(AbstractModel):
    """Mean–CVaR MIP solved with OR-Tools CBC.

    Builds the programme (P) described in the module docstring and exposes
    ``change_cvar_params`` to re-solve under different α / β without
    rebuilding the constraint matrix.

    Args:
        cvar_params:       CVaR confidence level and objective weight.
        r:                 (n × m) ndarray of gross returns (price ratios).
                           Rows are scenarios; columns are assets in the same
                           order as *price*.
        price:             pd.Series indexed by ticker with current prices.
        budget:            Available cash to invest on top of the current
                           portfolio equity (i.e. account.cash_onhand).
        current_portfolio: Existing Portfolio whose positions seed the budget
                           calculation.  Pass None to build from scratch.
        portfolio_delta:   Maximum total shares that may be reduced across
                           all current positions (turnover limit Δ).
                           0 = no sells allowed; large value = unconstrained.
        fractional:        Allow fractional share quantities.  When False,
                           x_j is integer unless the existing position is
                           already fractional.
        ignore:            Tickers excluded from the optimisation (forced to
                           retain their current position at most).
        must_buy:          Dict mapping ticker → minimum shares that must be
                           held in the solution.
        time_limit_ms:     Wall-clock time limit for the CBC solver in
                           milliseconds.  Raises RuntimeError if the limit is
                           reached without a feasible solution.  Default 30 s.
    """

    def __init__(
        self,
        cvar_params: CvarParameters,
        r: np.ndarray,
        price: pd.Series,
        budget: float,
        current_portfolio=None,
        portfolio_delta: float = 0,
        fractional: bool = True,
        ignore: list = [],
        must_buy: dict = {},
        time_limit_ms: int = 30_000,
    ):
        cvar_alpha = cvar_params.alpha
        assert cvar_alpha < 1.0, "CVaR alpha must be less than 1."
        cvar_beta = cvar_params.beta

        assert len(r) > 0 and len(r[0]) == len(price), (
            "Columns of r must match entries of price "
            f"(got {len(r[0])} vs {len(price)})."
        )

        self.r_bar = np.mean(r, axis=0)
        self.cov = np.cov(r, rowvar=False)
        n = len(r)          # number of scenarios
        stocks = price.index.to_list()

        portfolio_value = (
            0
            if current_portfolio is None
            else sum(
                price[s] * current_portfolio.get_position(s)
                for s in current_portfolio.assets
            )
        )
        B = portfolio_value + budget   # total budget

        solver = pywraplp.Solver.CreateSolver("CBC")
        solver.set_time_limit(time_limit_ms)

        # ── Decision variables ────────────────────────────────────────────

        # x_j: shares held in each asset (continuous or integer per asset)
        x = {}
        for s in stocks:
            x_lb = 0 if s not in must_buy else must_buy[s]
            x_ub = (
                B / price[s]
                if (s not in ignore and price[s] > 0)
                else np.maximum(0.0, current_portfolio.get_position(s))
            )
            if fractional or current_portfolio.position_is_fractional(s):
                x[s] = solver.NumVar(x_lb, x_ub, f"x_{s}")
            else:
                x[s] = solver.IntVar(x_lb, x_ub, f"x_{s}")

        # z_i: per-scenario shortfall above VaR threshold η
        z = {i: solver.NumVar(0.0, solver.infinity(), f"z_{i}") for i in range(n)}

        # η: Value at Risk threshold (unbounded; can be negative)
        eta = solver.NumVar(-solver.infinity(), solver.infinity(), "eta")

        # cash: uninvested portion of budget
        cash = solver.NumVar(0, B, "cash")

        # ── Constraints ───────────────────────────────────────────────────

        # Budget: Σ p_j x_j + cash = B
        solver.Add(sum(price[s] * x[s] for s in stocks) + cash == B)

        # CVaR linearisation: z_i ≥ L_i − η  where  L_i = −ρ_i
        for i in range(n):
            solver.Add(
                z[i]
                >= sum(
                    -r[i, j] * price[s] * x[s] / B
                    for (j, s) in enumerate(stocks)
                )
                - cash / B
                - eta
            )

        # Turnover limit: total shares reduced ≤ portfolio_delta
        if current_portfolio is not None:
            delta_x = {}
            for s in current_portfolio.assets:
                pos_s = current_portfolio.get_position(s)
                delta_x[s] = solver.NumVar(0.0, pos_s, f"delta_{s}")
                solver.Add(pos_s - x[s] <= delta_x[s])
            solver.Add(
                sum(delta_x[s] for s in current_portfolio.assets)
                <= portfolio_delta
            )

        # ── Objective: max β·E[ρ] − (1−β)·CVaR_α(L) ─────────────────────
        cvar_expr = eta + (1.0 / (n * (1 - cvar_alpha))) * sum(z[i] for i in range(n))
        exp_return = sum(
            self.r_bar[j] * price[s] * x[s] / B
            for (j, s) in enumerate(stocks)
        )
        solver.Maximize(cvar_beta * exp_return - (1 - cvar_beta) * cvar_expr)

        # ── Persist references for optimize() and change_cvar_params() ────
        self.solver = solver
        self.cvar = cvar_expr
        self.exp_return = exp_return
        self.cash = cash
        self.x = x
        self.z = z
        self.eta = eta
        self.cvar_beta = cvar_beta
        self.cvar_alpha = cvar_alpha
        self.stocks = stocks
        self.price = price
        self.n = n
        self.B = B
        self._time_limit_ms = time_limit_ms

    def optimize(self, mip_gap: float = 0.0001) -> tuple:
        """Solve (P) and return the portfolio DataFrame plus statistics.

        Args:
            mip_gap: Relative MIP optimality gap tolerance.  The solver
                     stops when the gap between the best integer solution
                     and the LP relaxation bound falls below this value.
                     Default 0.01 % (tight; increase to ~0.01 to trade
                     optimality for speed on large universes).

        Returns:
            sol_out: pd.DataFrame indexed by ticker with columns
                     price, qty, position (= price × qty), allocation
                     (fraction of total equity), side ('buy').
            stats:   dict with keys:
                       mean   – expected portfolio gross return (scalar)
                       std    – portfolio gross-return std dev (scalar)
                       VaR    – Value at Risk = −η  (positive ≈ loss)
                       CVaR   – CVaR of gross loss  (positive ≈ loss)
                       status – solver status string (e.g. 'OPTIMAL')

        Raises:
            RuntimeError: if the solver returns INFEASIBLE, ABNORMAL, or
                          NOT_SOLVED (including time-limit expiry without a
                          feasible point).
        """
        params = pywraplp.MPSolverParameters()
        params.SetDoubleParam(params.RELATIVE_MIP_GAP, mip_gap)

        status = self.solver.Solve(params)
        status_name = _STATUS_NAMES.get(status, str(status))

        if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
            raise RuntimeError(
                f"CVaR solver returned {status_name}. "
                "Check that the returns matrix is non-degenerate and that "
                "the budget is sufficient to buy at least one asset."
            )

        if status == pywraplp.Solver.FEASIBLE:
            log.warning(
                "CVaR solver reached time limit (%d ms) with a feasible but "
                "possibly sub-optimal solution (gap may exceed %.4f).",
                self._time_limit_ms, mip_gap,
            )

        log.debug(
            "CVaR solver: status=%s  objective=%.6f  cash=%.2f  wall_time=%d ms",
            status_name,
            self.solver.Objective().Value(),
            self.cash.solution_value(),
            self.solver.wall_time(),
        )

        x_sol = np.round(
            np.array([self.x[s].solution_value() for s in self.stocks]), 6
        )
        equity = x_sol * self.price
        total_equity = equity.sum()
        allocation = equity / total_equity if total_equity > 0 else equity * 0

        sol_out = pd.DataFrame({
            "price":      self.price,
            "qty":        x_sol,
            "position":   equity,
            "allocation": allocation,
            "side":       "buy",
        })

        stats = {
            "mean":   float(self.r_bar.dot(allocation)),
            "std":    float(np.sqrt(allocation.dot(self.cov.dot(allocation)))),
            "VaR":    float(-self.eta.solution_value()),
            "CVaR":   float(-self.cvar.solution_value()),
            "status": status_name,
        }
        return sol_out, stats

    def change_cvar_params(
        self,
        cvar_beta: float = None,
        cvar_alpha: float = None,
    ) -> tuple:
        """Update α and/or β, rebuild the objective, and re-solve.

        Only the objective function is modified; the constraint matrix is
        reused, so this is much cheaper than constructing a new model.

        Args:
            cvar_beta:  New β value (objective weight on expected return).
            cvar_alpha: New α value (CVaR confidence level).  Triggers a
                        rebuild of the CVaR expression inside the objective.

        Returns:
            Same (sol_out, stats) tuple as optimize().
        """
        if cvar_beta is not None:
            self.cvar_beta = cvar_beta
        if cvar_alpha is not None:
            self.cvar_alpha = cvar_alpha
            self.cvar = self.eta + (
                1.0 / (self.n * (1 - self.cvar_alpha))
            ) * sum(self.z[i] for i in range(self.n))

        log.debug(
            "Updating CVaR params: alpha=%.3f  beta=%.3f",
            self.cvar_alpha, self.cvar_beta,
        )
        self.solver.Maximize(
            self.cvar_beta * self.exp_return
            - (1 - self.cvar_beta) * self.cvar
        )
        return self.optimize()
