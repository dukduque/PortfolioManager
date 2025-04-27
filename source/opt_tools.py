"""
Created on Thu May 23 22:35:42 2019

@author: dduque

Implements several optimization models for
portfolio optimization.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp


class AbstractModel(ABC):
    """
    Abstract representation of an asset allocation model
    Attributes:
        m (object): a reference of a model that performs the optimization.
    """

    @abstractmethod
    def __init__(self):
        self.m = None

    @abstractmethod
    def optimize(self):
        pass


@dataclass
class CvarParameters:
    # CVaR level at risk. Fractile at risk. If L denotes the loss, 1-cvar_alpha
    # is the probability of having a L that exceeds the value at risk VaR(-L).
    # Typically this value is close to 1.
    alpha: float
    # Value for the convex combination between expected value and CVaR. If
    # beta = 0, this yields a CVaR minimization model.
    beta: float

def default_cvar_parameters() -> CvarParameters:
    return CvarParameters(alpha=0.90, beta=0.95)


class cvar_model_ortools(AbstractModel):
    def __init__(
        self,
        cvar_params,
        r,
        price,
        budget,
        current_portfolio=None,
        portfolio_delta=0,
        fractional=True,
        ignore=[],
        must_buy={},
    ):
        """
        Expectation/CVaR model (using OR-tools)
        Let x be the number of stock to purchase, at price p.
        E detones the expectation symbol and CVaR is computed
        at level cvar_alpha (>0.5).
        max beta E[sum_j r_j*p_j*x_j ] - (1-beta) CVaR(L)
            s.t.
            L = - sum_j r_j*p_j*x_j #The loss
            sum_j p_j*x_j <= budget

        Args:
            cvar_params (CvarParameters): CVaR parameters used to build the
                model.
            r (ndarray): returns data, where each column has the returns
                of a particular stock.
            price (DataFrame): price per stock.
            budget (float): total amount to be invested.
            current_portfolio (Portfolio): The current porfolio to revalance.
                If None, a new portfolio is created form scratch.
            portfolio_delta (float): number of shares by which the new
                portfolio can differ from the `current_portfolio`.
            fraction (bool): optional - if true (default), the portfolio is
                allow to have fractional number of stocks.
            ignore (list(str)): list of tickers to ignore in the optimization.
            must_buy (dict): A dictionary with tickers and the number of shares
                that must be bought from each ticker.
        """
        # CVaR paramters
        cvar_alpha = cvar_params.alpha
        assert cvar_alpha < 1.0, "CVaR alpha must be less than 1."
        cvar_beta = cvar_params.beta
        # Data prep
        assert len(r) > 0 and len(r[0]) == len(
            price
        ), "The number of securities in the returns array must match that in the price array"
        self.r_bar = np.mean(r, axis=0)
        self.cov = np.cov(r, rowvar=False)
        n = len(r)  # Number of returns
        stocks = price.index.to_list()
        portfolio_value = (
            0
            if current_portfolio is None
            else sum(
                price[s] * current_portfolio.get_position(s)
                for s in current_portfolio.assets
            )
        )
        new_portfolio_value = portfolio_value + budget
        solver = pywraplp.Solver.CreateSolver("CBC")

        # Number of shares to buy from each stock
        x = {}
        for s in stocks:
            x_lb = 0 if s not in must_buy else must_buy[s]
            x_ub = (
                new_portfolio_value / price[s]
                if (s not in ignore and price[s] > 0)
                else np.maximum(0.0, current_portfolio.get_position(s))
            )
            if fractional or current_portfolio.position_is_fractional(s):
                x[s] = solver.NumVar(x_lb, x_ub, f"x{s}")
            else:
                x[s] = solver.IntVar(x_lb, x_ub, f"x{s}")

        # Auxiliary variable to compute shortfall in cvar
        z = {}
        for j in range(n):
            z[j] = solver.NumVar(0.0, solver.infinity(), f"z{j}")

        # Value at risk
        eta = solver.NumVar(-solver.infinity(), solver.infinity(), "eta")

        # Cash
        cash = solver.NumVar(0, new_portfolio_value, "cash")

        # Portfolio budget contraint
        solver.Add(
            sum(price[s] * x[s] for s in stocks) + cash == new_portfolio_value
        )

        # CVaR computation, used in the objective function.
        cvar = eta + (1.0 / (n * (1 - cvar_alpha))) * sum(
            z[i] for i in range(n)
        )

        # CVaR linearlization
        for i in range(n):
            solver.Add(
                z[i]
                >= sum(
                    (
                        -(r[i, j]) * price[s] * x[s] / new_portfolio_value
                        for (j, s) in enumerate(stocks)
                    )
                )
                - cash / new_portfolio_value
                - eta
            )

        # Limit number of rebalancing sell transactions
        if current_portfolio is not None:
            delta_x = {}
            for s in current_portfolio.assets:
                position_s = current_portfolio.get_position(s)
                delta_x[s] = solver.NumVar(0.0, position_s, f"delta_x{s}")
                # solver.Add(x[s] - position_s <= delta_x[s])  # Buy
                solver.Add(position_s - x[s] <= delta_x[s])  # Sell
            solver.Add(
                sum(delta_x[s] for s in current_portfolio.assets)
                <= portfolio_delta
            )

        # Objective function
        exp_return = sum(
            self.r_bar[j] * price[s] * x[s] / new_portfolio_value
            for (j, s) in enumerate(stocks)
        )
        solver.Maximize(cvar_beta * exp_return - (1 - cvar_beta) * cvar)

        self.solver = solver
        self.cvar = cvar
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

    def optimize(self, mip_gap=0.0001):
        solver_parameters = pywraplp.MPSolverParameters()
        solver_parameters.SetDoubleParam(
            solver_parameters.RELATIVE_MIP_GAP, mip_gap
        )
        self.solver.Solve(solver_parameters)
        print("Objective func value:", self.solver.Objective().Value())
        print("Cash on hand:", self.cash.solution_value())
        x_sol = np.round(
            np.array([self.x[s].solution_value() for s in self.stocks]), 6
        )
        allocation = x_sol * self.price / np.sum(self.price * x_sol)
        sol_out = pd.DataFrame(
            {
                "price": self.price,
                "qty": x_sol,
                "position": x_sol * self.price,
                "allocation": allocation,
                "side": "buy",
            }
        )

        stats = {}
        stats["mean"] = self.r_bar.dot(allocation)
        stats["std"] = np.sqrt(allocation.dot(self.cov.dot(allocation)))
        stats["VaR"] = -self.eta.solution_value()
        stats["CVaR"] = -self.cvar.solution_value()

        return sol_out, stats

    def change_cvar_params(self, cvar_beta=None, cvar_alpha=None):
        self.cvar_beta = cvar_beta if cvar_beta is not None else self.cvar_beta
        self.cvar_alpha = (
            cvar_alpha if cvar_alpha is not None else self.cvar_alpha
        )
        print("Changing CVaR parameters:")
        print("CVaR_beta: %5.3f" % (self.cvar_beta))
        print("CVaR_alpha: %5.3f" % (self.cvar_alpha))

        if cvar_alpha is not None:
            self.cvar = self.eta + (
                1.0 / (self.n * (1 - self.cvar_alpha))
            ) * sum(self.z[i] for i in range(self.n))
            self.solver.Maximize(
                self.cvar_beta * self.exp_return
                - (1 - self.cvar_beta) * self.cvar
            )
            self.m.objective = (
                self.cvar_beta * self.exp_return
                - (1 - self.cvar_beta) * self.cvar
            )

        if cvar_beta is not None:
            self.solver.Maximize(
                self.cvar_beta * self.exp_return
                - (1 - self.cvar_beta) * self.cvar
            )

        return self.optimize()
