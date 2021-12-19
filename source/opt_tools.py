"""
Created on Thu May 23 22:35:42 2019

@author: dduque

Implements several optimization models for
portfolio optimization.
"""
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from itertools import product
from matplotlib import pyplot as plt
from pulp import LpProblem, LpVariable, LpMaximize, LpContinuous, LpInteger, lpSum, COIN_CMD
from pulp.apis.core import LpSolver_CMD
from ortools.linear_solver import pywraplp


class AbstractModel(ABC):
    '''
    Abstract representation of an asset allocation model
    Attributes:
        m (object): a reference of a model that performs the optimization.
    '''
    @abstractmethod
    def __init__(self):
        self.m = None
        pass
    
    @abstractmethod
    def optimize(self):
        pass


class cvar_model_ortools(AbstractModel):
    def __init__(self,
                 r,
                 price,
                 budget,
                 current_portfolio=None,
                 cvar_alpha=0.95,
                 cvar_beta=0.5,
                 cvar_bound=0,
                 portfolio_delta=0,
                 fractional=True,
                 ignore=[],
                 must_buy={}):
        '''
            Expectation/CVaR model (using PuLP)
            Let x be the number of stock to purchase, at price p.
            E detones the expectation symbol and CVaR is computed
            at level cvar_alpha (>0.5).
            max beta E[sum_j r_j*p_j*x_j ] - (1-beta) CVaR(L)
                s.t.
                L = - sum_j r_j*p_j*x_j #The loss
                sum_j p_j*x_j <= budget
                CVaR(L) >= cvar_bound (optional)

            Args:
                r (ndarray): returns data, where each column has the returns
                    of a particular stock.
                price (DataFrame): price per stock.
                budget (float): total amount to be invested.
                cvar_alpha (float): CVaR level at risk. Fractile at risk. If L denotes
                    the loss, 1-cvar_alpha is the probability of having a L that exceeds
                    the value at risk VaR(-L). Typically this value is close to 1.
                    Default value is 0.95.
                cvar_beta (float): optional - parameter used in the convex combination
                    for the objective function. beta=1 yields a full expectation
                    maximization model. beta = 0 yields a cvar minimization model. 
                    Default value is 0.5.
                cvar_bound (float): optional - a lower bound on the value cvar of the
                    portfolio. This parameters should be close (bellow) to the investment
                    budget for a risk-averse investor. Default value is zero.
                fraction (bool): optional - if true (default), the portfolio is allow to have
                    fractional numbers for the number of stocks on it.
        '''
        
        # Data prep
        assert len(r) > 0 and len(r[0]) == len(
            price
        ), 'The number of securities in the returns array must match that in the price array'
        self.r_bar = np.mean(r, axis=0)
        self.cov = np.cov(r, rowvar=False)
        n = len(r)  # Number of returns
        stocks = price.index.to_list()
        portfolio_value = 0 if current_portfolio is None else sum(
            price[s] * current_portfolio.get_position(s)
            for s in current_portfolio.assets)
        new_portfolio_value = portfolio_value + budget
        solver = pywraplp.Solver.CreateSolver('CBC')
        
        # Number of shares to buy from each stock
        x = {}
        for s in stocks:
            x_lb = 0 if s not in must_buy else must_buy[s]
            x_ub = np.max(new_portfolio_value /
                          price) if s not in ignore else np.maximum(
                              0.0, current_portfolio.get_position(s))
            if fractional or current_portfolio.position_is_fractional(s):
                x[s] = solver.NumVar(x_lb, x_ub, f'x{s}')
            else:
                x[s] = solver.IntVar(x_lb, x_ub, f'x{s}')
        
        # Auxiliary variable to compute shortfall in cvar
        z = {}
        for j in range(n):
            z[j] = solver.NumVar(0.0, solver.infinity(), f'z{j}')
        
        # Value at risk
        eta = solver.NumVar(-solver.infinity(), solver.infinity(), 'eta')
        
        # Cash
        cash = solver.NumVar(0, new_portfolio_value, 'cash')
        
        # Portfolio budget contraint
        solver.Add(
            sum(price[s] * x[s] for s in stocks) + cash == new_portfolio_value)
        
        # Risk constraint (>= becuase is a loss, i.e., want to bound loss from below)
        cvar = eta + (1.0 / (n * (1 - cvar_alpha))) * sum(z[i]
                                                          for i in range(n))
        #m += -cvar >= cvar_bound, 'cvar_ctr'
        
        # CVaR linearlization
        for i in range(n):
            solver.Add(
                z[i] >= sum((-(r[i, j]) * price[s] * x[s] / new_portfolio_value
                             for (j, s) in enumerate(stocks))) -
                cash / new_portfolio_value - eta)
        
        # Limit number of rebalancing sell transactions
        if current_portfolio is not None:
            delta_x = {}
            for s in current_portfolio.assets:
                position_s = current_portfolio.get_position(s)
                delta_x[s] = solver.NumVar(0.0, position_s, f'delta_x{s}')
                # solver.Add(x[s] - position_s <= delta_x[s])  # Buy
                solver.Add(position_s - x[s] <= delta_x[s])  # Sell
            solver.Add(
                sum(delta_x[s]
                    for s in current_portfolio.assets) <= portfolio_delta)
        
        # Objective function
        exp_return = sum(self.r_bar[j] * price[s] * x[s] / new_portfolio_value
                         for (j, s) in enumerate(stocks))
        exp_return = exp_return  # + cash / new_portfolio_value
        solver.Maximize(cvar_beta * exp_return - (1 - cvar_beta) * cvar)
        
        self.solver = solver
        self.cvar = cvar
        self.exp_return = exp_return
        self.cash = cash
        self.x = x
        self.z = z
        self.eta = eta
        self.cvar_bound = cvar_bound
        self.cvar_beta = cvar_beta
        self.cvar_alpha = cvar_alpha
        self.stocks = stocks
        self.price = price
        self.n = n
    
    def optimize(self, mip_gap=0.0001):
        solver_parameters = pywraplp.MPSolverParameters()
        solver_parameters.SetDoubleParam(solver_parameters.RELATIVE_MIP_GAP,
                                         mip_gap)
        self.solver.Solve(solver_parameters)
        print('Objective func value:', self.solver.Objective().Value())
        print('Cash on hand:', self.cash.solution_value())
        x_sol = np.round(
            np.array([self.x[s].solution_value() for s in self.stocks]), 6)
        allocation = x_sol * self.price / np.sum(self.price * x_sol)
        sol_out = pd.DataFrame({
            'price': self.price,
            'qty': x_sol,
            'position': x_sol * self.price,
            'allocation': allocation,
            'side': 'buy'
        })
        
        stats = {}
        stats['mean'] = self.r_bar.dot(allocation)
        stats['std'] = np.sqrt(allocation.dot(self.cov.dot(allocation)))
        stats['VaR'] = -self.eta.solution_value()
        stats['CVaR'] = -self.cvar.solution_value()
        
        return sol_out, stats
    
    def change_cvar_params(self,
                           cvar_beta=None,
                           cvar_alpha=None,
                           cvar_bound=None):
        self.cvar_bound = cvar_bound if cvar_bound is not None else self.cvar_bound
        self.cvar_beta = cvar_beta if cvar_beta is not None else self.cvar_beta
        self.cvar_alpha = cvar_alpha if cvar_alpha is not None else self.cvar_alpha
        print('Changing CVaR parameters:')
        print('CVaR_beta: %5.3f' % (self.cvar_beta))
        print('CVaR_alpha: %5.3f' % (self.cvar_alpha))
        print('CVaR_bound: %5.3f' % (self.cvar_bound))
        
        if cvar_bound is not None or cvar_alpha is not None:
            self.cvar = self.eta + (1.0 / (self.n *
                                           (1 - self.cvar_alpha))) * sum(
                                               self.z[i] for i in range(self.n))
            # self.m.constraints['cvar_ctr'] = -self.cvar >= self.cvar_bound
            self.solver.Maximize(self.cvar_beta * self.exp_return -
                                 (1 - self.cvar_beta) * self.cvar)
            self.m.objective = self.cvar_beta * self.exp_return - (
                1 - self.cvar_beta) * self.cvar
        
        if cvar_beta is not None:
            self.solver.Maximize(self.cvar_beta * self.exp_return -
                                 (1 - self.cvar_beta) * self.cvar)
        
        return self.optimize()
