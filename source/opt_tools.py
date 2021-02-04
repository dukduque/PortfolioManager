#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 22:35:42 2019

@author: dduque

Implements several optimiztion models for
porfolio optimization.
"""
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from itertools import product
from matplotlib import pyplot as plt
from pulp import LpProblem, LpVariable, LpMaximize, LpContinuous, LpInteger, lpSum, COIN_CMD
from pulp.solvers import PULP_CBC_CMD
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


class markowitz_dro_wasserstein(AbstractModel):
    def __init__(self, data, price, budget, delta_param, alpha_param, wasserstein_norm=1):
        '''
        Model from Blanchet et al. 2017
        DRO Markovitz reformulation from Wasserstein distance.
        '''
        from gurobipy import GRB, Model, quicksum
        r = np.mean(data, axis=0)
        cov = np.cov(data, rowvar=False)
        k = len(r)
        n = len(data)
        m = Model('opt_profolio')
        m.params.OutputFlag = 1
        m.params.TimeLimit = 100
        m.params.MIPGap = 0.01
        #m.params.NumericFocus = 3
        x = m.addVars(k, lb=0, ub=1, vtype=GRB.CONTINUOUS, name='x')
        norm_p = m.addVar(lb=0, ub=1, vtype=GRB.CONTINUOUS, name='norm')
        p_SD = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='p_var')
        m.update()
        
        sqrt_delta = np.sqrt(delta_param)
        m.addConstr((x.sum() == 1), 'portfolio_ctr')
        m.addConstr((quicksum(x[j] * r[j] for j in range(k)) >= alpha_param - sqrt_delta * norm_p), 'return_ctr')
        m.addConstr((p_SD * p_SD >= quicksum(cov[i, j] * x[i] * x[j] for i in range(k) for j in range(k))), 'SD_def')
        objfun = p_SD * p_SD + 2 * p_SD * sqrt_delta * norm_p + delta_param * norm_p * norm_p
        m.setObjective(objfun, GRB.MINIMIZE)
        
        if wasserstein_norm == 1:
            regularizer_norm = 'inf'
            m.addConstrs((norm_p >= x[j] for j in range(k)), 'norm_def')
        elif wasserstein_norm == 2:
            regularizer_norm = 2
            m.addConstr((norm_p * norm_p >= (quicksum(x[j] * x[j] for j in range(k)))), 'norm_def')
        elif wasserstein_norm == 'inf':
            regularizer_norm = 1
            #Note: this works since x>=0
            m.addConstr((norm_p == (quicksum(x[j] for j in range(k)))), 'norm_def')
        else:
            raise 'wasserstain norm should be 1,2, or inf'
        
        #optimize
        m.optimize()
        x_sol = np.array([x[j].X for j in range(k)])
        p_mean = r.dot(x_sol)
        p_var = x_sol.dot(cov.dot(x_sol))
        #print(x_sol, p_mean, p_var)
        #print('norms' , np.linalg.norm(x_sol) , norm_p.X)
        #return x_sol, p_mean, p_var


class cvar_model(AbstractModel):
    def __init__(self, r, price, budget, cvar_alpha=0.95, cvar_beta=0.5, cvar_bound=0, fractional=True):
        '''
            Expectation/CVaR model
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
                    the loss, 1-cvar_alpha is the probabily of have a L that exceeds 
                    the value at risk VaR(-L). Typically this value is close to 1.
                    Default value is 0.95.
                cvar_beta (float): optional - parameter used in the convex combination 
                    for the objective function. beta=1 yields a full expectation
                    maximization model. beta = 0 yields a cvar minimization model. 
                    Default value is 0.5.
                cvar_bound (float): optional - a lower bound on the value cvar of the
                    porfolio. This parameters should be close (bellow) to the investement
                    budget for a risk-averse investor. Default value is zero.
                fraction (bool): optional - if true (default), the portfolio is allow to have 
                    fractional numbers for the number of stocks on it. 
                
        '''
        self.m = None
        self.cvar = None
        self.exp_return = None
        self.x = None
        self.z = None
        self.eta = None
        
        self.r_bar = np.mean(r, axis=0)
        self.cov = np.cov(r, rowvar=False)
        
        n = len(r)  # Number of returns
        stocks = price.index.to_list()
        
        from gurobipy import GRB, Model, quicksum
        m = Model('opt_profolio_cvar')
        m.params.OutputFlag = 0
        #m.params.MIPGap = 0.00001
        
        #Number of shares to by
        varType = GRB.CONTINUOUS if fractional else GRB.INTEGER
        x = m.addVars(stocks, lb=0, ub=budget / price, vtype=varType, name='x')
        #Auxiliary variable to compute shortfall in cvar
        z = m.addVars(n, lb=0, vtype=GRB.CONTINUOUS, name='z')
        #Value at risk
        eta = m.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='eta')
        
        #cvar_bound = m.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='cvar')
        m.update()
        
        #Portfolio contraint
        m.addConstr((quicksum(price[s] * x[s] for s in stocks) <= budget), 'portfolio_budget')
        #Risk constraint (>= becuase is a loss, i.e., want to bound loss from below)
        cvar = eta + (1.0 / (n * (1 - cvar_alpha))) * z.sum()
        m.addConstr((-cvar >= cvar_bound), 'cvar_ctr')
        #CVaR linearlization
        m.addConstrs((z[i] >= quicksum(-(r[i, j]) * price[s] * x[s] for (j, s) in enumerate(stocks)) - eta
                      for i in range(n)), 'cvar_linear')
        #Objective function
        #m.setObjective(quicksum(self.r_bar[j]*price[j]*x[j] for j in range(k)), GRB.MAXIMIZE)
        exp_return = quicksum(self.r_bar[j] * price[s] * x[s] for (j, s) in enumerate(stocks))
        
        m.setObjective(cvar_beta * exp_return - (1 - cvar_beta) * cvar, GRB.MAXIMIZE)
        m.update()
        
        self.m = m
        self.cvar = cvar
        self.exp_return = exp_return
        self.x = x
        self.z = z
        self.eta = eta
        self.cvar_bound = cvar_bound
        self.cvar_beta = cvar_beta
        self.cvar_alpha = cvar_alpha
        self.stocks = stocks
        self.price = price
        self.n = n
        
        #return self.optimize()
    
    def optimize(self):
        self.m.optimize()
        print('Objective func value:', self.m.ObjVal)
        x_sol = np.array([self.x[j].X for j in self.stocks])
        allocation = x_sol * self.price / np.sum(self.price * x_sol)
        sol_out = pd.DataFrame({
            'price': self.price,
            'stock': x_sol,
            'position': x_sol * self.price,
            'allocation': allocation
        })
        
        stats = {}
        stats['mean'] = self.r_bar.dot(allocation)
        stats['std'] = np.sqrt(allocation.dot(self.cov.dot(allocation)))
        stats['VaR'] = -self.eta.X
        stats['CVaR'] = -self.cvar.getValue()
        
        return sol_out, stats
    
    def change_cvar_params(self, cvar_beta=None, cvar_alpha=None, cvar_bound=None):
        self.cvar_bound = cvar_bound if cvar_bound != None else self.cvar_bound
        self.cvar_beta = cvar_beta if cvar_beta != None else self.cvar_beta
        self.cvar_alpha = cvar_alpha if cvar_alpha != None else self.cvar_alpha
        print('Changing CVaR parameters:')
        print('CVaR_beta: %5.3f' % (self.cvar_beta))
        print('CVaR_alpha: %5.3f' % (self.cvar_alpha))
        print('CVaR_bound: %5.3f' % (self.cvar_bound))
        
        if cvar_bound is not None or cvar_bound is not None:
            self.m.remove(self.m.getConstrByName('cvar_ctr'))
            self.cvar = self.eta + (1.0 / (self.n * (1 - cvar_alpha))) * self.z.sum()
            self.m.addConstr((-self.cvar >= self.cvar_bound), 'cvar_ctr')
            self.m.setObjective(self.cvar_beta * self.exp_return - (1 - cvar_beta) * self.cvar, GRB.MAXIMIZE)
        
        if cvar_beta is not None:
            self.m.setObjective(self.cvar_beta * self.exp_return - (1 - cvar_beta) * self.cvar, GRB.MAXIMIZE)
        
        return self.optimize()


class cvar_model_pulp(AbstractModel):
    def __init__(self, r, price, budget, cvar_alpha=0.95, cvar_beta=0.5, cvar_bound=0, fractional=True):
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
                    the loss, 1-cvar_alpha is the probabily of having a L that exceeds
                    the value at risk VaR(-L). Typically this value is close to 1.
                    Default value is 0.95.
                cvar_beta (float): optional - parameter used in the convex combination
                    for the objective function. beta=1 yields a full expectation
                    maximization model. beta = 0 yields a cvar minimization model. 
                    Default value is 0.5.
                cvar_bound (float): optional - a lower bound on the value cvar of the
                    porfolio. This parameters should be close (bellow) to the investement
                    budget for a risk-averse investor. Default value is zero.
                fraction (bool): optional - if true (default), the portfolio is allow to have
                    fractional numbers for the number of stocks on it.
        '''
        self.m = None
        self.cvar = None
        self.exp_return = None
        self.x = None
        self.z = None
        self.eta = None
        
        self.r_bar = np.mean(r, axis=0)
        self.cov = np.cov(r, rowvar=False)
        
        n = len(r)  # Number of returns
        stocks = price.index.to_list()
        m = LpProblem(name='cvar_model', sense=LpMaximize)
        
        # Number of shares to by
        varType = LpContinuous if fractional else LpInteger
        x = LpVariable.dicts(name='x', indexs=stocks, lowBound=0, upBound=np.max(budget / price), cat=varType)
        
        # Auxiliary variable to compute shortfall in cvar
        z = LpVariable.dicts(name='z', indexs=range(n), lowBound=0, cat=LpContinuous)
        
        # Value at risk
        eta = LpVariable(name='eta', cat=LpContinuous)
        
        # cvar_bound = m.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='cvar')
        
        # Portfolio contraint
        m += lpSum([price[s] * x[s] for s in stocks]) <= budget, 'portfolio_budget'
        
        # Risk constraint (>= becuase is a loss, i.e., want to bound loss from below)
        cvar = eta + (1.0 / (n * (1 - cvar_alpha))) * lpSum(z)
        m += -cvar >= cvar_bound, 'cvar_ctr'
        
        # CVaR linearlization
        for i in range(n):
            m += z[i] >= lpSum(
                (-(r[i, j]) * price[s] * x[s] for (j, s) in enumerate(stocks))) - eta, 'cvar_linear_%i' % (i)
        
        # Objective function
        # m.setObjective(quicksum(self.r_bar[j]*price[j]*x[j] for j in range(k)), GRB.MAXIMIZE)
        exp_return = lpSum([self.r_bar[j] * price[s] * x[s] for (j, s) in enumerate(stocks)])
        
        m += cvar_beta * exp_return - (1 - cvar_beta) * cvar
        
        self.m = m
        self.cvar = cvar
        self.exp_return = exp_return
        self.x = x
        self.z = z
        self.eta = eta
        self.cvar_bound = cvar_bound
        self.cvar_beta = cvar_beta
        self.cvar_alpha = cvar_alpha
        self.stocks = stocks
        self.price = price
        self.n = n
        
        #return self.optimize()
    
    def optimize(self, mip_gap=0.001):
        cbc_solver = PULP_CBC_CMD(msg=1, fracGap=mip_gap)
        cbc_solver.solve(self.m)
        print('Objective func value:', self.m.objective.value())
        x_sol = np.array([self.x[j].value() for j in self.stocks])
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
        stats['VaR'] = -self.eta.value()
        stats['CVaR'] = -self.cvar.value()
        
        return sol_out, stats
    
    def change_cvar_params(self, cvar_beta=None, cvar_alpha=None, cvar_bound=None):
        self.cvar_bound = cvar_bound if cvar_bound is not None else self.cvar_bound
        self.cvar_beta = cvar_beta if cvar_beta is not None else self.cvar_beta
        self.cvar_alpha = cvar_alpha if cvar_alpha is not None else self.cvar_alpha
        print('Changing CVaR parameters:')
        print('CVaR_beta: %5.3f' % (self.cvar_beta))
        print('CVaR_alpha: %5.3f' % (self.cvar_alpha))
        print('CVaR_bound: %5.3f' % (self.cvar_bound))
        
        if cvar_bound is not None or cvar_alpha is not None:
            self.cvar = self.eta + (1.0 / (self.n * (1 - self.cvar_alpha))) * lpSum(self.z)
            self.m.constraints['cvar_ctr'] = -self.cvar >= self.cvar_bound
            self.m.objective = self.cvar_beta * self.exp_return - (1 - self.cvar_beta) * self.cvar
        
        if cvar_beta is not None:
            self.m.objective = self.cvar_beta * self.exp_return - (1 - self.cvar_beta) * self.cvar
        
        return self.optimize()


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
        
        # Data prep
        assert len(r) > 0 and len(
            r[0]) == len(price), 'The number of securities in the returns array must match that in the price array'
        self.r_bar = np.mean(r, axis=0)
        self.cov = np.cov(r, rowvar=False)
        n = len(r)  # Number of returns
        stocks = price.index.to_list()
        portfolio_value = 0 if current_portfolio is None else sum(price[s] * current_portfolio.get_position(s)
                                                                  for s in current_portfolio.assets)
        new_portfolio_value = portfolio_value + budget
        solver = pywraplp.Solver.CreateSolver('CBC')
        
        # Number of shares to buy from each stock
        x = {}
        for s in stocks:
            x_lb = 0 if s not in must_buy else must_buy[s]
            x_ub = np.max(new_portfolio_value /
                          price) if s not in ignore else np.maximum(0.0, current_portfolio.get_position(s))
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
        solver.Add(sum(price[s] * x[s] for s in stocks) + cash == new_portfolio_value)
        
        # Risk constraint (>= becuase is a loss, i.e., want to bound loss from below)
        cvar = eta + (1.0 / (n * (1 - cvar_alpha))) * sum(z[i] for i in range(n))
        #m += -cvar >= cvar_bound, 'cvar_ctr'
        
        # CVaR linearlization
        for i in range(n):
            solver.Add(z[i] >= sum((-(r[i, j]) * price[s] * x[s] / new_portfolio_value
                                    for (j, s) in enumerate(stocks))) - cash / new_portfolio_value - eta)
        
        # Limit number of rebalancing sell transactions
        if current_portfolio is not None:
            delta_x = {}
            for s in current_portfolio.assets:
                position_s = current_portfolio.get_position(s)
                delta_x[s] = solver.NumVar(0.0, position_s, f'delta_x{s}')
                # solver.Add(x[s] - position_s <= delta_x[s])  # Buy
                solver.Add(position_s - x[s] <= delta_x[s])  # Sell
            solver.Add(sum(delta_x[s] for s in current_portfolio.assets) <= portfolio_delta)
        
        # Objective function
        exp_return = sum(self.r_bar[j] * price[s] * x[s] / new_portfolio_value for (j, s) in enumerate(stocks))
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
    
    def optimize(self, mip_gap=0.001):
        # self.solver.parameters.RELATIVE_MIP_GAP = 1.0
        # print('Param changed ', param_change)
        # self.solver.EnableOutput()
        # self.solver.set_time_limit(3000)
        #        pywraplp.MPSolverParameters.SetDoubleParam(param=pywraplp.MPSolverParameters.RELATIVE_MIP_GAP, value=1)
        p1 = pywraplp.MPSolverParameters()
        p1.SetDoubleParam(p1.RELATIVE_MIP_GAP, mip_gap)
        self.solver.Solve(p1)
        print('Objective func value:', self.solver.Objective().Value())
        print('Cash on hand:', self.cash.solution_value())
        x_sol = np.round(np.array([self.x[s].solution_value() for s in self.stocks]), 6)
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
    
    def change_cvar_params(self, cvar_beta=None, cvar_alpha=None, cvar_bound=None):
        self.cvar_bound = cvar_bound if cvar_bound is not None else self.cvar_bound
        self.cvar_beta = cvar_beta if cvar_beta is not None else self.cvar_beta
        self.cvar_alpha = cvar_alpha if cvar_alpha is not None else self.cvar_alpha
        print('Changing CVaR parameters:')
        print('CVaR_beta: %5.3f' % (self.cvar_beta))
        print('CVaR_alpha: %5.3f' % (self.cvar_alpha))
        print('CVaR_bound: %5.3f' % (self.cvar_bound))
        
        if cvar_bound is not None or cvar_alpha is not None:
            self.cvar = self.eta + (1.0 / (self.n * (1 - self.cvar_alpha))) * sum(self.z[i] for i in range(self.n))
            # self.m.constraints['cvar_ctr'] = -self.cvar >= self.cvar_bound
            self.solver.Maximize(self.cvar_beta * self.exp_return - (1 - self.cvar_beta) * self.cvar)
            self.m.objective = self.cvar_beta * self.exp_return - (1 - self.cvar_beta) * self.cvar
        
        if cvar_beta is not None:
            self.solver.Maximize(self.cvar_beta * self.exp_return - (1 - self.cvar_beta) * self.cvar)
        
        return self.optimize()


class ssd_model_pulp(AbstractModel):
    '''
    Model based on second order stochastic dominance
    '''
    def __init__(self,
                 r,
                 price,
                 budget,
                 benchmark,
                 current_portfolio=None,
                 portfolio_delta=0,
                 fractional=True,
                 ignore=[],
                 must_buy={}):
        '''
        Constructor of an SSD model
        Args:
                returns (DataFrame): returns data, where each column has the returns
                    of a particular stock.
                price (DataSeries): price of the stocks
                benchmark (DataSeries): allocation of each stock in a benchmark. The number
                    of elements in the series should match the number of columns in returns.
        '''
        self.m = None
        self.x = None
        self.z = None
        self.ssd = None
        
        self.r_bar = np.mean(r, axis=0)
        self.cov = np.cov(r, rowvar=False)
        
        Y = benchmark * budget  # Distribution of the benchmark
        Y = np.percentile(Y, q=[10, 25, 50, 75, 90])  # [i * 1 for i in range(1, 101)])
        Y.sort()
        print(Y)
        
        nY = len(Y)  # Number of returns
        n = len(r)  # Number of returns
        stocks = price.index.to_list()
        m = LpProblem(name='SSD_model', sense=LpMaximize)
        
        # Number of shares to by
        varType = LpContinuous if fractional else LpInteger
        x = LpVariable.dicts(name='x', indexs=stocks, lowBound=0, upBound=np.max(budget / price), cat=varType)
        
        # Auxiliary variable to compute shortfall in SSD
        z_index = list(product(range(nY), range(n)))
        z = LpVariable.dicts(name='z', indexs=z_index, lowBound=0, cat=LpContinuous)
        
        # Slack variable of the SSD constraint
        ssd = LpVariable.dicts(name='s', indexs=range(nY), cat=LpContinuous)
        
        # Min max variable
        min_ssd = LpVariable(name='s', cat=LpContinuous)
        print('Done with vars')
        # Portfolio contraint
        m += lpSum([price[s] * x[s] for s in stocks]) <= budget, 'portfolio_budget'
        print('Loading SSD ctrs: ', len(z_index))
        # Slack computation of SSD constraint
        ij_counter = 0
        X = [lpSum((r[j, k] * price[s] * x[s] for (k, s) in enumerate(stocks))) for j in range(n)]
        for (i, j) in z_index:
            m += z[i, j] >= Y[i] - X[j], 'ctr_z_%i_%i' % (i, j)
            ij_counter += 1
            if ij_counter % 1000 == 0:
                print(ij_counter, '  at ', str((i, j)))
        
        print('Finished SSD ctrs: ', len(z_index))
        for i in range(nY):
            exp_Y_shortfall_i = sum(np.maximum(0, Y[i] - Y[j]) for j in range(nY)) / nY
            m += ssd[i] == exp_Y_shortfall_i - (lpSum(z[i, j] for j in range(n)) / n), 'ssd_slack_%i' % (i)
            m += min_ssd <= ssd[i], 'min_max_ctr_%i' % (i)
        
        m += lpSum(ssd)  # min_ssd
        print('Done model')
        self.m = m
        self.x = x
        self.z = z
        self.ssd = ssd
        self.stocks = stocks
        self.price = price
        self.n = n
        
        self.Y = benchmark  # Distribution of the benchmark
        self.r = r
        
        # return self.optimize()
    
    def optimize(self, mip_gap=0.01):
        cbc_solver = PULP_CBC_CMD(msg=1, fracGap=mip_gap, maxSeconds=300)
        cbc_solver.solve(self.m)
        print('Objective func value:', self.m.objective.value())
        x_sol = np.array([self.x[j].value() for j in self.stocks])
        print([self.ssd[i].value() for i in self.ssd])
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
        
        Y = self.Y
        X = self.r.dot(allocation)
        plt.hist(X, bins=30, color='b', alpha=0.9)
        plt.hist(Y, bins=30, color='r', alpha=0.6)
        plt.tight_layout()
        plt.show()
        return sol_out, stats
