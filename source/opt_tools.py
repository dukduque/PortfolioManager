#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 22:35:42 2019

@author: dduque
"""
import numpy as np
import pandas as pd
from gurobipy import GRB, Model, quicksum

def markovitz_dro_wasserstein(data, delta_param, alpha_param, wasserstein_norm=1):
    '''
    Model from Blanchet et al. 2017
    DRO Markovitz reformulation from Wasserstein distance.
    '''
    
    r = np.mean(data, axis=0)
    cov = np.cov(data, rowvar=False)
    k = len(r)
    n = len(data)
    m = Model('opt_profolio')
    m.params.OutputFlag = 1
    #m.params.NumericFocus = 3
    x = m.addVars(k,lb=0,ub=1,vtype=GRB.CONTINUOUS,name='x')
    norm_p = m.addVar(lb=0,ub=1, vtype=GRB.CONTINUOUS, name='norm')
    p_SD = m.addVar(lb=0, vtype=GRB.CONTINUOUS, name='p_var')
    m.update()
    
    sqrt_delta =np.sqrt(delta_param)
    m.addConstr((x.sum()==1), 'portfolio_ctr')
    m.addConstr((quicksum(x[j]*r[j] for j in range(k)) >= alpha_param - sqrt_delta*norm_p ), 'return_ctr' )
    m.addConstr((p_SD*p_SD>=quicksum(cov[i,j]*x[i]*x[j] for i in range(k) for j in range(k))), 'SD_def')
    objfun = p_SD*p_SD + 2*p_SD*sqrt_delta*norm_p + delta_param*norm_p*norm_p
    m.setObjective(objfun, GRB.MINIMIZE)
    
    if wasserstein_norm == 1:
        regularizer_norm = 'inf'
        m.addConstrs((norm_p>=x[j] for j in range(k)), 'norm_def')
    elif wasserstein_norm ==2 :
        regularizer_norm = 2
        m.addConstr((norm_p*norm_p>=(quicksum(x[j]*x[j] for j in range(k)))), 'norm_def')
    elif wasserstein_norm == 'inf':
        regularizer_norm = 1
        #Note: this works since x>=0
        m.addConstr((norm_p==(quicksum(x[j] for j in range(k)))), 'norm_def')
    else:
        raise 'wasserstain norm should be 1,2, or inf'
        
    #optimize
    m.optimize()
    x_sol =np.array([x[j].X for j in range(k)])
    p_mean = r.dot(x_sol)
    p_var  = x_sol.dot(cov.dot(x_sol))
    #print(x_sol, p_mean, p_var)
    #print('norms' , np.linalg.norm(x_sol) , norm_p.X)

    return x_sol, p_mean, p_var




def cvar_model(r, price, budget, cvar_alpha=0.95, cvar_beta=0.5, cvar_bound=0, fractional = True):
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
    
    
    r_bar = np.mean(r, axis=0)
    cov = np.cov(r, rowvar=False)    
    n = len(r) #Number of returns
    k = len(r[0]) #Number of stocks
    m = Model('opt_profolio_cvar')
    m.params.OutputFlag = 1
    cvar_alpha = 0.95
    #Number of shares to by
    varType = GRB.CONTINUOUS if fractional else GRB.INTEGER
    x = m.addVars(k,lb=0,ub=budget/price, vtype=varType, name='x')
    #Auxiliary variable to compute shortfall in cvar
    z = m.addVars(n,lb=0, vtype=GRB.CONTINUOUS,name='z')
    #Value at risk
    eta = m.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='eta')
    
    #cvar_bound = m.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name='cvar')
    m.update()
    
    #Portfolio contraint
    m.addConstr((quicksum(price[i]*x[i] for i in range(k))<=budget), 'portfolio_budget')
    #Risk constraint (>= becuase is a loss, i.e., want to bound loss from below)
    cvar = eta+(1.0/(n*(1-cvar_alpha)))*z.sum()
    m.addConstr((-cvar>=cvar_bound), 'cvar_ctr')
    #CVaR linearlization
    m.addConstrs((z[i]>=quicksum(-(r[i,j])*price[j]*x[j] for j in range(k))-eta  for i in range(n)), 'cvar_linear')
    #Objective function
    #m.setObjective(quicksum(r_bar[j]*price[j]*x[j] for j in range(k)), GRB.MAXIMIZE)
    exp_return = quicksum(r_bar[j]*price[j]*x[j] for j in range(k))
    
    m.setObjective(cvar_beta*exp_return - (1-cvar_beta)*cvar, GRB.MAXIMIZE)
    m.update()
    
    m.optimize()
    x_sol =np.array([x[j].X for j in range(k)])
    allocation = x_sol*price/np.sum(price*x_sol)
    sol_out = pd.DataFrame({'price':price,'stock':x_sol,'position':x_sol*price,'allocation':allocation})
   
    stats = {}
    stats['mean'] = r_bar.dot(allocation)
    stats['std'] = np.sqrt(allocation.dot(cov.dot(allocation)))
    stats['VaR'] = -eta.X
    stats['CVaR'] = -cvar.getValue()
    

    return sol_out, stats


