#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 23 22:35:42 2019

@author: dduque
"""
import numpy as np
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