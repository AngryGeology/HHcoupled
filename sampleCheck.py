#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 16:03:36 2024
step walk and sampling excercise 
@author: jimmy
"""
import os, sys
import numpy as np 
import matplotlib.pyplot as plt 

# add custom modules 
# if 'RSUTRA' not in sys.path: 
sys.path.append(os.path.abspath('RSUTRA'))
    
from SUTRAhandler import giveValues, stepWalk
    
v0 = 0.01 
v1 = 10.0 
n = 10000

K0_uniform = giveValues(v0, v1, n)
K1_uniform = giveValues(v0, v1, n)
K0_log = giveValues(v0, v1, n, 'loguniform')
K1_log = giveValues(v0, v1, n, 'loguniform')

fig, ax = plt.subplots()

ax.scatter(K0_uniform, K1_uniform, c='b')
ax.scatter(K0_log, K1_log, c='r')

ax.set_xscale('log')
ax.set_yscale('log')

#%% step walk 
step = 0.5
nstep = 10000
a = [1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
S0_uniform = np.repeat(a, 1000)
S1_uniform = np.repeat(a, 1000)
S0_log = np.repeat(a, 1000)
S1_log = np.repeat(a, 1000)
nstep= len(S0_log)

# for i in range(nstep): 
#     S0_uniform[i] += stepWalk(step)
#     S1_uniform[i] += stepWalk(step)
#     # S0_log[i] += stepWalk(step,'lognormal')
#     # S1_log[i] += stepWalk(step,'lognormal')
#     step0 = np.random.lognormal(0,step,1)[0]
#     step1 = np.random.lognormal(0,step,1)[0]
#     u0 = np.log10(S0_log[i])+(step0)
#     S0_log[i] = 10**u0
#     u1 = np.log10(S1_log[i])+(step1)
#     S1_log[i] = 10**u1

for i in range(nstep): 
    S0_uniform[i] = stepWalk(S0_uniform[i], step)
    S1_uniform[i] = stepWalk(S1_uniform[i], step)
    S0_log[i] = stepWalk(S0_log[i], step,'lognormal')
    S1_log[i] = stepWalk(S1_log[i], step,'lognormal')


fig, ax = plt.subplots()

ax.scatter(S0_uniform, S1_uniform, marker='.', c='b')
ax.scatter(S0_log, S1_log, marker='.', c='r')

ax.set_xscale('log')
ax.set_yscale('log')

ax.scatter(a,a,c='k')