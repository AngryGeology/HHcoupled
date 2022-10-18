#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 15:05:42 2022
Petrophysical functions for coupling SUTRA and R2. 
@author: jimmy
"""
import numpy as np 
from solveWaxSmit import solveRtWVP# , solveRtSt 
theta_param = np.genfromtxt('petroFit/theta_fit.txt')
ssf_poly_param = np.genfromtxt('petroFit/SSF_(Disturbed)_fit.txt')
Rw = 1.013171225937183451e+01
# saturation - waxman smit parameters (gmc)
gparam = {1:{'theta':0.36, # zone 1 is SSF 
             'Pg':2.74,
             'Rw':Rw,
             'cec':11.0,
             'FF':18.5,
             'n':3.05},
          2:{'theta':0.48, # zone 2 is WMF 
             'Pg':2.74,
             'Rw':Rw,
             'cec':22.5,
             'FF':9.42,
             'n':2.74}}# wax smit parameters

# saturation - waxman smit parameters (sat)
sparam = {2:{'Rw':Rw, 
             'cec':22.5,
             'FF':232.321,
             'n':2.695}}

#%% error handling 
def rmse(d0,dm):
    N = len(d0)
    diff = d0 - dm
    sumup = np.sum(diff**2)
    return np.sqrt(sumup/N)

def chi2(meas_errors,residuals):
    n = len(residuals)
    xsum = 0 
    for i in range(n):
        x = (residuals[i]**2) / (meas_errors[i]**2) 
        xsum += x 
    return xsum/n   

#%% gmc and sat conversion 
def sat2gmc(sat,theta,ps,pw=1):
    numon = pw*theta*sat 
    denom = ps*(1-theta)
    return numon/denom 

def gmc2sat(gmc,theta,ps,pw=1):
    numon = ((1-theta)*ps)*gmc 
    denom = pw*theta
    return numon/denom 

#%% petro functions in terms of GMC 
def wmf_petro(sat):
    # convert saturation to GMC 
    gmc = sat2gmc(sat,gparam[2]['theta'],gparam[2]['Pg'])
    # convert gmc to Rt 
    Rt = solveRtWVP(gmc, gparam[2]['Rw'], gparam[2]['Pg'], 1.00,
                    theta_param, gparam[2]['cec'], gparam[2]['FF'], 
                    gparam[2]['n'])
    return Rt 

def ssf_petro(sat):
    # convert saturation to GMC 
    gmc = sat2gmc(sat,gparam[1]['theta'],gparam[1]['Pg'])
    # convert gmc to Rt 
    Rt = solveRtWVP(gmc, gparam[1]['Rw'], gparam[2]['Pg'], 1.00,
                    theta_param, gparam[1]['cec'], gparam[1]['FF'], 
                    gparam[1]['n'])
    return Rt 

# def wmf_petro_sat(sat):
#     # solve in terms of native saturation 
#     Rt = solveRtSt(sat, gparam[2]['Rw'], gparam[2]['cec'], gparam[2]['FF'], 
#                    gparam[2]['n'])
#     return Rt 

#%% polynomial functions 
def ssf_polyn(sat):
    sat[sat<0.375] = 0.375
    sat[sat>0.8] = 0.8 
    Rt = np.polyval(ssf_poly_param,sat)
    return Rt 


# sat = np.array([0.926268, 0.92682,0.927427, 0.928095,0.92883,0.929639,0.93053,0.93151,0.932588,0.933775,0.93508])
