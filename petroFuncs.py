#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 15:05:42 2022
Petrophysical functions for coupling SUTRA and R2. 
@author: jimmy
"""
import numpy as np 
from solveWaxSmit import solveRtWVP, solveRtSt 
theta_param = np.genfromtxt('petroFit/theta_fit.txt')

def parse_fit(fname):
    # parse fitted parameters 
    param = {}
    fh = open(fname,'r')
    lines = fh.readlines()
    fh.close() 
    for line in lines:
        if line =='':
            continue 
        info = line.split(':')
        key = info[0]
        values = [float(a) for a in info[-1].split()]
        if len(values) == 1: 
            param[key] = values[0]
        else:
            param[key] = np.array(values)
    return param 
    
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
sparam = {1:parse_fit('petroFit/SSF_(Disturbed)_fit.txt'),
          2:parse_fit('petroFit/WMF_(Insitu)_fit.txt')}

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

#%% solve wmf in terms saturation 
def wmf_petro_sat(sat):
    # solve in terms of native saturation 
    Rt = solveRtSt(sat, sparam[2]['Rw'], sparam[2]['cec'], sparam[2]['FF'], 
                   sparam[2]['n'])
    return Rt 

#%% solve ssf in terms of saturation with a polynomail 
def ssf_polyn(sat):
    min_sat = sparam[1]['min']#
    max_sat = sparam[1]['max']
    pfit = sparam[1]['pfit']
    pfit0 = sparam[1]['pfit0']
    pfit1 = sparam[1]['pfit1']
    cidx = (sat >= min_sat) & (sat <= max_sat) # central index 
    lidx = sat < min_sat # data left of where we actually have data 
    ridx = sat > max_sat # data right of where we have data 
    Rt = np.zeros_like(sat)
    Rt[cidx] = np.polyval(pfit,sat[cidx])
    Rt[lidx] = np.polyval(pfit0,sat[lidx])
    Rt[ridx] = np.polyval(pfit1,sat[ridx])
    
    return Rt 

