#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 15:05:42 2022
Petrophysical functions for coupling SUTRA and R2. 
@author: jimmy
"""
import numpy as np 
from solveWaxSmit import solveRtWVP#, solveRtSt 
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
sparam = {1:parse_fit('petroFit/SSF_fit.txt'),
          2:parse_fit('petroFit/WMF_fit.txt'),
          3:parse_fit('petroFit/Shallow_WMF_fit.txt')}

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

def vmc2sat(vwc,theta):
    return vwc/theta 

#%% generic fitting 
def powerLaw(x,a,k,c):
    return (1/(a*(x**k))) + c 

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
    Rt = powerLaw(sat, sparam[2]['a'], sparam[2]['k'], sparam[2]['c'])
    return Rt 

def ssf_petro_sat(sat):
    # solve in terms of native saturation 
    Rt = powerLaw(sat, sparam[1]['a'], sparam[1]['k'], sparam[1]['c'])
    return Rt 

def wmf_petro_sat_shallow(sat):
    # solve in terms of native saturation 
    Rt = powerLaw(sat, sparam[3]['a'], sparam[3]['k'], sparam[3]['c'])
    return Rt  

#%% temperature correction 
def temp_model(day, z, Tm=10.3, dT=15.54, d=2.26, phi=-1.91):
    """Fit a model to seasonal tempature variation. Default parameters are correct
    for Hollin Hill. 
    
    Parameters
    ------------
    day: float, np array
        Julian day
    z: float, np array
        Depth
    Tm: float
        Annual mean temperature
    dT: float
        The peak-to-trough amplitude of the temperature variation ΔT
    d: float
        A characteristic depth d at which ΔT has decreased by 1/e
    phi: float
        A phase offset φ to bring surface and air temperature into phase
    
    Returns
    -----------
    Tmodel(t,z): float, np array
        Temperature model 
        
    Notes
    -----------
    From Ulhemann et al (2017) - "From the field data, we obtained 
    Tmean = 10.03°C, ΔT = 15.54°C, d = 2.26 m, and ϕ =1.91."
    
        
    """
    Tmdl = (dT/2)*np.exp(-z/d)*np.sin(((2*np.pi*day)/365)+phi-(z/d)) 
    return Tm + Tmdl

def temp_uncorrect(res0,depth,day, tc=-2,Tstd=20):
    """
    Put lab derived resistivity back in terms of in field resistivity. 

    Parameters
    ----------
    res0 : nd array 
        DESCRIPTION.
    depth : float 
        DESCRIPTION.
    day : float, int 
        DESCRIPTION.
    tc : float, int, optional
        DESCRIPTION. The default is -2.
    Tstd : float, int, optional
        DESCRIPTION. The default is 20.

    Returns
    -------
    Rf: nd array 
        Uncorrected resistivity. 

    """
    Tmdl = temp_model(day, depth)
    rhs = 1 + ((tc/100)*(Tstd-Tmdl))
    return res0/rhs 

def temp_correct(doy,depth,Res,
                 c=-2.0,
                 TCor  = 20.0,      # to correct to assumed laboratory temperature
                 Tmean = 10.029,
                 dT = 15.54,
                 d = 2.264,
                 phi = -1.907):
    """Temperature correct timelapse resistivity volumes. Code has been converted
    from Ulhemann's matlab code. 
    
    Parameters
    -------------
    doy: int
        Day of year in which survey took place. 
    depth: array like
        An array of cell depths, corresponding to each cell in the modelling mesh
    Res: array like
        Array of resistivity values corresponding to each cell the mesh (must be 
        the same length as depth)
        
    "other": floats 
        All other parameters are taken from Seb's workflow, best to leave as
        default. These are the temperature model parameters. 
    
    Returns
    --------------
    R_tcor: array 
        Array of temperature corrected resistivity values 
    """
    depth = np.array(depth)
    Res = np.array(Res)
    
    R_tcor = Res*(1+(c/100)*(TCor - (Tmean + (dT/2)*np.exp(-abs(depth)/d)*np.sin((2*np.pi*doy/365) + phi - (abs(depth)/d)))))
    
    return R_tcor

