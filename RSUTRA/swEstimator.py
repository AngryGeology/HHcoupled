#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 14 12:27:40 2023
Script to simulate the effect of rainfall in 1d on Sw and hence near surface 
pore pressures. 
@author: jimmy
"""
import time 
import numpy as np 
import matplotlib.pyplot as plt 
from scipy.interpolate import interp1d 
plt.close('all')

#%% functions 
def invVGcurve(water_content,sat,res,alpha,n,m = None):
    """
    Inverse of van genutchen curve where we solve for suction not moisture content. 

    Parameters
    ----------
    water_content : float, nd array (of float)
        Water content as a fraction 
    sat : float 
        Saturated water content (normally 1).
    res : float 
        Residual water content.
    alpha : float 
        alpha parameters.
    n : float 
        Van genutchen n parameter
    m : float, optional
        M parameter. The default is None (computed as a function of n)

    Returns
    -------
    pressure: float, nd array 
        Matric potential (positive).

    """
    if m is None:
        m=1-(1/n)
        
    thetan = (water_content-res)/(sat-res)
    step1 = ((1/thetan)**(1/m)) - 1 
    step2 = step1**(1/n)
    
    return step2 *(1/alpha)

def relK(water_content,sat,res,n):
    """
    Compute unsaturated relative hydraulic conductivity according to van Genutchen's
    equation 8 (1980). 

    Parameters
    ----------
    water_content : float, nd array (of float)
        Water content as a fraction 
    sat : float 
        Saturated water content (normally 1).
    res : float 
        Residual water content.
    n : float 
        Van genutchen n parameter
    Returns
    -------
    Kr: float, nd array 
        relative permeability(positive).

    """
    theta = (water_content-res)/(sat-res)
    m=1-(1/n)                                                   
    LHS=np.sqrt(theta)
    MID=(1-(theta**(1/m)))**m
    RHS=(1-MID)**2 
    KR=LHS*RHS
    return KR 

def unsatFlow(k,kr,por,sw,u,dp,rho=1000,g=9.81):
    """
    Compute unsaturated flow according to a generalised darcy's law  

    Parameters
    ----------
    k : TYPE
        DESCRIPTION.
    kr : TYPE
        DESCRIPTION.
    por : TYPE
        DESCRIPTION.
    sw : TYPE
        DESCRIPTION.
    u : TYPE
        DESCRIPTION.
    dp : TYPE
        DESCRIPTION.
    rho : TYPE, optional
        DESCRIPTION. The default is 1000.
    g : TYPE, optional
        DESCRIPTION. The default is 9.81.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    lhs = (k*kr)/(por*sw*u)
    rhs = dp - (rho*g)
    return lhs*rhs 
    
def computeQFlux(sw,sat,res,por,alpha,n,k,u):
    """
    Compute fluxes

    Parameters
    ----------
    Suz : TYPE
        DESCRIPTION.
    maxSuz : TYPE
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    if sw<=res: # cant have unsaturated flow in this case 
        return 0 
    pc = invVGcurve(sw, sat, res, alpha, n)
    kr = relK(sw, sat, res, n)
    if kr<0: 
        kr = 0 
    elif kr>1:
        kr = 1 
    Qo = unsatFlow(k, kr, por, sw, u, pc)
    return Qo 
    
def computeEFlux(sw,Et,res,sat):
    """
    Compute evaporation flux 

    Parameters
    ----------
    sw : TYPE
        DESCRIPTION.
    Et : TYPE
        DESCRIPTION.
    res : TYPE
        DESCRIPTION.
    sat : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    if sw <= res: 
        return 0
    elif sw>=sat: 
        return Et 
    else:
        thetan = (sw-res)/(sat-res)
        return thetan*Et
    
    
def computeODE(Pe,Et,Suz,maxSuz,sat,res,por,alpha,n,k,u):
    """
    Compute ordinary differential equations. 

    Parameters
    ----------
    Pe : TYPE
        DESCRIPTION.
    Et : TYPE
        DESCRIPTION.
    Suz : TYPE
        DESCRIPTION.
    maxSuz : TYPE
        DESCRIPTION.

    Returns
    -------
    dSuz_dt : TYPE
        DESCRIPTION.

    """
    if Suz > maxSuz: 
        Suz = maxSuz 
        print('WARNING: unsaturated storage exceeds maximum storage')
    sw = Suz/maxSuz # Calculate saturation 
    Qo = computeQFlux(sw,sat,res,por,alpha,n,k,u) # unsaturated flux 
    Ea = computeEFlux(sw,Et,res,sat)
    dSuz_dt = Pe - Ea - Qo 
    if sw<=res and dSuz_dt < 0: 
        return 0.0 # ensure that storage cannot go below residual saturation 
    if sw==1.0 and dSuz_dt >= 0:
        return 0.0# ensure that storage cannot exceed capacity 
    return dSuz_dt

def euler(x0, y0, dx, nsteps=10, func=None, **kwargs):
    """Explicit Euler method

    Parameters
    ----------
    x0 : float, int
        Start condition on the x axis.
    y0 : float, int
        Start condition on the y axis.
    dx : float, int
        step in x axis (the finer the greater the approximation).
    nsteps : int, optional
        Number of steps to take. The default is 10.
    func: function
        Function which is to be solved at each step in the x direction. 
    kwargs:
        Keyword arguments which are passed to the function 

    Raises
    ------
    ValueError
        if nsteps is not an integer.

    Returns
    -------
    x : nd array
        Approximate x positions along function
    y : nd array
        Approximate y positions along function.

    """
    if not isinstance(nsteps, int):
        raise ValueError('Number of steps must be an integer')
    if func is None:
        def func(x,y): #place holder function 
            return y
    x = np.zeros(nsteps,dtype=float)
    y = np.zeros(nsteps,dtype=float)
    x[0] = x0
    y[0] = y0
    for i in range(1,nsteps):
        x[i] = x[i-1] + dx
        y[i] = y[i-1] + dx*func(x[i-1], y[i-1], **kwargs)
    return x, y

def backwardEuler(x0, y0, dx, nsteps=10, func=None, **kwargs):
    """Implicit Euler method. We have to solve for the unknown y variable 
    at each step, here an iterative minimisation scheme is used. Seems to work
    okay if the timestep is small enough but can also be vunrable to divergence. 

    Parameters
    ----------
    x0 : float, intinfil
        Start condition on the x axis.
    y0 : float, int
        Start condition on the y axis.
    dx : float, int
        step in x axis (the finer the greater the approximation).
    nsteps : int, optional
        Number of steps to take. The default is 10.
    func: function
        Function which is to be solved at each step in the x direction. 
    kwargs:
        Keyword arguments which are passed to the function 

    Raises
    ------
    ValueError
        if nsteps is not an integer.

    Returns
    -------
    x : nd array
        Approximate x positions along function
    y : nd array
        Approximate y positions along function.

    """
    if not isinstance(nsteps, int):
        raise ValueError('Number of steps must be an integer')
    if func is None:
        def func(x,y): #place holder function 
            return y
    x = np.zeros(nsteps,dtype=float)
    y = np.zeros(nsteps,dtype=float)
    x[0] = x0
    y[0] = y0
    flag = False
    for i in range(nsteps-1):
        #approximation calculation
        x[i+1] = x[i] + dx
        y_guess = y[i] # start at previous value
        y_calc = y[i] + dx*func(x[i+1], y_guess, **kwargs) #problem here is we don't know the next y value
        delta = y_calc - y_guess
        y_guess = y_guess + delta
        #minmisation while loop - doesnt work too well
        count=0
        while abs(delta)>0.0001: #works to 1e-4 accuracy
            y_calc = y[i] + dx*func(x[i+1], y_guess, **kwargs)
            delta = y_calc - y_guess
            y_guess = y_guess + delta
            count+=1
            if count>100:#break if 50 iterations exceeded (stops us from getting stuck indefinitely)
                flag = True
                break
        y[i+1] = y_guess
    
    if flag:
        print('WARNING: Maximum iterations exceeded in Implicit Euler scheme')
    return x, y

def estSwfromRainfall(Pr,Et,ts,sat,res,por,alpha,n,k,u,ifac=10,
                      maxSuz=None,t0=0,Suz0=None,backward=True):
    if maxSuz is None: 
        maxSuz = por
    if Suz0 is None: 
        Suz0 = maxSuz*res
    pfunc = interp1d(ts, Pr)
    efunc = interp1d(ts, Et) 
    #define a wrapper function that can be passed to the euler scheme 
    def f(t,Suz): #takes time and unsaturated storage as input 
        P = pfunc(t)
        E = efunc(t)
        diff = computeODE(P,E,Suz,maxSuz,sat,res,por,alpha,n,k,u)# compute ordinary differential equation 
        #... so it effectively decouples unsaturated storage from canopy storage
        return diff # for the differential with respect to unsat storage want the second differential (index=1)
    
    dt = (ts[1]-ts[0])/ifac
    maxtime = max(ts)
    nsteps = int(maxtime/dt) # the number of steps required to reach simulation end 
    c0 = time.time()
    print('Approxing unsat storage with euler scheme: ',end='')
    print('dT = {:6.4f}\tNsteps = {:5d}\t Duration ='.format(dt,nsteps),end='')
    if backward: 
        timesteps, approxSuz = backwardEuler(t0,Suz0,dt,nsteps,func=f)
    else:
        timesteps, approxSuz = euler(t0,Suz0,dt,nsteps,func=f)
    c1 = (time.time()-c0)*1000
    print('{:6.2f} ms'.format(c1))
    
    # return timesteps, approxSuz, approxSuz/maxSuz
    
    sfunc = interp1d(timesteps, approxSuz/maxSuz,fill_value='extrapolate')
    # print(max(timesteps),max(ts))
    
    return sfunc(ts)
