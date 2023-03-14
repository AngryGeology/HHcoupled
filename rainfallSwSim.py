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
# global var 
sat = 1
res = 0.2 
alpha = 0.01e-3
n=1.5
u=1e-3 
por=0.5
k=2e-13
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
    
def computeQFlux(Suz,maxSuz):
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
    if Suz > maxSuz: 
        Suz = maxSuz 
        print('WARNING: unsaturated storage exceeds maximum storage')
    sw = Suz/maxSuz 
    if sw<=res:
        return 0 
    pc = invVGcurve(sw, sat, res, alpha, n)
    kr = relK(sw,sat,res,n)
    if kr<0: 
        kr = 0 
    elif kr>1:
        kr = 1 
    Qo = unsatFlow(k, kr, por, sw, u, pc)
    return Qo 
    
def computeODE(Pe,Et,Suz,maxSuz):
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
    dSuz_dt = Pe - Et - computeQFlux(Suz, maxSuz)
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
            if count>50:#break if 50 iterations exceeded (stops us from getting stuck indefinitely)
                flag = True
                break
        y[i+1] = y_guess
    
    if flag:
        print('WARNING: Maximum iterations exceeded in Implicit Euler scheme')
    return x, y

#%% program 

# simulate precipitation using synthetic model 
def timeVaryPrecip(t,t_hat=2,sigma_t=0.24,Pmax=40):
    P = Pmax*np.exp(-((t_hat-t)/sigma_t)**2)
    return P

secinday = 60*60*24
ts = np.linspace(0,20,1000)# times at which to similate precipitation 
precipSim = timeVaryPrecip(ts) # simulate how precipitation varies over time using given function 
ts = ts*secinday
evapSim = np.zeros_like(precipSim)
#precipSim is in mm/day # convert to m/s
precipSim = (precipSim/1000)/secinday 
pfunc = interp1d(ts, precipSim)
efunc = interp1d(ts, evapSim) 


fig, ax = plt.subplots()
ax.plot(ts/secinday,precipSim)
ax.set_ylabel('Precipitation (m/s)')
ax.set_xlabel('Time (days)')

#%% solve for unsaturated zone storage
#initial conditions 

t0 = 0
maxSuz = por
Suz0 = maxSuz*0.2   
#define a wrapper function that can be passed to the euler scheme 
def f(t,Suz, #takes time and unsaturated storage as input 
      maxSuz=maxSuz):

    Pe = max(precipSim) # pfunc(t)
    Et = efunc(t)
    diff = computeODE(Pe,Et,Suz,maxSuz) # compute ordinary differential equation 
    #... so it effectively decouples unsaturated storage from canopy storage
    return diff # for the differential with respect to unsat storage want the second differential (index=1)

#make figure to store graphs 
fig03, ax = plt.subplots()

#work out time steps in days and number of steps needed to span 10 days
secInDay = 24*60*60 # number of seconds in a day 
secInHour = 60*60 # number of seconds in an hour
secIn5min = 5*60 # seconds in 5 minutes 
dt = [secInDay,secInHour,secIn5min]
# dt = [secInDay/secInDay, secInHour/secInDay, secIn5min/secInDay] 
maxtime = max(ts)
nsteps = [int(maxtime/dt[i]) for i in range(len(dt))] # the number of steps required to go up to 10 days 
color = ['r','g','b'] # a list of strings used to color code the resulting graphs 
markers = ['v','o',None] # a list of markers to use for each graph
#loop through and compute different stepping times 
for i in range(len(dt)):#time steps are the same as they were for part 4.3.2
    c0 = time.time()
    print('Approxing unsat storage with euler scheme: ',end='')
    print('dT = {:6.4f}\tNsteps = {:5d}\t Duration ='.format(dt[i],nsteps[i]),end='')
    # timesteps, approxUnsatStore = euler(t0,Suz0,dt[i],nsteps[i],func=f)
    timesteps, approxUnsatStore = backwardEuler(t0,Suz0,dt[i],nsteps[i],func=f) # uncomment to use backward scheme (NB the results are very unstable)
    label = 'dt={:6.4f}, N={:5d}'.format(dt[i],nsteps[i])
    ax.plot(timesteps,approxUnsatStore,c=color[i],label=label,marker=markers[i])
    c1 = (time.time()-c0)*1000
    print('{:6.2f} ms'.format(c1))
    
ax.legend()
# ax.set_title('Pe = 100 and Ep = 5')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Soil Storage (m)')
# ax.set_ylim([-0,1]) # set appropiate y axis values 
ax.grid(True)

fig04, ax = plt.subplots()
ax.plot(timesteps/secinday,approxUnsatStore/maxSuz,c='b')
ax.set_xlabel('Time (days)')
ax.set_ylabel('Sw (-)')