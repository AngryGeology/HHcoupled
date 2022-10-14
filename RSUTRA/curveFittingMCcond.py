#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  8 14:29:21 2022
Fit van Genutchen parameters with an MCMC approach 
@author: jimmy
"""
import string as string_module 
from random import random
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt  
from scipy.stats import norm
from tqdm import tqdm  
alphabet = string_module.ascii_uppercase

#%% Vg functions
def vgCurve(suction,res,sat,alpha,n,m=None):
    """van Genuchtan curve. 
    
    Parameters
    ------------
    suction: array, float
        Suction values in kPa. 
    vol_res: float
        residual value
    vol_sat: float
        saturated value
    alpha: float
        air entry pressure, can be fitted with a curve or found experimentally
    n: float
        van Genuchtan 'n' parameter. Not the same as archie's exponents. 
    m: float, optional
        van Genuchtan 'n' parameter. If left as none then m = 1 -1/n
    
    Returns
    -----------
    eff_vol: array, float
        normalised value 
    """
    if m is None:
        m = 1 - (1/n) # defualt definition of m
    
    dVol = sat - res # maximum change in water content
    denom = 1 + (alpha*suction)**n
    x = res + dVol/(denom**m)
    
    return x 

#inverse of the above 
def invVGcurve(water_content,res,sat,alpha,n,m = None):
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
        
    step1 = (water_content-res)/(sat-res)
    step2 = (1/step1)**(1/m)
    step3 = (step2 - 1)**(1/n) 
    step4 = step3/alpha 
    
    return step4  


def invNorm(x,sat,res):
    denom = res-sat 
    return (x*denom) + res 


#%% metropolis algorithm 
def metromcmc(func,step_size,theta_init, nsteps, 
              theta_max=None, theta_min=None, walkn=0,
              target_ar=0.25,method='adapted',chi=False):
    """
    Metropolis Monte carlo Markov chain approach

    Parameters
    ----------
    func : function 
        Likelihood function (takes 2 parameters) 
    step_size : float 
        Size of step in A and B directions 
    theta_init : list 
        Starting model 
    nsteps : int
        Number of steps in markov chain.
    theta_max : list, optional
        maximum model parameters (same length as model). The default is None.
    theta_min : list, optional
        minimum model parameters (same length as model). The default is None.
    walkn : int, optional
        Walk number (does not change the number of walks, this is simply for 
        reporting with tqdm). The default is 0.

    Returns
    -------
    data : dict 
        Output of MCMC algoritm.
    acceptance_rate : float
        Acceptance rate (%).

    """
    naccept = 0  # Number of accepted models 
    nparam = len(theta_init) # number of model parameters 
    theta = np.asarray(theta_init, dtype = float) # set starting model 
    step_size = np.asarray(step_size, dtype = float)
    data = {'step':np.arange(nsteps), 
            'Lf':[0.0]*nsteps,
            'mu':[0.0]*nsteps, # probability to accept 
            'alpha':[0.0]*nsteps,
            'Accept':[False]*nsteps,
            'ar':[0.0]*nsteps} 
    for i in range(nparam):
        data[alphabet[i]] = [0.0]*nsteps
    
    # check for min and max limit definitions 
    min_check = False 
    max_check = False 
    if theta_max is not None:
        max_check = True 
    if theta_min is not None:
        min_check = True 
    
    ar = 1
    mem_length = 10 
    ac = []
    sa_step = (1-target_ar)/nsteps 
    for i in tqdm(range(nsteps),desc='Walk%i'%walkn):
        accept = False 
        pi = float(func(*theta)) # initial probability 
        step = step_size * np.random.randn(nparam) # do random walk 
        theta_trial = theta + step # trial model 
        pt = float(func(*theta_trial)) # trial probability 
        
        if chi: # if in log space compute ratio in log space 
            alpha = np.exp(-pt + pi) # compute alpha 
        else: 
            alpha = pt/pi 

        if len(ac) < mem_length:
            a_fac = 1 
        else:
            # special bit 
            ar = sum(ac)/mem_length
            if ar == 0: # if acceptance rate is zero then default back to metropolis 
                a_fac = 1 
            elif target_ar == -1:
                a_fac = 1 
            elif method == 'adapted': 
                # ratchet acceptance rate down to match target acceptance rate 
                a_fac = target_ar/ar
            elif method == 'SA':
                a_fac = 1-(i*sa_step) 
            
        mu = random()
        if alpha >= 1:
            accept = True 
        elif alpha*a_fac > mu:
            accept = True  
            
        if np.isnan(pi):
            accept = False 
           
        # check trial is within limits 
        if max_check: 
            for j in range(nparam):
                if theta_trial[j] > theta_max[j]:
                    accept = False 
                    break 
        if min_check: 
            for j in range(nparam):
                if theta_trial[j] < theta_min[j]:
                    accept = False 
                    break 
           
        for j in range(nparam):
            data[alphabet[j]][i] = theta[j]
        
        data['Lf'][i] = pt 
        data['mu'][i] = mu 
        data['alpha'][i] = alpha 
        data['Accept'][i] = accept 
        data['ar'][i] = ar 

        # remove 1st entry from acceptance chain
        if len(ac) >= mem_length:
            _ = ac.pop(0)
            
        if accept:
            theta += step 
            naccept += 1 
            ac.append(1)
        else:
            ac.append(0) # add to acceptance chain 
            
    acceptance_rate = (naccept/nsteps)*100 
    
    print('Acceptance rate = ',acceptance_rate, '%')
            
    return data, acceptance_rate


#%% load in data? 
fpath = '/home/jimmy/phd/Hollin_Hill/Lab_work/Hyprop_results/East_backscarp_compiled_data.csv'
# fpath = '/home/jimmy/phd/Hollin_Hill/Lab_work/Hyprop_results/East-flowlobe_combined_compiled_data.csv'
# fpath = 'C:/Users/jimmy/Documents/2PhDProjects/Hollin_Hill/Lab_work/Hyprop_results/East-flowlobe_combined_compiled_data.csv'
df = pd.read_csv(fpath)

# display data 
fig, ax1 = plt.subplots(ncols=2)

tension = (df['Tension_top(kpa)']+df['Tension_bottom(kpa)']).values/2 
ax1[0].scatter(df['Time(s)'],df['Tension_top(kpa)'],marker='*',color=(0.2,0.2,0.2))
ax1[0].scatter(df['Time(s)'],df['Tension_bottom(kpa)'],marker='*',color=(0.4,0.4,0.4))
ax1[0].scatter(df['Time(s)'],tension,color='b')

# find cavitation phase 
idx = np.argmax(tension)
a = [False]*len(df)
for i in range(idx+1):
    a[i] = True 

ax1[0].scatter(df['Time(s)'][idx],tension[idx],marker='o',color='r')
df['Tension'] = tension 

df_filt = df[a]

ax1[1].scatter(df_filt['Conductivity(mS/cm)'],df_filt['Tension'],c='k')

res = min(df['Conductivity(mS/cm)'])
sat = max(df['Conductivity(mS/cm)'])
alpha = 0.5
vn = 2
# alpha = log['A'][-1]
# vn = log['B'][-1]
mdlGmc = vgCurve(df_filt['Tension'],res,sat,alpha,vn)

# ax1[1].plot(mdlGmc,df_filt['Tension'])

#%% model data 
data = df_filt['Tension'].values 
X = df_filt['Conductivity(mS/cm)'].values 
sigma = 5
step_size = np.array([0.0005,0.001,0.01])
theta_init = np.array([alpha,vn,0.90])
theta_max = np.array([2.0, 4.0, 0.95])
theta_min = np.array([0.001,0.5,0.80])
# theta_max = np.array([2.0, 4.0, 1.05])
# theta_min = np.array([0.001,0.5,0.92])

# function returns pi for trial model 
def func(alpha,vn,cs):
    # print(alpha,vn,cs)
    mdl = invVGcurve(X,res,sat,alpha,vn)
    mdl[np.isnan(mdl)] = 0 # cap min at 0 
    residuals = data - mdl 
    n = len(residuals)
    psum = 0 
    lsum = 0 
    c = -0.5*np.log(2*np.pi) # constant 
    std = sigma # standard deviation of point 
    var = std**2 # varaince of point 
    for i in range(n):
        res2 = residuals[i]**2 # square of residual 
        if std == 0 and res2==0: # in the case of a perfect data point normalised LR is 1 
            psum += 1 # so add 1 to psum and move on 
            continue 
        lli = c - np.log(std) - res2/(2*var) # log likelihood 
        lmle = c - np.log(std) # log max likelihood estimate 
        llr = lli - lmle # log likelihood ratio 
        lr = np.exp(llr) # convert back into normal space 
        psum += lr # add to total probability 
        lsum += np.exp(lli)
        
    # return psum/n 
    return lsum/n 
    # return psum/n, lsum/n # normalise to between 0 and 1 

# l = func(alpha,vn)

log, ar = metromcmc(func,step_size,theta_init, 20000, 
                    theta_max=theta_max, theta_min=theta_min, target_ar=0.2)
logd = pd.DataFrame(log)

fig,ax = plt.subplots()
ax.plot(log['Lf'])

#%% plot 
nbins = 100 
def scatter_hist(x, y, z, ax, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    # the scatter plot:
    ax.tricontourf(x, y, z)

    ax_histx.hist(x, nbins, density = True)
    ax_histy.hist(y, nbins, density = True, orientation='horizontal')

    ax_histx.set_ylabel('PDF')
    ax_histy.set_xlabel('PDF')
    
# definitions for the axes
idx = np.array(log['Accept'])
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
spacing = 0.005


rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom + height + spacing, width, 0.2]
rect_histy = [left + width + spacing, bottom, 0.2, height]

# start with a square Figure
fig = plt.figure(figsize=(8, 8))

ax = fig.add_axes(rect_scatter)
ax_histx = fig.add_axes(rect_histx, sharex=ax)
ax_histy = fig.add_axes(rect_histy, sharey=ax)

x = np.array(log['A'])[idx]
y = np.array(log['B'])[idx]
z = np.array(log['Lf'])[idx]
# use the previously defined function
scatter_hist(x, y, z, ax, ax_histx, ax_histy)

ax.set_xlabel('Alpha')
ax.set_ylabel('N')

plt.show()

#%% best model? 
# for i in range(len(log['Lf'])):
#     if np.isnan(log['Lf'][i]):
#         log['Lf'][i] = 0 
# i = np.argmax(log['Lf'])
# alpha = log['A'][i]
# vn = log['B'][i]
# mdlGmc = vgCurve(df_filt['Tension'],res,sat,alpha,vn)

# ax1[1].plot(mdlGmc,df_filt['Tension'],c='r',label='Best Fit')

# print('Alpha: ',alpha)
# print('N: ',vn)

burnin = int(len(log['A'])/2)

# burnin_slice = [burnin:-1]
#%% error bounds? 
alpha, alpha_std = norm.fit(log['A'][burnin:-1])
vn, vn_std = norm.fit(log['B'][burnin:-1])
cs, cs_std = norm.fit(log['C'][burnin:-1])

# plot the pdf for alpha 
n = len(log['A'])
xmin = np.min(log['A'])
xmax = np.max(log['A'])
x = np.linspace(xmin, xmax, 100)
alpha_p = norm.pdf(x, alpha, alpha_std)
ax_histx.plot(x,alpha_p,c='r')

# plot pdf for N 
xmin = np.min(log['B'])
xmax = np.max(log['B'])
x = np.linspace(xmin, xmax, 100)
vn_p = norm.pdf(x, vn, vn_std)
ax_histy.plot(vn_p,x,c='r')

print('Alpha: %f +/- %f'%(alpha,alpha_std))
print('N: %f +/- %f'%(vn, vn_std))
print('Sat: %f+/- %f'%(cs, cs_std))

fitX = vgCurve(df_filt['Tension'],res,sat,alpha,vn)
# ax1[1].plot(fitX,df_filt['Tension'],c='b',label='Norm Fit')

#%% simulate errors 
nsim = 5000
suction_err = np.zeros((200, nsim))
# alpha_per = np.random.randn(nsim)*alpha_std
# vn_per = np.random.randn(nsim)*vn_std
# uniform =(np.random.random(nsim)-0.5)*2
alpha_per = (np.random.random(nsim)-0.5)*2*alpha_std
vn_per = (np.random.random(nsim)-0.5)*2*vn_std
cs_per = (np.random.random(nsim)-0.5)*2*cs_std 
mdlX = np.linspace(min(X),max(X),200)
mdlY = invVGcurve(mdlX, res, cs, alpha, vn)
for i in range(nsim):
    suction_err[:,i] = invVGcurve(mdlX, res, cs + cs_per[i], alpha+alpha_per[i], vn+vn_per[i])
    
suction_err[np.isnan(suction_err)] = 0
suction_err_min = np.min(suction_err, axis=1)
suction_err_max = np.max(suction_err, axis=1)
#suction_err_min = invVGcurve(X,res,sat,alpha-alpha_std,vn-vn_std)#inv_vg_curve(cond_model,EC_sat,EC_res,mod[0]- perr[0]*3,mod[1]- perr[0]*3)
#suction_err_max = invVGcurve(X,res,sat,alpha+alpha_std,vn+vn_std)#inv_vg_curve(cond_model,EC_sat,EC_res,mod[0]+ perr[0]*3,mod[1]+ perr[0]*3)

ax1[1].plot(mdlX,mdlY,c='b',label='Norm Fit')
ax1[1].fill_between(mdlX, suction_err_min, suction_err_max, color = 'blue', alpha = 0.3)

ax1[0].set_ylabel('Pore pressure (kPa)')
ax1[0].set_xlabel('Time (s)')
ax1[1].set_xlabel('Conductivity (mS/m)')

#%% resistivity versus suction plot 
fig, ax2 = plt.subplots()

ax2.scatter(df_filt['Resistivity(ohm.m)'],df_filt['Tension'],c='k')
ax2.plot((1/mdlX)*10,mdlY,c='b',label='Norm Fit')
ax2.fill_between((1/mdlX)*10, suction_err_min, suction_err_max, color = 'blue', alpha = 0.3)

ax2.set_xlabel('Resistivity (ohm.m)')
ax2.set_ylabel('Soil suction (kPa)')

#%% resistivity versus gmc plot 
fig, ax3 = plt.subplots()
ax3.scatter(df['Resistivity(ohm.m)'], df['Gmc(pnct)']*100, c='k')
ax3.set_xlabel('Resistivity (ohm.m)')
ax3.set_ylabel('Gravimetric moisture content (%)')

