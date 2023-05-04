#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 17:03:50 2022
Curve fitting with least squares or n order polynomails 
(polynomial works better for the staithes)
fitting in terms of saturation and resistivity 
@author: jimmy
"""

# import modules and relevant data
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.optimize import curve_fit

# gmc solver modules
# import mcmcsolver as mcmc  # custom module
from solveWaxSmit import solveRtSt, solveTheta
from petroFuncs import gmc2sat, rmse, chi2, powerLaw

plt.close("all")

master = pd.read_csv('Data/petro/Resy-data-compiled.csv')
theta_param = np.genfromtxt('petroFit/theta_fit.txt')
# convert gmc string into float
gmc = [float(s.strip("%")) for s in master["Est. GMC (%)"]]
master["gmc(%)"] = gmc
d = [float(n.split('-')[-1].replace('H', '').replace('V', ''))
     for n in master['Sample']]
master['depth'] = d

# convert gmc into saturation
theta = [solveTheta(theta_param, g/100) for g in gmc]
sat = gmc2sat(np.array(gmc)/100, np.array(theta), 2.74)
master['sat'] = sat


# %% plot data
fig, ax = plt.subplots()

uni_names = np.unique(master["Sample"].values)
unwanted = ['HH01- 4.5V', #'HH01-1.5H',
            "HH02 - 3.1V", "HH02 - 6.2V"]
# unwanted = ['HH01- 4.5V'] 

cols = [(0.7, 0.7, 0.5),
        (0.2, 0.2, 0.2),
        (0.4, 0.7, 0.4)]

labels = []
lns = []

for name in uni_names:
    if name in unwanted:
        continue  # skip the unwanted results
    idx = name == master["Sample"]

    # choose colour of marker, WMF, ssf, wmf disturbed
    col = cols[0]
    label = 'SSF data'

    if 'HH02' in name:
        col = cols[1]
        label = 'WMF data (deep)'
        # label = 'WMF - I'
        if master['depth'][idx].values[-1] < 1.5:
            col = cols[2]
            label = 'WMF data (shallow)'
            
    if 'HH01' in name:
        col = cols[0]
        label = 'SSF data (deep)' 
        # label = 'WMF - I'
        if master['depth'][idx].values[-1] < 1.5:
            col = cols[0]
            label = 'SSF data (shallow)'

    # mark off if vertical or horizontal - chooses shape of marker
    mark = '1'
    if 'H' == name[-1]:
        mark = '+'
        # label += 'H'
    else:
        pass 
        # label += 'V'

    ln = ax.scatter(
        master["sat"][idx],
        master["TC Resistivity (Ohm.m)"][idx],
        label=label,
        marker=mark,
        color=col,
    )

    if label not in labels:
        labels.append(label)
        lns.append(ln)

ax.set_yscale("log")
# ax.legend()
ax.set_ylabel("Resistivity (ohm.m)")
ax.set_xlabel("Saturation (-)")
ax.set_xlim([0, 1])
ax.set_ylim([1, 1000])

# %% set up solver
Rt = master["TC Resistivity (Ohm.m)"].values

tofit01 = ["HH02 - 0.95H", "HH02 - 1.05V"]
# tofit02 = ["HH02 - 3.1V", "HH02 - 6.2V"]
tofit02 = ["HH02 - 3H", "HH02 - 6.1H"]#"HH02 - 3.1V", "HH02 - 6.2V"]

tofit03 = ['HH01-1.5V','HH01-1.5H']

fit_names = ["Shallow WMF fit", "WMF fit", "SSF fit"]
fit_columns = [tofit01, tofit02, tofit03]
fit_colour = [cols[2], cols[1], cols[0]]
fit_type = ['Gen', 'Gen', 'Gen']

# fitting parameters
Rw = 1 / 0.0987
cec = [22.5, 11.0]  # cec defined for each curve fit

for i, name in enumerate(fit_names):
    idxfit = np.array([False] * len(master), dtype=bool)
    for sample in fit_columns[i]:
        idx = sample == master["Sample"]
        idxfit[idx] = True

    data = master["TC Resistivity (Ohm.m)"].values[idxfit]
    errorest = data * 0.1
    X = sat[idxfit]
    mdlSat = np.linspace(0.01, 1.00, 100)  # for modelling synthetic curves

    if fit_type[i] == 'Waxman':  # go the route of least squares fitting
        print("\nAttempting Waxman least squares fit for %s" % fit_names[i])

        def yfunc(xdata, FF, n):
            mdl = solveRtSt(xdata, Rw, cec[i], FF, n)
            return mdl

        # provide MCMC output as a start
        popt, pcov = curve_fit(yfunc, X, data, p0=[10, 2])
        wsFF, wsn = popt[0], popt[1]
        FF_std, wsn_std = pcov[0, 0], pcov[1, 1]

        print("Fiting parameters:")
        print("FF: %f +/- %f" % (wsFF, FF_std))
        print("N: %f +/- %f" % (wsn, wsn_std))

        fh = open(os.path.join(
            'petroFit', fit_names[i].replace(' ', '_'))+'.txt', 'w')
        fh.write('cec: %f\n' % cec[i])
        fh.write('Rw: %f\n' % Rw)
        fh.write('FF: %f\n' % wsFF)
        fh.write('n: %f\n' % wsn)
        fh.close()

        mdl = yfunc(X, wsFF, wsn)
        mdlRt = yfunc(mdlSat, wsFF, wsn)

    elif fit_type[i] == 'Poly':  # otherwise fit a polynomial
        print("\nAttempting poly fit search for %s" % fit_names[i])
        pfit = np.polyfit(X, data, 8)
        mdl = np.polyval(pfit, X)
        # fit last few measurements
        pfit1 = np.polyfit(X[0:5], data[0:5], 1)

        # fit first few measurements
        pfit0 = np.polyfit(X[-5:-1], data[-5:-1], 1)

        print("Fiting parameters (polynomial):")
        for j in range(len(pfit)):
            print("P%i: %f" % (j+1, pfit[j]))

        # synthetic data
        lidx = mdlSat < min(X)  # values to left of data
        ridx = mdlSat > max(X)
        cidx = (mdlSat >= min(X)) & (mdlSat <= max(X))
        mdlRt = np.zeros_like(mdlSat)
        mdlRt[cidx] = np.polyval(pfit, mdlSat[cidx])
        mdlRt[lidx] = np.polyval(pfit0, mdlSat[lidx])
        mdlRt[ridx] = np.polyval(pfit1, mdlSat[ridx])

        fh = open(os.path.join(
            'petroFit', fit_names[i].replace(' ', '_'))+'.txt', 'w')
        fh.write('min: %f\n' % min(X))
        fh.write('max: %f\n' % max(X))
        fh.write('pfit: ')
        for a in pfit:
            fh.write(' %f' % a)
        fh.write('\n')
        fh.write('pfit0: ')
        for a in pfit0:
            fh.write(' %f' % a)
        fh.write('\n')
        fh.write('pfit1: ')
        for a in pfit1:
            fh.write(' %f' % a)
        fh.write('\n')
        fh.close()
    elif fit_type[i] == 'Gen':
        print("\nAttempting generic fit for %s" % fit_names[i])

        # provide MCMC output as a start
        popt, pcov = curve_fit(powerLaw, X, data, p0=[10, 2, 20])
        a, k, c = popt[0], popt[1], popt[2]
        # FF_std, wsn_std = pcov[0,0], pcov[1,1]

        print("Fiting parameters:")
        print("a: %f" % a)
        print("k: %f" % k)
        print('c: %f' % c)

        # uncomment to write to file

        fh = open(os.path.join('petroFit',fit_names[i].replace(' ','_'))+'.txt','w')
        fh.write('a: %f\n'%a)
        fh.write('k: %f\n'%k)
        fh.write('c: %f\n'%c)
        fh.close()

        mdl = powerLaw(X, a, k, c)
        mdlRt = powerLaw(mdlSat, a, k, c)

    # compute some stats
    rms = rmse(X, mdl)
    residuals = mdl - data
    chi = chi2(errorest, residuals)
    r2, _ = pearsonr(data, mdl)

    print('Fitting stats:')
    print('RMS: %f' % rms)
    print('CHi^2: %f' % chi)
    print('r^2: %f' % r2)
    print('N: %i' % len(X))
    fig1, ax1 = plt.subplots()
    ax1.scatter(X, data, c="k", marker="*")
    ax1.plot(mdlSat, mdlRt, c="b")

    ax1.set_xlabel("sat. (-)")
    ax1.set_ylabel("Resistivity (ohm.m)")

    ln = ax.plot(mdlSat, mdlRt, c=fit_colour[i], label=fit_names[i], lw=2)
    lns.append(ln[0])
    labels.append(fit_names[i])
    print("___________________________________")
    # break

ax.legend(lns, labels)
ax.grid(True, 'major', linestyle='--',
        color=(0.5, 0.5, 0.7, 0.3))  # major grid
ax.grid(True, 'minor', linestyle=':', 
        color=(0.5, 0.5, 0.7, 0.3))

fig.savefig('/home/jimmy/phd/Hollin_Hill/papers/paper3/figs/Fig03.png',dpi=600)## uncomment to save figure
