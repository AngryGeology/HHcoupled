#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 16:10:40 2022
Plot the results of an MCMC chain
@author: jimmy
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit
plt.close('all')

# %% main program parameters
# get mcmc result file
dirname = 'Models/HydroMCMCmulti'
# dirname = 'SyntheticStudy/Models/MCMC/'
# dirname = 'SyntheticStudy/Models/MCMC(no_error)/'
# dists = {0: ['gauss', 'bimodal'],
#          1: ['bimodal', 'bimodal']}
dists = {0: ['gauss', 'bimodal'],
         1: ['gauss', 'bimodal']}
pt_threshold = 0.022
# pt_threshold = 0.45

savfig = True 
nzones = 2

# %% functions
# fitting distributions
def gauss(x, mu, sigma, A):
    left = A/(sigma*np.sqrt(2*np.pi))
    right = np.exp(-0.5*((x-mu)/sigma)**2)
    return left*right


def bimodal(x, mu0, sigma0, A0, mu1, sigma1, A1):
    return gauss(x, mu0, sigma0, A0)+gauss(x, mu1, sigma1, A1)

# class to fit a distribution
class distribution():
    def __init__(self, xdata, dist='gauss', nbins=None):
        self.xdata = xdata
        self.nbins = nbins
        self.dist = dist
        if self.nbins is None:
            self.nbins = int(len(self.xdata)/10)
        self.param = None
        self.yfit, bins = np.histogram(
            self.xdata, bins=self.nbins, density=True)
        self.xfit = (bins[1:]+bins[:-1])/2
        self.mu0 = None
        self.sigma0 = None
        self.mu1 = None
        self.sigma1 = None

    def bar_plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        _ = ax.hist(self.xdata, bins=self.nbins)

    def fit(self):

        if self.dist == 'gauss':
            _mu0 = np.mean(self.xdata)
            _sigma0 = np.std(self.xdata)
            _A0 = _sigma0*5
            p0 = (_mu0, _sigma0, _A0)
            self.param, cov = curve_fit(gauss, self.xfit, self.yfit, p0=p0)
            self.mu0 = self.param[0]
            self.sigma0 = self.param[1]
        elif self.dist == 'bimodal':
            p = np.percentile(self.xdata, (1, 25, 50, 75, 90))
            _mu0 = p[1]
            _sigma0 = abs(p[0]-p[2])
            _A0 = _sigma0*5
            _mu1 = p[3]
            _sigma1 = abs(p[2]-p[4])
            _A1 = _sigma1*5
            p0 = (_mu0, _sigma0, _A0, _mu1, _sigma1, _A1)
            self.param, cov = curve_fit(bimodal, self.xfit, self.yfit, p0=p0,
                                        maxfev=10000)
            self.mu0 = self.param[0]
            self.sigma0 = self.param[1]
            self.mu1 = self.param[3]
            self.sigma1 = self.param[4]
        return self.param

    def plot(self, ax=None, c='r', ydom=False):
        if ax is None:
            fig, ax = plt.subplots()

        if self.dist == 'bimodal':
            model = bimodal(self.xfit, *self.param)
        elif self.dist == 'gauss':
            model = gauss(self.xfit, *self.param)
        if ydom:
            ax.plot(model, self.xfit, c=c)
        else:
            ax.plot(self.xfit, model, c=c)

# show histograms on 2 axis
def scatter_hist(x, y, ax, ax_histx, ax_histy, trans=False):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    if trans:
        color = (0, 0, 1, 0.5)
    else:
        color = (0, 0, 1, 1)

    ax_histx.hist(x, bins=100, density=True, color=color)
    ax_histy.hist(y, orientation='horizontal', bins=100, density=True,
                  color=color)

# fit testing
# data = np.concatenate([np.random.normal(5,5,10000),np.random.normal(-10,5,5000)])
# fig, ax = plt.subplots()
# D = distribution(data,'bimodal')
# D.bar_plot(ax)
# D.fit()
# D.plot(ax)

# %% program
fname = None
for f in os.listdir(dirname):
    if f == 'mergedMCMClog.csv':
        fname = os.path.join(dirname, f)
        break

df = pd.read_csv(fname)
stable = df['Stable']

# create figures
# definitions for the axes
left, width = 0.1, 0.65
bottom, height = 0.1, 0.65
spacing = 0.005

rect_scatter = [left, bottom, width, height]
rect_histx = [left, bottom + height + spacing, width, 0.2]
rect_histy = [left + width + spacing, bottom, 0.2, height]

# start with a square Figure
figs = {}
axs = {}
for i in range(nzones):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_axes(rect_scatter)
    ax_histx = fig.add_axes(rect_histx, sharex=ax)
    ax_histy = fig.add_axes(rect_histy, sharey=ax)
    ax_histx.set_ylabel('Probability Density')
    ax_histy.set_xlabel('Probability Density')
    figs[i] = fig
    axs[i] = ax
    axs['hist_x%i' % i] = ax_histx
    axs['hist_y%i' % i] = ax_histy

figs[nzones], axs[nzones] = plt.subplots()
axs[nzones].set_ylabel('Normalised liklehood')
axs[nzones].set_xlabel('Run')
cfig, cax = plt.subplots()

# plot liklihood data and paths
params = {}
for i in range(nzones):
    params[i] = {}
    n = i+1
    cmp = axs[i].tricontourf(df['alpha_%i' % n][stable],
                             df['vn_%i' % n][stable],
                             df['Pt'][stable])
    axs[i].set_xlabel('Alpha (1/m)')
    axs[i].set_ylabel('N (-)')
    if i == 0:
        cbar = plt.colorbar(cmp, ax=cax, location="bottom")
        cbar.set_label('Normalised Likelihood')
    # add full histogram
    x = df['alpha_%i' % n][stable]
    y = df['vn_%i' % n][stable]
    scatter_hist(x, y, axs[i], axs['hist_x%i' % i], axs['hist_y%i' % i], True)
    # add histograms used for fitting
    idx = df['Pt'] > pt_threshold
    x = df['alpha_%i' % n][stable][idx]
    y = df['vn_%i' % n][stable][idx]
    distx = distribution(x, dists[i][0], 100)
    disty = distribution(y, dists[i][1], 100)
    scatter_hist(x, y, axs[i], axs['hist_x%i' % i], axs['hist_y%i' % i])
    # fit a histogram (to the better fitting selections)
    try:
        px = distx.fit()

        # following code plots the fit on the histograms
        lx = np.linspace(min(x), max(x), 100)
        distx.plot(axs['hist_x%i' % i], 'r')
    except:
        print('Couldnt find optimal parameters for zone %i alpha'%n)
        px = np.full(6,np.nan)
    try:
        py = disty.fit()
        ly = np.linspace(min(y), max(y), 100)    
        disty.plot(axs['hist_y%i' % i], 'r', ydom=True)
    except:
        print('Couldnt find optimal parameters for zone %i N'%n)
        py = np.full(6,np.nan)
        
    params[i]['alpha'] = px[0]
    params[i]['alpha_std'] = px[1]
    params[i]['n'] = py[0]
    params[i]['n_std'] = py[1]

    print('Fitting statistics for zone %i:' % n)
    if dists[i][0] == 'bimodal':
        print('Alpha 1: %f +/- %f (1/m)' % (px[0], px[1]))
        print('Alpha 2: %f +/- %f (1/m)' % (px[3], px[4]))
    else:
        print('Alpha : %f +/- %f (1/m)' % (px[0], px[1]))
    if dists[i][1] == 'bimodal':
        print('N 1: %f +/- %f (-)' % (py[0], py[1]))
        print('N 2: %f +/- %f (-)' % (py[3], py[4]))
    else:
        print('N : %f +/- %f (-)' % (py[0], py[1]))
    print('\n')
    nsample = len(df['alpha_%i' % n][stable][idx])
    
    figure_file = os.path.join(dirname, 'mcmc_figure_zone%i.png' % i)
    if savfig:
        figs[i].savefig(figure_file)
        cfig.savefig(os.path.join(dirname, 'colorbar.png'))

print('Nsample = %i' % nsample)

for chain in np.unique(df['chain']):
    idx = (df['chain'] == chain) & (df['Pt'] > 0)
    color = (0.2, 0.2, 0.2, 0.5)
    for i in range(nzones):
        n = i+1
        x = df['alpha_%i' % n][idx]
        y = df['vn_%i' % n][idx]
        axs[i].plot(x, y, color=color)
    # if chain == 0:

    axs[nzones].plot(df['run'][stable][idx], df['Pt'][stable][idx])

for i in range(nzones):
    figure_file_wpaths = os.path.join(
        dirname, 'mcmc_figure_zone%i_wpaths.png' % i)
    if savfig:
        figs[i].savefig(figure_file_wpaths)

best_model_idx = np.argmax(df['Pt'])
for i in range(nzones):
    n = i+1
    model = [df['alpha_%i' % n][best_model_idx],
             df['vn_%i' % n][best_model_idx]]
    print('Best fit for zone %i' % n)
    print('Alpha : %f (1/m)' % (model[0]))
    print('N : %f (-)' % (model[1]))
