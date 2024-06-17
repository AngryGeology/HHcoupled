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
import seaborn as sns 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.stats import shapiro
plt.close('all')
# sns.set_theme(style="dark")

# %% main program parameters
# get mcmc result file

synth = False 
savfig = True 
contour = False 
style = True 
nzones = 2
nparam = 3 

if synth:
    dirname = 'SyntheticStudy/Models/MCMC/'    
    dists = {0: ['gauss', 'bimodal'],
              1: ['gauss', 'bimodal']}
    pt_threshold = 0.6
    simN = {0:1.9, 1:1.5}
    simA = {0:0.2, 1:0.1}
else:
    dirname = 'Models/_HydroMCMCmultiV2'
    # dirname = 'Models/HydroMCMC'
    dists = {0: ['bimodal', 'bimodal'],
             1: ['gauss', 'bimodal']}
    pt_threshold = 0.0245
    simN = None 
    simA = None 
    
xlim = [0.0, 1.1]
ylim = [1.1, 2.5]
cmap = 'turbo'

convert_cons= {'u':1.307e-3, #kg/ms  
               'p':1000, #kg/ms^3 
               'g':9.81} #m/s^2 

# %% functions
# fitting distributions
def gauss(x, mu, sigma, A):
    left = A/(sigma*np.sqrt(2*np.pi))
    right = np.exp(-0.5*((x-mu)/sigma)**2)
    return left*right


def bimodal(x, mu0, sigma0, A0, mu1, sigma1, A1):
    return gauss(x, mu0, sigma0, A0)+gauss(x, mu1, sigma1, A1)

def convertk2K(k, time_unit='day', space_unit='m'):
    # convert hydrualic conductivity (m/day) to permeability (m^2)
    # the following values are for water 
    u = convert_cons['u']
    p = convert_cons['p']
    g = convert_cons['g']
    secinday = 84*60*60
    
    # 
    # k = self.K/secinday 
    # perm = (k*u)/(p*g)
    K = (k*p*g)/u 
    # convert K from m/s to m/day 
    K *= secinday 
    
    return K 

class logger():
    def __init__(self, fout):
        self.fout = fout 
        self.fh = open(fout, 'w')
        self.fh.close() 
        
    def log(self,x):
        # log outputs 
        self.fh = open(self.fout, 'a')
        self.fh.write(x + '\n')
        self.close()
        print(x)
        
    def close(self):
        self.fh.close() 

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
        color = (0.2, 0.2, 0.2, 0.2)
    else:
        color = (0.0, 0.0, 1.0, 1.0)

    ax_histx.hist(x, bins=100, density=True, color=color, edgecolor=color)
    ax_histy.hist(y, orientation='horizontal', bins=100, density=True,
                  color=color, edgecolor=color)
    
# filter dataframe to remove burn in 
def get_burnin(df, pt_threshold):
    idx = np.array([False]*len(df),dtype=bool)
    chains = df.chain
    # figure out if chain reached threshold for convergence  
    chain_converged = []
    for chain in np.unique(chains):
        ii = chain == chains 
        max_pt = df.Pt[ii].max()
        idx_chain = np.array([False]*len(df[ii]),dtype=bool)
        if max_pt > pt_threshold:
            # need to find where threshold get crossed for the first time
            c = 0 
            for x in df.Pt[ii]:
                if x >= pt_threshold:
                    c += 1 
                    break 
            if c < 500: # cap c 
                c = 500 
            idx_chain[c:-1] = True 
            chain_converged.append(chain)
        idx[ii] = idx_chain
            
    return idx, chain_converged
    

# %% program
fname = None
fout = os.path.join(dirname,'stats.txt')
log = logger(fout)

for f in os.listdir(dirname):
    if f == 'mergedMCMClog.csv':
        fname = os.path.join(dirname, f)
        break
    
log.log('Fitting stats for McMC file: %s'%fname)
log.log('_'*32)
log.log(' ')

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
for i in range(nparam):
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

figs[nparam], axs[nparam] = plt.subplots()
axs[nparam].set_ylabel('Normalised liklehood')
axs[nparam].set_xlabel('Run')
cfig, cax = plt.subplots()

# plot liklihood data and paths
params = {}
for i in range(nzones):
    params[i] = {}
    n = i+1
    xi = df['alpha_%i' % n][stable].values 
    yi = df['vn_%i' % n][stable].values
    zi = df['k_%i'%n][stable].values
    
    # create matrix of likelihoods and density 
    binwidth = 0.025
    xn = int((xlim[1] - xlim[0])/binwidth)
    yn = int((ylim[1] - ylim[0])/binwidth)
    xg = np.linspace(xlim[0], xlim[1], xn)
    yg = np.linspace(ylim[0], ylim[1], yn)
    xgg, ygg = np.meshgrid(xg,yg)
    Lmat = np.zeros((yn-1, xn-1), dtype=float) # likelihood matrix 
    Dmat = np.zeros((yn-1, xn-1), dtype=int) # density matrix
    for ii in range(xgg.shape[0]-1):
        for jj in range(ygg.shape[1]-1):
            idx = (xi >= xgg[ii,jj]) & (xi < xgg[ii+1,jj+1]) & (yi >= ygg[ii,jj]) & (yi < ygg[ii+1,jj+1])
            A = df['Pt'][stable][idx]
            if len(A) > 0: 
                Lmat[ii,jj] = np.max(A)
                Dmat[ii,jj] = len(A)
            else:
                Lmat[ii,jj] = np.nan
                Dmat[ii,jj] = 0
                
    if contour: 
        cmp = axs[i].tricontourf(df['alpha_%i' % n][stable],
                                 df['vn_%i' % n][stable],
                                 df['Pt'][stable])
    elif style:
        vmin = np.min(df['Pt'][stable])
        vmax = np.max(df['Pt'][stable])

        cmp = axs[i].scatter(df['alpha_%i' % n][stable],
                             df['vn_%i' % n][stable],
                             c = df['Pt'][stable],
                             marker='.',cmap=cmap,
                             vmin=vmin,vmax=vmax)
        
        axs[i].pcolor(xgg,ygg,Lmat,cmap=cmap, alpha = 0.75, 
                      vmin=vmin,vmax=vmax)

    else: 
        cmp = axs[i].scatter(df['alpha_%i' % n][stable],
                             df['vn_%i' % n][stable],
                             c = df['Pt'][stable],
                             marker='.')
    if synth:
        axs[i].scatter(simA[i], simN[i], c='r', marker='+',s=30)
        
    axs[i].set_xlabel('Alpha (1/m)')
    axs[i].set_ylabel('N (-)')
    axs[i].set_xlim(xlim)
    axs[i].set_ylim(ylim)
    if i == 0:
        cbar = plt.colorbar(cmp, ax=cax, location="bottom")
        cbar.set_label('Normalised Likelihood')
    # add full histogram
    x = df['alpha_%i' % n][stable]
    y = df['vn_%i' % n][stable]
    scatter_hist(x, y, axs[i], axs['hist_x%i' % i], axs['hist_y%i' % i], True)
    
    # filter out the burnin 
    # idx = df['Pt'] > pt_threshold
    idx, chain_converged = get_burnin(df[stable],pt_threshold)
    
    # add histograms used for fitting
    x = df['alpha_%i' % n][stable][idx]
    y = df['vn_%i' % n][stable][idx]
    snormx_coef = shapiro(x).statistic
    snormy_coef = shapiro(y).statistic
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
        log.log('Couldnt find optimal parameters for zone %i alpha'%n)
        px = np.full(6,np.nan)
    try:
        py = disty.fit()
        ly = np.linspace(min(y), max(y), 100)    
        disty.plot(axs['hist_y%i' % i], 'r', ydom=True)
    except:
        log.log('Couldnt find optimal parameters for zone %i N'%n)
        py = np.full(6,np.nan)
        
    params[i]['alpha'] = px[0]
    params[i]['alpha_std'] = px[1]
    params[i]['n'] = py[0]
    params[i]['n_std'] = py[1]

    log.log('Fitting statistics for zone %i:' % n)
    if dists[i][0] == 'bimodal':
        log.log('Alpha 1: %f +/- %f (1/m)' % (px[0], px[1]))
        log.log('Alpha 2: %f +/- %f (1/m)' % (px[3], px[4]))
    else:
        log.log('Alpha : %f +/- %f (1/m)' % (px[0], px[1]))
    log.log('Shapiro-Wilk coefficient for Alpha: %3.2f'%snormx_coef)
    
    if dists[i][1] == 'bimodal':
        log.log('N 1: %f +/- %f (-)' % (py[0], py[1]))
        log.log('N 2: %f +/- %f (-)' % (py[3], py[4]))
    else:
        log.log('N : %f +/- %f (-)' % (py[0], py[1]))
    log.log('Shapiro-Wilk coefficient for N: %3.2f'%snormy_coef)
    log.log(' ')
    nsample = len(df['alpha_%i' % n][stable][idx])
    
    figure_file = os.path.join(dirname, 'mcmc_figure_zone%i.png' % i)
    if savfig:
        figs[i].savefig(figure_file)
        cfig.savefig(os.path.join(dirname, 'colorbar.png'))

# save statistics to file 
log.log('Nsample = %i' % nsample)

for chain in np.unique(df['chain']):
    idx = (df['chain'] == chain) & (df['Pt'] > 0)
    color = (0.2, 0.2, 0.2, 0.5)
    # for i in range(nzones):
    #     n = i+1
    #     x = df['alpha_%i' % n][idx]
    #     y = df['vn_%i' % n][idx]
    axs[nzones].plot(df['run'][stable][idx], df['Pt'][stable][idx], 
                     alpha = 0.5, label='chain{:0>2d}'.format(chain))

axs[nzones].set_xlim([min(df.run),max(df.run)])
axs[nzones].legend(bbox_to_anchor=(1.05, 1.05))
figs[nzones].set_tight_layout(True)
figs[nzones].set_size_inches([14,6])
if savfig:
    figure_file = os.path.join(dirname, 'liklehood_track.png')
    figs[nzones].savefig(figure_file)

best_model_idx = np.argmax(df['Pt'])
for i in range(nzones):
    n = i+1
    model = [df['alpha_%i' % n][best_model_idx],
             df['vn_%i' % n][best_model_idx]]
    log.log('Best fit for zone %i' % n)
    log.log('Alpha : %f (1/m)' % (model[0]))
    log.log('N : %f (-)' % (model[1]))
log.log('Chains converged: %i'%len(chain_converged))

#%% plot K 
fig, ax = plt.subplots()

ax.scatter(df['k_%i'%1].values, df['k_%i'%2].values, c=df['Pt'])

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('k - SSF (1/m^2)')
ax.set_ylabel('k - WMF (1/m^2)')


fig, ax = plt.subplots()

ax.scatter(df['vn_%i'%1].values, df['vn_%i'%2].values, c=df['Pt'])

# ax.set_xscale('log')
# ax.set_yscale('log')
ax.set_xlabel('vn - SSF')
ax.set_ylabel('vn - WMF')

fig, ax = plt.subplots()

ax.scatter(df['alpha_%i'%1].values, df['alpha_%i'%2].values, c=df['Pt'])

# ax.set_xscale('log')
# ax.set_yscale('log')
ax.set_xlabel('alpha - SSF')
ax.set_ylabel('alpha - WMF')
