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
# import seaborn as sns 
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker 
from scipy.optimize import curve_fit
from scipy.stats import shapiro
plt.close('all')
# sns.set_theme(style="dark")

# %% main program parameters
# get mcmc result file

synth = True 
savfig = True 
nzones = 2
nparam = 3 
show_modes = False 
show_means = True 

if synth:
    dirname = 'SyntheticStudy/Models/MCMC'    
    dists = {0: ['gauss', 'bimodal'],
             1: ['gauss', 'bimodal'],
             2: ['guass', 'guass']}
    pt_threshold = 0.5
    simN = {0:1.9, 1:1.5}
    simA = {0:0.2, 1:0.1}
    simK = {0:0.14, 1:0.013}
    zone_names = ['Sandstone','Mudstone']
else:
    dirname = 'Models/HydroMCMCmultiV2'
    # dirname = 'Models/HydroMCMC'
    dists = {0: ['bimodal', 'bimodal'],
             1: ['bimodal', 'bimodal'],
             2: ['bimodal', 'bimodal']}
    pt_threshold = 0.0225
    simN = None 
    simA = None 
    simK = None
    zone_names = ['SSF','WMF']
    
xlim = [0.001, 1.1]
ylim = [1.1, 2.5]
cmap = 'turbo'

convert_cons= {'u':1.307e-3, #kg/ms  
               'p':1000, #kg/ms^3 
               'g':9.81} #m/s^2 

# %% functions
# fitting distributions
def gauss(x, mu, sigma, A=1):
    left = A/(sigma*np.sqrt(2*np.pi))
    right = np.exp(-0.5*(((x-mu)**2)/(sigma**2)))
    return left*right


def bimodal(x, mu0, sigma0, A0, mu1, sigma1, A1):
    return gauss(x, mu0, sigma0, A0)+gauss(x, mu1, sigma1, A1)

def convertk2K(k):
    # convert permeability (m^2) to hydrualic conductivity (m/day) 
    # the following values are for water 
    u = convert_cons['u']
    p = convert_cons['p']
    g = convert_cons['g']
    secinday = 24*60*60
    
    K = (k*p*g)/u 
    # convert K from m/s to m/day 
    K *= secinday 
    
    return K 


def convertK2k(K):
    # convert hydrualic conductivity (m/day) to permeability (m^2)
    # the following values are for water 
    u = convert_cons['u']
    p = convert_cons['p']
    g = convert_cons['g']
    secinday = 24*60*60
    
    # convert K from m/day to m/s
    k = K/secinday 
    k = (k*u)/(p*g) 
    
    return k

def log_tick_formatter(val, pos=None):
    return f"$10^{{{int(val)}}}$"

def get_ilog_values(val):
    return 10**val 

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
    def __init__(self, xdata, dist='gauss', nbins=None, log=False):
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
        self.log = log 

    def bar_plot(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        _ = ax.hist(self.xdata, bins=self.nbins)

    def fit(self):

        if self.dist == 'gauss':
            _mu0 = np.mean(self.xdata)
            _sigma0 = np.std(self.xdata)
            _A0 = 1 #_sigma0*5
            p0 = (_mu0, _sigma0, _A0)
            b0 = (-np.inf, 0, -1)
            b1 = (np.inf, np.inf, 1)
            self.param, cov = curve_fit(gauss, self.xfit, self.yfit, p0=p0)
            if self.log: 
                parami =[0,1]
                for i in parami: 
                    self.param[i] = 10**self.param[i]

            self.mu0 = self.param[0]
            self.sigma0 = self.param[1]
        elif self.dist == 'bimodal':
            p = np.percentile(self.xdata, (1, 25, 50, 75, 90))
            _mu0 = p[1]
            _sigma0 = abs(p[0]-p[2])
            _A0 = 1 # _sigma0*5
            _mu1 = p[3]
            _sigma1 = abs(p[2]-p[4])
            _A1 = 1 # _sigma1*5
            p0 = (_mu0, _sigma0, _A0, _mu1, _sigma1, _A1)
            b0 = (-np.inf, 0, -1, -np.inf, 0, -1)
            b1 = (np.inf, np.inf, 1, np.inf, np.inf, 1)
            self.param, cov = curve_fit(bimodal, self.xfit, self.yfit, p0=p0,
                                        maxfev=10000, bounds=(b0,b1))
            if self.log: 
                parami = [0,1,3,4]
                for i in parami: 
                    self.param[i] = 10**self.param[i]

            self.mu0 = self.param[0]
            self.sigma0 = self.param[1]
            self.mu1 = self.param[3]
            self.sigma1 = self.param[4]
        return self.param

    def plot(self, ax=None, c='r', ydom=False):
        if ax is None:
            fig, ax = plt.subplots()
            
        param = self.param.copy() 
        if self.log: 
            for i in range(len(self.param)):
                if i == 2 or i == 5: 
                    continue 
                else:
                    param[i] = np.log10(param[i])

        if self.dist == 'bimodal':
            model = bimodal(self.xfit, *param)
        elif self.dist == 'gauss':
            model = gauss(self.xfit, *param)
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
            for i,x in enumerate(df.Pt[ii]):
                c += 1 
                run = df.run[ii].values[i]
                if run < 250:
                    continue 
                if x >= pt_threshold:
                    break 

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
log.log('Stats reported as <parameter>_<mode number> = <mu> +/- <sigma> | <amplitude>')

df = pd.read_csv(fname)
stable = df['Stable']

## liklehood track 
figl, axl = plt.subplots() 
figl.set_size_inches([14., 4.8])
for chain in np.unique(df['chain']):
    idx = (df['chain'] == chain) & (df['Pt'] > 0)
    axl.plot(df['run'][stable][idx], df['Pt'][stable][idx], 
              alpha = 0.5, label='chain{:0>2d}'.format(chain))
axl.set_xlabel('Run number')
axl.set_ylabel('Normalised Liklehood')
    
if savfig:
    figure_file = os.path.join(dirname, 'liklehood_track.png')
    figl.savefig(figure_file)

#%% create figures
fighi, axshi = plt.subplots(nrows=nparam, ncols=nzones) # --> will hold histogram plots 
fig2d, axs2d = plt.subplots(nrows=1, ncols=nzones) # --> will hold 2d plots 
fig3d, axs3d = plt.subplots(nrows=1, ncols=nzones) # --> will hold 3d plots 
cfig, cax = plt.subplots()

# make 3d axis actually 3d 
axs3d[0].remove()
axs3d[1].remove()
axs3d[0] = fig3d.add_subplot(1,nzones, 1, projection='3d')
axs3d[1] = fig3d.add_subplot(1,nzones, 2, projection='3d')

fighi.set_tight_layout(True)
fighi.set_size_inches([11,  7.5])
fig2d.set_tight_layout(True)
fig2d.set_size_inches([11,  5])
# fig3d.set_tight_layout(True)
fig3d.set_size_inches([11,  6])

for i in range(3):
    axshi[i,0].set_ylabel('Probably Density')

#%% main loop 

# get burnin 
idx, chain_converged = get_burnin(df[stable],pt_threshold)

# plot liklihood data and paths
params = {}
for i in range(nzones):
    params[i] = {}
    n = i+1
    xi = df['alpha_%i'%n][stable].values 
    yi = df['vn_%i'%n][stable].values
    _zi = df['k_%i'%n][stable].values
    zi = convertk2K(_zi) 
    pi = df['Pt'][stable]
    
    logxi = np.log10(xi)
    logzi = np.log10(zi)
    
    ## create 3D plots 
    axs3d[i].scatter(logxi, yi, logzi, c=pi, cmap=cmap)
    
    axs3d[i].set_xlabel('Alpha (1/m)')
    xlabel_pos = np.array([-3., -2., -1.,  0.])
    xlabel_val = 10**xlabel_pos
    axs3d[i].set_xticks(xlabel_pos, xlabel_val)
    axs3d[i].set_ylabel('N (-)')
    
    axs3d[i].set_zlabel('K (m/day)')
    zlabel_pos = np.array([ -2, -1.,  0., 1.0])
    zlabel_val = 10**zlabel_pos
    axs3d[i].set_zticks(zlabel_pos, zlabel_val)
    
    ## create matrix of likelihoods and density for better visualisation 
    binwidth = 0.025
    xn = int((xlim[1] - xlim[0])/binwidth)
    yn = int((ylim[1] - ylim[0])/binwidth)
    # xg = np.linspace(xlim[0], xlim[1], xn)
    xg  = np.linspace(np.log10(xlim[0]), np.log10(xlim[1]), xn)
    yg = np.linspace(ylim[0], ylim[1], yn)
    xgg, ygg = np.meshgrid(xg,yg)
    Lmat = np.zeros((yn-1, xn-1), dtype=float) # likelihood matrix 
    Dmat = np.zeros((yn-1, xn-1), dtype=int) # density matrix
    for ii in range(xgg.shape[0]-1):
        for jj in range(ygg.shape[1]-1):
            idx_gg = (logxi >= xgg[ii,jj]) & (logxi < xgg[ii+1,jj+1]) & (yi >= ygg[ii,jj]) & (yi < ygg[ii+1,jj+1])
            A = pi[idx_gg]
            if len(A) > 0: 
                Lmat[ii,jj] = np.max(A)
                Dmat[ii,jj] = len(A)
            else:
                Lmat[ii,jj] = np.nan
                Dmat[ii,jj] = 0
                
    vmin = np.min(df['Pt'][stable])
    vmax = np.max(df['Pt'][stable])
    
    cmp = axs2d[i].scatter(logxi, yi,
                           c = df['Pt'][stable],
                           marker='.',cmap=cmap,
                           vmin=vmin,vmax=vmax)
    
    axs2d[i].pcolor(xgg,ygg,Lmat,cmap=cmap, alpha = 0.75, 
                    vmin=vmin,vmax=vmax)

    if synth:
        axs2d[i].scatter(np.log10(simA[i]), simN[i], c='r', marker='+',s=30)
        xxA = [np.log10(simA[i]), np.log10(simA[i])]
        xxN = [simN[i], simN[i]]
        xxK = [np.log10(simK[i]), np.log10(simK[i])]
        axshi[0,i].plot(xxA, [0,2], c='k', linestyle='-')
        axshi[1,i].plot(xxN, [0,2], c='k', linestyle='-')
        axshi[2,i].plot(xxK, [0,2], c='k', linestyle='-')
        
        
    # sort axis labels 
    axs2d[i].set_title(zone_names[i])
    axs2d[i].set_xlabel('Alpha (1/m)')
    axs2d[i].set_ylabel('N (-)')
    axs2d[i].set_xlim(np.log10(xlim))
    axs2d[i].set_ylim(ylim)
    axs2d[i].set_xticks(xlabel_pos, xlabel_val)
    
    # grab colour bar 
    if i == 0:
        cbar = plt.colorbar(cmp, ax=cax, location="bottom")
        cbar.set_label('Normalised Likelihood')
    
    
    ## add full histograms
    axshi[0,i].hist(logxi[idx], bins=100, density=True)
    axshi[1,i].hist(yi[idx],bins=100, density=True)
    axshi[2,i].hist(logzi[idx],bins=100, density=True)
    
    # sort histogram axis 
    axshi[0,i].set_title(zone_names[i])
    axshi[0,i].set_xlabel('Alpha (1/m)')
    axshi[0,i].set_xticks(xlabel_pos, xlabel_val)
    axshi[1,i].set_xlabel('N (-)')
    axshi[2,i].set_xlabel('K (m/day)')
    axshi[2,i].set_xticks(zlabel_pos, zlabel_val)
    
    axshi[0,i].set_xlim([np.log10(0.001), np.log10(1.0)])
    axshi[1,i].set_xlim([1.0, 2.5])
    axshi[2,i].set_xlim([np.log10(0.001), np.log10(10.0)])
    
    for j in range(3):
        axshi[j,i].set_ylim([0,2])
        
    ## shapiro wilk tests
    try: 
        snormx_coef = shapiro(logxi).statistic
        snormy_coef = shapiro(yi).statistic
        snormz_coef = shapiro(logzi).statistic
    except: 
        snormx_coef = 0
        snormy_coef = 0
        snormz_coef = 0 
        
    ## means and stds 
    alpha_mean = np.mean(logxi[idx])
    alpha_std = np.std(logxi[idx])
    n_mean = np.mean(yi[idx])
    n_std = np.std(yi[idx])
    k_mean = np.mean(logzi[idx])
    k_std = np.std(logzi[idx])
    
    log.log('Mean and standard deviations for zone %i:' % n)
    log.log('Log10(Alpha): %f +/- %f'%(alpha_mean, alpha_std))
    log.log('N: %f +/- %f'%(n_mean, n_std))
    log.log('Log10(K): %f +/- %f'%(k_mean, k_std))
    log.log('')
    if show_means: 
        axshi[0,i].plot([alpha_mean, alpha_mean], [0,2], c='k', linestyle=':')
        axshi[1,i].plot([n_mean, n_mean], [0,2], c='k', linestyle=':')
        axshi[2,i].plot([k_mean, k_mean], [0,2], c='k', linestyle=':')
    
    
    ## try and fit curves 
    logx = logxi[idx]
    y = yi[idx]
    logz = logzi[idx]
    
    distx = distribution(logx, dists[0][i], 100, log=True)
    disty = distribution(y, dists[1][i], 100)
    distz = distribution(logz, dists[2][i], 100, log=True)
    
    # fit a histogram (to the better fitting selections)
    try:
        px = distx.fit()
        # following code plots the fit on the histograms
        lx = np.linspace(min(logx), max(logx), 100)
        distx.plot(axshi[0,i], 'r')
    except:
        log.log('Couldnt find optimal parameters for zone %i alpha'%n)
        px = np.full(6,np.nan)
        
    try:
        py = disty.fit()
        ly = np.linspace(min(y), max(y), 100)    
        disty.plot(axshi[1,i], 'r')
    except:
        log.log('Couldnt find optimal parameters for zone %i N'%n)
        py = np.full(6,np.nan)
        
    try:
        pz = distz.fit()
        lz = np.linspace(min(logz), max(logz), 100)    
        distz.plot(axshi[2,i], 'r')
    except:
        log.log('Couldnt find optimal parameters for zone %i K'%n)
        pz = np.full(6,np.nan)
        
    params[i]['alpha'] = px[0]
    params[i]['alpha_std'] = px[1]
    params[i]['n'] = py[0]
    params[i]['n_std'] = py[1]
    params[i]['K'] = pz[0]
    params[i]['K_std'] = pz[1]

    log.log('Fitting statistics (anti-logged) for zone %i:' % n)
    if dists[0][i] == 'bimodal':
        log.log('Alpha 1: %f +/- %f | %f (1/m)' % (px[0], px[1], px[2]))
        log.log('Alpha 2: %f +/- %f | %f (1/m)' % (px[3], px[4], px[5]))
        if show_modes: 
            axshi[0,i].plot(np.log10([px[0], px[0]]), [0,2], c='k', linestyle=':')
            axshi[0,i].plot(np.log10([px[3], px[3]]), [0,2], c='k', linestyle=':')
    else:
        log.log('Alpha : %f +/- %f | %f (1/m)' % (px[0], px[1], px[2]))
        if show_modes: 
            axshi[0,i].plot(np.log10([px[0], px[0]]), [0,2], c='k', linestyle=':')
    log.log('Shapiro-Wilk coefficient for Alpha: %3.2f'%snormx_coef)
    
    if dists[1][i] == 'bimodal':
        log.log('N 1: %f +/- %f | %f (-)' % (py[0], py[1], py[2]))
        log.log('N 2: %f +/- %f | %f (-)' % (py[3], py[4], py[5]))
        if show_modes: 
            axshi[1,i].plot([py[0], py[0]], [0,2], c='k', linestyle=':')
            axshi[1,i].plot([py[3], py[3]], [0,2], c='k', linestyle=':')
    else:
        log.log('N : %f +/- %f | %f (-)' % (py[0], py[1], py[2]))
        if show_modes: 
            axshi[1,i].plot([py[0], py[0]], [0,2], c='k', linestyle=':')
    log.log('Shapiro-Wilk coefficient for N: %3.2f'%snormy_coef)
    
    if dists[2][i] == 'bimodal':
        log.log('K 1: %f +/- %f | %f (-)' % (pz[0], pz[1], pz[2]))
        log.log('K 2: %f +/- %f | %f (-)' % (pz[3], pz[4], pz[5]))
        if show_modes: 
            axshi[2,i].plot(np.log10([pz[0], pz[0]]), [0,2], c='k', linestyle=':')
            axshi[2,i].plot(np.log10([pz[3], pz[3]]), [0,2], c='k', linestyle=':')
    else:
        log.log('K : %f +/- %f | %f (-)' % (pz[0], pz[1], pz[2]))
        if show_modes: 
            axshi[2,i].plot(np.log10([pz[0], pz[0]]), [0,2], c='k', linestyle=':')
    log.log('Shapiro-Wilk coefficient for K: %3.2f'%snormz_coef)
    log.log(' ')
    nsample = len(df['alpha_%i' % n][stable][idx])
    
## save figures 
if savfig:
    fig2d.savefig(os.path.join(dirname, 'mcmc_pcolor2d.png'), dpi= 600)
    fig3d.savefig(os.path.join(dirname, 'mcmc_scatter3d.png'), dpi= 600)
    cfig.savefig(os.path.join(dirname, 'colorbar.png'))
    fighi.savefig(os.path.join(dirname, 'mcmc_histograms.png'), dpi=600)

# # save statistics to file 
log.log('Nsample = %i' % nsample)



# axs[nzones].set_xlim([min(df.run),max(df.run)])
# axs[nzones].legend(bbox_to_anchor=(1.05, 1.05))
# figs[nzones].set_tight_layout(True)
# figs[nzones].set_size_inches([14,6])


best_model_idx = np.argmax(df['Pt'])
for i in range(nzones):
    n = i+1
    model = [df['alpha_%i' % n][best_model_idx],
             df['vn_%i' % n][best_model_idx],
             convertk2K(df['k_%i' % n][best_model_idx])]
    log.log('Best fit for zone %i' % n)
    log.log('Alpha : %f (1/m)' % (model[0]))
    log.log('N : %f (-)' % (model[1]))
    log.log('K : %f (m/day)' %model[2])
log.log('Chains converged: %i'%len(chain_converged))

#%% 2d plots for testing 
# fig, ax = plt.subplots()

# ax.scatter(df['k_%i'%1][stable].values, df['k_%i'%2][stable].values, c=df['Pt'][stable])

# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_xlabel('k - SSF (1/m^2)')
# ax.set_ylabel('k - WMF (1/m^2)')


# fig, ax = plt.subplots()

# ax.scatter(df['vn_%i'%1][stable].values, df['vn_%i'%2][stable].values, c=df['Pt'][stable])

# # ax.set_xscale('log')
# # ax.set_yscale('log')
# ax.set_xlabel('vn - SSF')
# ax.set_ylabel('vn - WMF')

# fig, ax = plt.subplots()

# ax.scatter(df['alpha_%i'%1][stable].values, df['alpha_%i'%2][stable].values, c=df['Pt'][stable])

# ax.set_xscale('log')
# ax.set_yscale('log')
# ax.set_xlabel('alpha - SSF')
# ax.set_ylabel('alpha - WMF')
