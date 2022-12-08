#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 13 16:10:40 2022
Plot the results of an MCMC chain
@author: jimmy
"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.stats import norm

plt.close('all')

# show histograms 
def scatter_hist(x, y, ax, ax_histx, ax_histy, trans=False):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)
    
    if trans:
        color = (0,0,1,0.5)
    else:
        color = (0,0,1,1)

    ax_histx.hist(x, bins=100, density=True, color=color)
    ax_histy.hist(y, orientation='horizontal', bins=100, density=True,
                  color=color)

# get mcmc result file 

# df = pd.read_csv('HydroMCMC/mergedMCMClog.csv')
df = pd.read_csv('/home/jimmy/phd/Hollin_Hill/Coupled/SyntheticStudy/Models/MCMC/mergedMCMClog.csv')
pt_threshold = 0.4500 
nzones = 2 
stable = df['Stable']

#%% create figures 
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
    axs['hist_x%i'%i] = ax_histx
    axs['hist_y%i'%i] = ax_histy
    
figs[nzones],axs[nzones] = plt.subplots()
axs[nzones].set_ylabel('Normalised liklehood')
axs[nzones].set_xlabel('Run')

    
#%% plot liklihood data and paths 
params = {}
for i in range(nzones): 
    params[i] = {} 
    n = i+1 
    axs[i].tricontourf(df['alpha_%i'%n][stable], 
                       df['vn_%i'%n][stable],
                       df['Pt'][stable])
    axs[i].set_xlabel('Alpha (1/m)')
    axs[i].set_ylabel('N (-)')
    # add full histogram 
    x = df['alpha_%i'%n][stable]
    y = df['vn_%i'%n][stable]
    scatter_hist(x, y, axs[i], axs['hist_x%i'%i], axs['hist_y%i'%i],True)
    # add histograms used for fitting 
    idx = df['Pt'] > pt_threshold
    x = df['alpha_%i'%n][stable][idx]
    y = df['vn_%i'%n][stable][idx]
    scatter_hist(x, y, axs[i], axs['hist_x%i'%i], axs['hist_y%i'%i])
    # fit a histogram (to the better fitting selections)
    px = norm.fit(x)
    py = norm.fit(y)
    # following code plots the fit on the histograms 
    lx = np.linspace(min(x), max(x), 100)
    ly = np.linspace(min(y), max(y), 100)
    pdfx = norm.pdf(lx, px[0], px[1])
    pdfy = norm.pdf(ly, py[0], py[1]) 
    axs['hist_x%i'%i].plot(lx,pdfx,c='r')
    axs['hist_y%i'%i].plot(pdfy,ly,c='r')
    
    params[i]['alpha'] = px[0]
    params[i]['alpha_std'] = px[1]
    params[i]['n'] = py[0]
    params[i]['n_std'] = py[1]
    
    print('Fitting statistics for zone %i:'%n)
    print('Alpha : %f +/- %f (1/m)'%(px[0],px[1]))
    print('N : %f +/- %f (-)'%(py[0],py[1]))
    print('\n')
    
    figs[i].savefig('Models/mcmc_figure_zone%i.png'%i)

for chain in np.unique(df['chain']):
    idx = (df['chain'] == chain) & (df['Pt']>0) 
    for i in range(nzones): 
        n = i+1 
        x = df['alpha_%i'%n][idx]
        y = df['vn_%i'%n][idx]
        axs[i].plot(x, y)
    # if chain == 0: 
    axs[nzones].plot(df['run'][stable][idx], df['Pt'][stable][idx])
        
for i in range(nzones):
    figs[i].savefig('Models/mcmc_figure_zone%i_wpaths.png'%i)
    