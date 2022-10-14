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

# show histograms 
def scatter_hist(x, y, ax, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)

    ax_histx.hist(x, bins=100)
    ax_histy.hist(y, orientation='horizontal', bins=100)

# get mcmc result file 

df = pd.read_csv('HydroMCMC/mergedMCMClog.csv')
nzones = 2 
stable = df['Pt']>0

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
    ax_histx.set_ylabel('Count')
    ax_histy.set_xlabel('Count')
    figs[i] = fig 
    axs[i] = ax 
    axs['hist_x%i'%i] = ax_histx
    axs['hist_y%i'%i] = ax_histy
    
#%% plot liklihood data and paths 
params = {}
for i in range(nzones): 
    params[i] = {} 
    n = i+1 
    axs[i].tricontourf(df['alpha-%i'%n][stable]*1000, 
                       df['vn-%i'%n][stable],
                       df['Pt'][stable])
    axs[i].set_xlabel('Alpha (1/m)')
    axs[i].set_ylabel('N (-)')
    # add histograms 
    x = df['alpha-%i'%n][stable]*1000
    y = df['vn-%i'%n][stable]
    scatter_hist(x, y, axs[i], axs['hist_x%i'%i], axs['hist_y%i'%i])
    # fit a histogram 
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

for chain in np.unique(df['chain']):
    idx = (df['chain'] == chain) & (df['Pt']>0) 
    for i in range(nzones): 
        n = i+1 
        x = df['alpha-%i'%n][idx]*1000
        y = df['vn-%i'%n][idx]
        axs[i].plot(x, y)  

    