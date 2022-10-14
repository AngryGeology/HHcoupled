#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 13:56:08 2022
Compile topography data 
@author: jimmy
"""
import sys, os 
from datetime import datetime
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.interpolate import LinearNDInterpolator
from tqdm import tqdm 
sys.path.append('/home/jimmy/phd/yolo/')
import yolo.gpsTools as gps


#%% file locations 
dirname = '/home/jimmy/phd/Hollin_Hill/Data/compiled_protocols_remapped'
resdir = 'resData'
topo_out = 'topoData'
elec_out = 'elecData'
locationfile = '/home/jimmy/phd/Hollin_Hill/Data/init_elec_locs.csv'
elec_df = pd.read_csv('/home/jimmy/phd/yolo/data/all_electrode_positions_jamyd91.csv',header=0)
surf_df = pd.read_csv('/home/jimmy/phd/yolo/data/all_topo_positions_jamyd91.csv',header=0)

#%% grab the xy positions needed for single line (maybe?)
elec = pd.read_csv(locationfile)
# correct for negative id numbers 
neg_idx = elec['electrode'] < 0 
pos_idx = elec['electrode'] > 0 
elec['electrode'][neg_idx] += 81 
elec['electrode'][pos_idx] += 80

fig, ax = plt.subplots()
ax.scatter(elec['x'],elec['y'])

idx = (elec['x'] < 25) & (elec['x'] > 15)
ax.scatter(elec['x'][idx],elec['y'][idx])
elec2keep = sorted(elec['electrode'].values[idx])
elec = elec[idx].reset_index()
subsample = 10
X = np.array([])
Y = np.array([]) 
for i in range(1,len(elec)):
    x = np.linspace(elec['x'][i-1],elec['x'][i],subsample)
    y = np.linspace(elec['y'][i-1],elec['y'][i],subsample)
    X = np.append(X,x)
    Y = np.append(Y,y)

ax.scatter(X,Y)

#%% find times where we need to find electrode positions / topo 
files = sorted(os.listdir(resdir))
relevant_dates = []
for f in files:
    if f.endswith('.dat'):
        fname = f.split('.')[0] 
        relevant_dates.append(fname)
        
#%% get positions 
dates = list(surf_df['Date'])

elec_x, elec_y, elec_z = gps.get_peg_pos(elec_df, 0)
topo_x, topo_y, topo_z = gps.get_peg_pos(surf_df, 0)

# interp line 
subsample = 150
yi = np.linspace(min(topo_y),max(topo_y),subsample)
xi = np.zeros_like(yi) + np.mean(X)
missing = []
fig2,ax2 = plt.subplots()
for i in tqdm(range(len(relevant_dates)),ncols=100):
    if relevant_dates[i] in dates: 
        j = dates.index(relevant_dates[i]) # get index of relevant date 
    else:
        j = 0 
        missing.append(relevant_dates[i])
        while relevant_dates[i-j] not in dates:
            j += 1 
        j = dates.index(relevant_dates[i-j]) 
    elec_x, elec_y, elec_z = gps.get_peg_pos(elec_df, j) # get electrode xy
    ex = [elec_x[k-1] for k in elec2keep]
    ey = [elec_y[k-1] for k in elec2keep]
    topo_x, topo_y, topo_z = gps.get_peg_pos(surf_df, j) # get topo xyz 
    itopo = np.c_[xi,yi] # create points where to interpolate topo 
    ielec = np.c_[ex,ey] # points where to find electrode elevation 
    points = np.c_[topo_x,topo_y,topo_z]
    ifunc = LinearNDInterpolator(points[:,0:2],points[:,2])
    zi = ifunc(itopo)
    ez = ifunc(ielec)
    
    elec_tmp = pd.DataFrame({'id':elec2keep,
                             'x':ex,
                             'y':ey,
                             'z':ez})
    topo_tmp = pd.DataFrame({'x':xi,
                             'y':yi,
                             'z':zi})
    ax2.plot(yi,zi)
    elec_tmp.to_csv(os.path.join(elec_out,relevant_dates[i]+'.csv'),index=False)
    topo_tmp.to_csv(os.path.join(topo_out,relevant_dates[i]+'.csv'),index=False)
    