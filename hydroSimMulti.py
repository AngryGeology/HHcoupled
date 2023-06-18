#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May  5 16:49:54 2023

@author: jimmy
"""

import os, sys, shutil, time 
from datetime import datetime
import numpy as np 
import pandas as pd 
from scipy.spatial import cKDTree
from scipy.stats import pearsonr
import matplotlib.pyplot as plt 
from tqdm import tqdm
# from joblib import Parallel, delayed 
# add custom modules 
if 'RSUTRA' not in sys.path: 
    sys.path.append('RSUTRA')
linux_r_path = '/home/jimmy/phd/resipy/src'
# win_r_path = r'C:\Users\boydj1\Software\resipy\src' 
win_r_path = r'C:\Users\jimmy\Documents\resipy\src'
if sys.platform == 'linux' and linux_r_path not in sys.path: 
    sys.path.append(linux_r_path)
if 'win' in sys.platform.lower() and win_r_path not in sys.path:
    sys.path.append(win_r_path)
from SUTRAhandler import handler, material, readNod
from petroFuncs import ssf_petro_sat, wmf_petro_sat, temp_uncorrect
from petroFuncs import wmf_petro_sat_shallow
import createInput as ci 
from createInput import secinday 
 
plt.close('all')

c0 = time.time() 
# create working environment 
exec_loc = '/home/jimmy/programs/SUTRA_JB/bin/sutra_s'
if 'win' in sys.platform.lower():
    # exec_loc = r'C:/Users/boydj1/Software/SUTRA/bin/sutra.exe'
    exec_loc = r'C:/Users/jimmy/Documents/Programs/SUTRA/bin/sutra.exe'

model_dir = 'Models'
sim_dir = os.path.join(model_dir,'HydroSimMulti')
for d in [model_dir,sim_dir]:
    if not os.path.exists(d):
        os.mkdir(d)
        

samples = 1000
ncpu = 16 
run_hydro = False 

#%% load in the data 
elec = ci.HH_getElec()
hydro_data, data_seq, sequences, survey_keys, rfiles, sdiy = ci.HH_data(12)
TimeStamp = np.arange(len(hydro_data))
times = np.asarray(TimeStamp, dtype=int)


## interpolate missing vwc values 
cosmos_vwc = hydro_data['VWC'].values
nanidx = cosmos_vwc<0
x = np.arange(len(cosmos_vwc))
cosmos_vwc[nanidx] = np.interp(x[nanidx],
                               x[np.invert(nanidx)],
                               cosmos_vwc[np.invert(nanidx)])

#%% create mesh with some pressure conditions 
mesh, zone_flags, dx, pressures, boundaries = ci.HH_mesh(False)

# create boundary conditions 
general_node = []
general_type = []
general_pressure  = []

# set left side as seepage boundary 
general_node += boundaries['left'].tolist()
general_type += ['seep']*len(boundaries['left'])
general_pressure += [0.0]*len(boundaries['left'])
left_side = {'idx':boundaries['left'],
             'color':'m',
             'label':'Seepage node'}

# set top side as seepage boundary 
general_node += boundaries['top'].tolist()
general_type += ['seep']*len(boundaries['top'])
general_pressure += [0.0]*len(boundaries['top'])
top_side = {'idx':boundaries['top'],
            'color':'b',
            'label':'Seepage and source node'}

# set basal nodes as pressure boundary 
general_node += boundaries['bottom'].tolist()
general_type += ['pres']*len(boundaries['bottom'])
general_pressure += pressures['base'].tolist()
bot_side = {'idx':boundaries['bottom'],
            'color':'k',
            'label':'Min.Pressure Node'}

# set top boundary as source nodes 
source_node = boundaries['top'] + 1 

# nodal starting pressures 
nodal_pressures = pressures['nodal']
nodal_temp = [0]*mesh.numnp

general_node = np.array(general_node) + 1
general_pressure = np.array(general_pressure) 

## create some flags for passing to show setup command 
node_flags = [left_side,top_side,bot_side]

#%% setup infiltration for SUTRA 
ntimes = len(hydro_data)
precip=hydro_data['PRECIP'].values/secinday # to get in mm/s == kg/s 
pet=hydro_data['PE'].values/secinday 
kc=hydro_data['Kc'].values 
tdx = sum(dx)
fluidinp, tempinp = ci.prepRainfall(dx,precip,pet,kc,len(source_node),ntimes)
#rainfall is scaled by the number of elements 

#%% create materials 
# SSF properties from mcmc search 
SSF = material(Ksat=0.64,theta_res=0.06,theta_sat=0.38,
                alpha=0.24,vn=1.30,name='STAITHES')
# WMF properties from mcmc search 
WMF = material(Ksat=0.013,theta_res=0.1,theta_sat=0.48,
                alpha=0.11,vn=1.64,name='WHITBY')
# RMF properties 
RMF = material(Ksat=0.64,theta_res=0.1,theta_sat=0.48,
               alpha=0.0126,vn=1.44,name='REDCAR')

SSF.setPetroFuncs(ssf_petro_sat,ssf_petro_sat)
WMF.setPetroFuncs(wmf_petro_sat_shallow, wmf_petro_sat)
RMF.setPetroFuncs(wmf_petro_sat, wmf_petro_sat)

#%% create MC param 

ssf_config = {'alpha':0.244049,
              'alpha_sigma':0.091054,
              'n':1.298371,
              'n_sigma':0.057363}

wmf_config = {'alpha':0.114789,
              'alpha_sigma':0.036240,
              'n1':1.640132,
              'n1_sigma':0.118476,
              'n2':2.141462,
              'n2_sigma':0.268010}

samples2 = int(samples/2)
ssf_a = np.random.normal(ssf_config['alpha'], ssf_config['alpha_sigma'], samples) 
ssf_n = np.random.normal(ssf_config['n'], ssf_config['n_sigma'], samples) 
wmf_a = np.random.normal(wmf_config['alpha'], wmf_config['alpha_sigma'], samples) 
wmf_n = np.concatenate([np.random.normal(wmf_config['n1'], wmf_config['n1_sigma'], samples2), 
                        np.random.normal(wmf_config['n2'], wmf_config['n2_sigma'], samples2)])

ssf_param = {'alpha':ssf_a,'vn':ssf_n}
wmf_param = {'alpha':wmf_a,'vn':wmf_n}

# setup MC param 
SSF.setMCparam(ssf_param)
WMF.setMCparam(wmf_param)

fig,axs = plt.subplots(2,2) 
for i in range(2):
    axs[1,i].set_xlabel('N (-)')
    axs[0,i].set_xlabel('alpha (1/m)')
axs[0,0].set_title('SSF')
axs[0,1].set_title('WMF')

nbin = 100 
axs[0,0].hist(ssf_a,bins=nbin)
axs[1,0].hist(ssf_n,bins=nbin)
axs[0,1].hist(wmf_a,bins=nbin)
axs[1,1].hist(wmf_n,bins=nbin)

#%% run hydrological models 
run_time = 0 
setup_time = 0 
nstep = ntimes + 1 
# get saturations 
def getSats(i):
    fdir = pdirs[i]
    fname = os.path.join(fdir,'sutra.nod')
    data, n = readNod(fname)
    array = np.zeros((mesh.numnp,nstep))
    for j in range(nstep):
        key = 'step%i'%j
        if key not in data.keys(): 
            return array
        array[:,j] = data[key]['Saturation']
    return array

if run_hydro: 
    h = handler(dname=sim_dir, ifac=1,tlength=secinday,iobs=1, 
                flow = 'transient',
                transport = 'transient',
                sim_type='solute')
    h.maxIter = 300
    h.rpmax = 5e5
    h.drainage = 1e-8
    h.cpu = ncpu 
    h.clearDir()
    h.clearMultiRun()
    h.setMesh(mesh)
    h.setEXEC(exec_loc)
    
    h.addMaterial(SSF,zone_flags['SSF'])
    h.addMaterial(WMF,zone_flags['WMF'])
    h.addMaterial(RMF,zone_flags['RMF'])
    
    h.setupInp(times=times, 
               source_node=source_node, 
               general_node=general_node, general_type=general_type, 
               source_val=0,
               solver='direct')
    
    h.pressure = general_pressure
    h.writeInp(maximise_io=True) # write input without water table at base of column
    h.writeBcs(times, source_node, fluidinp, tempinp)
    h.writeIcs(nodal_pressures, nodal_temp) # INITIAL CONDITIONS 
    h.writeVg()
    h.writeFil(ignore=['BCOP', 'BCOPG'])
    
    setupfig = h.showSetup(custom_flags=node_flags, return_fig=True) 
    
    setup_time = time.time() - c0 
    c0 = time.time() 
    
    ### run ###  
    h.setupMultiRun() 
    h.runMultiRun(return_data=False)
    run_time = time.time() - c0 

    #% read in results 
    satmatrix = np.zeros((mesh.numnp,nstep,samples))
    pdirs = [] 
    for e in sorted(os.listdir(sim_dir)):
        if os.path.isdir(os.path.join(sim_dir,e)):
            pdirs.append(os.path.join(sim_dir,e))
    
    c0 = time.time()
    # run in serial 
    for i in tqdm(range(samples),ncols=100,desc='Reading'):
        satmatrix[:,:,i] = getSats(i)
    retrieval_time = time.time() - c0 
    
    np.save(os.path.join(sim_dir,'satmatrix.npy'),satmatrix)
    
    # report on timings 
    print('Setup time : %f s'%setup_time)
    print('Completed %i runs in %f s (%f s per run)'%(samples,run_time,run_time/samples))
    print('Reading time : %f s'%retrieval_time) 


#%% load in results 
satmatrix = np.load(os.path.join(sim_dir,'satmatrix.npy'))

# get stable results? 
stable = [True]*satmatrix.shape[2]
tidx = np.array([False]*nstep,dtype=bool)
for i in range(1,ntimes):
    if hydro_data['datetime'][i].year == 2016:
        tidx[i+1] = True 
for i in range(satmatrix.shape[2]):
    a = satmatrix[1495,tidx,i]
    if np.all(a==1.0) or np.all(a==0.0):
        stable[i] = False 
    
# can't really filter due to how ram hungry satmatrix is 
# satmatrix = satmatrix[:,:,np.array(stable,dtype=bool)]

samples = np.count_nonzero(stable)
print('%i simulations stable'%samples)

#%% analysis 
def toMC(x):
    # convert to atterberg moisture content 
    # Pg = 2.73
    Pw = 1.00 
    Ww = WMF.theta_sat*Pw*x
    Wd = 1.3 # (1-WMF.theta_sat)*Pg
    return Ww / Wd

# get xy positions 
X = mesh.node[:,0]
Y = mesh.node[:,2]
tree = cKDTree(np.c_[X,Y])

# get index of point in wmf  
xp = 129; yp = 90
wmf_point = [xp, yp]
dist,idxs = tree.query(np.c_[[xp],[yp]])
idx_wmf = idxs[0]

# get index of point in ssf
xp = 25.92; yp = 65.77
ssf_point = [xp, yp]
dist,idxs = tree.query(np.c_[[xp],[yp]])
idx_ssf = idxs[0]


# fig, ax = plt.subplots()
# for i in range(samples):
#     ax.plot(satmatrix[idx,:,i])

stats_wmf = {
    'min':np.zeros(nstep),
    'max':np.zeros(nstep),
    'std':np.zeros(nstep),
    'avg':np.zeros(nstep)
    }

stats_ssf = {
    'min':np.zeros(nstep),
    'max':np.zeros(nstep),
    'std':np.zeros(nstep),
    'avg':np.zeros(nstep),
    }

# for wmf wanna do some extra stats in terms of moisture content so 
stats_wmf['avg_mc'] = np.zeros(nstep)
stats_wmf['std_mc'] = np.zeros(nstep)

for i in range(nstep):
    a = satmatrix[idx_wmf,i,stable]
    stats_wmf['min'][i] = np.min(a)
    stats_wmf['max'][i] = np.max(a)
    stats_wmf['std'][i] = np.std(a)
    stats_wmf['avg'][i] = np.mean(a)
    stats_wmf['avg_mc'][i] = np.mean(toMC(a))
    stats_wmf['std_mc'][i] = np.std(toMC(a))
    a = satmatrix[idx_ssf,i,stable]
    stats_ssf['min'][i] = np.min(a)
    stats_ssf['max'][i] = np.max(a)
    stats_ssf['std'][i] = np.std(a)
    stats_ssf['avg'][i] = np.mean(a)
    
fig, (ax,axc,axr) = plt.subplots(nrows=3)

# add rainfall 
axr.bar(hydro_data['datetime'],hydro_data['PRECIP'].values,
        color='b', label='Measured Rainfall')
axr.bar(hydro_data['datetime'],-hydro_data['PE'].values*hydro_data['Kc'].values,
        color='r', label='Evapotranspiration')

axr.set_ylabel('rainfall (mm/day)')
axr.set_xlabel('date')
axr.legend()


# track colours 
yellow = [0.8,0.8,0.4] 
purple = [0.8,0.4,0.8]
red = [0.8,0.2,0.2]    
grey = [0.2,0.2,0.2]


### add saturation tracks ### 
# ax.plot(hydro_data['datetime'], stats_wmf['min'][1:], 
#         linestyle='--', color=(0.5,0.5,1,0.5),label='min')
# ax.plot(hydro_data['datetime'], stats_wmf['max'][1:],
#         linestyle=':', color=(0.5,0.5,1,0.5),label='max')
std_max = stats_wmf['avg'][1:] + stats_wmf['std'][1:]
std_min = stats_wmf['avg'][1:] - stats_wmf['std'][1:]
ax.fill_between(hydro_data['datetime'], std_min, std_max, 
                color = purple+[0.5], label='WMF(std)')

std_max = stats_ssf['avg'][1:] + stats_ssf['std'][1:]
std_min = stats_ssf['avg'][1:] - stats_ssf['std'][1:]
ax.fill_between(hydro_data['datetime'], std_min, std_max, 
                color = yellow+[0.5], label='SSF(std)')

ax.plot(hydro_data['datetime'], stats_wmf['avg'][1:],
        linestyle='-', color=purple, label='WMF(mean)')
ax.plot(hydro_data['datetime'], stats_ssf['avg'][1:],
        linestyle='-', color=yellow, label='SSF(mean)')


ax.set_ylabel('Saturation (-)')
ax.legend()
ax.set_ylim([np.min(std_min),1])

# add cosmos track 
axc.plot(hydro_data['datetime'],hydro_data['VWC'],c='b',label='Cosmos')
axc.legend()
axc.set_ylabel('VWC(%)')
axc.set_ylim([20,np.max(hydro_data['VWC'])])


for e in (ax,axr,axc):
    e.set_xlim([datetime(2015,1,1),datetime(2017,1,1)])
    
#%% moisture content estimation in comparison to movements 

#load in some data
pegs = pd.read_csv('/home/jimmy/phd/yolo/data/peg_movements_jamyd91_2019-07-19.csv',header=[0,1])
atterberg = pd.read_csv('/home/jimmy/phd/Hollin_Hill/Data/merrit_plasticity_data.csv') 
doi = [False]*len(atterberg) # depth of interest 
for i in range(len(atterberg)):
    depth = float(atterberg['ID'][i].split()[1].replace('m',''))
    if depth < 3:
        doi[i] = True 
atterbergf = atterberg[doi]

# remember that: 
# MC = (Ww - Wd) / Wd

fig, (ax2,ax3) = plt.subplots(nrows=2)
### peg movement ###  
# extract dates 
def str2dat(string):
    dt = datetime.strptime(string, "%Y-%m-%d %H:%M:%S")
    return dt 
peg_dates = [str2dat(pegs['Unnamed: 1_level_0'][ 'Date'][i]) for i in range(len(pegs))]

# get and plot movement 
peg_id = '1'
movements = np.zeros(len(pegs),dtype=float)
ax1 = ax2.twinx()
for j in [11,12,13,24,25,26,29,30,31]:
    peg_id = '%i'%(j)
    for i in range(1,len(pegs)):
        x0 = pegs[peg_id]['x'][i-1]
        y0 = pegs[peg_id]['y'][i-1]
        x1 = pegs[peg_id]['x'][i]
        y1 = pegs[peg_id]['y'][i]
        sqdist = ((x0-x1)**2) + ((y0-y1)**2)
        dist = np.sqrt(sqdist)
        movements[i] = dist 
    ax2.bar(peg_dates,movements,color='r',width=10)
ax2.set_ylabel('Maximum measured peg movement (m)')

### moisture content estimation ###  
std_max = stats_wmf['avg_mc'][1:] + stats_wmf['std_mc'][1:]
std_min = stats_wmf['avg_mc'][1:] - stats_wmf['std_mc'][1:]
std_max[std_max>np.max(stats_wmf['avg_mc'])] = np.max(stats_wmf['avg_mc'])

ax1.plot(hydro_data['datetime'],stats_wmf['avg_mc'][1:]*100,
         label='WMF(est)(mean)',color=grey)
ax1.fill_between(hydro_data['datetime'], std_min*100, std_max*100, 
                 color = grey+[0.5], label='WMF(est)(std)')


# Ed's testing in the lab suggests the plastic limit is 34% 
ax1.plot(hydro_data['datetime'],np.ones(ntimes)*34,color=red,
         linestyle='--', label='Plastic Limit (%)')

# from andy merrit's paper we get that the plastic limit is ~32 - 40%
PL = atterbergf['Plastic Limit (%)']
ax1.fill_between(hydro_data['datetime'], 
                 np.min(PL), np.max(PL), 
                 color = red+[0.2], label='Plastic Limit (Merrit et al, 2013)')
ax1.legend(loc='upper left')
ax1.set_xlim([datetime(2015,1,1),datetime(2017,1,1)])
ax1.set_ylabel('Gravimetric Moisture Content')
ax1.set_xlabel('Date')

#%% count times each simulation exceeded limit X in 2016
pnct = [0]*nstep
for i in range(nstep):
    a = satmatrix[idx_wmf,i,stable]
    gmc = toMC(a)
    count = np.count_nonzero(gmc>0.34)
    pnct[i] = (count/samples)*100
    
ax3.bar(hydro_data['datetime'],pnct[1:],color='k')
ax3.set_ylabel('Simulations Exceeding Threshold (%)')
ax3.set_xlabel('Date')
ax3.set_xlim([datetime(2015,1,1),datetime(2017,1,1)])

#%% plot up peg positions 
# fig, ax3 = plt.subplots()
# for j in range(45):
#     peg_id = '%i'%(j+1)
#     px = pegs[peg_id]['x'][0]
#     py = pegs[peg_id]['y'][0]
#     ax3.scatter(px,py)
#     ax3.text(px,py,s=peg_id)

#%% show mesh 
fig, axmesh = plt.subplots()
mesh.show(ax=axmesh,attr='zone',electrodes=False,color_map='Greys',vmax=4,edge_color=grey)
axmesh.scatter(wmf_point[0],wmf_point[1],label='WMF sample',color=purple,edgecolors='k')
axmesh.scatter(ssf_point[0],ssf_point[1],label='SSF sample',color=yellow,edgecolors='k')
axmesh.legend() 

#%% plot scatter plot of cosmos and wmf saturation 
fig,ax4 = plt.subplots()
fidx = hydro_data['VWC'] > 0 
ax4.scatter(hydro_data['VWC'][fidx],stats_wmf['avg'][1:][fidx])
ax4.set_ylabel('Saturation (modelled) (-)')
ax4.set_xlabel('VWC (measured) (%)')

pr=pearsonr(hydro_data['VWC'][fidx],stats_wmf['avg'][1:][fidx])

print('pearson correlation coefficient is %f'%pr[0])




