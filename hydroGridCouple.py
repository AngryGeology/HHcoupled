#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 15:59:20 2023
Hydrological simulation of Hollin Hill, with option of simulating resistivity too! 
@author: jimmy
"""
import os, sys, shutil, time 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
# add custom modules 
if 'RSUTRA' not in sys.path: 
    sys.path.append('RSUTRA')
linux_r_path = '/home/jimmy/phd/resipy/src'
win_r_path = r'C:\Users\boydj1\Software\resipy\src' 
if sys.platform == 'linux' and linux_r_path not in sys.path: 
    sys.path.append(linux_r_path)
if 'win' in sys.platform.lower() and win_r_path not in sys.path:
    sys.path.append(win_r_path)
from SUTRAhandler import handler, material, normLike
from petroFuncs import ssf_petro_sat, wmf_petro_sat 
import createInput as ci 
from createInput import secinday 
from resipy import Project
from resipy.r2in import write2in # needed for forward modelling 
 
plt.close('all')

c0 = time.time() 
# create working environment 
exec_loc = '/home/jimmy/programs/SUTRA_JB/bin/sutra'
if 'win' in sys.platform.lower():
    exec_loc = r'C:/Users/boydj1/Software/SUTRA/bin/sutra.exe'

model_dir = 'Models'
sim_dir = os.path.join(model_dir,'HydroGridSearch')
datadir = os.path.join(sim_dir,'SimData')
for d in [model_dir,sim_dir,datadir]:
    if not os.path.exists(d):
        os.mkdir(d)

#%% load in the data 
elec = ci.HH_getElec()
hydro_data, data_seq, sequences, survey_keys, rfiles = ci.HH_data(12)
TimeStamp = np.arange(len(hydro_data))
times = np.asarray(TimeStamp, dtype=int)

#%% create mesh with some pressure conditions 
mesh, zone_flags, dx, pressures, boundaries = ci.HH_mesh(True)

# create boundary conditions 
general_node = []
general_type = []
general_pressure  = []

# set left side as seepage boundary 
general_node += boundaries['left'].tolist()
general_type += ['seep']*len(boundaries['left'])
general_pressure += [0.0]*len(boundaries['left'])

# set basal nodes as pressure boundary 
general_node += boundaries['bottom'].tolist()
general_type += ['pres']*len(boundaries['bottom'])
general_pressure += pressures['base'].tolist()

# set top boundary as source nodes 
source_node = boundaries['top'] + 1 

# nodal starting pressures 
nodal_pressures = pressures['nodal']
nodal_temp = [0]*mesh.numnp

general_node = np.array(general_node) + 1
general_pressure = np.array(general_pressure) 

#%% setup infiltration for SUTRA 
ntimes = len(hydro_data)
precip=hydro_data['PRECIP'].values/secinday # to get in mm/s == kg/s 
pet=hydro_data['PE'].values/secinday 
numnp = mesh.numnp 
tdx = sum(dx)
fluidinp, tempinp = ci.prepRainfall(tdx,precip,pet,numnp,ntimes)
#rainfall is scaled by the number of elements 

# show input 
fig,ax = plt.subplots()
ax.bar(hydro_data['datetime'],hydro_data['PRECIP'].values,color='b')
ax.bar(hydro_data['datetime'],-hydro_data['PE'].values*0.4,color='r')
ax.set_ylabel('rainfall (mm/day)')
ax.set_xlabel('date')

#%% create materials 
SSF = material(Ksat=0.14,theta_res=0.06,theta_sat=0.38,
               alpha=0.2,vn=1.52,name='STAITHES')
WMF = material(Ksat=0.013,theta_res=0.1,theta_sat=0.48,
               alpha=0.1,vn=1.32,name='WHITBY')
RMF = material(Ksat=0.13,theta_res=0.1,theta_sat=0.48,
               alpha=0.0126,vn=1.44,name='REDCAR')

SSF.setPetro(ssf_petro_sat)
WMF.setPetro(wmf_petro_sat)
RMF.setPetro(wmf_petro_sat)

#%% create handler 
h = handler(dname=sim_dir, ifac=1,tlength=secinday,iobs=1, 
            flow = 'transient',
            transport = 'transient',
            sim_type='solute')
h.maxIter = 300
h.rpmax = 1e4 
h.drainage = 1e-8
h.clearDir()
h.setMesh(mesh)
h.setEXEC(exec_loc)
h.cpu = 20 # number of processors to use 

h.addMaterial(SSF,zone_flags['SSF'])
h.addMaterial(WMF,zone_flags['WMF'])
h.addMaterial(RMF,zone_flags['RMF'])

h.setupInp(times=times, 
           source_node=source_node, 
           general_node=general_node, general_type=general_type, 
           source_val=0,
           solver='direct')

h.pressure = general_pressure
h.writeInp() # write input without water table at base of column
h.writeBcs(times, source_node, fluidinp, tempinp)
h.writeIcs(nodal_pressures, nodal_temp) # INITIAL CONDITIONS 
h.writeVg()
h.writeFil(ignore=['BCOP', 'BCOPG'])

h.showSetup() 

setup_time = time.time() - c0 
c0 = time.time() 

#%% setup R2 project for res modelling 
k = Project(dirname=sim_dir)
k.setElec(elec)
k.createMesh(cl_factor=4)

h.setRproject(k)
h.setupRparam(data_seq, write2in, survey_keys, seqs=sequences)
depths, node_depths = h.getDepths()

#%% setup grid search 
# want to examine VG parameters for SSF only for now 
alpha_SSF = np.linspace(0.005, 2.0, 10)
alpha_WMF = np.linspace(0.01, 0.1, 10)
vn_SSF = np.linspace(1.05, 2, 10)
vn_WMF = np.linspace(1.05, 2, 10)

a0,n0 = np.meshgrid(alpha_SSF,vn_SSF)# ,alpha_WMF,vn_WMF)

ssf_param = {'alpha':a0.flatten(),'vn':n0.flatten()}
# wmf_param = {'alpha':a1.flatten(),'vn':n1.flatten()}

SSF.setMCparam(ssf_param)
# WMF.setMCparam(wmf_param)

h.setupMultiRun() 

#%% run sutra 
 # run multi threaded for resistivity part 
c0 = time.time() 
h.runMultiRun()
run_keys = h.getMultiRun()
hydro_run_time = time.time() - c0

#%% run R2 
# now setup R2 folders 
c0 = time.time() 
h.setupRruns(write2in,run_keys,survey_keys,sequences)

#now go through and run folders 
h.runResFwdmdls(run_keys)
data_store = h.getFwdRunResults(run_keys)
res_run_time = time.time() - c0 

#%% analyse plot up results 
#now go through and run folders 
likelihoods = [0]*len(data_store.keys())
for i,key in enumerate(data_store.keys()):
    fwd_data = data_store[key]
    residuals = np.abs(fwd_data['tr'].values - data_seq['tr'].values)
    likelihoods[i] = normLike(data_seq['error'].values, residuals)
    
alpha = []
vn = []
for run in run_keys:
    alpha.append(h.runparam[run]['alpha'][0]) 
    vn.append(h.runparam[run]['vn'][0])
    
# plot and save 
fig,ax = plt.subplots()
levels = np.linspace(0,max(likelihoods),100)
cax = ax.tricontourf(alpha,vn,likelihoods,levels=levels)
ax.set_xlabel('alpha (1/m)')
ax.set_ylabel('n')
cbar = plt.colorbar(cax) 
cbar.set_label('Normalised Likelihood')

best_fit = np.argmax(likelihoods)
ssf_alpha = alpha[best_fit]
ssf_vn = vn[best_fit]
ax.scatter([ssf_alpha],[ssf_vn])

fig.savefig(os.path.join(h.dname,'result.png'))
df = pd.DataFrame({'alpha':alpha,
                   'vn':vn,
                   'normLike':likelihoods})
df.to_csv(os.path.join(h.dname,'result.csv'),index=False)

#%% report some timings
print('Setup time : %f'%setup_time)
print('Hydrological model runtime: %f'%hydro_run_time)
print('Resistivity modelling runtime: %f'%res_run_time)