#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 13:10:02 2023

@author: jimmy
"""
import os, sys, shutil, time 
from datetime import timedelta
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from joblib import Parallel, delayed
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
from SUTRAhandler import handler, material 
from petroFuncs import ssf_petro_sat, wmf_petro_sat
import createInput as ci 
from createInput import secinday 
from resipy import Project
from resipy.r2in import write2in # needed for forward modelling 

c0 = time.time() 
# create working environment 
exec_loc = '/home/jimmy/programs/SUTRA_JB/bin/sutra'
if 'win' in sys.platform.lower():
    # exec_loc = r'C:/Users/boydj1/Software/SUTRA/bin/sutra.exe'
    exec_loc = r'C:/Users/jimmy/Documents/Programs/SUTRA/bin/sutra.exe'
    
nchain = 12
ncpu = 16
nstep = 1001 
    
synth_dir = 'SyntheticStudy'
model_dir = os.path.join(synth_dir,'Models')
sim_dir = os.path.join(model_dir,'MCMC')
for d in [model_dir,sim_dir]:
    if not os.path.exists(d):
        os.mkdir(d)

#%% load in the data 
elec = ci.Sy_getElec()
hydro_data, data_seq, sequences, survey_keys, rfiles, sdiy = ci.Sy_data(ncpu)
TimeStamp = np.arange(len(hydro_data))
times = np.asarray(TimeStamp, dtype=int)

#%% create mesh with some pressure conditions 
mesh, zone_flags, dx, pressures, boundaries = ci.Sy_mesh(False)

# create boundary conditions 
general_node = []
general_type = []
general_pressure  = []

# set left side as seepage boundary 
general_node += boundaries['left'].tolist()
general_type += ['seep']*len(boundaries['left'])
general_pressure += [0.0]*len(boundaries['left'])

# set top side as seepage boundary 
general_node += boundaries['top'].tolist()
general_type += ['seep']*len(boundaries['top'])
general_pressure += [0.0]*len(boundaries['top'])

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
kc=hydro_data['Kc'].values 
tdx = sum(dx)
fluidinp, tempinp = ci.prepRainfall(dx,precip,pet,kc, len(source_node),ntimes)

#%% create materials 
SSF = material(Ksat=0.14,theta_res=0.06,theta_sat=0.38,
               alpha=0.1317,vn=2.2,name='STAITHES')
WMF = material(Ksat=0.013,theta_res=0.1,theta_sat=0.48,
               alpha=0.012,vn=1.44,name='WHITBY')

SSF.setPetroFuncs(ssf_petro_sat, ssf_petro_sat)
WMF.setPetroFuncs(wmf_petro_sat, wmf_petro_sat)

# want to examine VG parameters for SSF and WMF 
alpha_SSF = [0.001, 0.02, 2.0] # LOWER LIMIT, STEP SIZE, UPPER LIMIT  
alpha_WMF = [0.001, 0.02, 2.0] 
vn_SSF = [1.1, 0.02, 2.5]
vn_WMF = [1.1, 0.02, 2.5]

ssf_param = {'alpha':alpha_SSF,'vn':vn_SSF}
wmf_param = {'alpha':alpha_WMF,'vn':vn_WMF}

SSF.setMCparam(ssf_param)
WMF.setMCparam(wmf_param)

#%% run mcmc 
def run(i):
    chain_dir = os.path.join(sim_dir,'chain%i'%(i+1))
    #create handler 
    h = handler(dname=chain_dir, ifac=1,tlength=secinday,iobs=1, 
                flow = 'transient',
                transport = 'transient',
                sim_type='solute')
    
    h.maxIter = 300
    h.rpmax = 5e5  
    h.drainage = 1e-8
    h.clearDir()
    h.setMesh(mesh)
    h.setEXEC(exec_loc)
    h.cpu = 1 # number of processors to use 
    
    h.addMaterial(SSF,zone_flags['SSF'])
    h.addMaterial(WMF,zone_flags['WMF'])
    
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
    
    h.showSetup(True) 
    
    # setup R2 project for res modelling 
    k = Project(dirname=chain_dir)
    k.setElec(elec)
    k.createMesh(cl_factor=4)
    
    h.setRproject(k)
    h.setupRparam(data_seq, write2in, survey_keys, seqs=sequences)
    depths, node_depths = h.getDepths()
    
    # run single mcmc single chain 
    chainlog, ar = h.mcmc(nstep,0.234)
    df = pd.DataFrame(chainlog)
    df.to_csv(os.path.join(h.dname,'chainlog.csv'),index=False)


#%% run multiple mcmc chains in parallel 
print('Running... %i chains for %i iterations'%(nchain,nstep))
Parallel(n_jobs=ncpu)(delayed(run)(i) for i in range(nchain))
print('Done')
