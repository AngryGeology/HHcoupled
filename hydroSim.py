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
from SUTRAhandler import handler, material 
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
sim_dir = os.path.join(model_dir,'HydroSim')
datadir = os.path.join(sim_dir,'SimData')
for d in [model_dir,sim_dir,datadir]:
    if not os.path.exists(d):
        os.mkdir(d)


model_res = False 

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

#%% run sutra 
h.runSUTRA()  # run
h.getResults()  # get results
hydro_run_time = time.time() - c0

#%% step10, plot results (if you want to)
## plot up results (on mesh only)
# get max/min temperature
data = h.nodResult
n = h.resultNsteps
tmin = np.inf
tmax = -np.inf
pmin = np.inf
pmax = -np.inf
for i in range(n):
    P = data['step%i' % i]['Pressure']
    if min(P) < pmin:
        pmin = min(P)
    if max(P) > pmax:
        pmax = max(P)
  
# plots 
h.attribute = 'Saturation'
h.vmax = 1.01
h.vmin = 0.2
h.vlim = [0.0, 1.01]
h.plot1Dresults(iobs=10)
sw_ssf = h.get1Dvalues(58.8, 74.8)[1:]
sw_wmf = h.get1Dvalues(129, 90)[1:]
axt = ax.twinx()
axt.plot(hydro_data['datetime'],sw_ssf,c='y',label='ssf')
axt.plot(hydro_data['datetime'],sw_wmf,c='m',label='wmf')
axt.set_ylabel('Sw (-)')

h.plotMeshResults(cmap='RdBu',iobs=10)

# h.attribute = 'Pressure'
# h.vmax = pmax
# h.vmin = pmin
# h.vlim = [pmin, pmax]
# h.plot1Dresults()

#%% run resistivity runs
res_run_time = 0 
if model_res:  
    # setup R2 project 
    k = Project(dirname=sim_dir)
    k.setElec(elec)
    k.createMesh(cl_factor=4)
    
    h.setRproject(k)
    h.setupRparam(data_seq, write2in, survey_keys, seqs=sequences)
    depths, node_depths = h.getDepths()
    
    ssf_param = {'alpha':[SSF.alpha],'vn':[SSF.vn]}
    wmf_param = {'alpha':[WMF.alpha],'vn':[WMF.vn]}
    
    # setup MC param 
    SSF.setMCparam(ssf_param)
    WMF.setMCparam(wmf_param)
    SSF.setPetro(ssf_petro_sat)
    WMF.setPetro(wmf_petro_sat)
    
    h.setupMultiRun() 
    h.cpu = 12 # run multi threaded for resistivity part 
    h.runMultiRun()
    run_keys = h.getMultiRun()
    
    # now setup R2 folders 
    c0 = time.time() 
    h.setupRruns(write2in,run_keys,survey_keys,sequences)
    
    #now go through and run folders 
    h.runResFwdmdls(run_keys)
    data_store = h.getFwdRunResults(run_keys)
    res_run_time = time.time() - c0 
    
    ## copy across the data files to a dedicated directory 
    fwddir = os.path.join(sim_dir,'r00000') 
    for f in sorted(os.listdir(fwddir)):
        if not 'forward' in f:
            continue 
        if 'R2' in f:
            continue 
        if '_model' in f: 
            continue 
        # work out index 
        idx = f.replace('forward','').replace('.dat','')
        i = int(idx)
        shutil.copy(os.path.join(fwddir,f),
                    os.path.join(datadir,rfiles[i]))

#%% report some timings
print('Setup time : %f'%setup_time)
print('Hydrological model runtime: %f'%hydro_run_time)
print('Resistivity modelling runtime: %f'%res_run_time)