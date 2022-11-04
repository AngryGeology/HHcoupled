#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 13:46:16 2022
Synthetic warm up hydrological model of a synthetic hill slope with similar 
attributes to hollin hill. 
@author: jimmy
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from scipy.spatial import cKDTree

# add custom modules 
if 'RSUTRA' not in sys.path: 
    sys.path.append('RSUTRA')
linux_r_path = '/home/jimmy/phd/resipy/src'
win_r_path = r'C:\Users\boydj1\Software\resipy\src' 
if sys.platform == 'linux' and linux_r_path not in sys.path: 
    sys.path.append(linux_r_path)
if 'win' in sys.platform.lower() and win_r_path not in sys.path:
    sys.path.append(win_r_path)
from resipy import meshTools as mt
from SUTRAhandler import handler, material 

# setup script variables and functions 
secinday = 24*60*60 
plt.close('all')

exec_loc = '/home/jimmy/programs/SUTRA_JB/bin/sutra'
if 'win' in sys.platform.lower():
    exec_loc = r'C:/Users/boydj1/Software/SUTRA/bin/sutra.exe'
    
    
#create directory structure 
masterdir='SyntheticStudy'
modeldir= os.path.join(masterdir,'Models')
datadir=os.path.join(masterdir,'Data')
warmupdir = os.path.join(modeldir,'WarmUp')
for dname in [masterdir,modeldir,datadir,warmupdir]:
    if not os.path.exists(dname):
        os.mkdir(dname)

# %% step 0, load in relevant files
rainfall = pd.read_csv(os.path.join('Data/Rainfall', 'COSMOS_2015-2016.csv'))
rainfall = pd.concat([rainfall[::2]]*5).reset_index()
rainfall.loc[:,'EFF_RAIN'] = 5 # assume 5 mm rainfall for warmup period 
# read in topo data
topo = pd.read_csv('Data/topoData/2016-01-08.csv')

# fit a generic polyline to get a similar slope profile of the actual hill 
tx = topo['y'].values # actually grid is located against the y axis, make this the local x coordinate in 2d 
p = np.polyfit(tx,topo['z'].values,1) # fit elevation to along ground distance data 
tz = np.polyval(p,tx) # fit the modelled topography 


# %% step 1, setup/create the mesh 
# create quad mesh
moutput = mt.quadMesh(tx, tz, elemx=1, pad=0, fmd=10,zf=1.1,zgf=1.1)
mesh = moutput[0]  # ignore the other output from meshTools here
numel = mesh.numel  # number of elements
# say something about the number of elements
print('Number of mesh elements is %i' % numel)
meshx = mesh.df['X']  # mesh X locations
meshz = mesh.df['Z']  # mesh Z locations
zone = np.ones(mesh.numel, dtype=int)+1 

# make anything above the median elevation be our WMF analogy, anything below SSF 
medianz = 75# np.median(tz)
inside = meshz < medianz

zone[inside] = 1 # assign zone 1 to staithes 
zone[np.invert(inside)] = 2 # assign zone 2 to WMF 

# now to map zones to their respective identification numbers by element 
zone_id = np.arange(mesh.numel,dtype=int)

ssf_idx = zone_id[zone==1] #zone ssf 
wmf_idx = zone_id[zone==2] #zone wmf 

# get the surface of the mesh
xz = mesh.extractSurface(False, False)
cell_depths = mesh.computeElmDepth()

maxx = np.max(mesh.node[:,0]) # these max/min values will be used for ... 
minx = np.min(mesh.node[:,0]) # boundary conditions later on 

mesh.dat(os.path.join(modeldir,'mesh.dat'))

#%% step 2 create materials 
# STAITHES SANDSTONE 
SSF = material(Ksat=0.11,theta_res=0.06,theta_sat=0.38,
               alpha=0.14,vn=2.22 ,name='STAITHES')
# WHITBY MUDSTONE 
WMF = material(Ksat=0.013,theta_res=0.1,theta_sat=0.48,
               alpha=0.01,vn=1.09,name='WHITBY')

#%% step 3, setup handler 
## create handler
h = handler(dname=warmupdir, ifac=1,tlength=secinday,iobs=100, 
            flow = 'transient',
            transport = 'transient',
            sim_type='solute')
h.maxIter = 100
h.rpmax = 10e3  
h.drainage = 1e-8
h.clearDir()
h.setMesh(mesh)
h.setEXEC(exec_loc)

## compute cell depths 
depths, node_depths = h.getDepths()

#%% step 4 add materials to handler 
h.addMaterial(SSF,ssf_idx)
h.addMaterial(WMF,wmf_idx)

#%% step 5, boundary conditions (note not all of these are active)
# find surface nodes
zone_node = h.mesh.ptdf['zone'].values 
tree = cKDTree(mesh.node[:, (0, 2)])
dist, source_node = tree.query(np.c_[xz[0], xz[1]])
source_node += 1

general_node = np.array([],dtype=int)
general_type = []
pres_node = np.array([],dtype=int) 

#find nodes on left side of mesh, set as drainage boundary 
left_side_idx = (mesh.node[:,0] == minx) & (mesh.node[:,2] > 45)
left_side_node = mesh.node[left_side_idx]
dist, left_node = tree.query(left_side_node[:,[0,2]])
general_node = np.append(general_node,left_node + 1) 
general_type = general_type + ['seep']*len(left_node)

# find nodes at base of mesh 
max_depth = max(cell_depths)+1
dist, base_node = tree.query(np.c_[xz[0], xz[1]-max_depth])
# pres_node = np.append(pres_node,base_node+1)
general_node = np.append(general_node, base_node+1)
general_type = general_type + ['pres']*len(base_node)

# find mid points for computing cross sectional area for infiltration 
dx = np.zeros(len(xz[0]))
dx[0] = np.abs(xz[0][1] - xz[0][0])
for i in range(1,len(xz[0])):
    dx[i] = np.abs(xz[0][i] - xz[0][i-1])
    
# that should be the mesh setup now

# %% step 6, doctor rainfall and energy input for SUTRA 
# rainfall is given in mm/day, so convert to kg/s
rain = rainfall['EFF_RAIN'].values  # rain in mm/d
infil = (rain/1000)/secinday #in kg/s # check this #### 
# infil = np.full(len(rain),(1000*0.01)/secinday) 
# infil[infil<0] = 0 # cap minimum at zero? 
infil[np.isnan(infil)] = 0 # put NAN at zero 

ntimes = len(rainfall)
TimeStamp = np.arange(ntimes)
times = np.asarray(TimeStamp, dtype=int)
Temps = np.zeros_like(TimeStamp) + 0
Temp_surface = np.zeros_like(TimeStamp) + 0

# now populate matrices
fluidinp = np.zeros((len(TimeStamp), len(source_node)))  # fluid input
tempinp = np.zeros((len(TimeStamp), len(source_node)))  # fluid temperature
surftempinp = np.zeros((len(TimeStamp), len(source_node)))  # surface temperature
for i in range(len(source_node)):
    m = dx[i]/2
    fluidinp[:, i] = infil*m
    tempinp[:, i] = Temps
    surftempinp[:, i] = Temp_surface
    
## plot infiltration 
fig, ax = plt.subplots()
ax.plot(TimeStamp,infil) 

#%% step 7, setup initial conditions 
pres = np.zeros(mesh.numnp)

# compute pressure below water table 
wt_depth = 5 #min(node_depths[pres==0])
bl_depth = node_depths - wt_depth 
bl_idx = pres==0 
pres[bl_idx] = (9810*bl_depth[bl_idx])#*10e3  

h.pressure = max(pres)

pressure_vals = [] # empty array 

temp = [0]*mesh.numnp 
# pres[np.isnan(pres)] = 0 

#%% step 8, write inputs for SUTRA 
h.setupInp(times=times, 
           source_node=source_node, 
           pressure_node=pres_node, pressure_val=pressure_vals, 
           general_node=general_node, general_type=general_type, 
           source_val=infil[0]*(dx/2))
h.writeInp() # write input without water table at base of column
h.writeBcs(times, source_node, fluidinp, tempinp)
h.writeIcs(pres, temp) # INITIAL CONDITIONS 
h.writeVg()
h.writeFil(ignore=['BCOP', 'BCOPG'])

h.showSetup() 

#%% step9, run sutra 
h.runSUTRA()  # run
h.getResults()  # get results

#%% step10, plot results 
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
h.plot1Dresults()
h.plotMeshResults(cmap='RdBu')

h.attribute = 'Pressure'
h.vmax = pmax
h.vmin = pmin
h.vlim = [pmin, pmax]
h.plot1Dresults()
h.plotMeshResults()

# save mesh output 
# mesh.flipYZ()
mesh.vtk(os.path.join(modeldir,'mesh.vtk'))

#%% get init conditions for real run 
nstep = h.resultNsteps
step = pd.DataFrame(h.nodResult['step%i'%(nstep-1)])
step['zone'] = zone_node 
step.to_csv(os.path.join(h.dname,'warm.csv'),index=False)
np.savetxt(os.path.join(modeldir,'p.txt'),p)
