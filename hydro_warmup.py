#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 13:46:16 2022
Hydrological model of Hollin Hill warm up period. 5 years at constant 5mm 
rainfall. 
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
from petroFuncs import ssf_petro_sat, wmf_petro_sat, temp_uncorrect

# setup script variables and functions 
secinday = 24*60*60 
plt.close('all')

exec_loc = '/home/jimmy/programs/SUTRA_JB/bin/sutra'
if 'win' in sys.platform.lower():
    exec_loc = r'C:/Users/boydj1/Software/SUTRA/bin/sutra.exe'
    
    
model_dir = 'Models'
sim_dir = os.path.join(model_dir,'HydroWarmUp')
for d in [model_dir,sim_dir]:
    if not os.path.exists(d):
        os.mkdir(d)

# %% step 0, load in relevant files
rainfall = pd.read_csv(os.path.join('Data/Rainfall', 'COSMOS_2015-2016.csv'))
rainfall = pd.concat([rainfall[::2]]*2).reset_index()
# compute day in year 
sdiy = []
c = 0 
for i in range(len(rainfall)):
    c+=1 
    sdiy.append(c)
    if c>365: 
        c = 0 

# read in topo data
topo = pd.read_csv('Data/topoData/2016-01-08.csv')


rainfall.loc[0:int(len(rainfall)/2),'EFF_RAIN'] = 0
rainfall.loc[int(len(rainfall)/2):,'EFF_RAIN'] = 5

# LOAD IN EXTENT OF WMF/SSF (will be used to zone the mesh)
poly_ssf = np.genfromtxt('interpretation/SSF_poly_v5.csv',delimiter=',')
poly_ssf_ext = np.genfromtxt('interpretation/SSF_poly_ext.csv',delimiter=',')
poly_dogger = np.genfromtxt('interpretation/Dogger_poly.csv',delimiter=',')

maxx = np.max(topo['y'])
minx = np.min(topo['y'])

# %% step 1, setup/create the mesh 
# create quad mesh
moutput = mt.quadMesh(topo['y'], topo['z'], elemx=1, pad=1, fmd=10,zf=1.1,zgf=1.1)
mesh = moutput[0]  # ignore the other output from meshTools here
numel = mesh.numel  # number of elements
# say something about the number of elements
print('Number of mesh elements is %i' % numel)
meshx = mesh.df['X']  # mesh X locations
meshz = mesh.df['Z']  # mesh Z locations
path = mpath.Path(poly_ssf)  # get path of staithes extent 

# id points which are SSF 
inside = path.contains_points(np.c_[meshx, meshz])
# create a zone, 1 for staithes and 2 for whitby (which is taken to represent RMF too)
zone = np.ones(mesh.numel, dtype=int)+1 
zone[inside] = 1 # assign zone 1 to staithes 

# assign extended zone outside of main investigation to staithes too 
path = mpath.Path(poly_ssf_ext)
inside = path.contains_points(np.c_[meshx, meshz])
zone[inside] = 1 

# assign dogger formation, and anything right of x = 280 
path = mpath.Path(poly_dogger)
inside = path.contains_points(np.c_[meshx, meshz])
zone[inside] = 3 
inside = meshx > 280
zone[inside] = 3 

# assign RMF formation (anything underneath the stiathes) or left of x = -10 
inside = meshx < -10 
zone[inside] = 4 
for x in np.unique(meshx):
    idxx = meshx == x 
    zc = zone[idxx] # zone column 
    if 1 in zc and 2 in zc:
        cz = meshz[idxx] # column z coordinate, find minimum of zone 1 z coord 
        cz_min = np.min(cz[zc==1])
        for elm_id in mesh.df['elm_id'][idxx]:
            if zone[elm_id-1] == 2 and meshz[elm_id-1] < cz_min:
                zone[elm_id-1] = 4

# now to map zones to their respective identification numbers by element 
zone_id = np.arange(mesh.numel,dtype=int)

ssf_idx = zone_id[zone==1] #zone ssf 
wmf_idx = zone_id[zone==2] #zone wmf 
dog_idx = zone_id[zone==3] #zone dog 
rmf_idx = zone_id[zone==4] #zone rmf  

# get the surface of the mesh
xz = mesh.extractSurface(False, False)
cell_depths = mesh.computeElmDepth()

maxx = np.max(mesh.node[:,0]) # these max/min values will be used for ... 
minx = np.min(mesh.node[:,0]) # boundary conditions later on 

#%% step 2 create materials 
SSF = material(Ksat=0.144e0,theta_res=0.06,theta_sat=0.38,
               alpha=0.1317,vn=2.2,name='STAITHES')
WMF = material(Ksat=0.013,theta_res=0.1,theta_sat=0.48,
               alpha=0.012,vn=1.44,name='WHITBY')
DOG = material(Ksat=0.309,theta_res=0.008,theta_sat=0.215,
               alpha=0.05,vn=1.75,name='DOGGER')
RMF = material(Ksat=0.075e0,theta_res=0.1,theta_sat=0.48,
               alpha=0.0126,vn=1.44,name='REDCAR')

SSF.setPetro(ssf_petro_sat)
WMF.setPetro(wmf_petro_sat)
DOG.setPetro(ssf_petro_sat)
RMF.setPetro(wmf_petro_sat)

#%% step 3, setup handler 
## create handler
h = handler(dname=sim_dir, ifac=1,tlength=secinday,iobs=100, 
            flow = 'transient',
            transport = 'transient',
            sim_type='solute')
h.maxIter = 200
h.rpmax = 1e4  
h.drainage = 1e-2
h.clearDir()
h.setMesh(mesh)
h.setEXEC(exec_loc)

## compute cell depths 
depths, node_depths = h.getDepths()

#%% step 4 add materials to handler 
h.addMaterial(SSF,ssf_idx)
h.addMaterial(WMF,wmf_idx)
h.addMaterial(DOG,dog_idx)
h.addMaterial(RMF,rmf_idx)

#%% step 5, boundary conditions (note not all of these are active)
# find surface nodes
zone_node = h.mesh.ptdf['zone'].values 
tree = cKDTree(mesh.node[:, (0, 2)])
dist, source_node = tree.query(np.c_[xz[0], xz[1]])
source_node += 1

general_node = np.array([],dtype=int)
general_type = []
pres_node = []

# find nodes on right side of mesh 
b1902x = mesh.node[:,0][np.argmin(np.sqrt((mesh.node[:,0]-106.288)**2))]
right_side_idx = (mesh.node[:,0] == maxx) # & (zone_node == 2)
# right_side_idx = (mesh.node[:,0] == b1902x) 
right_side_node = mesh.node[right_side_idx]
right_side_topo = max(right_side_node[:,2])
right_side_wt = right_side_topo - 7.5
rs_delta = right_side_topo - right_side_wt 
right_side_node_sat = right_side_node[right_side_node[:,2]<(right_side_topo-rs_delta)]
dist, right_node = tree.query(right_side_node_sat[:,[0,2]])
# compute pressure on right hand side of mesh 
right_side_bl= right_side_wt - mesh.node[right_node][:,2]
right_pressure_val = max(9810*right_side_bl)

#find nodes on left side of mesh, set as drainage boundary 
left_side_idx = (mesh.node[:,0] == minx)
left_side_node = mesh.node[left_side_idx]
dist, left_node = tree.query(left_side_node[:,[0,2]])
general_node = np.append(general_node,left_node + 1) 
general_type = general_type + ['seep']*len(left_node)
pres_node = pres_node + [0]*len(left_node)

# hold pressure at borehole 1901 
b1901x = mesh.node[:,0][np.argmin(np.sqrt((mesh.node[:,0]-26.501)**2))]
b1901_idx = (mesh.node[:,0] == b1901x)
b1901_node = mesh.node[b1901_idx]
b1901_topo = max(b1901_node[:,2])
b1901_wt = b1901_topo - 5.7
b1901_delta = b1901_topo - b1901_wt 
b1901_node_sat = b1901_node[b1901_node[:,2]<(b1901_topo-b1901_delta)]
dist, b1901_node = tree.query(b1901_node_sat[:,[0,2]])
b1901_bl= b1901_wt - mesh.node[b1901_node][:,2]
b1901_pressure = max(9810*b1901_bl)

# find nodes at base of mesh 
max_depth = max(cell_depths)+1
dist, base_node = tree.query(np.c_[xz[0], xz[1]-max_depth])
# pres_node = np.append(pres_node,base_node+1)
general_node = np.append(general_node, base_node+1)
general_type = general_type + ['pres']*len(base_node)
p = np.polyfit([b1901x,b1902x],[b1901_pressure,right_pressure_val],1)
X = mesh.node[:,0][base_node]
base_pressures = np.polyval(p, X)
pres_node = pres_node + base_pressures.tolist()

# make top nodes a seepage boundary? 
# general_node = np.append(general_node,source_node) 
# general_type = general_type + ['seep']*len(source_node)

# find nodes at base of ssf 
ssf_x = np.unique(mesh.node[:,0][zone_node==1])
ssf_z = np.zeros_like(ssf_x)
for i,x in enumerate(ssf_x):
    idx = (mesh.node[:,0]==x) & (zone_node==1)
    z = mesh.node[:,2][idx]
    ssf_z[i] = np.min(z)
dist, ssf_base_node = tree.query(np.c_[ssf_x, ssf_z])
# pres_node = np.append(pres_node, ssf_base_node+1) 

# find mid points for computing cross sectional area for infiltration 
dx = np.zeros(len(xz[0]))
dx[0] = np.abs(xz[0][1] - xz[0][0])
for i in range(1,len(xz[0])):
    dx[i] = np.abs(xz[0][i] - xz[0][i-1])
    
# that should be the mesh setup now

# %% step 6, doctor rainfall and energy input for SUTRA 
# rainfall is given in mm/day, so convert to kg/s
rain = rainfall['EFF_RAIN'].values  # rain in mm/d
infil = (rain*1e-3)/secinday #in kg/s # check this #### 
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
wt_depth = 8 #min(node_depths[pres==0])
bl_depth = node_depths - wt_depth 
bl_idx = pres==0 
pres[bl_idx] = (9810*bl_depth[bl_idx])#*10e3  
pres[np.isnan(pres)] = 0 

# pressure values for pressure nodes 
pressure_vals = None 

temp = [0]*mesh.numnp 
pres[np.isnan(pres)] = 0 

#%% step 8, write inputs for SUTRA 
h.setupInp(times=times, 
           source_node=source_node, 
           #pressure_node=pres_node, pressure_val=pressure_vals, 
           general_node=general_node, general_type=general_type, 
           source_val=infil[0]*(dx/2))
h.pressure = pres_node # assigned to general pressure nodes 
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

h.callPetro()
h.callTempCorrect(temp_uncorrect, sdiy[::100]+[sdiy[-2],sdiy[-1]])
h.attribute = 'Resistivity'
h.vmax = 120
h.vmin = 5
h.vlim = [h.vmin, h.vmax]
h.plot1Dresults()
h.plotMeshResults()


# save mesh output 
mesh.flipYZ()
mesh.vtk(os.path.join(h.dname,'mesh.vtk'))

#%% get init conditions for real run 
nstep = h.resultNsteps
step = pd.DataFrame(h.nodResult['step%i'%(nstep-1)])
step['zone'] = zone_node 
step.to_csv(os.path.join(h.dname,'warm.csv'),index=False)
