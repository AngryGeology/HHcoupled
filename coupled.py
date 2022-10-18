#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 13:46:16 2022
Basic hydrological model of hollin hill 
@author: jimmy
"""
import os, sys 
from datetime import datetime 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from scipy.spatial import cKDTree
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
if 'RSUTRA' not in sys.path: 
    sys.path.append('RSUTRA')
if '/home/jimmy/phd/resipy/src' not in sys.path: 
    sys.path.append('/home/jimmy/phd/resipy/src')
from resipy import Survey, Project 
from resipy.r2in import write2in # needed for forward modelling 
from resipy import meshTools as mt
from SUTRAhandler import handler, material, secinday, normLike
# from SUTRAhandler import invVGcurve
from petroFuncs import ssf_petro, wmf_petro, wmf_petro_sat, ssf_polyn

# %% step 0, load in relevant files
rainfall = pd.read_csv(os.path.join('Rainfall', 'COSMOS_2015-2016.csv'))

# read in topo data
topo = pd.read_csv('/home/jimmy/phd/Hollin_Hill/Coupled/topoData/2016-01-08.csv')
elec = pd.read_csv('/home/jimmy/phd/Hollin_Hill/Coupled/elecData/2016-01-08.csv')

# read in warm up
warmup = pd.read_csv('HydroWarmUp/warm.csv')

# LOAD IN EXTENT OF WMF/SSF (will be used to zone the mesh)
poly_ssf = np.genfromtxt('interpretation/SSF_poly_v3.csv',delimiter=',')
poly_ssf_ext = np.genfromtxt('interpretation/SSF_poly_ext.csv',delimiter=',')
poly_dogger = np.genfromtxt('interpretation/Dogger_poly.csv',delimiter=',')

# add label to elec 
elec['label'] = [str(elec['id'][i]) for i in range(len(elec))]
elecx = elec['y'].values 
elec.loc[:,'y'] = 0 
elec.loc[:,'x'] = elecx 

#%% step 1 is to match up resistivity survey dates with when we have rainfall 

maxx = np.max(topo['y'])
minx = np.min(topo['y'])

# get resistivity files and translate file names into dates 
def str2dat(fname):
    if '.dat' in fname: 
        string = fname.replace('.dat','')
    else:
        string = fname 
    dt = datetime.strptime(string, "%Y-%m-%d")
    return dt 
    
rfiles = []
sdates = [] # survey dates 
for f in sorted(os.listdir('resData')):
    if f.endswith('.dat'):
        rfiles.append(f)
        sdates.append(str2dat(f))
        
# find times when resistivity and rainfall data are there
nobs = len(rainfall)
rdates = [str2dat(rainfall['DATE_TIME'][i]) for i in range(nobs)]
sflag = [False]*len(rainfall) # flag to determine if there is a resistivity survey 
suidx = [-1]*nobs # corresponding survey index 
for i in range(nobs):
    date = rdates[i]
    delta = [abs((date - sdate).days) for sdate in sdates]
    if min(delta) == 0:
        idx = np.argmin(delta)
        suidx[i] = idx 
        sflag[i] = True 
        
        
# %% step 2, create a sequence of data and estimated errors 
fig,ax = plt.subplots()
data_seq = pd.DataFrame()
sequences = [] 
for i,f in enumerate(rfiles): 
    s = Survey(os.path.join('resData',f),ftype='ProtocolDC') # parse survey 
    s.fitErrorPwl(ax) # fit reciprocal error levels 
    # extract the abmn, data, error information and date 
    df = s.df[['a','b','m','n','recipMean','resError']]
    df = df.rename(columns={'recipMean':'tr',
                            'resError':'error'})
    df['sidx'] = i 
    sequences.append(df[['a','b','m','n']].values)
    data_seq = pd.concat([data_seq,df]) # append frame to data sequence 
    ax.cla() 
    
plt.close(fig)
# we will come back to these data later     

# %% step 3, setup/create the mesh for the hydrological model 
# create quad mesh
moutput = mt.quadMesh(elec['x'], elec['z'], elemx=1, pad=0, fmd=10,zf=1.2,zgf=1.1)
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

#%% step 4 create materials 
SSF = material(Ksat=0.11,theta_res=0.06,theta_sat=0.38,
               alpha=0.1317,vn=2.5,name='STAITHES')
WMF = material(Ksat=0.013,theta_res=0.1,theta_sat=0.48,
               alpha=0.0126,vn=1.44,name='WHITBY')
DOG = material(Ksat=0.309,theta_res=0.008,theta_sat=0.215,
               alpha=0.05,vn=1.75,name='DOGGER')
RMF = material(Ksat=0.013,theta_res=0.1,theta_sat=0.48,
               alpha=0.0126,vn=1.44,name='REDCAR')

SSF.setPetro(ssf_polyn)
WMF.setPetro(wmf_petro_sat)
DOG.setPetro(ssf_petro)
RMF.setPetro(wmf_petro)

#%% step 5, setup handler 
## create handler
h = handler(dname='HydroGeophys', ifac=1,tlength=secinday,iobs=1, 
            flow = 'transient',
            transport = 'transient',
            sim_type='solute')
h.maxIter = 300
h.rpmax = 5e3  
h.drainage = 1e-2
h.clearDir()
h.setMesh(mesh)
h.setEXEC('/home/jimmy/programs/SUTRA_JB/bin/sutra')

## compute cell depths 
depths, node_depths = h.getDepths()

#%% step 6 add materials to handler 
h.addMaterial(SSF,ssf_idx)
h.addMaterial(WMF,wmf_idx)
h.addMaterial(DOG,dog_idx)
h.addMaterial(RMF,rmf_idx)

#%% step 7, boundary conditions (note not all of these are active)
# find surface nodes
zone_node = h.mesh.ptdf['zone'].values 
tree = cKDTree(mesh.node[:, (0, 2)])
dist, source_node = tree.query(np.c_[xz[0], xz[1]])
source_node += 1

general_node = np.array([],dtype=int)
general_type = []
pres_node = np.array([],dtype=int) 

# find nodes on right side of mesh 
b1902x = mesh.node[:,0][np.argmin(np.sqrt((mesh.node[:,0]-106.288)**2))]
right_side_idx = (mesh.node[:,0] == maxx) # & (zone_node == 2)
# right_side_idx = (mesh.node[:,0] == b1902x) & (zone_node == 2)
right_side_node = mesh.node[right_side_idx]
right_side_topo = max(right_side_node[:,2])
right_side_wt = 82.4 
rs_delta = right_side_topo - right_side_wt 
right_side_node_sat = right_side_node[right_side_node[:,2]<(right_side_topo-rs_delta)]
dist, right_node = tree.query(right_side_node_sat[:,[0,2]])
# pres_node = np.append(pres_node, right_node + 1) 

cutoff = 50
#find nodes on left side of mesh, set as drainage boundary below 55 od 
left_side_idx = (mesh.node[:,0] == minx) & (mesh.node[:,2] < cutoff)
left_side_node = mesh.node[left_side_idx]
dist, left_node = tree.query(left_side_node[:,[0,2]])
general_node = np.append(general_node,left_node + 1) 
general_type = general_type + ['seep']*len(left_node)

#find nodes on left side of mesh, set as seepage boundary above 55 od 
left_side_idx = (mesh.node[:,0] == minx) & (mesh.node[:,2] > cutoff)
left_side_node = mesh.node[left_side_idx]
dist, left_node = tree.query(left_side_node[:,[0,2]])
general_node = np.append(general_node,left_node + 1) 
general_type = general_type + ['seep']*len(left_node)

# hold pressure at borehole 1901 
b1901_wt = 60.0 
b1901x = mesh.node[:,0][np.argmin(np.sqrt((mesh.node[:,0]-26.501)**2))]
b1901_idx = (mesh.node[:,0] == b1901x) & (zone_node==1)
b1901_node = mesh.node[b1901_idx]
b1901_topo = max(b1901_node[:,2])
b1901_delta = b1901_topo - b1901_wt 
b1901_node_sat = b1901_node[b1901_node[:,2]<(b1901_topo-b1901_delta)]
dist, b1901_node = tree.query(b1901_node_sat[:,[0,2]])
# pres_node = np.append(pres_node, b1901_node+1)

# find nodes at base of mesh to hold a minimum pressure 
max_depth = max(cell_depths)+1
dist, base_node = tree.query(np.c_[xz[0], xz[1]-max_depth])
general_node = np.append(general_node, base_node+1)
general_type = general_type + ['pres']*len(base_node)

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

# %% step 8, doctor rainfall and energy input for SUTRA 
# rainfall is given in mm/day, so convert to kg/s
rain = rainfall['EFF_RAIN'].values # rain in mm/d
infil = (rain/1000)/secinday #in kg/s # check this #### 

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

#%% step 9, setup initial conditions (read from warmup run)
ppres = warmup['Pressure'].values 
ppoints = np.c_[warmup['X'].values, warmup['Y'].values]
ipoints = np.c_[h.mesh.node[:,0], h.mesh.node[:,2]]
ifunc = LinearNDInterpolator(ppoints, ppres)
nfunc = NearestNDInterpolator(ppoints, ppres)
pres = ifunc(ipoints)
nanidx = np.isnan(pres)
pres[nanidx] = nfunc(ipoints[nanidx])
pressure_vals = None 

# compute pressure on right hand side of mesh 
right_side_bl= right_side_wt - mesh.node[right_node][:,2]
right_pressure_val = (9810*right_side_bl)
h.pressure = max(right_pressure_val)
# h.cold = False 

#%% step 10, write inputs for SUTRA and run 
h.setupInp(times=times, 
           source_node=source_node, 
           pressure_node=pres_node, pressure_val=pressure_vals, 
           general_node=general_node, general_type=general_type, 
           source_val=infil[0]*(dx/2))
h.writeInp() # write input without water table at base of column
h.writeBcs(times, source_node, fluidinp, tempinp)
temp = [0]*mesh.numnp 
h.writeIcs(pres, temp) # INITIAL CONDITIONS 
h.writeVg()
h.writeFil(ignore=['BCOP', 'BCOPG'])

h.showSetup() 

# run sutra 
h.runSUTRA()  # run
h.getResults()  # get results

#%% create MC runs 
# want to examine VG parameters for SSF and WMF 
alpha_SSF = np.linspace(0.5e-4, 4.0e-4,15)
alpha_WMF = np.linspace(0.1e-4, 1.0e-4,10)
vn_SSF = np.linspace(1.2, 2,10)
vn_WMF = np.linspace(1.2, 2,5)

a0,n0 = np.meshgrid(alpha_SSF,vn_SSF)# ,alpha_WMF,vn_WMF)

ssf_param = {'alpha':a0.flatten(),'vn':n0.flatten()}
# wmf_param = {'alpha':a1.flatten(),'vn':n1.flatten()}

SSF.setMCparam(ssf_param)
# WMF.setMCparam(wmf_param)

h.setupMultiRun() 

#%% run MC runs 
h.cpu = 20 
h.runMultiRun()
run_keys = h.getMultiRun()


#%% setup ResIPy project and resistivity runs 
k = Project(dirname='HydroGeophys')
k.setElec(elec)
surface = np.c_[topo['y'],topo['z']]
k.createMesh(cl_factor=4, surface=surface)

h.setRproject(k)
survey_keys = np.arange(h.resultNsteps)[np.array(sflag)==True]+1

# we need to add one becuase the steps start from a initialising run where t=0 
h.setupRruns(write2in,run_keys,survey_keys,sequences)
h.runResFwdmdls(run_keys)
data_store = h.getFwdRunResults(run_keys)

likelihoods = [0]*len(data_store.keys())
for i,key in enumerate(data_store.keys()):
    fwd_data = data_store[key]
    residuals = np.abs(fwd_data['tr'].values - data_seq['tr'].values)
    likelihoods[i] = normLike(data_seq['error'].values, residuals)
    
alpha = []
vn = []
for run in run_keys:
    alpha.append(h.runparam[run]['alpha'][0]*1000) # * by 10^3 to get back into 1/m
    vn.append(h.runparam[run]['vn'][0])
    
#%% plot and save 
fig,ax = plt.subplots()
cax = ax.tricontourf(alpha,vn,likelihoods)
ax.set_xlabel('alpha (1/m)')
ax.set_ylabel('n')
cbar = plt.colorbar(cax) 
cbar.set_label('Normalised Likelihood')
fig.savefig(os.path.join(h.dname,'result.png'))
df = pd.DataFrame({'alpha':alpha,
                   'vn':vn,
                   'normLike':likelihoods})
df.to_csv(os.path.join(h.dname,'result.csv'),index=False)

#%% create the best model 
best_fit = np.argmax(likelihoods)
ssf_alpha = alpha[best_fit]
ssf_vn = vn[best_fit]

print(ssf_alpha,ssf_vn)

#%% clear runs 
h.clearMultiRun()
    