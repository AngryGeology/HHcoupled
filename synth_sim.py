#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 13:46:16 2022
Synthetic data simulation for synthetic hill slope study 
@author: jimmy
"""
import os, sys, shutil 
from datetime import datetime 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import matplotlib.path as mpath
from scipy.spatial import cKDTree
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator

# add custom modules 
if 'RSUTRA' not in sys.path: 
    sys.path.append('RSUTRA')
linux_r_path = '/home/jimmy/phd/resipy/src'
win_r_path = r'C:\Users\boydj1\Software\resipy\src' 
if sys.platform == 'linux' and linux_r_path not in sys.path: 
    sys.path.append(linux_r_path)
if 'win' in sys.platform.lower() and win_r_path not in sys.path:
    sys.path.append(win_r_path)
from resipy import Survey, Project 
from resipy import meshTools as mt
from resipy.r2in import write2in # needed for forward modelling 
from SUTRAhandler import handler, material 
from petroFuncs import ssf_petro_sat, wmf_petro_sat 

# setup script variables and functions 
secinday = 24*60*60 
plt.close('all')

exec_loc = '/home/jimmy/programs/SUTRA_JB/bin/sutra_s'
if 'win' in sys.platform.lower():
    exec_loc = r'C:/Users/boydj1/Software/SUTRA/bin/sutra.exe'
    
    
#create directory structure 
masterdir='SyntheticStudy'
modeldir= os.path.join(masterdir,'Models')
datadir=os.path.join(masterdir,'Data')
simdir = os.path.join(modeldir,'Synth')
# directories should have been made already 
for dname in [masterdir,modeldir,datadir,simdir]:
    if not os.path.exists(dname):
        os.mkdir(dname)

# %% step 0, load in relevant files
rainfall = pd.read_csv(os.path.join('Data/Rainfall', 'COSMOS_2015-2016.csv'))
mesh = mt.readMesh(os.path.join(modeldir,'mesh.vtk')) # read in quad mesh
warmup = pd.read_csv(os.path.join(modeldir,'WarmUp','warm.csv'))
elec = pd.read_csv('Data/elecData/2016-01-08.csv')
p = np.genfromtxt(os.path.join(modeldir,'p.txt')) # matrix describing slope angle 

# add label to elec 
elec['label'] = [str(elec['id'][i]) for i in range(len(elec))]
elecx = elec['y'].values 
elec.loc[:,'y'] = 0 
elec.loc[:,'x'] = elecx 
elec.loc[:,'z'] = np.polyval(p,elecx)
elec.to_csv(os.path.join(modeldir,'electrodes.csv'))

#%% step 0.5 ,handle resistivity data 
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
for f in sorted(os.listdir('Data/resData')):
    if f.endswith('.dat'):
        rfiles.append(f)
        sdates.append(str2dat(f))
        
# find times when resistivity and rainfall data are there
nobs = len(rainfall)
rdates = [str2dat(rainfall['DATE_TIME'][i]) for i in range(nobs)]
sflag = [False]*nobs # flag to determine if there is a resistivity survey 
suidx = [-1]*nobs # corresponding survey index 
for i in range(nobs):
    date = rdates[i]
    delta = [abs((date - sdate).days) for sdate in sdates]
    if min(delta) == 0:
        idx = np.argmin(delta)
        suidx[i] = idx 
        sflag[i] = True 

# sflag = [False] + sflag # add one here because first returned survey is time 0 
survey_keys = np.arange(nobs)[np.array(sflag)==True]
        
# create a sequence of data and estimated errors 
fig,ax = plt.subplots()
data_seq = pd.DataFrame()
sequences = [] 
for i,f in enumerate(rfiles): 
    s = Survey(os.path.join('Data/resData',f),ftype='ProtocolDC') # parse survey 
    s.fitErrorPwl(ax) # fit reciprocal error levels 
    # extract the abmn, data, error information and date 
    df = s.df[['a','b','m','n','recipMean','resError']]
    keepidx = s.df['irecip'] > 0 # filter out reciprocals for synthetic study 
    df = df.rename(columns={'recipMean':'tr',
                            'resError':'error'})[keepidx].reset_index()
    df['sidx'] = i 
    sequences.append(df[['a','b','m','n']].values)
    data_seq = pd.concat([data_seq,df]) # append frame to data sequence 
    ax.cla() 
    
plt.close(fig)

# %% step 1, setup/create the mesh 

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

#%% step 2 create materials 
# STAITHES SANDSTONE 
SSF = material(Ksat=0.11,theta_res=0.06,theta_sat=0.38,
               alpha=0.14,vn=2.2 ,name='STAITHES')
# WHITBY MUDSTONE 
WMF = material(Ksat=0.013,theta_res=0.1,theta_sat=0.48,
               alpha=0.05,vn=1.4,name='WHITBY')

#%% step 3, setup handler 
## create handler
h = handler(dname=simdir, ifac=1,tlength=secinday,iobs=1, 
            flow = 'transient',
            transport = 'transient',
            sim_type='solute')
h.maxIter = 500
h.rpmax = 5e3  
h.drainage = 5e-2
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
# compute maximum pressure below water table 
pres = np.zeros(mesh.numnp)
wt_depth = 5 #min(node_depths[pres==0])
bl_depth = node_depths - wt_depth 
bl_idx = pres==0 
pres[bl_idx] = (9810*bl_depth[bl_idx])#*10e3  
h.pressure = max(pres)

# getwarmup values for start of model 
ppres = warmup['Pressure'].values 
ppoints = np.c_[warmup['X'].values, warmup['Y'].values]
ipoints = np.c_[h.mesh.node[:,0], h.mesh.node[:,2]]
ifunc = LinearNDInterpolator(ppoints, ppres)
nfunc = NearestNDInterpolator(ppoints, ppres)
pres = ifunc(ipoints)
nanidx = np.isnan(pres)
pres[nanidx] = nfunc(ipoints[nanidx])


pressure_vals = [] # empty array 
temp = [0]*mesh.numnp 

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
# h.runSUTRA()  # run()
# want to examine VG parameters for SSF and WMF,
# going to recycle the multi / parallel running code here 
ssf_param = {'alpha':[SSF.alpha],'vn':[SSF.vn]}
wmf_param = {'alpha':[WMF.alpha],'vn':[WMF.vn]}

# setup materials for resitivity-gmc relationships 
SSF.setMCparam(ssf_param)
WMF.setMCparam(wmf_param)
SSF.setPetro(ssf_petro_sat)
WMF.setPetro(wmf_petro_sat)

h.setupMultiRun() 
h.cpu = 1
h.runMultiRun()
run_keys = h.getMultiRun()

#%% step10, simulate resistivities 
# setup ResIPy project and resistivity runs 
k = Project(dirname=simdir)
k.setElec(elec)
k.createMesh(cl_factor=4)

h.setRproject(k)
# we need to add one becuase the steps start from a initialising run where t=0 
survey_keys = np.arange(h.resultNsteps)[np.array(sflag)==True]+1

# now setup R2 folders 
h.setupRruns(write2in,run_keys,survey_keys,sequences,ncpu=1)

#now go through and run folders 
h.runResFwdmdls(run_keys)
data_store = h.getFwdRunResults(run_keys)

# %% copy across the data files 
fwddir = os.path.join(simdir,'r00000') 
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

