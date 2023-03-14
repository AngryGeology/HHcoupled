#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 13:46:16 2022
Hydrological and geophysical simulation of Hollin Hill using best fit parameters  
and rainfall. 
@author: jimmy
"""
import os,sys,shutil 
from datetime import datetime 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.path as mpath
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
from resipy.r2in import write2in # needed for forward modelling 
from resipy import meshTools as mt
from SUTRAhandler import handler, material , invVGcurve
from petroFuncs import ssf_petro_sat, wmf_petro_sat 

# setup script variables and functions 
secinday = 24*60*60 
plt.close('all')

exec_loc = '/home/jimmy/programs/SUTRA_JB/bin/sutra'
if 'win' in sys.platform.lower():
    exec_loc = r'C:/Users/boydj1/Software/SUTRA/bin/sutra.exe'
    
    
model_dir = 'Models'
sim_dir = os.path.join(model_dir,'HydroExp')
datadir = os.path.join(sim_dir,'SimData')
for d in [model_dir,sim_dir,datadir]:
    if not os.path.exists(d):
        os.mkdir(d)

# %% step 0, load in relevant files
rainfall = pd.read_csv(os.path.join('Data/Rainfall', 'COSMOS_2015-2016.csv'))

ifac = 2
# rainfall = rainfall.loc[rainfall.index.repeat(temporalrefine)].reset_index()
moisture = pd.read_csv(os.path.join('Data/Rainfall','COSMOS_vwc_2014-2016.csv')) 
# moisture = moisture.loc[moisture.index.repeat(temporalrefine)].reset_index()
# read in warm up
warmup = pd.read_csv('Models/HydroWarmUp/warm.csv')
# read in topo data
topo = pd.read_csv('Data/topoData/2016-01-08.csv')
elec = pd.read_csv('Data/elecData/2016-01-08.csv')
# LOAD IN EXTENT OF WMF/SSF (will be used to zone the mesh)
poly_ssf = np.genfromtxt('interpretation/SSF_poly_v3.csv',delimiter=',')
poly_ssf_ext = np.genfromtxt('interpretation/SSF_poly_ext.csv',delimiter=',')
poly_dogger = np.genfromtxt('interpretation/Dogger_poly.csv',delimiter=',')

maxx = np.max(topo['y'])
minx = np.min(topo['y'])

# doctor elec 
elec['label'] = [str(elec['id'][i]) for i in range(len(elec))]
elecx = elec['y'].values 
elec.loc[:,'y'] = 0 
elec.loc[:,'x'] = elecx 

#%% step 0.5 
# get resistivity files and translate file names into dates 
def str2dat(fname):
    if '.dat' in fname: 
        string = fname.replace('.dat','')
    else:
        string = fname 
    dt = datetime.strptime(string, "%Y-%m-%d")
    return dt 
    
rfiles = [] # resistivity files 
sdates = [] # survey dates 
sdiy = [] 
for f in sorted(os.listdir('Data/resData')):
    if f.endswith('.dat'):
        rfiles.append(f)
        dt = str2dat(f)
        sdates.append(str2dat(f))
        # compute day in year 
        year = dt.year 
        ref_dt = datetime(year-1,12,31)
        sdiy.append((dt - ref_dt).days)  
        
# find times when resistivity and rainfall data are there
nobs = len(moisture)
rdates = [str2dat(moisture['DATE_TIME'][i].split()[0]) for i in range(nobs)]
days_elasp = [0]*nobs 
sflag = [False]*nobs # flag to determine if there is a resistivity survey 
suidx = [-1]*nobs # corresponding survey index 
for i in range(nobs):
    date = rdates[i]
    days_elasp[i] = (date-rdates[0]).days  
    delta = [abs((date - sdate).days) for sdate in sdates]
    if min(delta) == 0:
        idx = np.argmin(delta)
        suidx[i] = idx 
        sflag[i] = True 

# sflag = [False] + sflag # add one here because first returned survey is time 0 ??
survey_keys = np.arange(nobs)[np.array(sflag)==True]

## interpolate missing vwc values 
nanidx = moisture['VWC']<0
x = np.array(days_elasp)
moisture.loc[nanidx,'VWC'] = np.interp(x[nanidx],x[np.invert(nanidx)],
                                       moisture['VWC'][np.invert(nanidx)])
moisture['sw'] = (moisture['VWC']/max(moisture['VWC']))

# %% step 1, setup/create the mesh 
# create quad mesh
moutput = mt.quadMesh(topo['y'][0::2].values, topo['z'][0::2].values, 
                      elemx=1, pad=5, fmd=10,zf=1.1,zgf=1.1)
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
# zone[inside] = 3 
# inside = meshx > 280
# zone[inside] = 3 

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
SSF = material(Ksat=0.144,theta_res=0.06,theta_sat=0.38,
               alpha=0.13,vn=1.52,name='STAITHES')
WMF = material(Ksat=0.013,theta_res=0.1,theta_sat=0.48,
               alpha=0.01,vn=1.32,name='WHITBY')
DOG = material(Ksat=0.309,theta_res=0.008,theta_sat=0.215,
               alpha=0.05,vn=1.75,name='DOGGER')
RMF = material(Ksat=0.13,theta_res=0.1,theta_sat=0.48,
               alpha=0.0126,vn=1.44,name='REDCAR')

SSF.setPetro(ssf_petro_sat)
WMF.setPetro(wmf_petro_sat)
DOG.setPetro(ssf_petro_sat)
RMF.setPetro(wmf_petro_sat)

#%% step 3, setup handler 
## create handler
h = handler(dname=sim_dir, ifac=ifac,tlength=secinday,
            iobs=ifac, 
            flow = 'transient',
            transport = 'transient',
            sim_type='solute')
h.maxIter = 300
h.rpmax = 5e3
h.drainage = 1e-8
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
# right_side_idx = (mesh.node[:,0] == maxx) # & (zone_node == 2)
right_side_idx = (mesh.node[:,0] == b1902x) 
right_side_node = mesh.node[right_side_idx]
right_side_topo = max(right_side_node[:,2])
right_side_wt = right_side_topo - 5.0 
rs_delta = right_side_topo - right_side_wt 
right_side_node_sat = right_side_node[right_side_node[:,2]<(right_side_topo-rs_delta)]
dist, right_node = tree.query(right_side_node_sat[:,[0,2]])
# compute pressure on right hand side of mesh 
right_side_bl= right_side_wt - mesh.node[right_node][:,2]
right_pressure_val = max(9810*right_side_bl)
# pres_node =np.asarray(np.append(pres_node,right_node),dtype=int)
# pressure_vals = 9810*right_side_bl


#find nodes on left side of mesh, set as drainage boundary 
left_side_idx = (mesh.node[:,0] == minx)
left_side_node = mesh.node[left_side_idx]
dist, left_node = tree.query(left_side_node[:,[0,2]])
general_node = np.append(general_node,left_node + 1) 
general_type = general_type + ['seep']*len(left_node)
# pres_node = pres_node + [0]*len(left_node)

# hold pressure at borehole 1901 
b1901x = mesh.node[:,0][np.argmin(np.sqrt((mesh.node[:,0]-26.501)**2))]
b1901_idx = (mesh.node[:,0] == b1901x)
b1901_node = mesh.node[b1901_idx]
b1901_topo = max(b1901_node[:,2])
b1901_wt = b1901_topo - 5.7
b1901_delta = b1901_topo - b1901_wt 
b1901_node_sat = b1901_node[b1901_node[:,2]<(b1901_topo-b1901_delta)]
dist, b1901_node = tree.query(b1901_node_sat[:,[0,2]])
b1901_bl=b1901_wt - mesh.node[b1901_node][:,2]
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
# pres_node = pres_node + base_pressures.tolist()

# find mid points for computing cross sectional area for infiltration 
dx = np.zeros(len(xz[0]))
dx[0] = np.abs(xz[0][1] - xz[0][0])
for i in range(1,len(xz[0])):
    dx[i] = np.abs(xz[0][i] - xz[0][i-1])

# %% step 6, doctor rainfall and energy input for SUTRA 
# rainfall is given in mm/day, so convert to kg/s
# rain = rainfall['EFF_RAIN'].values  # rain in mm/d
# # rain[rain<0]=0
# infil = (rain*1e-3)/secinday #in kg/s # check this #### 
# infil[np.isnan(infil)] = 0 # put NAN at zero 

# ntimes = len(rainfall)
# TimeStamp = np.arange(ntimes)
# times = np.asarray(TimeStamp, dtype=int)
# Temps = np.zeros_like(TimeStamp) + 0
# Temp_surface = np.zeros_like(TimeStamp) + 0

# # now populate matrices
# fluidinp = np.zeros((len(TimeStamp), len(source_node)))  # fluid input
# tempinp = np.zeros((len(TimeStamp), len(source_node)))  # fluid temperature
# surftempinp = np.zeros((len(TimeStamp), len(source_node)))  # surface temperature
# for i in range(len(source_node)):
#     m = dx[i]
#     fluidinp[:, i] = infil*m
#     tempinp[:, i] = Temps
#     surftempinp[:, i] = Temp_surface
    
# ## plot infiltration 
# fig, ax = plt.subplots()
# ax.plot(TimeStamp,infil) 

#%% step 6 doctor moisture content data for sutra 
ntimes = len(moisture)
TimeStamp = np.arange(ntimes)
times = np.asarray(TimeStamp, dtype=int)
Temps = np.zeros_like(TimeStamp) + 0
Temp_surface = np.zeros_like(TimeStamp) + 0
nsurface = len(source_node)
zoneinp = h.mesh.ptdf['zone'][source_node-1]

swinp = np.zeros((len(TimeStamp), len(source_node))) 
presinp = np.zeros((len(TimeStamp), len(source_node)))  # fluid input
P = np.zeros_like(zoneinp,dtype=float)
for i in range(len(TimeStamp)):
    sw = moisture['sw'][i]*0.99
    swinp[i,:] = sw 
    P[zoneinp==1] = SSF.pres(np.full(np.count_nonzero(zoneinp==1),sw))
    P[zoneinp==2] = WMF.pres(np.full(np.count_nonzero(zoneinp==2),sw))
    P[zoneinp==3] = DOG.pres(np.full(np.count_nonzero(zoneinp==3),sw))
    P[zoneinp==4] = RMF.pres(np.full(np.count_nonzero(zoneinp==4),sw))
    presinp[i, :] = -P*1000 # pressure in PA 
    
# ## plot moisture versus time  
fig, ax = plt.subplots()
ax.plot(TimeStamp,moisture['sw']) 
ax.set_ylabel('Sw (-)')
ax.set_xlabel('Day')

#%% step 6.5 estimate Sw from rainfall 
fig, ax = plt.subplots() 

rain=rainfall['EFF_RAIN'].values
sw_dates=[str2dat(rainfall['DATE_TIME'][i]) for i in range(len(rainfall))]
sw_times = np.arange(len(rain))*secinday 
rain[np.isnan(rain)] = 0 
Pr = (rain*1e-3)/secinday
Et = np.zeros_like(Pr)
sw_ssf = SSF.estSw(Pr, Et, sw_times)
sw_wmf = WMF.estSw(Pr, Et, sw_times)
l0 = ax.plot(sw_dates,sw_ssf,c='y',label='SSF Sw')
l1 = ax.plot(sw_dates,sw_wmf,c='k',label='SSF Sw')
ax.set_ylabel('Sw (-)')
ax.set_xlabel('Times(days)')
axt = ax.twinx()
l2 = axt.bar(sw_dates,rain,label='Rainfall')
axt.set_ylabel('Effective Rainfall (m/s)')

# lines = [l0,l1,l2[0]]
# labels = ['SSF','WMF','Rainfall']
ax.legend()

#%% step 7, setup initial conditions 
ppres = warmup['Pressure'].values 
ppoints = np.c_[warmup['X'].values, warmup['Y'].values]
ipoints = np.c_[h.mesh.node[:,0], h.mesh.node[:,2]]
ifunc = LinearNDInterpolator(ppoints, ppres)
nfunc = NearestNDInterpolator(ppoints, ppres)
pres = ifunc(ipoints)
nanidx = np.isnan(pres)
pres[nanidx] = nfunc(ipoints[nanidx]) 

# compute pressure on right hand side of mesh 
right_side_bl= right_side_wt - mesh.node[right_node][:,2]
right_pressure_val = 9810*right_side_bl
h.pressure = np.append(np.zeros(len(left_node)),base_pressures)# max(right_pressure_val)

pressure_vals = pres[source_node-1]

temp = [0]*mesh.numnp 
pres[np.isnan(pres)] = 0 

#%% step 8, write inputs for SUTRA 
h.setupInp(times=times, 
           # source_node=source_node, 
           pressure_node=source_node, pressure_val=pressure_vals, 
           general_node=general_node, general_type=general_type)
# h.pressure = pres_node 
h.writeInp(variable_pressure=True) # write input without water table at base of column
# h.writeBcs(times, source_node, fluidinp, tempinp)
h.writeBcsPres(times, source_node, presinp)
h.writeIcs(pres, temp) # INITIAL CONDITIONS 
h.writeVg()
h.writeFil(ignore=['BCOP', 'BCOPG'])

h.showSetup() 

#%% step9, run sutra 
h.runSUTRA()  # run
h.getResults()  # get results

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
h.plot1Dresults()
h.plotMeshResults(cmap='RdBu')

h.attribute = 'Pressure'
h.vmax = pmax
h.vmin = pmin
h.vlim = [pmin, pmax]
# h.plot1Dresults()
# h.plotMeshResults()

# save mesh output 
# mesh.flipYZ()
mesh.vtk(os.path.join(h.dname,'mesh.vtk'))

#%% step 11 simulate geophysical responses 
## create a sequence of data and estimated errors 
# fig,ax = plt.subplots()
# data_seq = pd.DataFrame()
# sequences = [] 
# for i,f in enumerate(rfiles): 
#     s = Survey(os.path.join('Data/resData',f),ftype='ProtocolDC') # parse survey 
#     s.fitErrorPwl(ax) # fit reciprocal error levels 
#     # extract the abmn, data, error information and date 
#     df = s.df[['a','b','m','n','recipMean','resError']]
#     df = df.rename(columns={'recipMean':'tr',
#                             'resError':'error'})
#     df['sidx'] = i 
#     sequences.append(df[['a','b','m','n']].values)
#     data_seq = pd.concat([data_seq,df]) # append frame to data sequence 
#     ax.cla() 
# and setup R2 project 
# k = Project(dirname=sim_dir)
# k.setElec(elec)
# surface = np.c_[topo['y'],topo['z']]
# k.createMesh(cl_factor=4, surface=surface)

# h.setRproject(k)
# h.setupRparam(data_seq, write2in, survey_keys, seqs=sequences)
# ssf_param = {'alpha':[SSF.alpha],'vn':[SSF.vn]}
# wmf_param = {'alpha':[WMF.alpha],'vn':[WMF.vn]}

# # setup materials for resitivity-gmc relationships 
# SSF.setMCparam(ssf_param)
# WMF.setMCparam(wmf_param)
# SSF.setPetro(ssf_petro_sat)
# WMF.setPetro(wmf_petro_sat)

# h.setupMultiRun() 
# h.cpu = 12 # run multi threaded for resistivity part 
# h.runMultiRun()
# run_keys = h.getMultiRun()

# # simulate resistivities 
# # we need to add one becuase the steps start from a initialising run where t=0 
# survey_keys = np.arange(h.resultNsteps)[np.array(sflag)==True]+1

# # now setup R2 folders 
# h.setupRruns(write2in,run_keys,survey_keys,sequences)

# #now go through and run folders 
# h.runResFwdmdls(run_keys)
# data_store = h.getFwdRunResults(run_keys)

# # %% copy across the data files 
# fwddir = os.path.join(sim_dir,'r00000') 
# for f in sorted(os.listdir(fwddir)):
#     if not 'forward' in f:
#         continue 
#     if 'R2' in f:
#         continue 
#     if '_model' in f: 
#         continue 
#     # work out index 
#     idx = f.replace('forward','').replace('.dat','')
#     i = int(idx)
#     shutil.copy(os.path.join(fwddir,f),
#                 os.path.join(datadir,rfiles[i]))

