#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 15:34:31 2022
Timelapse processing of 2D sections for Hollin Hill 
@author: jimmy
"""
import os, sys 
from datetime import datetime
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy.interpolate import LinearNDInterpolator
sys.path.append('/home/jimmy/phd/resipy/src')
from resipy.Project import Project 
from resipy.meshTools import points2vtk, quadMesh
from petroFuncs import temp_correct# temp_uncorrect
ncpu = 20
invert_real = True 
invert_simu = True  

#%% load in electrode and topography data 
files = sorted(os.listdir('Data/resData'))
efiles = sorted(os.listdir('Data/elecData'))
tfiles = sorted(os.listdir('Data/topoData'))

elec = pd.read_csv(os.path.join('Data/elecData',efiles[0]))
topo = pd.read_csv(os.path.join('Data/topoData',tfiles[0]))
elec['label'] = [str(elec['id'][i]) for i in range(len(elec))]
elec['x'] = elec['y'].values 
elec['y'] = 0 
topo['x'] = topo['y'].values 
topo['y'] = 0 

## find target date for baseline 
target = datetime(2015,4,15)
idx = -1 
doy = [0]*len(files)#'days of year'
for i,f in enumerate(files):
    dinfo = f.split('.')[0].split('-')
    fdate = datetime(int(dinfo[0]),
                     int(dinfo[1]),
                     int(dinfo[2]))
    delta = abs((fdate - target).days)
    doy[i]= (fdate - datetime(fdate.year-1,12,31)).days 
    if abs(delta) < 1 and idx==-1:
        idx = files.index(f)
        
    
flist = [os.path.join('Data/resData',f) for f in files] # merge files into full path 
flist = [flist[idx]]+flist # put baseline at top of list 
doy = [doy[idx]]+doy 

    
#%% setup resipy project 
k = Project('Models/timeLapse2D')
# k.createSurvey(flist[0],ftype='ProtocolDC')
k.createTimeLapseSurvey(flist,ftype='ProtocolDC')
k.setElec(elec)
# k.filterRecip(5)
# k.elec2distance(True)

#%% create pseudo sections 
pseudo_dir=os.path.join(k.dirname,'pseudo')
if not os.path.exists(pseudo_dir):
    os.mkdir(pseudo_dir)

# get reference apparent resistivity 
S = k.surveys[0]
S.filterRecip(5,False)
K = S.computeKborehole()
ie = S.df.irecip>0 # forward index 
xpos,ypos,zpos = S._computePseudoDepth()
refapp = S.df['resist'][ie]*K[ie]
mpkg = quadMesh(k.elec.x,k.elec.y,elemx=2,fmd=20,pad=0)
qmesh = mpkg[0]
qmesh.show()
ifunc = LinearNDInterpolator(np.c_[xpos[ie],-zpos[ie]],refapp.values,fill_value=-1)
iappref = ifunc(qmesh.df[['X','Z']].values)
qmesh.addAttribute(iappref, 'App.Ref.Res')
rapp_cache = []

# loop through and get other apparent resistivities
for i in range(len(k.surveys)):
    S = k.surveys[i]
    S.filterRecip(5,False)
    K = S.computeKborehole()
    ie = S.df.irecip>0 # forward index 
    xpos,ypos,zpos = S._computePseudoDepth()
    app = S.df['resist'][ie]*K[ie]
    points2vtk(xpos[ie], ypos[ie], -zpos[ie],
               file_name=os.path.join(pseudo_dir,'p{:0>3d}.vtk'.format(i)),
               data={'Apparent Resistivity':app})
    ifunc = LinearNDInterpolator(np.c_[xpos[ie],-zpos[ie]],app.values,fill_value=-1)
    iapp = ifunc(qmesh.df[['X','Z']].values)
    idiff = ((iapp-iappref)/iappref)*100
    qmesh.addAttribute(iapp, 'App.Res')
    qmesh.addAttribute(idiff, 'App.Res.Diff')
    qmesh.vtk(os.path.join(pseudo_dir,'pcolor{:0>3d}.vtk'.format(i)))
    rapp_cache.append(iapp)
    
    

#%% setup mesh 
surf_array = np.c_[topo['x'].values, topo['y'].values, topo['z'].values][np.isnan(topo['z'])==False]
k.createMesh(cl=1,cl_factor=5, surface=surf_array)
k.showMesh()
depths = k.mesh.computeElmDepth()

#%% inversion settings and preprocessing 
k.fitErrorPwl()
errors = []
for s in k.surveys:
    errors.append(s.df['resError'])
plt.close('all')
k.err = True 
k.param['reg_mode'] = 2 
k.param['alpha_s'] = 0 
k.bigSurvey.fitErrorPwl()
errorModel = k.bigSurvey.errorModel 

#%% invert 
if invert_real: 
    k.invert(parallel=True,ncores=ncpu) # INVERT 
    
    # grab reference temperature correction 
    mesh = k.meshResults[idx+1]
    depths = mesh.computeElmDepth()
    ref1 = mesh.df['Resistivity(ohm.m)'].values
    ref0 = temp_correct(ref1, depths, doy[idx+1])
    #loop through other meshes 
    for i,mesh in enumerate(k.meshResults):
        if i==0:
            continue 
        res1 = mesh.df['Resistivity(ohm.m)'].values
        res0 = temp_correct(res1, depths, doy[i])
        mesh.addAttribute(res0,'ResistivityTc(ohm.m)')
        diff = ((res0-ref0)/ref0)*100
        mesh.addAttribute(diff,'DifferenceTc(pnct)')
    
    # save result? 
    k.saveVtks('Models/timeLapse2D/result')
    os.rename('Models/timeLapse2D/result/time_step0001.vtk',
              'Models/timeLapse2D/result/baseline.vtk')

#%% invert the synthetic data set 
simpath = 'Models/HydroSim/SimData'
if not os.path.exists(simpath):
    sys.exit() 
    
files = sorted(os.listdir(simpath))
efiles = sorted(os.listdir('Data/elecData'))
tfiles = sorted(os.listdir('Data/topoData'))

elec = pd.read_csv(os.path.join('Data/elecData',efiles[0]))
topo = pd.read_csv(os.path.join('Data/topoData',tfiles[0]))

elec['label'] = [str(elec['id'][i]) for i in range(len(elec))]
elec['x'] = elec['y'].values 
elec['y'] = 0 
topo['x'] = topo['y'].values 
topo['y'] = 0 

flist = [os.path.join(simpath,f) for f in files] # merge files into full path 
flist = [flist[idx]]+flist # put baseline at top of list 
    
# setup resipy project 
k = Project('Models/SyntheticDataInversion')
k.createTimeLapseSurvey(flist,ftype='ProtocolDC')
k.setElec(elec)

# get reference apparent resistivity 
S = k.surveys[0]
K = S.computeKborehole()
ie = S.df.irecip>0 # forward index 
xpos,ypos,zpos = S._computePseudoDepth()
refapp = S.df['resist']*K
ifunc = LinearNDInterpolator(np.c_[xpos,-zpos],refapp.values,fill_value=-1)
iappref = ifunc(qmesh.df[['X','Z']].values)
qmesh.addAttribute(iappref, 'App.Ref.Res')

# create pseudo sections 
pseudo_dir=os.path.join(k.dirname,'pseudo')
if not os.path.exists(pseudo_dir):
    os.mkdir(pseudo_dir)
for i in range(len(k.surveys)):
    S = k.surveys[i]
    K = S.computeKborehole()
    xpos,ypos,zpos = S._computePseudoDepth()
    app = S.df['resist']*K 
    points2vtk(xpos, ypos, -zpos,
               file_name=os.path.join(pseudo_dir,'p{:0>3d}.vtk'.format(i)),
               data={'Apparent Resistivity':app})
    rapp = rapp_cache[i]
    ifunc = LinearNDInterpolator(np.c_[xpos,-zpos],app.values,fill_value=-1)
    iapp = ifunc(qmesh.df[['X','Z']].values)
    idiff = ((iapp-iappref)/iappref)*100
    qmesh.addAttribute(iapp, 'App.Res')
    qmesh.addAttribute(idiff, 'App.Res.Diff')
    rapp_diff = ((iapp-rapp)/rapp)*100
    qmesh.addAttribute(rapp_diff, 'Meas.Res.Diff')
    qmesh.vtk(os.path.join(pseudo_dir,'pcolor{:0>3d}.vtk'.format(i)))
    
    
# setup mesh 
surf_array = np.c_[topo['x'].values, topo['y'].values, topo['z'].values][np.isnan(topo['z'])==False]
k.createMesh(cl=1,cl_factor=5, surface=surf_array)
k.showMesh()

# inversion settings and preprocessing 
# k.fitErrorPwl()
for i,s in enumerate(k.surveys): 
    s.df.loc[:,'resError'] = errors[i]
k.err = True 
k.param['reg_mode'] = 2 
k.param['alpha_s'] = 0 
k.param['a_wgt'] = 0
k.param['b_wgt'] = 0
k.bigSurvey.errorModel = errorModel 

if invert_simu: 
    # invert 
    k.invert(parallel=True,ncores=ncpu) 
    # save result? 
    k.saveVtks('Models/SyntheticDataInversion/result')
    
    os.rename('Models/SyntheticDataInversion/result/time_step0001.vtk',
              'Models/SyntheticDataInversion/result/baseline.vtk')
    
#%% convert list into python animation track 
fdates = [f.replace('.dat','') for f in files] # merge files into full path 

fh = open('Models/pseudo_timestamps.txt','w')
for x in fdates:
    fh.write(x)
    fh.write('\n')
fh.close() 
