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
sys.path.append('/home/jimmy/phd/resipy/src')
from resipy.Project import Project 
from resipy.meshTools import points2vtk
ncpu = 20
invert_real = False 

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
for f in files:
    dinfo = f.split('.')[0].split('-')
    fdate = datetime(int(dinfo[0]),
                     int(dinfo[1]),
                     int(dinfo[2]))
    delta = (fdate - target).days 
    if abs(delta) < 10:
        idx = files.index(f)
        break
    
flist = [os.path.join('Data/resData',f) for f in files] # merge files into full path 
flist = [flist[idx]]+flist # put baseline at top of list 
    
#%% setup resipy project 
k = Project('Models/timeLapse2D')
# k.createSurvey(flist[0],ftype='ProtocolDC')
k.createTimeLapseSurvey(flist,ftype='ProtocolDC')
k.setElec(elec)
k.filterRecip(5)
# k.elec2distance(True)

#%% create pseudo sections 
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

#%% setup mesh 
surf_array = np.c_[topo['x'].values, topo['y'].values, topo['z'].values][np.isnan(topo['z'])==False]
k.createMesh(cl=1,cl_factor=5, surface=surf_array)
k.showMesh()

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
    k.invert(parallel=True,ncores=ncpu) 
    # k.showResults()
    
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
# invert 
k.invert(parallel=True,ncores=ncpu) 
# save result? 
k.saveVtks('Models/SyntheticDataInversion/result')

os.rename('Models/SyntheticDataInversion/result/time_step0001.vtk',
          'Models/SyntheticDataInversion/result/baseline.vtk')