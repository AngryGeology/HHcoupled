#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 12:51:46 2022
Convert warm up model into resistivity and invert it 
@author: jimmy
"""
#%% load modules 
import os, shutil 
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.path as mpath
from scipy.spatial import cKDTree
sys.path.append('RSUTRA')
sys.path.append('/home/jimmy/phd/resipy/src')
from resipy.Project import Project 
from resipy.parsers import protocolParser
from SUTRAhandler import handler
from SUTRAhandler import invVGcurve, vgCurve 
from solveWaxSmit import solveRtWVP

plt.close('all')

#%% load in files and parameters 
# load up warm up results 
# df = pd.read_csv('HydroWarmUp/warm.csv')
df = pd.read_csv('HydroBestFit/frun.csv')
# read in topo and elec data
topo = pd.read_csv('topoData/2016-01-08.csv')
elec = pd.read_csv('elecData/2016-01-08.csv')
elec['x'] = elec['y']
elec.loc[:,'y'] = 0 
e,data = protocolParser('resData/2016-01-08.dat')
seq = data[['a','b','m','n']].values 

theta_fit = np.genfromtxt('/home/jimmy/phd/Hollin_Hill/papers/paper2/codes/theta_fit.txt')

tparam = {1:{'res':0.24,
             'sat':0.48,
             'a':0.16,
             'n':1.22},
          2:{'res':0.166,
             'sat':0.89,
             'a':0.113,
             'n':1.46}}# tension parameters 

Rw = 1.013171225937183451e+01
sparam = {1:{'theta':0.36,
             'Pg':2.74,
             'Rw':Rw,
             'cec':11.0,
             'FF':18.5,
             'n':3.05},
          2:{'theta':0.45,
             'Pg':2.74,
             'Rw':Rw,
             'cec':22.5,
             'FF':9.42,
             'n':2.74}}# wax smit parameters 

# NEW PARAM after talk with andy
sparam[1]['FF'] = 24.05 
sparam[1]['n'] = 3.91
sparam[2]['FF'] = 13.60
sparam[2]['n'] = 10.06

# one of 2 lines belows needs to be commented 
# fix for if using actual data sequence 
elec['label'] = [str(elec['id'][i]) for i in range(len(elec))]
# fix for if using resipy generated sequence 
# elec.sort_values('x',inplace=True)

#%% setup resipy project 
# k = Project(dirname='HydroWarmUp')
k = Project(dirname='HydroBestFit')
k.createSurvey('resData/2016-01-08.dat',ftype='ProtocolDC')
irecip = k.surveys[0].df['irecip']
k.fitErrorPwl()
error = k.surveys[0].df['resError']
seq=seq[irecip>-1]
k.setElec(elec)
k.createMesh(typ='trian', cl=0.75, cl_factor=5)
k.showMesh() 
# k.showPseudo()
fwd = False 

#%% map pressures to resistivity 
ipoints = df[['X','Y']].values 
tree = cKDTree(ipoints)
dist, nidx = tree.query(k.mesh.elmCentre[:,[0,2]])
mesh_pressure = df['Pressure'][nidx].values 
mesh_zone = df['zone'][nidx].values 
mesh_saturation = df['Saturation'][nidx].values 
mesh_resist = df['Resistivity'][nidx].values 
k.mesh.addAttribute(mesh_resist, 'res0')
k.mesh.addAttribute(mesh_pressure,'Pressure')
k.mesh.addAttribute(mesh_zone, 'zone')
# k.mesh.show(attr='zone')

#%% pressure way 
# cond = np.zeros(k.mesh.numel)
# for i in range(k.mesh.numel):
#     zid = int(mesh_zone[i])
#     if mesh_pressure[i] < 0:
#         cond[i] = vgCurve(-mesh_pressure[i]/1000,
#                           tparam[zid]['res'],
#                           tparam[zid]['sat'],
#                           tparam[zid]['a'],
#                           tparam[zid]['n'])
#     else:
#         cond[i] = tparam[zid]['sat']

# k.mesh.addAttribute(cond, 'Conductivity')
# fwd_res = 1/(cond/10)
# # fwd_res[mesh_zone==1] = 80
# # fwd_res[mesh_zone==2] = 20 
# k.mesh.addAttribute(fwd_res, 'fwd_res')

#%% saturation/gmc way 
def sat2gmc(sat,theta,ps,pw=1):
    numon = pw*theta*sat 
    denom = ps*(1-theta)
    return numon/denom 

# fwd_res = np.zeros(k.mesh.numel)
# for i in range(k.mesh.numel):
#     zid = int(mesh_zone[i])
#     # slight fudge to deal with the fact the hydro model has 4 zones! 
#     if zid == 4:
#         zid = 2
#     if zid == 3:
#         zid = 1 
#     gmc = sat2gmc(mesh_saturation[i],
#                   sparam[zid]['theta'],
#                   sparam[zid]['Pg'])
#     # (double[:] gmc, double Rw, double Pg, double Pw,
#     #              double[:] theta_param, double cec, double FF, double n):
#     fwd_res[i] = solveRtWVP(np.array([gmc]),sparam[zid]['Rw'],sparam[zid]['Pg'],1.0,
#                             theta_fit, sparam[zid]['cec'], sparam[zid]['FF'], 
#                             sparam[zid]['n'])
    
    
# k.mesh.addAttribute(fwd_res, 'fwd_res')
#%% forward model 
k.setStartingRes({1:mesh_resist})
k.sequence = seq
# k.createSequence([('dpdp1', 1, 8),
#                   ('dpdp2', 2, 8)])
k.forward(0) 
k.surveys[0].df.loc[:,'resError'] = error 
k.showPseudo()
fwd = True 
k.err = True 

#%% invert model 
# k.setRefModel(np.full(k.mesh.numel,100))
# k.param['a_wgt'] = 0.01
# k.param['b_wgt'] = 0.02
# k.param['target_decrease'] = 0.1 
if not fwd: # if not running the forward model then do error analysis 
    k.fitErrorPwl()
    k.filterRecip(10)
    k.err = True 
k.invert() 
k.showResults(len(k.meshResults)-1)

#%% get real data/inversion 
fpath = '/home/jimmy/phd/Hollin_Hill/Coupled/timeLapse2D/invdir/f001_res.vtk'
fname = os.path.split(fpath)[-1]
shutil.copy(fpath, os.path.join(k.dirname,'real_inversion.vtk'))
shutil.copy(os.path.join(k.dirname,'f001_res.vtk'),
            os.path.join(k.dirname,'synthetic_inversion.vtk'))

#%% WMF param 
# Number of successful chains: 43
# burnin id: 430000
# Alpha: 0.113470 +/- 0.002382
# N: 1.466579 +/- 0.005234
# Sat: 0.891706+/- 0.037381
# RMS: 56.591249
# CHi^2: 1093.016394
# r^2: 0.997188

#%% staithes param 
# Alpha: 0.161573 +/- 0.012460
# N: 1.226883 +/- 0.008078
# Sat: 0.481590+/- 0.020311
# Res: 0.249228
# RMS: 46.625941
# CHi^2: 1179.566270
# r^2: 0.973260
