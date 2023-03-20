#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 13:58:28 2023
Mesh creation for coupled modelling 
@author: jimmy
"""
import os, sys 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.path as mpath
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
# add custom modules 
if 'RSUTRA' not in sys.path: 
    sys.path.append('RSUTRA')

from SUTRAhandler import material 
from petroFuncs import ssf_petro_sat, wmf_petro_sat 

# add custom modules 
linux_r_path = '/home/jimmy/phd/resipy/src'
win_r_path = r'C:\Users\boydj1\Software\resipy\src' 
if sys.platform == 'linux' and linux_r_path not in sys.path: 
    sys.path.append(linux_r_path)
if 'win' in sys.platform.lower() and win_r_path not in sys.path:
    sys.path.append(win_r_path)
from resipy import meshTools as mt

#%% create hollin hill mesh 
def HH_mesh(show=False): 
    """
    Create mesh and starting pore pressures 

    Parameters
    ----------
    show : bool, optional
        Flag to show mesh. The default is False.

    Returns
    -------
    None.

    """
    # read in topo data
    topo = pd.read_csv('Data/topoData/2016-01-08.csv')
    # LOAD IN EXTENT OF WMF/SSF (will be used to zone the mesh)
    poly_ssf = np.genfromtxt('interpretation/SSF_poly_v3.csv',delimiter=',')
    poly_ssf_ext = np.genfromtxt('interpretation/SSF_poly_ext.csv',delimiter=',')
    # poly_dogger = np.genfromtxt('interpretation/Dogger_poly.csv',delimiter=',')
    
    moutput = mt.quadMesh(topo['y'].values[0::2], topo['z'].values[0::2],
                          elemx=1, pad=5, fmd=15,zf=1.1,zgf=1.1)
    
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
    # path = mpath.Path(poly_dogger)
    # inside = path.contains_points(np.c_[meshx, meshz])
    # zone[inside] = 3 
    # inside = meshx > 280
    # zone[inside] = 3 
    
    # assign RMF formation (anything underneath the stiathes) or left of x = -10 
    inside = meshx < -10 
    zone[inside] = 3 
    for x in np.unique(meshx):
        idxx = meshx == x 
        zc = zone[idxx] # zone column 
        if 1 in zc and 2 in zc:
            cz = meshz[idxx] # column z coordinate, find minimum of zone 1 z coord 
            cz_min = np.min(cz[zc==1])
            for elm_id in mesh.df['elm_id'][idxx]:
                if zone[elm_id-1] == 2 and meshz[elm_id-1] < cz_min:
                    zone[elm_id-1] = 3
    
    # now to map zones to their respective identification numbers by element 
    zone_id = np.arange(mesh.numel,dtype=int)
    
    ssf_idx = zone_id[zone==1] #zone ssf 
    wmf_idx = zone_id[zone==2] #zone wmf 
    # dog_idx = zone_id[zone==3] #zone dog 
    rmf_idx = zone_id[zone==3] #zone rmf  
    
    zone_id[ssf_idx] = 1
    zone_id[wmf_idx] = 2 
    zone_id[rmf_idx] = 3 
    
    zone_flags = {'SSF':ssf_idx,
                  'WMF':wmf_idx,
                  'RMF':rmf_idx}
    
    # get the surface of the mesh
    surf_x,surf_z = mesh.extractSurface(False, False)
    cell_depths = mesh.computeElmDepth()
    
    maxx = np.max(mesh.node[:,0]) # these max/min values will be used for ... 
    minx = np.min(mesh.node[:,0]) # boundary conditions later on 
    
    mesh.addAttribute(zone_id, 'zone') # add the zone to the mesh 
    # mesh.node2ElemAttr(zone_id, 'zone') # add the zone as a element property 
    
    ## boundary conditions (note not all of these are active)
    # find surface nodes
    # zone_node = mesh.ptdf['zone'].values 
    tree = cKDTree(mesh.node[:, (0, 2)])
    dist, top_node = tree.query(np.c_[surf_x, surf_z])
    
    # find nodes at base of mesh 
    max_depth = max(cell_depths)+1
    dist, base_node = tree.query(np.c_[surf_x, surf_z-max_depth])
    
    # find nodes on left side of mesh, set as drainage boundary 
    left_side_bool = mesh.node[:,0] == minx
    left_side_node = mesh.node[left_side_bool]
    dist, left_node = tree.query(left_side_node[:,[0,2]])
    
    # find nodes on left side of mesh 
    right_side_bool = mesh.node[:,0] == maxx 
    right_side_node = mesh.node[right_side_bool]
    dist, right_node = tree.query(right_side_node[:,[0,2]])
    
    # find nodes at borehole 1902 
    b1902x = mesh.node[:,0][np.argmin(np.sqrt((mesh.node[:,0]-106.288)**2))]
    b1902_idx = mesh.node[:,0] == b1902x
    b1902_node = mesh.node[b1902_idx]
    b1902_topo = max(b1902_node[:,2])
    b1902_wt = b1902_topo - 5.0 
    # compute pressure at base of mesh below b1902 
    b1902_delta= b1902_wt - np.min(b1902_node[:,2])
    b1902_pressure_val = 9810*b1902_delta
    
    # hold pressure at borehole 1901 
    b1901x = mesh.node[:,0][np.argmin(np.sqrt((mesh.node[:,0]-26.501)**2))]
    b1901_idx = (mesh.node[:,0] == b1901x)
    b1901_node = mesh.node[b1901_idx]
    b1901_topo = max(b1901_node[:,2])
    b1901_wt = b1901_topo - 5.7
    # compute pressure at base of mesh below b1902 
    b1901_delta= b1901_wt - np.min(b1901_node[:,2])
    b1901_pressure_val = 9810*b1901_delta
    
    # compute pressure for every node 
    wfunc = interp1d([b1901x,b1902x,160],
                     [b1901_wt,b1902_wt,89],
                     # fill_value=(b1901_wt,b1902_wt)
                      fill_value='extrapolate'
                     )
    wt = wfunc(mesh.node[:,0])
    # wt[mesh.node[:,0]<b1901x] = b1901_wt
    wt[mesh.node[:,0]>159] = 89
    delta = wt - mesh.node[:,2]
    nodal_pressures = delta*9810

    # compute basal pressure assuming watertable known at  1901 and 1902 
    base_pressure = nodal_pressures[base_node]
    
    # find mid points for computing cross sectional area for infiltration 
    dx = np.zeros(len(surf_x))
    dx[0] = np.abs(surf_x[1] - surf_x[0])
    for i in range(1,len(surf_x)):
        dx[i] = np.abs(surf_x[i] - surf_x[i-1])
    
    if show: 
        fig, ax = plt.subplots()
        mesh.show(attr='zone',ax=ax)
        ax.plot(mesh.node[:,0],wt,c='b')
        
    return mesh, zone_flags, nodal_pressures, base_pressure
    
def HH_mat():
    SSF = material(Ksat=0.144,theta_res=0.06,theta_sat=0.38,
                   alpha=0.062,vn=1.52,name='STAITHES')
    WMF = material(Ksat=0.013,theta_res=0.1,theta_sat=0.48,
                   alpha=0.3,vn=1.32,name='WHITBY')
    
    DOG = material(Ksat=0.309,theta_res=0.008,theta_sat=0.215,
                   alpha=0.05,vn=1.75,name='DOGGER')
    RMF = material(Ksat=0.13,theta_res=0.1,theta_sat=0.48,
                   alpha=0.0126,vn=1.44,name='REDCAR')
    
    SSF.setPetro(ssf_petro_sat)
    WMF.setPetro(wmf_petro_sat)
    DOG.setPetro(ssf_petro_sat)
    RMF.setPetro(wmf_petro_sat)
    return SSF,WMF,DOG,RMF 

# HH_mesh(True)