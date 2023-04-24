#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 13:58:28 2023
Mesh creation for coupled modelling 
@author: jimmy
"""
import os, sys 
from datetime import datetime 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import matplotlib.path as mpath
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
from joblib import Parallel, delayed
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
from resipy import Survey

secinday = 24*60*60 

#%% create hollin hill mesh and data 
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
    poly_ssf = np.genfromtxt('Domain/SSF_poly_v3.csv',delimiter=',')
    poly_ssf_ext = np.genfromtxt('Domain/SSF_poly_ext.csv',delimiter=',')
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
    rmf_idx = zone_id[zone==3] #zone rmf  
    
    zone_id[ssf_idx] = 1
    zone_id[wmf_idx] = 2 # SSF = material(Ksat=0.64,theta_res=0.06,theta_sat=0.38,
#                 alpha=0.64,vn=1.2,name='STAITHES')
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
    # zone_node = mesh.ptdf['zone'].values 
    tree = cKDTree(mesh.node[:, (0, 2)])
    dist, top_node = tree.query(np.c_[surf_x, surf_z])
    
    # find nodes at base of mesh 
    max_depth = max(cell_depths)+1
    dist, base_node = tree.query(np.c_[surf_x, surf_z-max_depth])
    
    # find nodes on left side of mesh, set as drainage boundary 
    left_side_bool = mesh.node[:,0] == minx
    left_max_z = np.max(mesh.node[:,2][left_side_bool])
    left_side_bool = (mesh.node[:,0] == minx) & (mesh.node[:,2]<left_max_z)
    left_side_node = mesh.node[left_side_bool]
    dist, left_node = tree.query(left_side_node[:,[0,2]])
    
    # find nodes on left side of mesh 
    right_side_bool = mesh.node[:,0] == maxx 
    right_max_z = np.max(mesh.node[:,2][right_side_bool])
    right_side_bool = (mesh.node[:,0] == maxx) & (mesh.node[:,2]<right_max_z)
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
    # nodal_pressures[nodal_pressures<0] = 0 
    nodal_pressures[np.isnan(nodal_pressures)] = 0 

    # compute basal pressure assuming watertable known at  1901 and 1902 
    base_pressure = nodal_pressures[base_node]
    
    # find mid points for computing cross sectional area for infiltration 
    dx = np.zeros(len(surf_x))
    dx[0] = np.abs(surf_x[1] - surf_x[0])
    dx[-1] = np.abs(surf_x[-1] - surf_x[-2])
    for i in range(1,len(surf_x)-1):
        dx0 = np.abs(surf_x[i] - surf_x[i-1])
        dx1 = np.abs(surf_x[i] - surf_x[i+1])
        dx[i] = (dx0+dx1)/2 
    
    if show: 
        fig, ax = plt.subplots()
        mesh.show(attr='zone',ax=ax)
        ax.plot(mesh.node[:,0],wt,c='b')
        
    boundaries = {'top':top_node,
                  'bottom':base_node,
                  'left':left_node,
                  'right':right_node}
    
    pressures = {'nodal':nodal_pressures,
                 'base':base_pressure}
        
    return mesh, zone_flags, dx, pressures, boundaries  
    
def HH_mat():
    """
    HH materials 

    Returns
    -------
    SSF : TYPE
        DESCRIPTION.
    WMF : TYPE
        DESCRIPTION.
    DOG : TYPE
        DESCRIPTION.
    RMF : TYPE
        DESCRIPTION.

    """
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

def str2dat(fname):
    if '.dat' in fname: 
        string = fname.replace('.dat','')
    else:
        string = fname 
    dt = datetime.strptime(string, "%Y-%m-%d")
    return dt 

def HH_data(ncpu=1,show=False):
    """
    Read in hollin hill hydro and transfer resistance data 

    Parameters
    ----------
    tr : bool, optional
        DESCRIPTION. The default is True.
    ncpu : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    hydro_data = pd.read_csv('Data/Rainfall/HydroForcing.csv')
    
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
    nobs = len(hydro_data)
    rdates = [str2dat(hydro_data['DATE_TIME'][i]) for i in range(nobs)]
    hdiy = [0]*nobs 
    sflag = [False]*nobs # flag to determine if there is a resistivity survey 
    suidx = [-1]*nobs # corresponding survey index 
    for i in range(nobs):
        date = rdates[i]
        delta = [abs((date - sdate).days) for sdate in sdates]
        if min(delta) == 0:
            idx = np.argmin(delta)
            suidx[i] = idx 
            sflag[i+1] = True # need to add one here as the first time step in sutra is time 0 
        year = date.year 
        ref_dt = datetime(year-1,12,31)
        hdiy[i] = (date - ref_dt).days
        
    hydro_data['datetime'] = rdates 
    hydro_data['diy'] = hdiy 
    
    # sflag = [False] + sflag # add one here because first returned survey is time 0 ??
    survey_keys = np.arange(nobs)[np.array(sflag)==True]
        
    ## create a sequence of data and estimated errors 
    data_seq = pd.DataFrame()
    sequences = [] 
    def loop(i):
        f = rfiles[i]
        s = Survey(os.path.join('Data/resData',f),ftype='ProtocolDC',debug=False) # parse survey 
        s.filterRecip(5,False)
        fig = s.fitErrorPwl() # fit reciprocal error levels 
        # extract the abmn, data, error information and date 
        df = s.df[['a','b','m','n','recipMean','resError']]
        ie = s.df['irecip'].values >= 0 # reciprocal + non-paired
        df = s.df[ie]
        df = df.rename(columns={'recipMean':'tr',
                                'resError':'error'})
        df['sidx'] = i 
        sequence = df[['a','b','m','n']].values
        # sequences.append(df[['a','b','m','n']].values)
        # data_seq = pd.concat([data_seq,df]) # append frame to data sequence 
        plt.close(fig) 
        return df, sequence 
    
    if ncpu <= 1: 
        pout = [loop(i) for i in range(len(rfiles))]
    else: 
        nruns = len(rfiles)
        pout=Parallel(n_jobs=ncpu)(delayed(loop)(i) for i in range(nruns))
    
    for i in range(len(rfiles)):
        data_seq = pd.concat([data_seq,pout[i][0]])
        sequences.append(pout[i][1])
        
    if show: 
        fig, ax = plt.subplots(nrows=3) 
        miny = -max(hydro_data['PE'].values)
        maxy = max(hydro_data['PRECIP'].values) 
        for i in range(3):
            ax[i].bar(rdates,hydro_data['PRECIP'].values,color='b')
            ax[i].bar(rdates,-hydro_data['PE'].values,color='r')
            for j in range(len(sdates)):
                ax[i].plot([sdates[j],sdates[j]],[miny,maxy],c=(0.5,0.5,0.5,0.5))
            ax[i].set_ylabel('Rainfall / p.Et (mm/day)')
            limits = [datetime(2014+i,1,1),datetime(2014+i,12,31)]
            ax[i].set_xlim(limits)
            ax[i].set_ylim([miny,maxy])            
        ax[-1].set_xlabel('Datetime')
        
        return fig 
                
    return hydro_data, data_seq, sequences, survey_keys, rfiles, sdiy

def HH_getElec():
    elec = pd.read_csv('Data/elecData/2016-01-08.csv')
    # doctor elec 
    elec['label'] = [str(elec['id'][i]) for i in range(len(elec))]
    elecx = elec['y'].values 
    elec.loc[:,'y'] = 0 
    elec.loc[:,'x'] = elecx 
    
    return elec 

#%% create synthetic mesh and data 
def Sy_mesh(show=False):
    topo = pd.read_csv('Data/topoData/2016-01-08.csv')
    
    # fit a generic polyline to get a similar slope profile of the actual hill 
    tx = topo['y'].values # actually grid is located against the y axis, make this the local x coordinate in 2d 
    p = np.polyfit(tx,topo['z'].values,1) # fit elevation to along ground distance data 
    tz = np.polyval(p,tx) # fit the modelled topography 
    
    moutput = mt.quadMesh(tx[0::2], tz[0::2], elemx=1, pad=5, fmd=15,
                          zf=1.1,zgf=1.1)
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
    
    zone_id[ssf_idx] = 1
    zone_id[wmf_idx] = 2 
    
    zone_flags = {'SSF':ssf_idx,
                  'WMF':wmf_idx}
    
    # get the surface of the mesh
    surf_x,surf_z = mesh.extractSurface(False, False)
    cell_depths = mesh.computeElmDepth()
    
    maxx = np.max(mesh.node[:,0]) # these max/min values will be used for ... 
    minx = np.min(mesh.node[:,0]) # boundary conditions later on 
    
    mesh.addAttribute(zone_id, 'zone') # add the zone to the mesh 
    # mesh.node2ElemAttr(zone_id, 'zone') # add the zone as a element property 
    
    ## boundary conditions (note not all of these are active)
    tree = cKDTree(mesh.node[:, (0, 2)])
    dist, top_node = tree.query(np.c_[surf_x, surf_z])
    
    # find nodes at base of mesh 
    max_depth = max(cell_depths)+1
    dist, base_node = tree.query(np.c_[surf_x, surf_z-max_depth])
    
    # find nodes on left side of mesh, set as drainage boundary 
    left_side_bool = mesh.node[:,0] == minx
    left_max_z = np.max(mesh.node[:,2][left_side_bool])
    left_side_bool = (mesh.node[:,0] == minx) & (mesh.node[:,2]<left_max_z)
    left_side_node = mesh.node[left_side_bool]
    dist, left_node = tree.query(left_side_node[:,[0,2]])
    left_side_wt = tz[np.argmin(tx)]-5 
    
    # find nodes on left side of mesh 
    right_side_bool = mesh.node[:,0] == maxx 
    right_max_z = np.max(mesh.node[:,2][right_side_bool])
    right_side_bool = (mesh.node[:,0] == maxx) & (mesh.node[:,2]<right_max_z)
    right_side_node = mesh.node[right_side_bool]
    dist, right_node = tree.query(right_side_node[:,[0,2]])   
    right_side_wt = tz[np.argmax(tx)]-5  
    
    # compute pressure for every node 
    wfunc = interp1d([min(tx),max(tx)],
                     [left_side_wt,right_side_wt],
                      fill_value='extrapolate'
                     )
    wt = wfunc(mesh.node[:,0])
    wt[mesh.node[:,0]<min(tx)] = left_side_wt
    wt[mesh.node[:,0]>max(tx)] = right_side_wt
    delta = wt - mesh.node[:,2]
    nodal_pressures = delta*9810
    # nodal_pressures[nodal_pressures<0] = 0 
    nodal_pressures[np.isnan(nodal_pressures)] = 0 

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
        
    boundaries = {'top':top_node,
                  'bottom':base_node,
                  'left':left_node,
                  'right':right_node}
    
    pressures = {'nodal':nodal_pressures,
                 'base':base_pressure}
    
    return mesh, zone_flags, dx, pressures, boundaries 

def Sy_data(ncpu=1,show=False):
    """
    Read in hollin hill hydro and transfer resistance data 

    Parameters
    ----------
    tr : bool, optional
        DESCRIPTION. The default is True.
    ncpu : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
    hydro_data = pd.read_csv('Data/Rainfall/HydroForcing.csv')
    datadir = 'SyntheticStudy/DataErr'
    rfiles = [] # resistivity files 
    sdates = [] # survey dates 
    sdiy = [] 
    for f in sorted(os.listdir(datadir)):
        if f.endswith('.dat'):
            rfiles.append(f)
            dt = str2dat(f)
            sdates.append(str2dat(f))
            # compute day in year 
            year = dt.year 
            ref_dt = datetime(year-1,12,31)
            sdiy.append((dt - ref_dt).days)  
            
    # find times when resistivity and rainfall data are there
    nobs = len(hydro_data)
    rdates = [str2dat(hydro_data['DATE_TIME'][i]) for i in range(nobs)]
    hdiy = [0]*nobs 
    sflag = [False]*nobs # flag to determine if there is a resistivity survey 
    suidx = [-1]*nobs # corresponding survey index 
    for i in range(nobs):
        date = rdates[i]
        delta = [abs((date - sdate).days) for sdate in sdates]
        if min(delta) == 0:
            idx = np.argmin(delta)
            suidx[i] = idx 
            sflag[i+1] = True # need to add one here as the first time step in sutra is time 0 
        year = date.year 
        ref_dt = datetime(year-1,12,31)
        hdiy[i] = (date - ref_dt).days
        
    hydro_data['datetime'] = rdates 
    hydro_data['diy'] = hdiy 
    
    # sflag = [False] + sflag # add one here because first returned survey is time 0 ??
    survey_keys = np.arange(nobs)[np.array(sflag)==True]
        
    ## create a sequence of data and estimated errors 
    data_seq = pd.DataFrame()
    
    sequences = [] 
    def loop(i):
        f = rfiles[i]
        s = Survey(os.path.join(datadir,f),ftype='ProtocolDC',debug=False) # parse survey 
        # extract the abmn, data, error information and date 
        df = s.df[['a','b','m','n','resist']]
        ie = s.df['irecip'].values >= 0 # reciprocal + non-paired
        df = s.df[ie]
        df = df.rename(columns={'resist':'tr'})
        df['sidx'] = i 
        df['error'] = np.abs(df['tr'])*0.05
        sequence = df[['a','b','m','n']].values
        return df, sequence 
    
    if ncpu <= 1: 
        pout = [loop(i) for i in range(len(rfiles))]
    else: 
        nruns = len(rfiles)
        pout=Parallel(n_jobs=ncpu)(delayed(loop)(i) for i in range(nruns))
    
    for i in range(len(rfiles)):
        data_seq = pd.concat([data_seq,pout[i][0]])
        sequences.append(pout[i][1])
        
    if show: 
        fig, ax = plt.subplots(nrows=3) 
        miny = -max(hydro_data['PE'].values)
        maxy = max(hydro_data['PRECIP'].values) 
        for i in range(3):
            ax[i].bar(rdates,hydro_data['PRECIP'].values,color='b')
            ax[i].bar(rdates,-hydro_data['PE'].values,color='r')
            for j in range(len(sdates)):
                ax[i].plot([sdates[j],sdates[j]],[miny,maxy],c=(0.5,0.5,0.5,0.5))
            ax[i].set_ylabel('Rainfall / p.Et (mm/day)')
            limits = [datetime(2014+i,1,1),datetime(2014+i,12,31)]
            ax[i].set_xlim(limits)
            ax[i].set_ylim([miny,maxy])            
        ax[-1].set_xlabel('Datetime')
        
        return fig 
                
    return hydro_data, data_seq, sequences, survey_keys, rfiles, sdiy

def Sy_getElec():
    topo = pd.read_csv('Data/topoData/2016-01-08.csv')
    elec = pd.read_csv('Data/elecData/2016-01-08.csv')
    # doctor elec 
    elec['label'] = [str(elec['id'][i]) for i in range(len(elec))]
    elecx = elec['y'].values 
    elec.loc[:,'y'] = 0 
    elec.loc[:,'x'] = elecx 
    # fit a generic polyline to get a similar slope profile of the actual hill 
    tx = topo['y'].values # actually grid is located against the y axis, make this the local x coordinate in 2d 
    p = np.polyfit(tx,topo['z'].values,1) # fit elevation to along ground distance data 
    elec.loc[:,'z'] = np.polyval(p,elecx) # fit the modelled topography 
    return elec 

#%% prep rainfall 
def prepRainfall(dx,precip,pet,kc,numnp,ntimes, show=False):
    
    # deal with et 
    et = pet*kc
    effrain = precip - et 
    tdx = sum(dx)
    infil = effrain*tdx
    
    # now populate matrices
    fluidinp = np.zeros((ntimes, numnp),dtype=float)  # fluid input
    tempinp = np.zeros((ntimes, numnp),dtype=float)  # fluid temperature
    for i in range(numnp):
        m = dx[i]/tdx
        if i==0 or i==(numnp-1):
            m = 0.5*(dx[i]/tdx)
        fluidinp[:, i] = infil*m
        
    if show:
        fig, ax = plt.subplots()
        steps = np.arange(ntimes)
        ax.bar(steps,infil)

    return fluidinp, tempinp 

