#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 11:31:51 2022
Compile line 3 resistivity data at Hollin Hill for 2015 to 2016  
@author: jimmy
"""
import os, sys  
from datetime import datetime
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
sys.path.append('/home/jimmy/phd/resipy/src')
from resipy.parsers import protocolParser

def bisectionSearch(arr, var):
    """Efficent search algorithm for sorted list of postive ints 
    
    Parameters
    -------------
    arr: list 
        sorted list 
    var: int
        item to be searched / indexed. If not found False is returned 
    """
    L = 0
    n = len(arr)
    R = n-1
    m = 0 
    while L <= R:
        m = int((L+R)/2)
        if arr[m]<var:
            L = m+1
        elif arr[m]>var:
            R = m-1
        else:
            return m
    return -1

def writeProtocol(fname,a,b,m,n,tr):
    fh = open(fname,'w')
    nmeas = len(a)
    fh.write('%i\n'%nmeas)
    for i in range(nmeas):
        line = '{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:f}\n'.format(i+1,
                                                             a[i],b[i],
                                                             m[i],n[i],
                                                             tr[i])
        fh.write(line)
    fh.close()
    return

dirname = '/home/jimmy/phd/Hollin_Hill/Data/compiled_protocols_remapped'
outputdir = 'resData'
locationfile = '/home/jimmy/phd/Hollin_Hill/Data/init_elec_locs.csv'


#%% which electrodes do i need for line 3? 

elec = pd.read_csv(locationfile)
# correct for negative id numbers 
neg_idx = elec['electrode'] < 0 
pos_idx = elec['electrode'] > 0 
elec['electrode'][neg_idx] += 81 
elec['electrode'][pos_idx] += 80

fig, ax = plt.subplots()
ax.scatter(elec['x'],elec['y'])

idx = (elec['x'] < 25) & (elec['x'] > 15)
ax.scatter(elec['x'][idx],elec['y'][idx])
elec2keep = sorted(elec['electrode'].values[idx])

#%% date limits 
d0 = datetime(2015,1,1)
d1 = datetime(2016,12,31)

#%% filter out files 
files = os.listdir(dirname)

for i, f in enumerate(files):
    if not f.endswith('.dat'): # ignore if not a dat file 
        continue 
    # parse datatime information 
    dinfo = f.split('.')[0].split('-')
    fdate = datetime(int(dinfo[0]),
                     int(dinfo[1]),
                     int(dinfo[2]))
    # skip if outside desired date range 
    if fdate < d0:
        continue
    if fdate > d1:
        continue 
    # parse protocal and filter out 3d lines 
    _, df = protocolParser(os.path.join(dirname,f))
    nmeas = len(df)
    i2keep = [False]*nmeas 
    for j in range(nmeas):
        abmn = [df['a'][j],df['b'][j],df['m'][j],df['n'][j]]
        c = 0 
        for e in abmn:
            if bisectionSearch(elec2keep,int(e)) > -1:
                c += 1 
        if c == 4:
            # keep the line 
            i2keep[j] = True 
    df = df[i2keep] # filter out measurements on other lines 
    # write to file 
    if len(df) < 100:# skip files with next to no measurements 
        continue 
    writeProtocol(os.path.join(outputdir,f),
                  np.asarray(df['a'].values,dtype=int),
                  np.asarray(df['b'].values,dtype=int),
                  np.asarray(df['m'].values,dtype=int),
                  np.asarray(df['n'].values,dtype=int),
                  np.asarray(df['resist'].values,dtype=float)) 
    