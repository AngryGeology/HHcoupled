#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 15:41:24 2022
compute effective rainfall for cosmos data 
@author: jimmy
"""
import os 
from datetime import datetime 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from tqdm import tqdm 

prfile = '/home/jimmy/phd/Hollin_Hill/Data/cosmos/COSMOS-UK_HOLLN_HydroSoil_SH_2013-2018.csv'
pefile = '/home/jimmy/phd/Hollin_Hill/Data/cosmos/COSMOS-UK_HOLLN_HydroSoil_Daily_2013-2018.csv'

prdf = pd.read_csv(prfile)
pedf = pd.read_csv(pefile)

def str2dt(string):
    s1 = string.split()
    s2 = s1[0].split('-')
    s3 = s1[1].split(':')
    y = int(s2[0]) 
    M = int(s2[1])
    d = int(s2[2])
    h = int(s3[0])
    m = int(s3[1])
    return datetime(y,M,d,h,m)

def date2id(date):
    template = '{:4d}{:0>2d}{:0>2d}' # YYYYMMDDhhmmss  
    fid = template.format(date.year, # populate template with numbers 
                          date.month,
                          date.day)
    return int(fid)

def bisectionSearch(arr, var):
    """Efficent search algorithm for sorted array of postive ints 
    """
    L = 0
    n = len(arr)
    R = n-1
    while L <= R:
        m = int((L+R)/2)
        if arr[m]<var:
            L = m+1
        elif arr[m]>var:
            R = m-1
        else:
            return m
    return -1

prdates = [str2dt(prdf['DATE_TIME'][i]) for i in range(len(prdf))]
pedates = [str2dt(pedf['DATE_TIME'][i]) for i in range(len(pedf))]


# find each day we have PE data and get the precipitation  
pe = pedf['PE'].values 
pr = prdf['PRECIP'].values 
prmatched = np.array([0]*len(pe),dtype=float) # holds the values of precipitation matched to the potential evap 
effrain = np.array([0]*len(pe), dtype=float) # holds effective rainfall values 
filteridx = [False]*len(pe)#holds if in the desired time range 

for i in tqdm(range(len(pedf))):
    d0 = pedates[i] # get date of potential evap
    psum = 0 
    # perform some QC 
    if abs(pe[i]) > 9000:
        pe[i] = float('nan')
    for j in range(len(prdf)): 
        d1 = prdates[j]
        # check if same day as precip measurement, otherwise continue 
        if d1.year != d0.year:
            continue 
        if d1.month != d0.month:
            continue 
        if d1.day != d0.day:
            continue 
        if abs(pr[j]) > 9000:
            pr[j] = float('nan')
        psum += pr[j]
    
    prmatched[i] = psum
    effrain[i] = psum - pe[i]
    
    if d0 > datetime(2014,12,31) and d0 < datetime(2017,1,1):
        filteridx[i] = True 
    
#%% infall nan 
peids = np.array([date2id(pedates[i]) for i in range(len(pedf))])
nanidx = np.isnan(pe)
numidx = np.invert(nanidx)

effrain[nanidx] = np.interp(peids[nanidx],peids[numidx],effrain[numidx])

#%% plot 
fig, ax = plt.subplots()
ax.plot(pedates,prmatched)
ax.plot(pedates,effrain)


#%% EFF RAINFALL OUTPUT 
df = {'DATE_TIME':pedates,
      'PE':pe,
      'PRECIP':prmatched,
      'EFF_RAIN':effrain}

df = pd.DataFrame(df)
df.to_csv(os.path.join('Rainfall','COSMOS_2013-2018.csv'),index=False)

# now filter down to just 2015  and 2016 
df[filteridx].to_csv(os.path.join('Rainfall','COSMOS_2015-2016.csv'),index=False)
