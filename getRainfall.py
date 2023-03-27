#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 15:41:24 2022
compute effective rainfall for cosmos data 
@author: jimmy
"""
import os 
from datetime import datetime,timedelta
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
vwc = pedf['COSMOS_VWC'].values 
prmatched = np.array([0]*len(pe),dtype=float) # holds the values of precipitation matched to the potential evap 
effrain = np.array([0]*len(pe), dtype=float) # holds effective rainfall values 
filteridx = [False]*len(pe)#holds if in the desired time range 

for i in tqdm(range(len(pedf))):
    d0 = pedates[i] # get date of potential evap
    psum = 0 
    # perform some QC 
    if abs(pe[i]) > 9000:
        pe[i] = 0 # float('nan')
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
            pr[j] = 0 # float('nan')
        psum += pr[j]
        # break 
    
    prmatched[i] = psum
    effrain[i] = psum - pe[i]
    
    if d0 > datetime(2014,6,30) and d0 < datetime(2017,1,1):
        filteridx[i] = True 
    
#%% infall nan 
peids = np.array([date2id(pedates[i]) for i in range(len(pedf))])
nanidx = np.isnan(pe)
numidx = np.invert(nanidx)

effrain[nanidx] = np.interp(peids[nanidx],peids[numidx],effrain[numidx])

#%% plot 
fig, ax = plt.subplots()
ax.bar(pedates,prmatched,color='b')
ax.bar(pedates,-pe,color='r')

#%% scan 7 day intervals to get Kc 
lookup={'low':0.8,'mod':0.6,'hi':0.4,'v.hi':0.3}
def getAllenKc(rval):
    if rval < 3:
        Kc = lookup['low']
    elif rval >= 3 and rval < 5:
        Kc = lookup['mod']
    elif rval >= 5 and rval < 7:
        Kc = lookup['hi']
    else:
        Kc = lookup['v.hi']
    return Kc

nobs = len(pedates)
rval = 0 
Kc = 0 
kcmatched = [0]*nobs 
wdmatched = [0]*nobs 

for i in range(nobs): 
    wdmatched[i] = pedates[i].weekday()
pstore = []
istore = []
c = 0 
i = 0 
while i<(nobs):
    for j in range(7):
        weekday = pedates[i].weekday() 
        pstore.append(prmatched[i]) 
        istore.append(i)
        # print(pstore)
        c+=1
        i+=1 
        if weekday == 6:
            break 
        if i == nobs:
            break 
    rval = max(pstore) # max rainfall event in week 
    Kc = getAllenKc(rval)
    for j in istore: 
        kcmatched[j] = Kc

    # reset 
    istore = [] 
    pstore = []
    c = 0 

axt = ax.twinx()
axt.plot(pedates, kcmatched,color=(0.2,0.2,0.2,0.5))
axt.set_ylabel('Crop coefficient (-)')


#%% EFF RAINFALL OUTPUT 
df = {'DATE_TIME':pedates,
      'PE':pe,
      'VWC':vwc, 
      'PRECIP':prmatched,
      'Kc':kcmatched, 
      'EFF_RAIN':effrain}

df = pd.DataFrame(df)
df.to_csv(os.path.join('Data','Rainfall','COSMOS_2013-2018.csv'),index=False)

# now filter down to just 2014  and 2016 
df_fil = df[filteridx].reset_index() 
df_fil.drop(columns='index',inplace=True)
    

#%% create a dummy dataset for 6 months at the beginning of 2014. 
pedates_warm = []
warm_date = datetime(2014,1,1)
warm_dates = [] 
delta = timedelta(days=1)
c = 0 
while warm_date<df_fil['DATE_TIME'][0]:
    warm_dates.append(warm_date)
    warm_date += delta 
    c += 1 
   
df_warm = {
    'DATE_TIME':warm_dates,
    'PE':np.zeros(c,dtype=float),
    'VWC':np.zeros(c,dtype=float), 
    'PRECIP':np.zeros(c,dtype=float)+0.4,
    'Kc':np.zeros(c,dtype=float), 
    'EFF_RAIN':np.zeros(c,dtype=float)}

df_warm = pd.DataFrame(df_warm)

#%% output 
master = pd.concat([df_warm,df_fil])
master.to_csv(os.path.join('Data','Rainfall','HydroForcing.csv'),index=False)
