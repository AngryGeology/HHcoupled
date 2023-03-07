#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 18:38:28 2023
Add error to simulated data files 
@author: jimmy
"""
import os 
import numpy as np 
import pandas as pd 
from RSUTRA.SUTRAhandler import protocolParser

datadir = 'SyntheticStudy/Data'
datadirwerr = 'SyntheticStudy/DataErr'

def to2protocol(fname, df):
    fh = open(fname,'w')
    template = '{:d}\t{:d}\t{:d}\t{:d}\t{:d}\t{:f}'
    fh.write('%i\n'%len(df))
    for i in range(len(df)):
        line = template.format(i+1, 
                               int(df['a'][i]),
                               int(df['b'][i]),
                               int(df['m'][i]),
                               int(df['n'][i]),
                               df['resist'][i])
        fh.write(line)
        fh.write('\n')
    fh.close()
    
if not os.path.exists(datadirwerr):
    os.mkdir(datadirwerr)

for f in os.listdir(datadir):
    if not f.endswith('.dat'):
        continue 
    _, df = protocolParser(os.path.join(datadir,f))
    tr = df['resist'].values 
    err = np.abs(tr)*0.02*np.random.randn(len(df))
    tre = tr + err 
    df.loc[:,'resist'] = tre 
    to2protocol(os.path.join(datadirwerr,f), df)

    
    
