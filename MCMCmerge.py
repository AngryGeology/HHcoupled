# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 13:23:54 2022
Merge MCMC logs 
@author: jimmy
"""
import os 
import pandas as pd 

dirname = 'Models/HydroMCMCmulti'
# dirname = 'SyntheticStudy/Models/MCMC'

entries = os.listdir(dirname)
chain_dirs = []
chain_ids = []
for e in entries: 
    if 'chain' in e and os.path.isdir(os.path.join(dirname,e)):
        chain_dirs.append(os.path.join(dirname,e))
        chain_ids.append(int(e.replace('chain','')))
        
master_df = pd.DataFrame()
for i,dname in enumerate(chain_dirs): 
    fpath = os.path.join(dname,'chainlog.csv')
    if not os.path.exists(fpath):
        continue 
    df = pd.read_csv(fpath)
    df['chain'] = chain_ids[i]
    master_df = pd.concat([master_df,df])
    
master_df.to_csv(os.path.join(dirname,'mergedMCMClog.csv'),index=False)
        