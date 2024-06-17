# -*- coding: utf-8 -*-
"""
Created on Tue Nov 22 13:23:54 2022
Merge MCMC logs 
@author: jimmy
"""
import os 
import pandas as pd 

dirname = 'Models/_HydroMCMCmultiV2'
# dirname = 'SyntheticStudy/Models/MCMC'

def log2csv(fname): 
    fh = open(fname,'r')
    lines = fh.readlines()
    fh.close() 

    headers = ['run','Chi^2','Pt','mu','alpha','Accept','Stable','ar'] 
    data = {}

    for head in headers: 
        data[head] = [] 

    for i,line in enumerate(lines):
        if line.strip() == 'Initial trial is unstable solution! Try a different starting model':
            return {}
        
        line = line.strip() 
        if line == '':
            continue 
        if 'Run' != line.split()[0]: 
            continue 
        else:
            run = int(line.split()[-1])
        for j in range(10):
            c = i + 1 + j
            if c == len(lines):
                break 
            if 'Parameters:' in lines[c]:
                dataeter_line = lines[c+1].split('\t')
                parameters = {}
                for v in dataeter_line: 
                    if v == '\n':
                        break 
                    u = v.split('=')[0].strip()
                    s = u.split('_')[0]
                    n = int(u.split('_')[1])+1
                    p = '%s_%i'%(s,n)
                    a = v.split('=')[-1].strip()
                    parameters[p] = float(a)
        chi2 = 0 
        ar = 0 
        if run == 0:
            # get first pt value 
            Pt = float(lines[i+1].split('=')[-1])
            accept = True 
            mu = 0
            alpha = 0 
            stable = True 
            # continue 
        elif 'Proposed model is unstable!' in lines[i+2]:
            Pt = 0
            accept = False 
            mu = 0
            alpha = 0 
            stable = False 
        elif 'Proposed trial model is out of limits' in lines[i+1]:
            Pt = 0
            accept = False 
            mu = 0
            alpha = 0 
            stable = False 
        else:
            try: 
                Pt = float(lines[i+3].strip().split('=')[-1])
                if lines[i+7].strip().split('=')[-1] == 'True':
                    accept = True
                else:
                    accept = False 
                alpha_mu = lines[i+4].strip().split(',')
                mu = float(alpha_mu[1].split('=')[-1])
                alpha = float(alpha_mu[0].split('=')[-1])
                stable = True 
            except: 
                Pt = 0
                accept = False 
                mu = 0
                alpha = 0 
                stable = False 
                
        data['run'].append(run)
        data['Chi^2'].append(chi2)
        data['Pt'].append(Pt)
        data['mu'].append(mu)
        data['alpha'].append(alpha)
        data['Accept'].append(accept)
        data['Stable'].append(stable)
        data['ar'].append(ar)
        
        for key in parameters.keys():
            if key not in data.keys():
                data[key] = []
            data[key].append(parameters[key])
            
    return data 

entries = sorted(os.listdir(dirname))
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
        #attempt to recover data from the chain log 
        fpath2 = os.path.join(dname,'chain.log')
        df = pd.DataFrame(log2csv(fpath2))
        if len(df) == 0:
            continue 
    else:
        df = pd.read_csv(fpath)
    df['chain'] = chain_ids[i]
    master_df = pd.concat([master_df,df])
    
master_df.to_csv(os.path.join(dirname,'mergedMCMClog.csv'),index=False)

