#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  7 15:00:16 2021
SUTRA HANDLER   V2 
@author: jimmy
"""
import os, platform, warnings, shutil  
from subprocess import PIPE, Popen
import datetime as dt
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd 
from tqdm import tqdm
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
# custom module 
from swEstimator import estSwfromRainfall 

secinday = 24*60*60 

# function for writing log outputs to 
def logger(text,logf):
    fh = open(logf,'a')
    fh.write(text)
    fh.write('\n')
    fh.close() 
        

#%% matric potential functions 
def normparam(val,sat_val,res_val) -> float: 
    """ normalised parameter
    """
    val = np.array(val)
    return (val-res_val)/(sat_val-res_val)

def vgCurve(suction,ECres,ECsat,alpha,n,m=None):
    """van Genuchtan curve. 
    
    Parameters
    ------------
    suction: array, float
        Suction values in kPa. 
    vol_res: float
        residual value
    vol_sat: float
        saturated value
    alpha: float
        air entry pressure, can be fitted with a curve or found experimentally
    n: float
        van Genuchtan 'n' parameter. Not the same as archie's exponents. 
    m: float, optional
        van Genuchtan 'n' parameter. If left as none then m = 1 -1/n
    
    Returns
    -----------
    eff_vol: array, float
        normalised value 
    """
    if m is None:
        m = 1 - (1/n) # defualt definition of m
    
    dVol = ECsat - ECres # maximum change in water content
    denom = 1 + (alpha*suction)**n
    eff_vol = ECres + dVol/(denom**m)
    
    return eff_vol

def vgCurvefit(suction,res,sat,alpha,n,m=None):
    """van Genuchtan curve used in curve fitting. 
    
    Parameters
    ------------
    suction: array, float
        Suction values in kPa. 
    res: float
        residual value
    sat: float
        saturated value
    alpha: float
        air entry pressure, can be fitted with a curve or found experimentally
    n: float
        van Genuchtan 'n' parameter. Not the same as archie's exponents. 
    m: float, optional
        van Genuchtan 'n' parameter. If left as none then m = 1 -1/n
    
    Returns
    -----------
    val: array, float
        EC or water content value (not normalised)
    """
    if m is None:
        m = 1 - (1/n) # defualt definition of m
    
    dVol = sat - res # maximum change in water content value 
    denom = 1 + (alpha*suction)**n
    eff = res + dVol/(denom**m) # effective water content value 
    
    return (eff*dVol )+ res

def normVGcurve(suction,alpha,n,m=None):
    """Van Genuchtan curve but returns normalised parameters. 
    
    Parameters
    ------------
    suction: array, float
        Suction values in kPa. 
    alpha: float
        Air entry pressure, can be fitted with a curve or found experimentally
    n: float
        Van Genuchtan 'n' parameter. Not the same as archie's exponents. 
    m: float, optional
        Van Genuchtan 'm' parameter. If left as none then m = 1 -1/n
    
    Returns
    -----------
    norm_param: array, float
        normalised water content 
    """
    if m is None:
        m = 1 - (1/n) # defualt definition of m
    
    denom = 1 + (alpha*suction)**n
    norm_param = denom**-m
    return norm_param

#inverse of the above 
def invVGcurve(water_content,sat,res,alpha,n,m = None):
    """
    Inverse of van genutchen curve where we solve for suction not moisture content. 

    Parameters
    ----------
    water_content : float, nd array (of float)
        Water content as a fraction 
    sat : float 
        Saturated water content (normally 1).
    res : float 
        Residual water content.
    alpha : float 
        alpha parameters.
    n : float 
        Van genutchen n parameter
    m : float, optional
        M parameter. The default is None (computed as a function of n)

    Returns
    -------
    pressure: float, nd array 
        Matric potential (positive).

    """
    if m is None:
        m=1-(1/n)
        
    thetan = (water_content-res)/(sat-res)
    step1 = ((1/thetan)**(1/m)) - 1 
    step2 = step1**(1/n)
    
    return step2*(1/alpha)

def waxmanSmit(saturation,F,sigma_w,sigma_s): # convert moisture content to resistivity via waxman smits 
    """
    
    Parameters
    ----------
    saturation: float
        Saturation (in fraction)
    F: float
        Formation factor 
    sigma_w: float 
        Pore fluid conductivity 
    sigma_s: float 
        Grain surface conductivity 
    """    
    sigma = (1/F)*((sigma_w*saturation**2)+(sigma_s*saturation))
    
    return sigma 

def invWaxmanSmit(sigma,F,sigma_w,sigma_s):#working!
    """Convert true rock resistivity into a water saturation according waxman-smit
    model.
    
    Parameters
    ---------- 
    sigma: float 
        total rock conductivity 
    F: float
        Formation factor 
    sigma_w: float 
        Pore fluid conductivity 
    sigma_s: float 
        Grain surface conductivity 
    
    Returns 
    ---------- 
    Sw: float
        water (or conductive phase) saturation 
    """
    #minimization scheme start
    trial_Sw=0.5#trial Sw

    #first calculation
    calc_Sw=((sigma_w*(trial_Sw**2))-(sigma*F))/-sigma_s
    delta_Sw=calc_Sw-trial_Sw
    trial_Sw=trial_Sw+delta_Sw
    #minmisation while loop 
    count=0
    while abs(delta_Sw)>0.0001:
        calc_Sw=((sigma_w*(trial_Sw**2))-(sigma*F))/-sigma_s
        delta_Sw=calc_Sw-trial_Sw
        trial_Sw=trial_Sw+delta_Sw
        count+=1
        if count>30:
            print('warning')
            break
    return trial_Sw

#%% statistics / utility functions 
def rmse(d0,dm):
    N = len(d0)
    diff = d0 - dm
    sumup = np.sum(diff**2)
    return np.sqrt(sumup/N)

def chi2(meas_errors,residuals):
    n = len(residuals)
    xsum = 0 
    for i in range(n):
        x = (residuals[i]**2) / (meas_errors[i]**2) 
        xsum += x 
    return xsum/n   

def chi2_log(meas_errors,residuals):
    n = len(residuals)
    r = np.matrix(np.log10(np.abs(residuals))).T
    W = np.matrix(np.diag(1/np.log10(meas_errors**2)))
    X_2 = r.T*W.T*W*r
    X2 = float(X_2)*(1/n)
    return X2     

def lfunc(meas_errors,residuals):
    comp1 = -0.5*np.log(np.pi*2*np.abs(meas_errors))
    comp2 = -0.5*((residuals**2) / (meas_errors**2))
    return -np.sum(comp1 + comp2)

def llfunc(meas_errors, residuals): 
    n = len(residuals)
    c = -0.5*np.log(2*np.pi) # constant 
    lsum = 0 
    for i in range(n):
        std = abs(meas_errors[i]) 
        var = meas_errors[i]**2
        res2 = residuals[i]**2 
        # li = np.exp(-res2/(2*var)) / np.sqrt(np.pi*2*var)
        lli = c - np.log(std) - res2/(2*var)
        lsum += lli 
    
    return lsum 

def normLike(meas_errors,residuals):
    """
    Compute the normalised likelihood ratio for a model. 

    Parameters
    ----------
    meas_errors : array like 
        Data errors.
    residuals : array like 
        d0 - dm.

    Returns
    -------
    normalised_likelihood: float 
        sum of normalised likelihoods over the number of measurements. 

    """
    n = len(residuals)
    psum = 0 
    c = -0.5*np.log(2*np.pi) # constant 
    for i in range(n):
        std = abs(meas_errors[i]) # standard deviation of point 
        var = meas_errors[i]**2 # varaince of point 
        res2 = residuals[i]**2 # square of residual 
        if std == 0 and res2==0: # in the case of a perfect data point normalised LR is 1 
            psum += 1 # so add 1 to psum and move on 
            continue 
        
        lli = c - np.log(std) - res2/(2*var) # log likelihood 
        lmle = c - np.log(std) # log max likelihood estimate 
        llr = lli - lmle # log likelihood ratio 
        lr = np.exp(llr) # convert back into normal space 
        psum += lr # add to total probability 
        
    return psum/n # normalise to between 0 and 1 

def convertTimeUnits(x,unit='sec'):
    if unit =='sec':
        return x 
    elif unit =='min':
        return x/60 
    elif unit =='hour':
        return x/(60*60)
    elif unit =='day':
        return x/(24*60*60)
    else:
        return x 
    
#%% return monte-carlo values 
def giveValues(v0,v1,n,dist='uniform'):
    if dist == 'ordered':
        return np.linspace(v0,v1,n)
    elif dist == 'logordered':
        return 10**np.linspace(np.log10(v0),np.log10(v1),n)
    elif dist =='uniform':
        return np.random.uniform(v0,v1,n)
    elif dist == 'loguniform':
        return 10**np.random.uniform(np.log10(v0),np.log10(v1),n)
    elif dist == 'normal':
        loc = (v0+v1)/2 
        scale = (v1-v0)/2 
        return np.random.normal(loc,scale,n)
    else: # shouldnt happen 
        raise Exception('Distribution type is unknown!')
    
def stepWalk(iarg, size, dist='normal'):
    """
    Walk a 'random' step

    Parameters
    ----------
    iarg : float
        Starting value 
    size : float 
        Size of normal step (if lognormal then size is in terms of log space)
    dist : str, optional
        Type of scale, use lognormal to walk randomly in log space. 
        The default is 'normal'.

    Raises
    ------
    Exception
        If dist is of unkown type. 

    Returns
    -------
    u: float 
        New value for random walk 

    """
    step = size * np.random.randn(1)[0]
    if dist == 'normal':
        return iarg + step # do random walk 
    elif dist == 'lognormal':
        u = np.log10(iarg) + step 
        return 10**u 
    else: # shouldnt happen 
        raise Exception('Distribution type is unknown!')
        
def checkRange(v,v0,v1):
    """
    Return False if value is outside of range (v0 to v1) or negative. Else
    return True. 

    """
    if v < v0: 
        return False 
    elif v > v1:
        return False 
    elif v < 0: 
        return False 
    else:
        return True 

#%% read and write functions 
def readNod(fname): 
    # open file 
    fh = open(fname,'r')
    
    #header lines 
    for i in range(3):
        _ = fh.readline()
    
    # first find the number of nodes 
    line = fh.readline()
    info = line.split()
    numnp = None 
    for i in range(len(info)):
        if 'Nodes' in info[i]:
            numnp = int(info[i-1])
    if numnp is None:
        raise ValueError("Couldn't parse the number of nodes, aborting...")
    
    #find nth time step 
    n = 0 
    c = 100 # fail safe if cannot find the next timelapse array 
    data = {}
    breakout = False 
    while True:
        line = fh.readline()
        c = 0 
        while 'TIME STEP' not in line:
            line = fh.readline()
            c+=1 
            if c==100 and n>0:
                breakout = True 
                break
        if breakout:
            break 
            
        #time step info 
        info = line.replace('#','').split()
        try:
            _ = float(info[4])
            _ = float(info[7])
        except:
            info[4] = 'nan'
            info[7] = 'nan'
            
        stepinfo = {'TIME STEP':int(info[2]),'Duration':float(info[4]),
                    'Time':float(info[7])}
        
        # get names of colums 
        _ = fh.readline()
        names_line = fh.readline().replace('#','')
        columns = names_line.split()
        ncol = len(columns)
        
        #create a structure for storing information 
        step = {}
        for column in columns:
            step[column] = []
            
        for i in range(numnp):
            values = fh.readline().split() # values for each node at step 
            for i in range(ncol):
                column = columns[i]
                if values[i] == 'NaN': # catch nan
                    values[i] = 'nan'
                if column == 'Node':
                    step[column].append(int(values[i]))
                else:
                    step[column].append(float(values[i]))
        
        data['step%iinfo'%(n)]=stepinfo
        data['step%i'%(n)]=step
        n+=1 
        
        if fh.readline() == '':# or c==100:
            break 
        
    fh.close()
    return data,n 

def readMassBlnc(fname):
    fh = open(fname,'r')
    # tline = 'T E M P O R A L   C O N T R O L   A N D   S O L U T I O N   C Y C L I N G   D A T A'
    tline = 'MAXIMUM NUMBER OF TIMES AFTER INITIAL TIME'
    fline = 'F L U I D   M A S S   B U D G E T' # fluid line 
    # find number of time steps 
    line = fh.readline()
    c = 0 
    ntime = 0 
    while True:
        if tline in line: 
            ntime = int(line.strip().split()[0])
            line = fh.readline()
            break 
        if c>10000:
            raise Exception('Couldnt find number of time steps')
            break 
        line = fh.readline()
        c+=1 
    
    times = [0]*ntime 
    massinPc = [0]*ntime # mass in due to change in pressure  
    massinCc = [0]*ntime # mass in due to change in concentration  
    massinSs = [0]*ntime # mass in at source / sink nodes  
    massinPn = [0]*ntime # mass in pressure nodes  
    massinGn = [0]*ntime # mass in at generalised flow nodes   
    
    massotPc = [0]*ntime # mass out due to change in pressure  
    massotCc = [0]*ntime # mass out due to change in concentration  
    massotSs = [0]*ntime # mass out at source / sink nodes  
    massotPn = [0]*ntime # mass out pressure nodes  
    massotGn = [0]*ntime # mass out at generalised flow nodes   
    i=0 
    break_condition = 0 
    while True: 
        if fline in line: 
            tmp = line.strip().split()
            ti = tmp.index('STEP')+1
            times[i] = int(tmp[ti].replace(',',''))
            # now to parse mass in and out 
            for j in range(20):
                line = fh.readline()
                tmp = line.strip().split() 
                if 'PRESSURE CHANGE' in line:
                    massinPc[i] = float(tmp[-3])
                    massotPc[i] = float(tmp[-2])
                if 'CONCENTRATION CHANGE' in line:
                    massinCc[i] = float(tmp[-3])
                    massotCc[i] = float(tmp[-2])
                if 'SOURCES AND SINKS' in line:
                    massinSs[i] = float(tmp[-3])
                    massotSs[i] = float(tmp[-2])
                if 'SPECIFIED P NODES' in line:
                    massinPn[i] = float(tmp[-3])
                    massotPn[i] = float(tmp[-2])
                if 'GEN.-FLOW NODES' in line:
                    massinGn[i] = float(tmp[-3])
                    massotGn[i] = float(tmp[-2])
            i+=1 
        if line =='':
            break_condition += 1 
        else:
            break_condition = 0
        if break_condition > 10000:
            break 
        line = fh.readline()
    fh.close() 
    
    # output for parsing 
    output = {
        'times':times,
        'massinPc':massinPc,  
        'massinCc':massinCc,  
        'massinSs':massinSs,  
        'massinPn':massinPn,  
        'massinGn':massinGn,   
        'massotPc':massotPc,  
        'massotCc':massotCc,  
        'massotSs':massotSs,  
        'massotPn':massotPn,  
        'massotGn':massotGn,
        }
    return output  

#%% run sutra (in parallel)
def doSUTRArun(wd,execpath=None,return_data=True): # single run of sutra 
    if wd is None:
        raise Exception('Working directory needs to be set')
    if execpath is None:
        raise Exception('Executable path needs to be set')
    
    if platform.system() == "Windows":#command line input will vary slighty by system 
        cmd_line = [execpath]
    elif platform.system() == 'Linux':
        cmd_line = [execpath] # using linux version if avialable (can be more performant)
        if '.exe' in execpath: # assume its a windows executable 
            cmd_line.insert(0,'wine') # use wine to run instead 
    else:
        raise OSError('Unsupported operating system') # if this even possible? BSD maybe. 

    ERROR_FLAG = False # changes to true if sutra causes an error, and results will not be read 

    p = Popen(cmd_line, cwd=wd, stdout=PIPE, stderr=PIPE, shell=False)#run gmsh with ouput displayed in console
    while p.poll() is None:
        line = p.stdout.readline().rstrip()
        if line.decode('utf-8') != '':
            # dump(line.decode('utf-8'))
            if 'ERROR' in line.decode('utf-8'):
                ERROR_FLAG = True 
    
    if ERROR_FLAG:
        if return_data: 
            return {},0 # return an empty result
        else:
            return 
    
    # now read in result as a dictionary 
    if return_data: 
        files = os.listdir(wd)
        fname = '.nod'
        for f in files:
            if f.endswith('.nod'):
                fname = os.path.join(wd,f) #  we have found the .nod file 
                break 
        data, n = readNod(fname) 
        
        return data,n 

#%% run R2/R3t (in parrallel probably)
# protocol parser for 2D/3D and DC/IP
def protocolParser(fname, ip=False, fwd=False):
    """
    <type>     <ncol>
    DC 2D         6
    DC 2D + err   7
    DC 2D + fwd   7
    IP 2D         7
    IP 2D + err   9
    IP 2D + fwd   8
    DC 3D         10
    DC 3D + err   11
    DC 3D + fwd   11
    IP 3D         11
    IP 3D + err   13
    IP 3D + fwd   12
    
    format:
    R2   :5,7,7,7,7,20,15
    cR2  :4,4,4,4,4,16,14,16
    R3t  :5,7,4,7,4,7,4,7,4,20,15
    cR3t :5,7,4,7,4,7,4,7,4,20,15,15
    """
    # method 1: np.genfromtxt and fallback to pd.read_fwf
    try:
        # this should work in most cases when there is no large numbers
        # that mask the space between columns
        x = np.genfromtxt(fname, skip_header=1) # we don't know if it's tab or white-space
    except Exception as e: # if no space between columns (because of big numbers or so, we fall back)
        # more robust but not flexible to other format, this case should only be met in fwd
        # we hope to be able to determine ncols from genfromtxt()
        a = np.genfromtxt(fname, skip_header=1, max_rows=2)
        threed = a.shape[1] >= 10
        if threed is False and ip is False: # 2D DC
            x = pd.read_fwf(fname, skiprows=1, header=None, widths=[5,7,7,7,7,20,15]).values
        elif threed is False and ip is True: # 2D IP
            x = pd.read_fwf(fname, skiprows=1, header=None, widths=[4,4,4,4,4,16,14,16]).values
        elif threed is True and ip is False: # 3D DC
            x = pd.read_fwf(fname, skiprows=1, header=None, widths=[5,7,4,7,4,7,4,7,4,20,15]).values
        elif threed is True and ip is True: # 3D IP
            x = pd.read_fwf(fname, skiprows=1, header=None, widths=[5,7,4,7,4,7,4,7,4,20,15,15]).values
        else:
            raise ValueError('protocolParser Error:', e)
    
    if len(x.shape) == 1: # a single quadrupole
        x = x[None,:]
    if fwd:
        x = x[:,:-1] # discard last column as it is appRes
    if ip:
        colnames3d = np.array(['index','sa','a','sb','b','sm', 'm','sn','n','resist','ip','magErr','phiErr'])
        colnames2d = np.array(['index','a','b','m','n','resist','ip','magErr','phiErr'])
    else:
        colnames3d = np.array(['index','sa','a','sb','b','sm', 'm','sn','n','resist','magErr'])
        colnames2d = np.array(['index','a','b','m','n','resist','magErr'])
    ncol = x.shape[1]
    if ncol <= len(colnames2d): # it's a 2D survey
        colnames = colnames2d[:ncol]
    else: # it's a 3D survey
        colnames = colnames3d[:ncol]
        
    df = pd.DataFrame(x, columns=colnames)
    df = df.astype({'a':int, 'b':int, 'm':int, 'n':int})
    if 'sa' in df.columns:
        df = df.astype({'sa':int, 'sb':int, 'sm':int, 'sn':int})
        elec = np.vstack([df[['sa','a']].values, df[['sb','b']].values,
                          df[['sm','m']].values, df[['sn','n']].values])
        uelec = np.unique(elec, axis=0)
        dfelec = pd.DataFrame(uelec, columns=['string', 'elec'])
        dfelec = dfelec.sort_values(by=['string','elec']).reset_index(drop=True)
        dfelec['label'] = dfelec['string'].astype(str) + ' ' + dfelec['elec'].astype(str)
        dfelec = dfelec.drop(['string', 'elec'], axis=1)
        df['a2'] = df['sa'].astype(str) + ' ' + df['a'].astype(str)
        df['b2'] = df['sb'].astype(str) + ' ' + df['b'].astype(str)
        df['m2'] = df['sm'].astype(str) + ' ' + df['m'].astype(str)
        df['n2'] = df['sn'].astype(str) + ' ' + df['n'].astype(str)
        df = df.drop(['a','b','m','n','sa','sb','sm','sn'], axis=1)
        df = df.rename(columns={'a2':'a','b2':'b','m2':'m','n2':'n'})
    else:
        uelec = np.unique(df[['a','b','m','n']].values.flatten()).astype(int)
        dfelec = pd.DataFrame(uelec, columns=['label'])
        dfelec = dfelec.astype({'label': str})
    dfelec['x'] = np.arange(dfelec.shape[0])
    dfelec['y'] = 0
    dfelec['z'] = 0
    dfelec['buried'] = False
    dfelec['remote'] = False
    df = df.astype({'a':str, 'b':str, 'm':str, 'n':str})
    if 'ip' not in df.columns:
        df['ip'] = np.nan
    return dfelec, df

def runR2runs(wd,execpath=None,surrogate='resistivity.dat',return_data=False,
              clean=True):
    """
    Run R2 (or R3t) for multiple forward runs

    Parameters
    ----------
    wd : str
        Path to R2 working directory.
    execpath : str
        Path to executable file, needs to be defined. The default is None.
    surrogate: str, optional 
        Name of surrogate resistivity file which detials the starting resistivities 
    
    Raises
    ------
    Exception
        DESCRIPTION.
    OSError
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.
    TYPE
        DESCRIPTION.

    """
    if wd is None:
        raise Exception('Working directory needs to be set')
    if execpath is None:
        raise Exception('Executable path needs to be set')
    
    if platform.system() == "Windows":#command line input will vary slighty by system 
        cmd_line = [execpath]
    
    elif platform.system() == 'Linux':
        cmd_line = [execpath] # using linux version if avialable (can be more performant)
        if '.exe' in execpath: # assume its a windows executable 
            cmd_line.insert(0,'wine') # use wine to run instead 
    else:
        raise OSError('Unsupported operating system') # if this even possible? BSD maybe. 
    
    ERROR_FLAG = False 
    
    entries = os.listdir(wd)
    steps = []
    n = 0 
    for e in entries:
        # determine if resistivity vector 
        if 'step' in e:
            steps.append(e)
            n+=1 
            
    data = {}
    for i,step in enumerate(sorted(steps)):
        shutil.copy(os.path.join(wd,step),
                    os.path.join(wd,surrogate))
        shutil.copy(os.path.join(wd,'protocol%i.dat'%i),
                    os.path.join(wd,'protocol.dat'))
        
        p = Popen(cmd_line, cwd=wd, stdout=PIPE, stderr=PIPE, shell=False)#run R2 with ouput displayed in console
        while p.poll() is None and not ERROR_FLAG:
            line = p.stdout.readline().rstrip()
            if line.decode('utf-8') != '':
                # dump(line.decode('utf-8'))
                if 'ERROR' in line.decode('utf-8'):
                    ERROR_FLAG = True 
        if ERROR_FLAG:
            return {},0 # return an empty result if error thrown, otherwise ... 
        
        # wait till finish and parse result
        fwd_file = 'R2_forward.dat'
        if 'R3t.exe' in execpath:
            fwd_file = 'R3t_forward.dat'
        
        shutil.copy(os.path.join(wd,fwd_file),
                    os.path.join(wd,'forward%i.dat'%i))
        
        if clean:
            # clean up excess files 
            os.remove(os.path.join(wd,step))
            os.remove(os.path.join(wd,'protocol%i.dat'%i))
            
        if return_data:
            elec,df = protocolParser(os.path.join(wd,fwd_file))
            data[i] = df['resist'].values 
    
    if return_data: 
        return data,n 

#%% define a material for sutra 
class material:
    convert_cons= {'u':1.307e-3, #kg/ms  
                   'p':1000, #kg/ms^3 
                   'g':9.81} #m/s^2 
    
    def __init__(self, Ksat, theta_res, theta_sat, alpha, vn, 
                 res = None, sat = None, name='Material', perm=None):
        self.name = name # name of material 
        
        # hydrualic properties 
        self.K = Ksat # saturated hydrualic conductivity 
        self.sat = sat # saturated saturation value (normally 1)
        self.res = res # residual saturation value 
        self.theta_sat = theta_sat # saturated volumetric water content value (same as porosity)
        self.theta_res = theta_res # residual volumetric water content value 
        self.perm = perm # permeability, otherwise estimated from hydrualic conductivity assuming its in m/day 
        # van genutchen parameters 
        self.alpha = alpha 
        self.vn = vn 
        
        # petro physical translation 
        self.petro_func = None 
        self.shallow_threshold = 1.0
        
        # zone properties 
        self.zone = 0 # this will get reassigned when interfacing SUTRA handler 
        
        # MC parameter handling 
        self.MCparam = {} 
        self.niter = None # number of iterations / model runs 
        self.pdist = 'normal'
        self.MCpdist = {}
        
        # Sw estimation 
        self.thick = 1 
        
        if self.sat is None: 
            self.sat = 1 
        
        if self.res is None: 
            self.res = self.theta_res/self.theta_sat
            
        if self.perm is None: 
            self.convertk2K()
    
    def convertk2K(self,time_unit='day', space_unit='m', return_value = False):
        """
        Convert hydrualic conductivity into permeability (todo)

        Parameters
        ----------
        time_unit : TYPE
            DESCRIPTION.
        space_unit : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        # convert hydrualic conductivity (m/day) to permeability (m^2)
        # the following values are for water 
        u = self.convert_cons['u']
        p = self.convert_cons['p']
        g = self.convert_cons['g']
        
        # convert K to m/s from m/day 
        k = self.K/secinday 
        perm = (k*u)/(p*g)
        self.perm = perm 
        
        if 'K' in self.MCparam.keys(): 
            ks = self.MCparam['K']
            perms = [(_k*u)/(p*g) for _k in ks]
            if 'K' in self.MCpdist.keys():
                pdist = self.MCpdist['K']
                del self.MCpdist['K']
            else:
                pdist = 'normal'
            self.MCpdist['k'] = pdist 
            self.MCparam['k'] = perms
            if 'log' in pdist: # middle param value in terms of log space 
                # so grab K distribution size before deleting it! 
                self.MCparam['k'][1] = self.MCparam['K'][1]
            del self.MCparam['K'] # remove conductivity key (so that sutra handler works)
        
        if return_value:  
            return  perm # in m^2 
        
    def setPetro(self,func):
        """
        Set petrophysical transfer function, takes saturation as only input. 

        Returns
        -------
        None.

        """
        if not callable(func):
            raise Exception("Input must be a callable object")
        # if not isinstance(param,dict):
        #     raise Exception('Param input must be a dictionary')
        
        self.petro_func = func 
        # self.petro_param = param 
        
    def setPetroFuncs(self,func0,func1):
        """
        Set petrophysical transfer functions for more than one depth, 
        takes saturation as only input. 

        Returns
        -------
        None.

        """
        self.petro_func = func1 
        self.petro_func_shallow = func0 
        self.petro_func_deep = func1 
        
    def petro(self,S,d=None):
        """
        Solve resistivity in terms of saturation 

        Parameters
        ----------
        S : float array 
            Saturation values.

        Returns
        -------
        Resistivity: float array 
            Resistivity values 

        """
        if d is None: 
            return self.petro_func(S)
        res = np.zeros_like(S)
        sidx = d < self.shallow_threshold
        didx = np.invert(sidx)
        res[didx]=self.petro_func_deep(S[didx])
        res[sidx]=self.petro_func_shallow(S[sidx])
        return res 
    
    def pres(self,S):
        """
        Solve pressure in terms of pressure 

        Parameters
        ----------
        S : float array 
            Saturation values. 
        Returns
        -------
        P: float array 
            Pressure values 

        """
        P = invVGcurve(S,self.sat,self.res,self.alpha,self.vn)
        return P 
    
    def setMCparam(self,param, pdist = None):
        """
        Set Monte Carlo parameters 

        Returns
        -------
        None.

        """
        if not isinstance(param,dict):
            raise Exception('Param input must be a dictionary')
            
        lengths = [0]*len(param.keys())
        for i,key in enumerate(param.keys()):
            lengths[i] = len(param[key])
        
        if any(np.array(lengths) != max(lengths)):
            raise Exception('All parameters should be the same length!')
            
        self.niter = max(lengths)
        self.MCparam = param 
        self.MCpdist = {} 
        if pdist is None:
            for key in self.MCparam.keys():
                self.MCpdist[key] = 'normal'
        else:
            self.MCpdist = pdist 
        
        if 'K' in self.MCparam.keys(): 
            self.convertk2K()
        
    def estSw(self,Pr,Et,ts,start_sw=None):
        """
        Estimate Sw from rainfall  

        Parameters
        ----------
        Pr : Array like 
            Rainfall in m/s 
        Et : Array like 
            Evapotranspiration in m/s 
        ts : Array like 
            Time steps in seconds 

        Returns
        -------
        sw : Array 
            Saturation fraction estimation.
        """
        maxSuz = self.thick*self.theta_sat
        Suz0 = maxSuz*self.res 
        if not start_sw is None:
            Suz0 = maxSuz*start_sw 
        sw = estSwfromRainfall(
            Pr,Et,ts,self.sat,self.res,
            self.theta_sat,self.alpha*1e-3,self.vn,
            self.perm, self.convert_cons['u'], 
            ifac=24,maxSuz=maxSuz,Suz0=Suz0
            ) 
        return sw 
        
    
#%% master class for handling sutra 
class handler: 
    
    # parameter arguments now handled at a material level 
    # parg0 = {} # lower bound for parameter ranges in monte carlo searches 
    # parg1 = {} # upper bound for parameter ranges in monte carlo searches 
    # pargs = {} # starting place for mcmc approach 
    # psize = {} # step sizes for mc approach 
    
    pdist = {'k':'uniform',
             'theta':'uniform', 
             'res':'uniform',
             'sat':'uniform',
             'alpha':'uniform',
             'vn':'normal'} # types of distributions (pdfs) used for the monte carlo modelling 

    setupinp = None # store for initial model setup (populated by setupInp) 
    
    pdirs = [] # directorys for running parallel runs 
    nruns = 0 
    
    template = 'r{:0>5d}' # template for parallel runs 
    
    def __init__(self,dname='SUTRAwd',name='sutra', title = None, 
                 subtitle = None, tlength=3600, ifac=5, ncpu=6, iobs=None,
                 saturated = False, flow = 'transient', transport='transient',
                 sim_type='solute', cold = True):
        if not os.path.exists(dname):
            os.mkdir(dname)
        self.dname = dname 
        self.name = name 
        self.title = title 
        self.subtitle = subtitle 
        if self.title is None: # set run title 
            self.title = 'SUTRA run for %s'%name 
        if self.subtitle is None: # if no subtitle just assign the date 
            self.subtitle = str(dt.datetime.now())
        self.tlength = tlength # length of each time step (inseconds)
        self.ifac = ifac # factor which to break down, internal time cycling in sutra 
        self.maxIter = 500 
        if iobs == None: # set the factor at which to make observations 
            self.iobs = self.ifac 
        else:
            self.iobs = iobs 
            
        self.logf = None # logging file (set during mcmc approach)
            
        # model parameters 
        self.param = {'res':[],
                      'sat':[],
                      'alpha':[],
                      'vn':[]} # van genutchen parameter arrays (populated with default values)
    
        self.waxman = {'F':None,
                       'sigma_w':None,
                       'sigma_s':None}
        
        # model type 
        self.sim_type = sim_type
        self.saturated = saturated # True if only saturated flow, otherwise unsaturated 
        self.flow = flow # flow type, steady or transient
        self.transport = transport # transport type 
        self.cold = cold # set to cold for initial conditions to be read from previous model 
        self.rpmax = 1000
        self.rumax = 1000 
        self.drainage = 1e-6 
        self.pressure = None 
        
        # 'setable' attributes 
        self.mesh = None 
        self.nzones = 0 
        self.execpath = None 
        self.closeFigs = True 
        self.materials = [] 
        
        # attributes used in plotting 
        self.attribute = 'Saturation'
        self.vmin = -0.01
        self.vmax = 1.01 
        self.xlim = None 
        self.ylim = None 
        self.vlim = None 
        
        # results variables 
        self.nodResult = {} 
        self.nodResultMulti = {} 
        self.resultNsteps = 0
        self.resFwdMdls = {} # holds forward model results from R family of codes 
        self.massBalance = None 
        
        # set the number of cpus to use in parallised functions 
        self.ncpu = ncpu 
        
        # mcmc coupling parameters relating to using R2 
        self.project = None 
        self.seqs = None 
        self.tr = None 
        self.error = None  
        self.write2in = None  
        self.survey_keys = None 
        self.tfunc = None  
        self.diy = []
        
    def setDname(self,dname):
        self.dname = dname 
        if not os.path.exists(dname):
            os.mkdir(dname)
            
    # THESE SET FUNCTIONS MUST BE RUN FIRST 
    def setMesh(self,mesh):
        """
        Set mesh object for sutra handler 

        Parameters
        ----------
        mesh : class 
            Mesh class of ResIPy.meshTools.

        """
        self.mesh = mesh 
        self.mesh.df = pd.DataFrame()
        self.mesh.ptdf = pd.DataFrame()
        self.mesh.cellCentres()
        
        # set up necessary mesh parameters 
        self.mesh.df['zone'] = np.zeros(mesh.numel,dtype=int) # zone by element basis 
        self.mesh.ptdf['zone'] = np.zeros(mesh.numnp,dtype=int) # zone by node basis 
        self.mesh.df['perm'] = np.zeros(mesh.numel,dtype=float) # permeability by element basis 
        self.mesh.ptdf['por'] = np.zeros(mesh.numnp,dtype=float) # porosity by node basis 
        
        # get mesh node depths 
        sx, sz = mesh.extractSurface(False,False)
        cell_depths = mesh.computeElmDepth()
        node_depths = np.interp(mesh.node[:,0], sx, sz) - mesh.node[:,2]
        self.mesh.ptdf['depth'] = node_depths 
        self.mesh.df['depth'] = cell_depths
    
        
    def getSurfaceArea(self):
        """
        Gets the surface area multiplier to apply to source / sink nodes 

        Returns
        -------
        None.

        """
        if self.mesh is None: 
            raise Exception ('mesh class has not been set!')
        # assume 2d for now 
        X,Z = self.mesh.extractSurface() 
        dx = np.zeros(len(X))
        dx[0] = np.abs(X[1] - X[0])
        for i in range(1,len(X)):
            dx[i] = np.abs(X[i] - X[i-1])
        
        return dx 
       
    def getDepths(self):
        """
        Compute cell depths for a 2D problem 

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if self.mesh is None: 
            raise Exception ('mesh class has not been set!')
            
        depths = self.mesh.computeElmDepth()    
        depths_node  = self.mesh.ptdf['depths'].values 
        
        return depths, depths_node
    
    def addMaterial(self, material, zone_ids=None):
        if self.mesh is None: 
            raise Exception ('mesh class has not been set!')
        
        if material.perm is None: 
            raise Exception ('Material has no permeability!')
            
        if material.theta_sat is not None: 
            if 'theta_sat' not in self.param.keys():
                self.param['theta_sat'] = []
        if material.theta_res is not None: 
            if 'theta_res' not in self.param.keys():
                self.param['theta_res'] = []
            
        self.param['alpha'].append(material.alpha)
        self.param['theta_sat'].append(material.theta_sat)
        self.param['theta_res'].append(material.theta_res)
        self.param['res'].append(material.res)
        self.param['sat'].append(material.sat)
        self.param['vn'].append(material.vn)
        
        self.nzones+=1 
        self.materials.append(material)

        if self.nzones == 1: 
            #then its the first material added 
            self.mesh.df.loc[:,'zone'] = 1 
            self.mesh.ptdf.loc[:,'zone'] = 1 
            self.mesh.df.loc[:,'perm'] = material.perm  
            self.mesh.df.loc[:,'Ksat'] = material.K 
            self.mesh.ptdf.loc[:,'por'] = material.theta_sat  
            material.zone = self.nzones 
        else: 
            # then materials have already been added 
            self.mesh.df.loc[zone_ids,'zone'] = self.nzones 
            # lookup node zone using the mesh connection matrix 
            for i in range(self.mesh.numel):
                for j in range(4):
                    a = self.mesh.connection[i,j]
                    if self.mesh.ptdf['zone'][a] == 1: # if not yet set then assign zone 
                        self.mesh.ptdf.loc[a,'zone'] = self.mesh.df['zone'][i]
            self.mesh.df.loc[zone_ids,'perm'] = material.perm
            self.mesh.df.loc[zone_ids,'Ksat'] = material.K 
            node_idx = self.mesh.ptdf['zone'] == self.nzones
            self.mesh.ptdf.loc[node_idx,'por'] = material.theta_sat  
            material.zone = self.nzones
    
    
    # LEGACY WAY TO SET MESH PARAMETERS     
    def setZone(self,val=1):
        if self.mesh is None: 
            raise Exception ('mesh class has not been set!')
        if isinstance(val,float) or isinstance(val,int):
            self.mesh.ptdf['zone'] = [int(val)]*self.mesh.numnp 
        else:
            if self.mesh.numnp == len(val):    
                self.mesh.ptdf['zone'] = val 
            else:
                raise ValueError('mis match in node array lengths')
        self.nzones = np.unique(self.mesh.ptdf['zone'].values).shape[0] # get the number of zones 
        # set the zones for elements as well 
        self.mesh.node2ElemAttr(self.mesh.ptdf['zone'].values,'zone')
        #force values to be int 
        self.mesh.df['zone'] = np.asarray(np.round(self.mesh.df['zone']),dtype=int)
        
    
    def setPor(self,val=0.3):
        if self.mesh is None or self.nzones is None: 
            raise Exception ('mesh and zones need to be set first')
        if isinstance(val,float): # if single value then set to array like 
            self.mesh.ptdf['por'] = [val]*self.mesh.numnp 
        elif len(val) == self.nzones: # then assign by zone 
            self.mesh.ptdf['por'] = np.zeros(self.mesh.numnp)
            c = 0 
            for i in np.unique(self.mesh.ptdf['zone'].values):
                idx = self.mesh.ptdf['zone'].values == i
                self.mesh.ptdf.loc[idx,'por'] = val[c]
                c+=1 
        else: # assume that its the nodewise assigment 
            if len(val) != self.mesh.numnp:
                raise ValueError('mis match in node array lengths')
            self.mesh.ptdf['por'] = val 
            
    def setPerm(self,val=1e-10):
        if self.mesh is None or self.nzones is None: 
            raise Exception ('mesh and zones need to be set first')
        if isinstance(val,float): # if single value then set to array like 
            self.mesh.df['perm'] = [val]*self.mesh.numel 
        elif len(val) == self.nzones: # then assign by zone 
            self.mesh.df['perm'] = np.zeros(self.mesh.numel)
            c = 0 
            for i in np.unique(self.mesh.df['zone'].values):
                idx = self.mesh.df['zone'].values == i
                self.mesh.df.loc[idx,'perm'] = val[c]
                c+=1 
        else: # assume that its the nodewise assigment 
            if len(val) != self.mesh.numel:
                raise ValueError('mis match in element array lengths')
            self.mesh.df['perm'] = val 
            
    def setEXEC(self,execpath):
        if not isinstance(execpath,str):
            raise ValueError('input must be a string')
        if not os.path.exists(execpath):
            raise EnvironmentError('Executable file does not exist!')
        self.execpath = os.path.abspath(execpath) 

    def setupInp(self,times=[], source_node=[], source_val =None, 
                 pressure_node = [], pressure_val = None,
                 temp_node = [], temp_val = None, 
                 general_node = [], general_type = None, 
                 solver = 'DIRECT'): 
        """
        Setup input
        """
        self.setupinp = {'times':times,
                         'source_node':source_node,
                         'source_val':source_val,
                         'pressure_node':pressure_node,
                         'pressure_val':pressure_val,
                         'temp_node':temp_node,
                         'temp_val':temp_val, 
                         'general_node':general_node,
                         'general_type':general_type,
                         'solver':solver.upper()}
        
    # WRITE functions
    def writeInp(self,dirname=None,variable_pressure=False,maximise_io=False):
        """
        Write inp file for sutra run. 

        Parameters
        ----------
        dirname : TYPE, optional
            DESCRIPTION. The default is None.
        variable_pressure : TYPE, optional
            DESCRIPTION. The default is False.
        maximise_io : TYPE, optional
            Maximise input / output writing by restricting SUTRA to just outputting
            the important parameters. ie. pressure and saturation at each 
            timestep (and thats it)

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if self.setupinp is None:
            raise Exception('Initial setup has not been done yet!')
            
        times = self.setupinp['times']
        source_node = self.setupinp['source_node']
        source_val = self.setupinp['source_val']
        pressure_node = self.setupinp['pressure_node']
        pressure_val = self.setupinp['pressure_val']
        temp_node = self.setupinp['temp_node']
        temp_val = self.setupinp['temp_val']
        general_node = self.setupinp['general_node']
        general_type = self.setupinp['general_type']
        solver= self.setupinp['solver']
        
        # mesh stats 
        if self.mesh is None: 
            raise Exception ('mesh class has not been set!')
        else: 
            numnp = self.mesh.node.shape[0]
            numel = self.mesh.connection.shape[0]
    
        if dirname is None: 
            fh = open(os.path.join(self.dname, self.name+'.inp'), 'w')
        else:
            fh = open(os.path.join(dirname, self.name+'.inp'), 'w')
        # Title 1 see master document (sutra 2.2) pg 239 onwards
        fh.write('%s\n'%self.title)
        fh.write('%s\n'%self.subtitle)  # TITLE 2
    
        # Data Set 2A
        fh.write('# Data Set 2A\n')
        # Four words: sutra, version, and flow type
        fh.write("'SUTRA VERSION 3.0 %s TRANSPORT'\n"%self.sim_type.upper())
    
        # Data Set 2B
        fh.write('# Data Set 2B\n')
        fh.write("'2D IRREGULAR MESH'\n")  # mesh descriptor, NN1 NN2 (NN3)
    
        # Start_inp3
        # Data Set 3 - simulation control - see pg 243 of master doc and pg 51 of sutra 3.0 update doc
        fh.write('# Data Set 3\n')
        NUBC = len(temp_node)  # len(source_node)#0
        NPBC = len(pressure_node)
        NSOP = len(source_node)
        NPBG =  len(general_node) # general nodes used as seepage for now 
    
        # numnp(NN), numel(NE), NPBC, NUBC, NSOP, NSOU, NPBG NUBG NOBS (line changed as of sutra 3.0?)
        fh.write('%i %i %i %i %i 0 %i 0 0\n' % (numnp, numel, NPBC, NUBC, NSOP, NPBG))
    
        # Start_inp4
        # Data Set 4
        fh.write('# Data Set 4\n')
        # CUNSAT CSSFLOW CSSTRA CREAD ISTORE
        sat_text = 'UNSATURATED'
        if self.saturated:
            sat_text = 'SATURATED'
        cold_text = 'WARM'
        if self.cold: 
            cold_text = 'COLD'
        fh.write("'%s' '%s FLOW' '%s TRANSPORT' '%s'  9999\n"%(sat_text,
                                                                 self.flow.upper(),
                                                                 self.transport.upper(),
                                                                 cold_text))
    
        # Data Set 5
        fh.write('# Data Set 5\n')
        # UP (reccomend = 0) | GNUP GNUU also shown in docs but not needed as of sutra 3.0
        fh.write('0.\n')
    
        # Data Set 6
        tmax = times[-1]
        nsteps =len(times)
        scalt = self.tlength #scale in seconds 
        ifac = self.ifac 
        tstep = np.mean(np.diff(times))
        istep = tstep/ifac  # step between internal time steps
        isteps = np.arange(0, tmax+istep, istep) 
    
        fh.write('# Data Set 6\n')
        fh.write('2 1 1\n')  # NSCH NPCYC NUCYC
        # old way to internal scheduling, we need a TIME_STEPS though
        #SCHNAM SCHTYP CREFT SCALT NTMAX TIMEI TIMEL TIMEC NTCYC TCMULT TCMIN TCMAX
        fh.write("'TIME_STEPS' 'TIME CYCLE' 'ELAPSED' %i %i 0 1.e+99 1. 1 1. 1.e-20 1\n"%(scalt/ifac,len(isteps))) 
    
        self.resultNsteps = len(isteps) # number of time steps expected in results 
        # define time steps in input file here
        if self.flow.upper() == 'TRANSIENT':
            if variable_pressure: 
                fh.write("'Pressure' 'TIME LIST' 'ELAPSED' %3.2f %i\n" % (scalt, nsteps))
            else: 
                fh.write("'Rainfall' 'TIME LIST' 'ELAPSED' %3.2f %i\n" % (scalt, nsteps))
            c = 0
            for i in range(nsteps):
                fh.write("%3.2f " % (times[i]))
                if c == 10:
                    fh.write('\n')
                    c = 0
                c += 1
            fh.write('\n')
    
        fh.write('-\n')  # signals the last schedule
    
        # Data Set 7A
        fh.write('# Data Set 7A\n')
        fh.write('%i %e %e\n'%(self.maxIter,self.rpmax,self.rumax))  # ITRMAX RPMAX RUMAX
    
        if solver == 'DIRECT':
            # Data Set 7B - equation solver for pressure solution (i guess direct is okay for small problems)
            fh.write('# Data Set 7B\n')
            fh.write("'DIRECT'\n")  # CSOLVP | ITRMXP TOLP
        
            # Data Set 7C - equation solver for transport solution
            fh.write('# Data Set 7C\n')
            fh.write("'DIRECT'\n")  # CSOLVU | ITRMXU TOLU
        else: # use non linear solver 
            # Data Set 7B - equation solver for pressure solution (using non direct methods)
            fh.write('# Data Set 7B\n')
            fh.write("'%s' 100 %e\n"%(solver,1e-5)) #CSOLVP | ITRMXP TOLP
        
            # Data Set 7C - equation solver for transport solution 
            fh.write('# Data Set 7C\n')
            # fh.write("'%s' 1000 %e\n"%(solver,1e2)) #CSOLVP | ITRMXP TOLP
            fh.write("'GMRES' 1000 1e2\n")
    
        # Start_inpe  - output options
        fh.write('# Data Set 8\n')

        if maximise_io:
            # NPRINT CNODAL CELMNT CINCID CPANDS CVEL CCORT CBUDG CSCRN CPAUSE
            fh.write("%i 'N' 'N' 'N' 'N' 'N' 'N' 'Y' 'N' 'N' 'Data Set 8A'\n"%self.iobs)
            fh.write("%i 'X' 'Y' 'S' 'P' '-' 'Data Set 8B'\n"%self.iobs)
            fh.write("%i '-' 'Data Set 8C'\n"%self.iobs) 
        else:
            # NPRINT CNODAL CELMNT CINCID CPANDS CVEL CCORT CBUDG CSCRN CPAUSE
            fh.write("%i 'N' 'Y' 'N' 'Y' 'N' 'N' 'Y' 'Y' 'N' 'Data Set 8A'\n"%self.iobs)
            # NCOLPR NCOL ..
            fh.write("%i 'N' 'X' 'Y' 'U' 'S' 'P' '-' 'Data Set 8B'\n"%self.iobs)
            fh.write("%i 'E' 'VX' 'VY' '-' 'Data Set 8C'\n"%self.iobs)  # LCOLPR LCOL ..
    
        # Start_inp8D
        # OMIT when there are no observation points
    
        # Start_inp9 - output controls for boundary conditions
        # NBCFPR NBCSPR NBCPPR NBCUPR CINACT
        for i in range(6):
            fh.write("%i "%self.iobs) # output on time steps where data is 
        fh.write("Y Y Y 'Data Set 8E'\n") #NBCFPR NBCSPR NBCPPR NBCUPR CINACT
    
        # fluid properties
        fh.write('# Data Set 9\n')
        # COMPFL CW SIGMAW RHOWØ URHOWØ DRWDU VISCØ
        fh.write("4.47e-10 1. 1.e-09 1000. 0. 700. 0.001 'Data Set 9'\n") 
        
        # solid matrix properties
        fh.write('# Data Set 10\n')
        # COMPMA CS SIGMAS RHOS
        # fh.write("1.e-08 840. 1632. 2740. 'Data Set 10'\n")
        fh.write("1.e-08 0. 0. 2740. 'Data Set 10'\n") #COMPMA CS SIGMAS RHOS
    
        # adsorption parameters
        fh.write("'NONE' 'Data Set 11'\n")  # ADSMOD
    
        # Start_inp12
        # production of energy or solute mass
        fh.write('# Data Set 12\n')
        fh.write("0. 0. 0. 0. 'Data Set 12'\n")  # PRODFØ PRODSØ PRODF1 PRODS1
    
        # oreintation of gravity vector
        fh.write('# Data Set 13\n')
        fh.write("0. -9.81 0 'Data Set 13'\n")  # GRAVX GRAVY GRAVZ
    
        ### write out node matrix (dataset 14) ###            
        fh.write('# Data Set 14 (mesh nodes)\n')
        # NODE SCALX SCALY SCALZ PORFAC
        fh.write("'NODE' 1. 1. 1. 1. 'Data Set 14A'\n")
        zone = self.mesh.ptdf['zone']
        nodePor = self.mesh.ptdf['por']
        for i in range(numnp):
            # II NREG(II) X(II) Y(II) (Z(II)) POR(II)
            line = "%i %i %f %f 1. %f\n" % (
                i+1, zone[i], self.mesh.node[i, 0], self.mesh.node[i, 2], nodePor[i])
            fh.write(line)
    
        ### scale factors for element data ###
        fh.write('# Data Set 15 (element scalers)\n')
        # character PMAXFA PMINFA ANG1FA ALMAXF ALMINF ATMAXF ATMINF
        fh.write("'ELEMENT' 1 1 1 1 1 1 1\n"),
        elemPerm = self.mesh.df['perm']
        elemZone = self.mesh.df['zone']
        for i in range(numel):
            # L LREG(L) PMAX(L) PMIN(L) ANGLE1(L) ALMAX(L) ALMIN(L) ATMAX(L) ATMIN(L)
            line = "%i %i %e %e 0. 1.0 1.0 1.0 1.0\n" %(i+1,elemZone[i],elemPerm[i],elemPerm[i])
            fh.write(line)
        
    
        ### write out source nodes ###
        if NSOP > 0:
            if source_val is None: 
                source_val = np.zeros(NSOP)
            elif isinstance(source_val,float) or isinstance(source_val,int):
                source_val = np.full(NSOP,source_val)
            for i in range(NSOP):
                line = "%i %.8e 0.0 'Data Set 17'\n" %(source_node[i],source_val[i])
                fh.write(line)
            fh.write("0 'Data Set 17'\n")
        
        ### write out pressure nodes ### 
        if NPBC>0: 
            if pressure_val is None:
                pressure_val = np.zeros(NPBC) 
            elif isinstance(pressure_val,float) or isinstance(pressure_val,int):
                pressure_val = np.full(NPBC,pressure_val)
            for i in range(NPBC):
                line = "%i %f 0. 'Data Set 19'\n"%(pressure_node[i],pressure_val[i])
                fh.write(line)
            fh.write("0 'Data Set 19'\n")
    
        ### write out temperatue nodes ###
        if NUBC > 0:
            for i in range(NUBC):
                line = "%i %f 'Data Set 20'\n" %(temp_node[i],temp_val[i])
                fh.write(line)
            fh.write("0 'Data Set 20'\n")
    
        ### add general flow / drainage nodes / seepage nodes ###
        if NPBG > 0:
            if general_type is None: 
                general_type = [None]*NPBG 
            if self.pressure is None: 
                self.pressure = np.max(pressure_val)
            if isinstance(self.pressure,int) or isinstance(self.pressure,float):
                self.pressure = [self.pressure]*len(general_node)
            
            fname21a = self.name+'.inp21A'
            fh.write("# Start new dataset 21A here\n")
            fh.write("@INSERT 95 '%s'\n"%fname21a)
            fh2 = open(os.path.join(self.dname, fname21a), 'w')
            for i in range(NPBG):
                #IPBG PBG1 QPBG1 PBG2 QPBG2 CPQL1 CPQL2 UPBGI CUPBGO UPBGO 
                if general_type[i] == 'seep':
                    line = "%i -1. 0. 0. 0. 'N' 'P' 0. 'REL' 0. 'Data Set 21A'\n"%general_node[i]
                elif general_type[i]  == 'drain' and 'Ksat' in self.mesh.df.columns:
                    line = "%i -1. 0. 100000. -%e 'N' 'P' 0. 'REL' 0. 'Data Set 21A'\n"%(general_node[i], self.mesh.df['Ksat'][general_node[i]])
                elif general_type[i]  == 'drain':
                    line = "%i -1. 0. 100000. -%e 'N' 'P' 0. 'REL' 0. 'Data Set 21A'\n"%(general_node[i], self.drainage)
                elif general_type[i] == 'pres':
                    line = "%i %e, 0. %e -1. 'P' 'P' 0. 'REL' 0. 'Data Set 21A'\n"%(general_node[i], 
                                                                                   self.pressure[i],
                                                                                   self.pressure[i]+(2*9180))
                elif general_type[i] == 'evap':
                    P1 = self.mesh.ptdf['depth'][general_node[i]-1]*-1*9180
                    line = "%i %e, 0. %e %e 'Q' 'N' 0. 'REL' 0. 'Data Set 21A'\n"%(general_node[i], 
                                                                                   P1,
                                                                                   -1,
                                                                                   self.pressure[i])
                    
                else: # standard general node, to do add functionality for this     
                    line = "%i -1. 0. 0. 0. 'N' 'N' 0. 'REL' 0. 'Data Set 21A'\n"%general_node[i]
                    
                fh2.write(line)
                # only doing general nodes for seepage for now 
            fh2.write("0 'Data Set 21A'\n")
            fh2.close()
        
        ### write out connection matrix ###
        fh.write('# Data Set 22 (connection matrix)\n')
        fh.write("'INCIDENCE'\n")
        for i in range(numel):
            line = "%i " % (i+1)  # LL IIN ...
            for j in range(4):
                line += "%i " % (self.mesh.connection[i, j] + 1)
            line += "\n"
            fh.write(line)
        fh.close()
    
    
    def writeBcs(self, times, source_node, rainfall, temps, surface_temp=None):
        """ 
        Datasets 3, 4, 5, and 6 of a “.bcs” file correspond to datasets 17, 18, 19, and 20, respectively,
        of the “.inp” file. (For example, “.bcs” dataset 5 and “.inp” dataset 19 both define
        specified-pressure nodes.) The formats of these four “.bcs” datasets parallel those of the
        corresponding “.inp” datasets. This function is for writing rainfall to file. 

        Parameters
        ----------
        times : TYPE
            DESCRIPTION.
        source_node : TYPE
            DESCRIPTION.
        rainfall : TYPE
            DESCRIPTION.
        temps : TYPE
            DESCRIPTION.
        surface_temp : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """

        nsteps = len(times)
        nnodes = len(source_node)
        fh = open(os.path.join(self.dname, self.name+'.bcs'), 'w')
        ### time steps ###
        fh.write('# Dataset 1\n')
        fh.write("'Rainfall'\n")
    
        ### time dependent variables ###
        fh.write('# Dataset 3\n')
        NUBC = 0
        NSOP = len(source_node)
        if surface_temp is not None:
            NUBC = len(source_node)
    
        ### write out time steps (for sources) ###
        for i in range(nsteps):
            fh.write("'ts{:0>6d}' ".format(times[i]))
            if i == 0:
                # NSOP1, NSOU1, NPBC1, NUBC1, NPBG1, NUBG1 (according to sutra 3.0 docuementation)
                fh.write('%i 0 0 %i 0 0\n' % (NSOP, 0))
            else:
                fh.write('%i 0 0 %i 0 0\n' % (NSOP, NUBC))
            for j in range(nnodes):
                line = "%i %.8e %.8e\n" % (
                    source_node[j], rainfall[i, j], temps[i, j])
                fh.write(line)
            fh.write('0\n')
            # write out recorded surface temperatures
            if surface_temp is not None and i != 0:
                fh.write('# Dataset 6\n')
                for j in range(nnodes):
                    line = "%i %f\n" % (source_node[j], surface_temp[i, j])
                    fh.write(line)
                fh.write('0\n')
        fh.close()
        
    def writeBcsPres(self, times, pres_node, pressure):
        """
        Datasets 3, 4, 5, and 6 of a “.bcs” file correspond to datasets 17, 18, 19, and 20, respectively,
        of the “.inp” file. (For example, “.bcs” dataset 5 and “.inp” dataset 19 both define
        specified-pressure nodes.) The formats of these four “.bcs” datasets parallel those of the
        corresponding “.inp” datasets. This function is for writing pressures to file. 

        Parameters
        ----------
        times : TYPE
            DESCRIPTION.
        pres_node : TYPE
            DESCRIPTION.
        pressure : TYPE
            DESCRIPTION.
        temps : TYPE
            DESCRIPTION.
        surface_temp : TYPE, optional
            DESCRIPTION. The default is None.
        """
        nsteps = len(times)
        nnodes = len(pres_node)
        fh = open(os.path.join(self.dname, self.name+'.bcs'), 'w')
        ### time steps ###
        fh.write('# Dataset 1\n')
        fh.write("'Pressure'\n")
        ### time dependent variables ###
        NPBC = len(pres_node)
        
        ### write out time steps (for sources) ###
        for i in range(nsteps):
            fh.write("'ts{:0>6d}' ".format(times[i]))
            # NSOP1, NSOU1, NPBC1, NUBC1, NPBG1, NUBG1 (according to sutra 3.0 docuementation)
            fh.write('0 0 %i 0 0 0\n'%(NPBC))
            fh.write('# Dataset 5\n')
            for j in range(nnodes):
                line = "%i %.8e %.8e\n"%(pres_node[j], pressure[i, j], 0)
                fh.write(line)
            fh.write('0\n')
            
    def writeGenBcs(self, times, general_node, rainfall):
        """
        Datasets 3, 4, 5, and 6 of a “.bcs” file correspond to datasets 17, 18, 19, and 20, respectively,
        of the “.inp” file. (For example, “.bcs” dataset 5 and “.inp” dataset 19 both define
        specified-pressure nodes.) The formats of these four “.bcs” datasets parallel those of the
        corresponding “.inp” datasets. This function is for writing rainfall as generalised nodes to file. 

        Parameters
        ----------
        times : TYPE
            DESCRIPTION.
        pres_node : TYPE
            DESCRIPTION.
        pressure : TYPE
            DESCRIPTION.
        temps : TYPE
            DESCRIPTION.
        surface_temp : TYPE, optional
            DESCRIPTION. The default is None.
        """
        nsteps = len(times)
        nnodes = len(general_node)
        fh = open(os.path.join(self.dname, self.name+'.bcs'), 'w')
        ### time steps ###
        fh.write('# Dataset 1\n')
        fh.write("'Rainfall'\n")
        ### time dependent variables ###
        NPBG = len(general_node)
        
        ### write out time steps (for sources) ###
        for i in range(nsteps):
            fh.write("'ts{:0>6d}' ".format(times[i]))
            # NSOP1, NSOU1, NPBC1, NUBC1, NPBG1, NUBG1 (according to sutra 3.0 docuementation)
            fh.write('0 0 0 0 %i 0\n'%(NPBG))
            fh.write('# Dataset 7A\n')
            for j in range(nnodes):
                # IPBG1, PBG11, QPBG11, PBG21, QPBG21, CPQL11, CPQL21, UPBGI1, CUPBGO1, UPBGO1
                line = "%i -1. %.8e 0 %.8e 'N' 'N' 0 'REL' 0\n"%(general_node[j], rainfall[i, j], rainfall[i, j])
                fh.write(line)
            fh.write('0\n')
        
    
    def writeIcs(self, pres, temps):
        # just needs the pressures and temperatures at each node
        fh = open(os.path.join(self.dname, self.name+'.ics'), 'w')
        fh.write('0 # starting time for init conditions\n')
        fh.write('# Dataset 2 - pressure conditions \n')
        # nb: this can be acquired with a single run of a steady state system
        fh.write("'NONUNIFORM'\n")
        for i in range(len(pres)):
            fh.write('%f\n' % pres[i])
    
        fh.write('# Dataset 3 - temperature conditions \n')
        # nb: this can be acquired with a single run of a steady state system
        fh.write("'NONUNIFORM'\n")
        for i in range(len(temps)):
            fh.write('%f\n' % temps[i])
        fh.close()
        
    def writeVg(self,swres=None,swsat=None,alpha=None,vn=None,dname=None,
                alpha_scale=1e-3):
        """
        Write unsaturated parameters to file 

        Parameters
        ----------
        swres : TYPE, optional
            DESCRIPTION. The default is None.
        swsat : TYPE, optional
            DESCRIPTION. The default is None.
        alpha : TYPE, optional
            DESCRIPTION. The default is None.
        vn : TYPE, optional
            DESCRIPTION. The default is None.
        dname : TYPE, optional
            DESCRIPTION. The default is None.
        alpha_scale: float
            Multiplier to apply to alpha to make it in terms of Pa not kPa. 
            Default is 1e-3. 

        """
        if swres is None: # override with default param 
            swres = self.param['res']
            swsat = self.param['sat']
            alpha = self.param['alpha']
            vn = self.param['vn']
            
        if dname is None: # fall back onto default directory 
            fpath = os.path.join(self.dname,'param.vg')
        else:
            fpath = os.path.join(dname,'param.vg')
        fh = open(fpath,'w')
        fh.write('%i\n'%self.nzones)
        for i in range(self.nzones):
            line = '{:d} {:f} {:f} {:16f} {:f}\n'.format(i+1,swres[i],swsat[i],
                                                         alpha[i]*alpha_scale,
                                                         vn[i])
            fh.write(line)
            
            # nb: alpha gets divided by 1000 as its normally in terms of kpa, not pa
        fh.close()
        
    # def writeVg_OLD(self, swres, alpha, vn):
    #     fh = open(os.path.join(self.dname,'param.vg'),'w')
    #     nzones = len(swres)
    #     fh.write('%i\n'%nzones)
    #     for i in range(nzones):
    #         line = '{:d} {:f} {:f} {:f}\n'.format(i+1,swres[i],alpha[i],vn[i])
    #         fh.write(line)
    #     fh.close()
    
    
    def writeFil(self, ignore=['BCOP', 'BCOPG']):
        # write out master file
        exts = ['INP', 'BCS', 'ICS', 'LST', 'RST',
                'NOD', 'ELE', 'OBS', 'BCOP', 'BCOPG', 'SMY'] # FILE EXTENSIONS 
        iunits = [50,    52,  55,     60,     66,
                  70,     76,     80,    92,       94,   98]# FILE READ UNITS (MAX SHOULD BE 98)
    
        fh = open(os.path.join(self.dname, 'SUTRA.FIL'), 'w')
        for i in range(len(exts)):
            e = exts[i]
            u = iunits[i]
            if e not in ignore:
                line = "{:_<7s} {:_<3d} '{:s}.{:s}'\n" .format(
                    e, u, self.name, e.lower())
                line = line.replace('_', ' ')
                # print(line.strip())
                fh.write(line)
        fh.close()
    
    # directory management 
    def clearDir(self):
        # clears directory of files made during runs 
        exts = ['INP', 'BCS', 'ICS', 'LST', 'RST',
                'NOD', 'ELE', 'OBS', 'BCOP', 'BCOPG', 'SMY'] # FILE EXTENSIONS 
        extsl = [e.lower() for e in exts] # lower case extensions 
        
        files = os.listdir(self.dname)
        for f in files:
            if f in exts or f in extsl: 
                os.remove(os.path.join(self.dname,f))
                
    def clearDirs(self):
        template = 'r{:0>5d}'
        for i in range(self.nruns): # loop through the number of runs 
            dpath = os.path.join(self.dname,template.format(i))
            if os.path.exists(dpath):
                shutil.rmtree(dpath) # remove directory 
                
    # run sutra 
    def runSUTRA(self,show_output=True,dump=print): # single run of sutra 
        if self.execpath is None:
            raise Exception('Executable path needs to be set first!')
        
        cwd = os.getcwd()
        if platform.system() == "Windows":#command line input will vary slighty by system 
            cmd_line = [self.execpath]
        
        elif platform.system() == 'Linux':
            cmd_line = [self.execpath] # using linux version if avialable (can be more performant)
            if '.exe' in self.execpath: # assume its a windows executable 
                cmd_line.insert(0,'wine') # use wine to run instead 
        else:
            raise Exception('Unsupported operating system') # if this even possible? BSD maybe. 
    
        ERROR_FLAG = False # changes to true if sutra causes an error 
        # change working directory 
        os.chdir(self.dname)
        # p = Popen(['cd',self.dname], stdout=PIPE, stderr=PIPE, shell=False)
        if show_output: 
            p = Popen(cmd_line, stdout=PIPE, stderr=PIPE, shell=False)#run gmsh with ouput displayed in console

            while p.poll() is None:
                line = p.stdout.readline().rstrip()
                if line.decode('utf-8') != '':
                    dump(line.decode('utf-8'))
                    if 'ERROR' in line.decode('utf-8'):
                        ERROR_FLAG = True 
        else:
            p = Popen(cmd_line, stdout=PIPE, stderr=PIPE, shell=False)
            p.communicate() # wait to finish
            
        os.chdir(cwd)
        
        if ERROR_FLAG:
            raise Exception('Looks like SUTRA run has failed, check inputs!')
            
    def showSetup(self,save=False,custom_flags=None, return_fig=False):
        """
        Show the model setup. 

        Parameters
        ----------
        save : bool, optional
            If true then the figure is closed but saved to the working 
            directory. The default is False.

        Returns
        -------
        None.

        """
        source_node = self.setupinp['source_node']
        pressure_node = self.setupinp['pressure_node'] 
        temp_node = self.setupinp['temp_node'] 
        general_node = self.setupinp['general_node'] 
        
        NUBC = len(temp_node)  # len(source_node)#0
        NPBC = len(pressure_node)
        NSOP = len(source_node)
        NPBG = len(general_node) # general nodes used as seepage for now 
        
        fig, ax = plt.subplots()
        fig.set_size_inches(12.0 , 8)

        # get node positions 
        self.mesh.show(ax=ax, attr='zone',color_map='Greys',electrodes=False,
                       xlim=[min(self.mesh.node[:,0]),max(self.mesh.node[:,0])],
                       vmin=0,vmax=4)
        
        nx = self.mesh.node[:,0]
        nz = self.mesh.node[:,2]
        
        if custom_flags is None: 
            if NUBC > 0:    
                ax.scatter(nx[temp_node-1],nz[temp_node-1],c='r',label='temp node')
                
            if NPBC > 0:
                ax.scatter(nx[pressure_node-1],nz[pressure_node-1],c='m',label='pressure node')
            
            if NSOP > 0:
                ax.scatter(nx[source_node-1],nz[source_node-1],c='b',label='source node')
                
            if NPBG > 0:
                ax.scatter(nx[general_node-1],nz[general_node-1],c='k',label='general node')
        elif not isinstance(custom_flags, list):
            raise TypeError('Custom flags should be a list of dictionary')
        else:
            for custom in custom_flags: 
                idx = custom['idx']
                label = custom['label']
                color = custom['color']
                ax.scatter(nx[idx],nz[idx],c=color,label=label)
        
        ax.legend() 
        tdx = max(nx) - min(nx)
        ax.set_xlim([min(nx) - 0.05*tdx, max(nx) + 0.05*tdx])
        
        if save: 
            fig.savefig(os.path.join(self.dname,'setup.png'))
            # plt.close(fig)
        if return_fig: 
            return fig 
            
    #get result functions 
    def getNod(self):
        #get nod file 
        files = os.listdir(self.dname)
        fname = '.nod'
        for f in files:
            if f.endswith('.nod'):
                fname = os.path.join(self.dname,f) # then we have found the .nod file 
                break 
        data, n = readNod(fname)
        self.nodResult = data 
        self.resultNsteps = n 
        print('%i time steps read'%n)  
        
    def getMsBnc(self):
        # parse lst file to get mass balance information 
        files = os.listdir(self.dname)
        fname = '.lst'
        for f in files:
            if f.endswith('.lst'):
                fname = os.path.join(self.dname,f) # then we have found the .nod file 
                break 
        parse = readMassBlnc(fname)
        df = pd.DataFrame(parse)
        self.massBalance = df 
        
    def getElem(self):
        files = os.listdir(self.dname)
        fname = '.ele'
        for f in files:
            if f.endswith('.ele'):
                fname = os.path.join(self.dname,f) # then we have found the .nod file 
                break 
        return 
        
    # get results 
    def getResults(self):
        self.getNod()
        # self.getElem()
        self.getMsBnc()
    
    # plotting functions 
    def plotNod(self, n=0): 
        """
        Plots an invidual time step of the nodewise simulation.

        Parameters
        ----------
        n : int, optional
            index of time step. The default is 0.
        attr : str, optional
            name of parameter to be plotted. The default is 'Saturation'.
        close: bool, optional 
            closes the figure once it has been plotted (result is saved inside 
            of working directory)

        """
        # plot sutra data 
        step=self.nodResult['step%i'%n]

        X = np.array(step['X']) # y values 
        Y = np.array(step['Y']) # x values 
        arr = np.array(step[self.attribute]) # array of node values 
        
        if any(np.isnan(arr)==True):
            return 
        
        fig, ax = plt.subplots()
        
        levels = np.linspace(self.vmin, self.vmax, 50)
        cax = ax.tricontourf(X,Y,arr,levels=levels)
        plt.colorbar(cax)
        timeprint = self.nodResult['step%iinfo'%n]['Time']
        ax.set_title('Step = %i, Time = %s (sec)'%(n,str(timeprint))) 
        if self.xlim is None:
            self.xlim = [np.min(X), np.max(X)]
        if self.ylim is None:
            self.ylim = [np.min(Y), np.max(Y)]
            
        ax.set_xlim(self.xlim)
        ax.set_ylim(self.ylim)
        
        dirname = os.path.join(self.dname,self.attribute)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
            
        fig.savefig(os.path.join(dirname,'{:0>3d}.png'.format(n)))
        
        
        if self.closeFigs:
            plt.close(fig)
    
    def plotResults(self,parallel=False):
        # note that the parrallel option for this function doesnt work very well 
        n = self.resultNsteps
        desc = 'Plotting steps'
        warnings.filterwarnings("ignore")
        if not parallel: 
            for i in tqdm(range(n),ncols=100, desc=desc):
                self.plotNod(i)
        else:
            Parallel(n_jobs=self.ncpu)(delayed(self.plotNod)(i) for i in tqdm(range(n),ncols=100, desc=desc))#process over multiple cores 
        warnings.filterwarnings("default")
        
    def plotMesh(self,n=0, time_units='day',cmap='Spectral'):
        # covert to resistivity and map to elements
        step=self.nodResult['step%i'%n]
        node_arr = np.array(step[self.attribute])
        arr = self.mesh.node2ElemAttr(node_arr,self.attribute)

        fig, ax = plt.subplots()
        fig.set_size_inches(12.0 , 8)

        self.mesh.show(ax=ax, attr=self.attribute,
                       vmin=self.vmin,vmax=self.vmax,
                       color_map=cmap,zlim=self.ylim,
                       electrodes=False,edge_color=None)

        timeprint = self.nodResult['step%iinfo'%n]['Time']
        timeprint = convertTimeUnits(timeprint,time_units)
        ax.set_title('Step = %i, Time = %s (%s)'%(n,str(timeprint),time_units)) 
        dirname = os.path.join(self.dname,self.attribute)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
            
        fig.savefig(os.path.join(dirname,'{:0>3d}.png'.format(n)))
        
        if self.closeFigs:
            plt.close(fig)
            
    def saveMesh(self,n=0):
        step=self.nodResult['step%i'%n]
        potential_arrays = ['Saturation','Pressure','Resistivity']
        for name in potential_arrays:
            if name in step.keys():
                node_arr = np.array(step[name])
                _ = self.mesh.node2ElemAttr(node_arr,name)
                
        dirname = os.path.join(self.dname,self.attribute)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
            
        self.mesh.vtk(os.path.join(dirname,'ts{:0>3d}.vtk'.format(n)))
        
    def saveMeshes(self,survey_keys=None):
        n = self.resultNsteps
        desc = 'Saving meshes steps'
        warnings.filterwarnings("ignore")
        if survey_keys is None: 
            steps = np.arange(n)
        else:
            steps = survey_keys 
        for i in tqdm(range(n),ncols=100, desc=desc):
            if i not in steps:
                continue 
            self.saveMesh(i)
    
    def plotMeshResults(self,time_units='day',cmap='Spectral',iobs=10):
        n = self.resultNsteps
        desc = 'Plotting steps'
        warnings.filterwarnings("ignore")
        steps = np.arange(n)[0::iobs].tolist()
        for i in tqdm(range(n),ncols=100, desc=desc):
            if i not in steps:
                continue 
            self.plotMesh(i,time_units,cmap)
            
    def plot1Dresult(self,n=0,x=None,time_units='day'):
        step=self.nodResult['step%i'%n]
        X = np.array(step['X']) # x values 
        Y = np.array(step['Y']) # elevation values 
        if x is None: 
            #get middle of array 
            xm = np.median(self.mesh.node[:,0])
            i = np.argmin(np.abs(X - xm))
            x = X[i]

        arr = np.array(step[self.attribute]) # array of node values 
        
        if any(np.isnan(arr)==True):
            return 
        
        fig, ax = plt.subplots()
        
        idx = X == x # index of array 
        ax.plot(arr[idx], Y[idx],marker='x')
        timeprint = self.nodResult['step%iinfo'%n]['Time']
        timeprint = convertTimeUnits(timeprint,time_units)
        ax.set_title('Step = %i, Time = %s (%s)'%(n,str(timeprint),time_units)) 
        # if self.xlim is None:
        #     self.xlim = [np.min(X), np.max(X)]
        if self.vlim is None:
            self.vlim = [np.min(arr),np.max(arr)]
        if self.ylim is None:
            self.ylim = [np.min(Y), np.max(Y)]
            
        ax.set_xlim(self.vlim)
        ax.set_ylim(self.ylim)
        ax.set_xlabel(self.attribute)
        ax.set_ylabel('Depth (m)')
        
        dirname = os.path.join(self.dname,self.attribute)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
            
        fig.savefig(os.path.join(dirname,'oneD{:0>3d}.png'.format(n)))
        
        if self.closeFigs:
            plt.close(fig)
            
    def get1Dvalues(self,xp,yp):
        a = np.array([0]*self.resultNsteps,dtype=float)
        step=self.nodResult['step%i'%0]
        X = np.array(step['X']) # x values 
        Y = np.array(step['Y']) # elevation values 
        tree = cKDTree(np.c_[X,Y])
        dist,idx = tree.query(np.c_[[xp],[yp]])
        for i in range(self.resultNsteps): 
            step=self.nodResult['step%i'%i]
            arr = np.array(step[self.attribute]) # array of node values 
            a[i] = arr[idx[0]]
            # if any(np.isnan(arr)==True):
            #     return 
            
        return a 

            
    def plot1Dresults(self,clean_dir = True, iobs=1):
        n = self.resultNsteps
        desc = 'Plotting steps'
        warnings.filterwarnings("ignore")

        dirname = os.path.join(self.dname,self.attribute)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        elif clean_dir:
            files = os.listdir(dirname)
            for f in files: # remove old files 
                if f.endswith('.png'):
                    os.remove(os.path.join(dirname,f))
        steps = np.arange(n)[0::iobs].tolist()
        for i in tqdm(range(n),ncols=100, desc=desc):
            if i not in steps:
                continue 
            self.plot1Dresult(i)
        

    #%% Petrophysical functions 
    def callPetro(self):
        """
        Call petrophysical transfer functions for each material 

        Returns
        -------
        None.

        """
        for m in self.materials:
            if m.petro_func is None:
                raise Exception('Material %s has no petrophysical transfer functions associated with it!')
        if 'depths' not in self.mesh.ptdf.keys():
            _ = self.mesh.computeElmDepth()    
        depths_node  = self.mesh.ptdf['depths'].values 
        res0 = np.zeros(self.mesh.numnp) 
        res_elem = np.zeros((self.mesh.numel,self.resultNsteps)) # array to hold resistivity values 
        zones = self.mesh.ptdf['zone'].values 
        
        for i in range(self.resultNsteps):
        # for i in tqdm(range(self.resultNsteps),ncols=100): # generally this bit of code is very quick 
            sat = np.array(self.nodResult['step%i'%i]['Saturation']) # pressure in kpa 
            sat[sat>1] = 1 # cap saturation at 1 
            for zone in range(self.nzones):
                zidx = zones == (zone+1) 
                m = self.materials[zone]
                res0[zidx] = m.petro(sat[zidx], depths_node[zidx])
            
            # add moisture values to data 
            self.nodResult['step%i'%i]['Resistivity'] = res0.tolist() 
        
            # and map to elements ??
            # res_elem[:,i] = self.mesh.node2ElemAttr(res0,'res%i'%i)
        
        return res_elem  
    
    # correct resistivity for temperature? 
    def callTempCorrect(self,tfunc,diys):
        """
        Correct resistivity for temperature changes (or rather put the corrected
        resistivities back to thier insitu values). 

        Parameters
        ----------
        tfunc : function 
            Fuction which takes resistivity as its first argument and date in 
            year as second non optional argument, and cell depths as its third 
            argument. 
        diys : array like 
            list of the julian day in year for each resistivity survey, should be 
            between 1 and 365. 

        """
        if 'depths' not in self.mesh.ptdf.keys():
            _ = self.mesh.computeElmDepth()    
        depths_node  = self.mesh.ptdf['depths'].values 
        res_elem = np.zeros((self.mesh.numel,self.resultNsteps)) 
        
        for i in range(self.resultNsteps):
        # for i in tqdm(range(self.resultNsteps),ncols=100): # generally this bit of code is very quick 
            res0 = np.array(self.nodResult['step%i'%i]['Resistivity']) # pressure in kpa 
            res1 = tfunc(res0, depths_node, diys[i])
            
            # add moisture values to data 
            self.nodResult['step%i'%i]['Resistivity'] = res1.tolist() 
        
            # and map to elements 
            # res_elem[:,i] = self.mesh.node2ElemAttr(res0,'res%i'%i)
        
        return res_elem  
        
   
    
    #%% several run handling (monte carlo approach)
    def createPdir(self, runno, to_copy=[], pargs={}):
        """
        Create a monte  carlo run modelling directory
        
        Parameters 
        ----------
        runno: int
            Number of model run, use to create the name of the directory
        to_copy: list, optional 
            Which files to copy from the base directory of SUTRA
        pargs: dict, optional 
            Creates the text file detailing run parameters. 
        Returns
        -------
        dpath: str 
            Directory path 
        """ 
            
        if len(to_copy) == 0: 
            exts = ['INP', 'INP21A','BCS', 'ICS', 'RST', 
                    'BCOP', 'BCOPG', 'FIL', 'VG']
            files = os.listdir(self.dname)
           
            for f in files: # find files needed to run sutra 
                ext = f.split('.')[-1]
                if ext.upper() in exts: # add it to files which need copying 
                    to_copy.append(f)
        
        dpath = os.path.join(self.dname,self.template.format(runno))
            
        if not os.path.exists(dpath):
            os.mkdir(dpath) # create directory
            
        self.pdirs.append(dpath)
        # copy across files to parrallel run directory (files need to be written first)
        for j in range(len(to_copy)):
            source_path = os.path.join(self.dname,to_copy[j])
            copy_path = os.path.join(dpath,to_copy[j])
            shutil.copy(source_path,copy_path)
            
        # write run parameters 
        text = '{:<6s}\t'.format('')
        for i in range(self.nzones):
            text += '{:<8s}\t'.format('zone %i'%i)
        text += '\n'
        
        for key in pargs.keys():
            text += '{:<6s}\t'.format(key)
            for i in range(self.nzones):
                if pargs[key][i] == '-':
                    text += '{:<8s}\t'.format(pargs[key][i])
                else: 
                    text += '{:<8e}\t'.format(pargs[key][i])
            text += '\n'
            
        fh = open(os.path.join(dpath,'pargs.txt'),'w')
        fh.write(text)
        fh.close() 
        
        return dpath 
    
    def setupMultiRun(self):
        """
        Setup to handle changes to the permeability, porosity and VG parameters  
        Sets up directories for multiple runs with perturbed values. Number of 
        runs decided by the perturbation to the material properites. 

        """
        self.pdirs = []      
        self.nruns = 0 

        # time to see what model run parameters are being perturbed each run 
        maltered = [False]*len(self.materials)
        miter = [-1]*len(self.materials)
        for i,m in enumerate(self.materials):
            if len(m.MCparam.keys()) > 0: 
                maltered[i] = True 
                miter[i] = m.niter       
            
        self.nruns = max(miter)
        self.runparam = {}
        
        # for i in range(self.nruns):
        for i in tqdm(range(self.nruns),ncols=100,desc='Creating Dirs'):
            alpha = [-1]*self.nzones 
            swres = [-1]*self.nzones 
            swsat = [-1]*self.nzones 
            vn = [-1]*self.nzones 
            pargs = {'k':['-']*self.nzones,
                     'theta':['-']*self.nzones,
                     'res':['-']*self.nzones,
                     'sat':['-']*self.nzones,
                     'alpha':['-']*self.nzones,
                     'vn':['-']*self.nzones} 
            # loop through to get run param 
            for zone in range(self.nzones):
                zidx = self.mesh.ptdf['zone'] == (zone+1) # node zone idx 
                eidx = self.mesh.df['zone'] == (zone+1) # element zone idx 
                m = self.materials[zone]
                 # gather petrubed parameters 
                if 'k' in m.MCparam.keys():
                    self.mesh.ptdf.loc[zidx,'perm'] = m.MCparam['k'][i]
                    pargs['k'][zone] = m.MCparam['k'][i]
                
                if 'theta' in m.MCparam.keys():
                    self.mesh.df.loc[eidx,'por'] = m.MCparam['theta'][i]
                    pargs['theta'][zone] = m.MCparam['theta'][i]
                    
                if 'alpha' in m.MCparam.keys():
                    alpha[zone] = m.MCparam['alpha'][i]
                    pargs['alpha'][zone] = m.MCparam['alpha'][i]
                else:
                    alpha[zone] = m.alpha
                
                if 'vn' in m.MCparam.keys():
                    vn[zone] = m.MCparam['vn'][i]
                    pargs['vn'][zone] = m.MCparam['vn'][i]
                else:
                    vn[zone] = m.vn 
                
                if 'res' in m.MCparam.keys():
                    swres[zone] = m.MCparam['res'][i]
                    pargs['res'][zone] = m.MCparam['res'][i]
                else:
                    swres[zone] = m.res 
                
                if 'sat' in m.MCparam.keys():
                    swsat[zone] = m.MCparam['sat'][i]
                    pargs['sat'][zone] = m.MCparam['sat'][i]
                else:
                    swsat[zone] = m.sat  
                    
            self.runparam[i] = pargs 
            # create directory
            to_copy_ext = ['INP21A','BCS', 'ICS', 'RST', 'BCOP', 'BCOPG', 'FIL']
            to_copy = []
            files = os.listdir(self.dname)
            for f in files: # find files needed to run sutra 
                ext = f.split('.')[-1]
                if ext.upper() in to_copy_ext: # add it to files which need copying 
                    to_copy.append(f)
                    
            rdir = self.createPdir(i,to_copy,pargs) # run directory 
            self.writeInp(rdir,maximise_io=True)
            self.writeVg(swres,swsat,alpha,vn,dname=rdir) 
        
            
    def runMultiRun(self,return_data=False):
        """
        Run SUTRA for multiple runs (uses joblib for parallisation)

        """
        if self.ncpu == 0 or self.ncpu == 1: # run single threaded with a basic for loop 
            pout = []
            for i in tqdm(range(self.nruns),ncols=100):    
                out = doSUTRArun(self.pdirs[i],self.execpath)
                pout.append(out)
        else: # run in parallel 
            pout = Parallel(n_jobs=self.ncpu)(delayed(doSUTRArun)(self.pdirs[i],self.execpath,return_data) for i in tqdm(range(self.nruns),ncols=100,desc='Running'))
        
        # nb storing all this information in memory is too computationally expensive for bigger problems 
        if return_data: 
            for i,p in enumerate(pout):
                if p[1] == self.resultNsteps:
                    self.nodResultMulti[i] = p[0]
            return pout 
        
    def getMultiRun(self,return_data=False):
        """
        Get number of successful model runs, the criteria for this is that 
        the model has the same number of steps in it as the baseline model. 

        Parameters
        ----------
        return_data : TYPE, optional
            Return nodewise data for each run and append it to the class variable.
            "self.nodResultMulti". The default is False as it requires a lot of 
            memory to do this. 

        Returns
        -------
        run_success : list
            Index of successful runs.

        """
        run_success = [] # keys of successful runs 
        for runno in tqdm(range(self.nruns),ncols=100,desc='Retrieving Runs'):
            dpath = os.path.join(self.dname,self.template.format(runno))
            if os.path.isdir(dpath):
                for f in os.listdir(dpath):
                    if f.lower().endswith('.nod'):
                        # print(os.path.join(dpath,f))
                        try: 
                            data,n = readNod(os.path.join(dpath,f)) 
                            success = n >= self.resultNsteps
                        except: 
                            success = False 
                        if return_data and success:
                            self.nodResultMulti[runno] = data 
                        if success:
                            run_success.append(runno)
                        break 
        return run_success
                        
    
    def clearMultiRun(self):
        """
        Clear directories made for multiple sutra runs. 

        """
        entries = os.listdir(self.dname)
        for e in entries:
            dpath = os.path.join(self.dname,e)
            if os.path.isdir(dpath):
                if 'pargs.txt' in os.listdir(dpath):
                    shutil.rmtree(dpath)
                    
    #%% coupled modelling with R2 family of codes for MC methods 
    def setRproject(self,project):
        """
        Set Resipy project

        Parameters
        ----------
        project : class 
            Resipy class with mesh and electrodes already set.

        """
        # if not isinstance(project,type):
        #     raise Exception('project is not a class type')
        if project.elec is None: 
            raise Exception('Resipy project does not have electrodes')
        if project.mesh is None:
            raise Exception('Resipy project has no mesh')

        self.project = project 
        
    def setupRruns(self, write2in, run_keys, survey_keys, seqs=[],ncpu=None, 
                   tfunc=None, diy=[]):
        """
        Setup R2(3t) runs for 

        Parameters
        ----------
        write2in : function 
            DESCRIPTION.
        run_keys : list 
            DESCRIPTION.
        survey_keys : list 
            DESCRIPTION.
        seqs : TYPE, optional
            DESCRIPTION. The default is [].
        ncpu: int
            Number of cores to spread the creation of R2 files across. 
        tfunc : function, optional
            DESCRIPTION. The default is None.
        diy : list, optional
            Day in year where surveys are run. The default is []. Ignored unless
            tfunc is set. 

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if ncpu is None: 
            ncpu = self.ncpu 
        self.survey_keys = survey_keys 
        if not callable(write2in):
            raise Exception('Write2in should be a python function')
        if self.project is None: 
            raise Exception('No resipy project has been set')
        tcorrect = False # flag to do temperature correction 
        if tfunc is not None:
            tcorrect = True # do correction if function provided 
            if not callable(tfunc):
                raise Exception('Temperature function is not callable')
        
        # set project parameters up for forward modelling 
        self.project.param['job_type'] = 0
        self.project.param['num_regions'] = 0
        self.project.param['num_poly'] = 0 
        self.project.param['res0File'] = 'resistivity.dat' 
        
        if run_keys is None: 
            run_keys = self.nodResultMulti.keys()
        if survey_keys is None:
            survey_keys = [i for i in range(self.resultNsteps)]
        
        # flag_3d = False # only 2D coupling handled for now, 3D would require 
        # extra development 
        
        # local mesh zonation and other parameters 
        zone=self.mesh.ptdf['zone'].values 
        mx = self.mesh.node[:,0]
        mz = self.mesh.node[:,2]
        depths_node  = self.mesh.ptdf['depths'].values 
        
        mesh = self.project.mesh
        rx = self.project.mesh.df['X'].values 
        # ry = self.project.mesh.df['Y'].values 
        rz = self.project.mesh.df['Z'].values 
        
        ipoints = np.c_[rx,rz] # if 3d this will need a third dimension 
        
        def loop(k):
            run = run_keys[k] 
            # note this will only do runs in stable solutions 
            dpath = os.path.join(self.dname,self.template.format(run))
            # read in step nodewise data 
            data = {}
            for f in os.listdir(dpath):
                if f.lower().endswith('.nod'):
                    data,n = readNod(os.path.join(dpath,f)) 
                    break 
            if len(data.keys())==0:
                raise Exception('Cannot find .nod file from sutra run')
            
            count = 0 
            # now to go and make all the resistivities at different steps where a survey took place 
            for i in survey_keys:
                # need to write, mesh file, .in file, resistivity file, and protocal file  
                key = 'step%i'%i
                satNode = np.array(data[key]['Saturation'])
                resNode = np.zeros_like(satNode)
                for j,m in enumerate(self.materials):
                    zidx = zone == m.zone 
                    resNode[zidx] = m.petro(satNode[zidx], depths_node[zidx])
                    if tcorrect:
                        resNode[zidx] = tfunc(resNode[zidx], depths_node[zidx], diy[count])

                # convert saturation to resistivity #
                ifunc = LinearNDInterpolator(np.c_[mx,mz],resNode) # interpolate to resipy mesh 
                nfunc = NearestNDInterpolator(np.c_[mx,mz],resNode) # use nearest lookup where nan occurs 
                res0 = ifunc(ipoints)
                nanidx = np.isnan(res0)
                res0[nanidx] = nfunc(ipoints[nanidx])
                fh = open(os.path.join(dpath,key+'.dat'),'w')
                for j in range(len(res0)):
                    line = "\t{: 10.5e}\t{: 10.5e}\t{: 10.5e}\t{: 10.5e}\n".format(rx[j],rz[j],res0[j],np.log10(res0[j]))
                    fh.write(line)
                fh.close() 

                mesh.dat(os.path.join(dpath,'mesh.dat'))
                write2in(self.project.param,dpath,typ=self.project.typ)
                
                # set sequence 
                seq = seqs[count]
                fh = open(os.path.join(dpath,'protocol%i.dat'%count),'w')
                fh.write('%i\n'%seq.shape[0])
                for j in range(seq.shape[0]):
                    line = '{:d} {:} {:} {:} {:}'.format(j+1,
                                                         seq[j,0],seq[j,1],
                                                         seq[j,2],seq[j,3])
                    line += '\n'
                    fh.write(line)
                fh.close() 
                count += 1 
                
        # if only one run key present then just loop once 
        if len(run_keys) == 1: 
            loop(0)
            return 
        # if using just one cpu do loop in serial 
        if ncpu <= 1:
            for k in tqdm(range(len(run_keys)),ncols=100,desc='Setting R2 files'):
                loop(k)
            return # jump out of function here 
        
        # otherwise run in parallel 
        Parallel(n_jobs=self.ncpu)(delayed(loop)(k) for k in tqdm(range(len(run_keys)),ncols=100,desc='Setting R2 files'))

                
    def runResFwdmdls(self,run_keys):
        execpath = os.path.join(self.project.apiPath,'exe','R2.exe')
        nrun = len(run_keys)
        if self.ncpu == 0 or self.ncpu == 1: # run single threaded with a basic for loop 
            for i in range(nrun): 
                runR2runs(self.pdirs[run_keys[i]],execpath) 
        else: # run in parallel 
            Parallel(n_jobs=self.ncpu)(delayed(runR2runs)(self.pdirs[run_keys[i]],execpath) for i in tqdm(range(nrun),ncols=100,desc='Running R2'))
        
    def runResFwdmdl(self,run_key):
        execpath = os.path.join(self.project.apiPath,'exe','R2.exe')
        pdir = os.path.join(self.dname,self.template.format(run_key))
        runR2runs(pdir,execpath,return_data=False,clean=True)

    def getFwdRunResults(self,run_keys):
        nsurveys = len(self.survey_keys)
        data_store = {}
        nruns = len(run_keys)
        for n in tqdm(range(nruns),desc='Retrieving R2 runs',ncols=100):
            run = run_keys[n]
            pdir = self.pdirs[run]
            data_seq = pd.DataFrame()
            for i in range(nsurveys):
                _, df = protocolParser(os.path.join(pdir,'forward%i.dat'%i))
                df = df.rename(columns={'resist':'tr'})
                df['sidx'] = i 
                data_seq = pd.concat([data_seq,df[['a','b','m','n','tr','sidx']]])
            data_store[run] = data_seq # hopefully this is not too memory intense ...  
            
        return data_store 
    
    def getFwdRunResult(self,run_key):
        nsurveys = len(self.survey_keys)
        pdir = os.path.join(self.dname,self.template.format(run_key)) # self.pdirs[run_key]
        data_seq = pd.DataFrame()
        for i in range(nsurveys):
            _, df = protocolParser(os.path.join(pdir,'forward%i.dat'%i))
            df = df.rename(columns={'resist':'tr'})
            df['sidx'] = i 
            data_seq = pd.concat([data_seq,df[['a','b','m','n','tr','sidx']]])
            
        return data_seq
            
    #%% markov chain monte carlo running 
    def mcmcProposer(self, ipargs, i, start=False):
        """
        Propose a model and create run directory for MCMC chains. 

        Parameters
        ----------
        ipargs : dict
            Initial parameters.
        i : int 
            Run indexer.
        start : Bool, optional
            If True then random values returned in a range, used to create 
            first model. The default is False.

        Returns
        -------
        pargs : dict 
            Proposed model.
        rdir : str
            Run directory.
        inrange : bool 
            Check on if values are in range or negative, False if these 
            conditions are true.

        """
        alpha = [-1]*self.nzones 
        swres = [-1]*self.nzones 
        swsat = [-1]*self.nzones 
        vn = [-1]*self.nzones 
        pargs = {'k':['-']*self.nzones,
                 'theta':['-']*self.nzones,
                 'res':['-']*self.nzones,
                 'sat':['-']*self.nzones,
                 'alpha':['-']*self.nzones,
                 'vn':['-']*self.nzones} 
        inrange = [True]*self.nzones # flag if parameters are within range 
        # loop through to get run param 
        for zone in range(self.nzones):
            zidx = self.mesh.ptdf['zone'] == (zone+1) # node zone idx 
            eidx = self.mesh.df['zone'] == (zone+1) # element zone idx 
            m = self.materials[zone]
             # gather petrubed parameters 
            if 'k' in m.MCparam.keys():
                v0 = m.MCparam['k'][0]
                v1 = m.MCparam['k'][2]
                size = m.MCparam['k'][1]
                # logger('mcmc proposer information for k_%i:'%(zone+1),self.logf) 
                if start: # if starting run then return random value 
                    pdist = 'uniform'
                    if 'log' in m.MCpdist['k']:
                        pdist = 'loguniform'
                    v = giveValues(v0, v1, 1, pdist)[0]
                    # logger('Input into give values',self.logf)
                    # logger('%e, %e, %i, %s'%(v0,v1,1,pdist),self.logf)
                else: 
                    v = stepWalk(ipargs['k'][zone], size, m.MCpdist['k'])
                    # logger('Input into step walk',self.logf)
                    # logger('%e, %f, %s'%(ipargs['k'][zone],size,m.MCpdist['k']),self.logf)
                self.mesh.ptdf.loc[zidx,'perm'] = v 
                pargs['k'][zone] = v 
                inrange[zone] = checkRange(v, v0, v1)
            
            if 'theta' in m.MCparam.keys():
                v0 = m.MCparam['theta'][0]
                v1 = m.MCparam['theta'][2]
                size = m.MCparam['theta'][1]
                if start: # if starting run then return random value 
                    pdist = 'uniform'
                    if 'log' in m.MCpdist['theta']:
                        pdist = 'loguniform'
                    v = giveValues(v0, v1, 1, pdist)[0]
                else: 
                    v = stepWalk(ipargs['theta'][zone], size, m.MCpdist['theta'])
                self.mesh.df.loc[eidx,'por'] = v 
                pargs['theta'][zone] = v 
                if inrange[zone]: #if true check if needs changing to false 
                    inrange[zone] = checkRange(v, v0, v1)
                
            if 'alpha' in m.MCparam.keys():               
                v0 = m.MCparam['alpha'][0]
                v1 = m.MCparam['alpha'][2]
                size = m.MCparam['alpha'][1]
                if start: # if starting run then return random value 
                    pdist = 'uniform'
                    if 'log' in m.MCpdist['alpha']:
                        pdist = 'loguniform'
                    v = giveValues(v0, v1, 1, pdist)[0]
                else: 
                    v = stepWalk(ipargs['alpha'][zone], size, m.MCpdist['alpha'])
                alpha[zone] = v 
                pargs['alpha'][zone] = v 
                if inrange[zone]:
                    inrange[zone] = checkRange(v, v0, v1)    
            else:
                alpha[zone] = m.alpha
            
            if 'vn' in m.MCparam.keys():
                v0 = m.MCparam['vn'][0]
                v1 = m.MCparam['vn'][2]
                size = m.MCparam['vn'][1]
                if start: # if starting run then return random value 
                    pdist = 'uniform'
                    if 'log' in m.MCpdist['vn']:
                        pdist = 'loguniform'
                    v = giveValues(v0, v1, 1, pdist)[0]
                else: 
                    v = stepWalk(ipargs['vn'][zone], size, m.MCpdist['vn'])
                vn[zone] = v 
                pargs['vn'][zone] = v 
                if inrange[zone]:
                    inrange[zone] = checkRange(v, v0, v1)
            else:
                vn[zone] = m.vn 
            
            if 'res' in m.MCparam.keys():
                v0 = m.MCparam['res'][0]
                v1 = m.MCparam['res'][2]
                size = m.MCparam['res'][1]
                if start: # if starting run then return random value 
                    pdist = 'uniform'
                    if 'log' in m.MCpdist['res']:
                        pdist = 'loguniform'
                    v = giveValues(v0, v1, 1, pdist)[0]
                else: 
                    v = stepWalk(ipargs['res'][zone], size, m.MCpdist['res'])
                swres[zone] = v 
                pargs['res'][zone] = v 
                if inrange[zone]:
                    inrange[zone] = checkRange(v, v0, v1)
            else:
                swres[zone] = m.res 
            
            if 'sat' in m.MCparam.keys():
                v0 = m.MCparam['sat'][0]
                v1 = m.MCparam['sat'][2]
                size = m.MCparam['sat'][1]
                if start: # if starting run then return random value 
                    pdist = 'uniform'
                    if 'log' in m.MCpdist['sat']:
                        pdist = 'loguniform'
                    v = giveValues(v0, v1, 1, pdist)[0]
                else: 
                    v = stepWalk(ipargs['sat'][zone], size, m.MCpdist['sat'])
                swsat[zone] = v 
                pargs['sat'][zone] = v 
                if inrange[zone]:
                    inrange[zone] = checkRange(v, v0, v1)
            else:
                swsat[zone] = m.sat  
                
        # create directory
        to_copy_ext = ['INP21A','BCS', 'ICS', 'RST', 'BCOP', 'BCOPG', 'FIL']
        to_copy = []
        files = os.listdir(self.dname)
        for f in files: # find files needed to run sutra 
            ext = f.split('.')[-1]
            if ext.upper() in to_copy_ext: # add it to files which need copying 
                to_copy.append(f)
        rdir = self.createPdir(i,to_copy,pargs) # run directory 
        self.writeInp(rdir)
        self.writeVg(swres,swsat,alpha,vn,dname=rdir) 

        return pargs, rdir, all(inrange)
    
    def setupRparam(self, tr, write2in, survey_keys, seqs=[], 
                    tfunc = None, diy= []):
        self.seqs = seqs
        self.tr = tr 
        self.write2in = write2in 
        self.survey_keys = survey_keys 
        self.tfunc = tfunc 
        self.diy = diy 
        
    def mcmc(self, nsteps=100,  target_ar = -1):
        """
        Run a markov chain monte carlo search. Needs updating to make use 
        of materials. 

        Parameters
        ----------
        tr : DataFrame, dict
            Transfer resistance data frame (real data). 
        nsteps : int, optional
            Maximum number of runs. The default is 100.
        target_ar : float, optional
            Target acceptance rate for adaptive metropolis hastings algorithm. 
            The default is -1. If negative default metropolis hastings is used. 

        Raises
        ------
        Exception
            If starting model is not stable.

        Returns
        -------
        chainlog: dict 
            Log of MCMC parameters and proposed models.
        ar: float 
            Acceptance rate. 

        """
        # steps: 
        # set mc trial parameters 
        # run trial parameters through SUTRA 
        # check if success 
        # convert to resistivity 
        # run resistivity through R2 
        # compute chi^2 / liklehood value 
        # run metropolis algorithm to accept or decline new model 
        # do 1-4 for initial model 
        # repeat for as many steps needed 
            
        # create log file 
        self.logf = os.path.join(self.dname, 'chain.log')
        fh = open(self.logf,'w')
        fh.close()

        # setup dataframe to log mcmc chain 
        self.chainlog = {}
        self.chainlog['run'] = [i for i in range(nsteps)]
        self.chainlog['Chi^2'] = [0.0]*nsteps 
        self.chainlog['Pt'] = [0.0]*nsteps 
        self.chainlog['mu'] = [0.0]*nsteps 
        self.chainlog['alpha'] = [0.0]*nsteps 
        self.chainlog['Accept'] = [False]*nsteps
        self.chainlog['Stable'] = [False]*nsteps 
        self.chainlog['ar'] = [0.0]*nsteps 
        
        pkeys = [] # altered parameter keys 
        pzones = [] # zones where parameters are altered 
        nparam = 0 
        for zone in range(self.nzones):# setup columns for storing model parameters in log 
            m = self.materials[zone]
            for key in m.MCparam.keys():
                self.chainlog['%s_%i'%(key,zone+1)] = [0.0]*nsteps
                pkeys.append(key)
                pzones.append(zone)
                nparam += 1 
                    
                
        # get starting place and set starting parameters for each mcmc chain 
        logger('______Starting MCMC search_______',self.logf)
        self.nruns = 0 # number of model runs 
        naccept = 0 # number of accepted models 
        stable = False # flag if initial model is stable, it must be in order to 
        
        # log distributions 
        logger('Parameters to be tested (and distributions):',self.logf)
        for zone in range(self.nzones):# setup columns for storing model parameters in log 
            m = self.materials[zone]
            for key in m.MCpdist.keys():
                logger('\t%s_%i = %s'%(key, zone+1, m.MCpdist[key]), self.logf)
        # continue to proposing new models in the mcmc chain. 
        logger('Generating initial model',self.logf)
        c = 0 # counter for trial model, ensures we dont get stuck in a infinite loop 
        maxc = 1000
        # run the first model 
        while not stable: # run until stable (or until max counter is reached)
            # failsafe statement 
            if c>maxc:
                break 
            c+=1 
            model, rdir, stable = self.mcmcProposer(None, 0,True)  # propose starting model 
            if not stable: # restart loop if not stable  
                logger('Initial model out of parameter range, selecting new parameters',self.logf)
                continue 
            # run init parameters 
            logger('Running SUTRA model...',self.logf)
            data,n = doSUTRArun(rdir,self.execpath)
            if n < self.resultNsteps: # n does not meet the expected number of steps 
                stable = False 
                logger('Initial SUTRA model is unstable, attempting to run again with new starting parameters',self.logf)
                continue 
            logger('Done.',self.logf)
        
        if not stable: # if no stable model is found for start then we must exit 
            logger('Initial trial is unstable solution! Try a different starting model',self.logf) # dont accept the trial model 
            return self.chainlog, naccept/nsteps
        
        # convert result to resistivity for R2/(or R3t) and setup forward modelling directory 
        logger('Writing out resistivity files',self.logf)
        self.setupRruns(self.write2in, [0], self.survey_keys, self.seqs,
                        tfunc=self.tfunc, diy=self.diy)

        # now run resistivity code 
        logger('Running forward resistivity models',self.logf)
        self.runResFwdmdl(0)
        
        # get result 
        logger('Retrieving forward resistivity runs',self.logf)
        data_seq = self.getFwdRunResult(0)
        
        # compute initial chi^2 value 
        d0 = self.tr['tr'].values # get real data values 
        d1 = data_seq['tr'].values # get synthetic data values 
        error = self.tr['error'].values # get error estimates 
        
        residuals = d0 - d1 
        X2 = chi2(error, residuals)
        Pi = normLike(error, residuals)
        
        self.chainlog['Chi^2'][0] = X2 
        self.chainlog['Pt'][0] = Pi  
        self.chainlog['Accept'][0] = True 
        self.chainlog['Stable'][0] = True 
        
        logger('\nRun %i'%0,self.logf)
        logger('Pi = %f'%Pi,self.logf)

        logoutput = 'Parameters: \n' 
        for j in range(nparam):
            key = pkeys[j]
            zone = pzones[j] # add one to get index starting at 1 
            self.chainlog['%s_%i'%(key,zone+1)][0] = model[key][zone]
            logoutput += '%s_%i = %f\t'%(key,zone,model[key][zone])
               
        logger(logoutput,self.logf) 
        logger('Done.',self.logf)
        
        
        # setup parameters for altering the acceptance rate on the fly 
        ar = 1 # acceptance rate 
        a_fac = 1 # alpha factor. modified according to the target acceptance rate 
        ac = [] # empty list for acceptance chain 
        
        logger('Starting search from initial model',self.logf)
        
        # for loop goes here 
        for i in range(1,nsteps):
            # delete old runs to save on file space 
            if i>3:
                dpath = os.path.join(self.dname,self.template.format(i-3)) 
                if os.path.isdir(dpath):
                    if 'pargs.txt' in os.listdir(dpath):
                        shutil.rmtree(dpath)
                    
            logger('\nRun %i'%i,self.logf)
            self.nruns += 1 
                
            accept = False 
            # do random walk for model parameters
            trial, rdir, inrange = self.mcmcProposer(model, i)
            if not inrange: # check trial is within limits (otherwise move on)
                accept = False 
                logger('Proposed trial model is out of limits',self.logf)
                logger('Accept = False',self.logf)
                ac.append(0) # add 0 to acceptance chain so that unstable models dont get unfairly weighted 
                continue 
            
            # run trial parameters  
            logger('Running SUTRA model...',self.logf)
            data,n = doSUTRArun(rdir,self.execpath)
            if n < self.resultNsteps: # n does not meet the expected number of steps 
                accept = False 
                logger('Proposed model is unstable!',self.logf)
                logger('Accept = False',self.logf)
                ac.append(0)
                continue 
            
            # convert result to resistivity for R2/(or R3t) and setup forward modelling directory 
            logger('Running R2 models',self.logf)
            self.setupRruns(self.write2in, [i], self.survey_keys, self.seqs,
                            tfunc=self.tfunc, diy=self.diy)
    
            # now run resistivity code 
            self.runResFwdmdl(i)
            # get result 
            data_seq = self.getFwdRunResult(i)
                
            # compute chi^2 and likelihood values  
            d1 = data_seq['tr'].values 
            if len(d1) != len(d0):
                accept = False 
                logger('Proposed model is unstable in R2!',self.logf)
                logger('Accept = False',self.logf)
                ac.append(0)
                continue 
            
            residuals = d0 - d1  # compare fout to 'real data' 
            
            X2 = chi2(error, residuals)
            Pt = normLike(error, residuals)
            
            logger('Pt = %f, Pi = %f'%(Pt,Pi),self.logf)
            
            ## adaptation to metropolis ## 
            if target_ar < 0: # perform default metropolis 
                a_fac = 1 # hence alpha factor is 1 
            elif len(ac) < 10: # not enough samples in acceptance chain yet 
                a_fac = 1 # alpha factor is 1 
            elif Pt <= Pi:
                ar = sum(ac)/i # current acceptance rate 
                if ar == 0: 
                    # if acceptance rate is zero then default back to metropolis 
                    a_fac = 1 
                else: 
                    # adjust acceptance rate to match target acceptance rate 
                    a_fac = a_fac*(target_ar/ar)
                if a_fac >= 1: # cap alpha factor at 1 
                    a_fac = 1 
                else: # if alpha factor < 1 mention it in console 
                    logger('Acceptance probability adjusted*',self.logf)
            
            ## metropolis algorithm (adapted) ##
            alpha = (Pt/Pi)*a_fac
            mu = np.random.random() 
            if Pt > Pi:
                accept = True 
            elif alpha > mu:
                accept = True  
                
            logger('Alpha = %f, mu = %f'%(alpha,mu),self.logf)
                
            self.chainlog['Chi^2'][i] = X2 
            self.chainlog['Pt'][i] = Pt 
            self.chainlog['mu'][i] = mu 
            self.chainlog['alpha'][i] = alpha 
            self.chainlog['Accept'][i] = accept 
            self.chainlog['Stable'][i] = True 
            self.chainlog['ar'][i] = ar 
            logoutput = 'Parameters: \n'
            for j in range(nparam):
                key = pkeys[j]
                zone = pzones[j] # add one to get index starting at 1 
                self.chainlog['%s_%i'%(key,zone+1)][i] = trial[key][zone]
                logoutput += '%s_%i = %e\t'%(key,zone+1,trial[key][zone])
            logger(logoutput,self.logf) 
                
            # decide if to accept model 
            if accept:
                logger('Accept = True',self.logf)
                model = trial.copy() # trial now becomes current model  
                Pi = Pt 
                naccept += 1 
                ac.append(1)
            else:
                ac.append(0)
                logger('Accept = False',self.logf)
            
        logger('Max steps reached: End of chain',self.logf)
            
        return self.chainlog, naccept/nsteps
    
    #%% write last time step to file for warm start up runs 
    def writeLastStep(self):
        nstep = self.resultNsteps
        step = pd.DataFrame(self.nodResult['step%i'%(nstep-1)])
        step.to_csv(os.path.join(self.dname,'warm.csv'))
        
    
#%% convert input file to mesh file 
def mesh2dat(fname):
    # read in mesh file
    fh = open(fname, 'r')
    dump = fh.readlines()
    fh.close()
    nlines = len(dump)
    
    node_idx = 0
    connec_idx = 0
    node_dump = []
    connec_dump = []
    for i in range(nlines):
        line = dump[i].strip()
        if 'NODE' in line:
            node_idx = i
            # find where first entry on line is no longer a digit
            j = node_idx+1
            numnp = 0
            while True:
                l = dump[j].strip()
                if l[0] == '#':
                    pass  # then ignore
                elif l[0].isdigit():
                    numnp += 1
                    node_dump.append(l)
                else:
                    break
                j += 1
        if 'INCIDENCE' in line:
            connec_idx = i
            j = connec_idx+1
            numel = 0
            while True:
                l = dump[j].strip()
                if l[0] == '#':
                    pass  # then ignore
                elif l[0].isdigit():
                    numel += 1
                    connec_dump.append(l)
                else:
                    break
                j += 1
                if j == nlines:  # break if at bottom of file
                    break
    
    # create arrays
    connec = np.zeros((numel, 4), dtype=int)
    node = np.zeros((numnp, 3), dtype=float)
    
    for i in range(numnp):
        info = node_dump[i].split()
        node[i, 0] = float(info[2])
        node[i, 2] = float(info[3])
    
    for i in range(numel):
        info = connec_dump[i].split()
        for j in range(4):
            connec[i, j] = int(info[j+1])-1
            
    #write to file 
    fh = open(fname.replace('.inp','.dat'))
    numel = connec.shape[0]
    numnp = node.shape[0]
    fh.write('%i %i 0\n'%(numel,numnp))
    for i in range(numel):
        fh.write('%i\t'%(i+1))
        for j in range(connec.shape[1]):
            fh.write('%i\t'%connec[i,j])
        fh.write('\n')
    for i in range(numnp):
        fh.write('%i\t'%(i+1))
        for j in range(node.shape[1]):
            fh.write('%f\t'%node[i,j])
        fh.write('\n')
    fh.close()


#%% legacy code 
    # def checkParam(self):
    #     if len(self.parg0.keys())==0:
    #         raise Exception('No input variables given to vary')

    #     allowed_param = ['k','theta','sat','res','alpha','vn']
        
    #     for key in self.parg0.keys():
    #         if key not in self.parg1.keys():
    #             raise Exception('Parameter %s not defined self.parg1')
    #         if key not in allowed_param:
    #             print('Parameters which can be varied = ')
    #             for a in allowed_param:
    #                 print(a)
    #             raise Exception('Parameter not allowed to be varied')
     
 # def pres2res(self):
 #     if self.resultNsteps is None:
 #         raise Exception('Must run a model and get results first before converting to resistivity')
 #     if 'ecres' not in self.param.keys():
 #         raise Exception('ecres not in defined in paramters')
 #     if 'ecsat' not in self.param.keys():
 #         raise Exception('ecsat not in defined in paramters')
     
 #     cond = np.zeros(self.mesh.numnp) # array to hold conductivity values 
 #     res = np.zeros((self.mesh.numel,self.resultNsteps))
 #     ECsat = self.param['ecsat'] # parameter arrays 
 #     ECres = self.param['ecres']
 #     alpha = self.param['alpha']
 #     vn = self.param['vn']
 #     for i in range(self.resultNsteps):
 #     # for i in tqdm(range(self.resultNsteps),ncols=100): # generally this bit of code is very quick 
 #         pres = np.array(self.nodResult['step%i'%i]['Pressure']) # pressure in kpa 
 #         sat_idx = pres >= 0 # where saturation level is 1 
 #         unsat_idx = pres < 0 
 #         c=0
 #         for j in np.unique(self.mesh.ptdf['zone'].values):
 #             sat_idx = (self.mesh.ptdf['zone'].values == j) & sat_idx # index relevant to zone 
 #             unsat_idx = (self.mesh.ptdf['zone'].values == j) & unsat_idx 
 #             cond[sat_idx] = ECsat[c] # saturated region just the saturated conductivity  
 #             normcond = vgCurve(np.abs(pres[unsat_idx]), # need absolute values for function 
 #                                ECres[c],ECsat[c],alpha[c],vn[c])
 #             # non - normalised conductivity 
 #             cond[unsat_idx] = (normcond*(ECsat[c]-ECres[c]))+ECres[c]
 #             c+=1 # add one to loop iterator index 
         
 #         # add conductivity values to data 
 #         self.nodResult['step%i'%i]['Conductivity'] = cond.tolist() 
 #         self.nodResult['step%i'%i]['Resistivity'] = (1/cond).tolist() 
 #         # covert to resistivity and map to elements 
 #         res[:,i] = self.mesh.node2ElemAttr(1/cond,'res%i'%i)
         
 #     return res 
 
 # def sat2res(self):
 #     if self.resultNsteps is None:
 #         raise Exception('Must run a model and get results first before converting to resistivity')

 #     if self.waxman['F'] is None:
 #         raise Exception('Formation factors not defined')
 #     if self.waxman['sigma_w'] is None:
 #         raise Exception('Pore fluid conductivity not defined')
 #     if self.waxman['sigma_s'] is None:
 #         raise Exception('Grain surface conductivity not defined')
         
 #     cond = np.zeros(self.mesh.numnp) # array to hold conductivity values 
 #     res = np.zeros((self.mesh.numel,self.resultNsteps))
     
 #     multirun = False 
 #     if len(self.nodResultMulti.keys())>0:
 #         multirun = True 
         
 #     for i in range(self.resultNsteps):
 #     # for i in tqdm(range(self.resultNsteps),ncols=100): # generally this bit of code is very quick 
 #         sat = np.array(self.nodResult['step%i'%i]['Saturation']) # pressure in kpa 
 #         sat[sat>1] = 1 # cap saturation at 1 
 #         c=0
 #         cond = np.zeros_like(sat)
 #         for j in np.unique(self.mesh.ptdf['zone'].values):
 #             idx = self.mesh.ptdf['zone'].values == j 
 #             cond[idx] = waxmanSmit(sat[idx],
 #                               self.waxman['F'][c],
 #                               self.waxman['sigma_w'][c],
 #                               self.waxman['sigma_s'][c])
 #             c+=1 # add one to loop iterator index 
         
 #         # add conductivity values to data 
 #         self.nodResult['step%i'%i]['Conductivity'] = cond.tolist() 
 #         self.nodResult['step%i'%i]['Resistivity'] = (1/cond).tolist() 
 #         # covert to resistivity and map to elements 
 #         res[:,i] = self.mesh.node2ElemAttr(1/cond,'res%i'%i)
         
 #         if not multirun:
 #             continue 
         
 #         # catch mutliple runs results as well 
 #         for key in self.nodResultMulti.keys():
 #             sat = np.array(self.nodResultMulti[key]['step%i'%i]['Saturation']) # pressure in kpa 
 #             sat[sat>1] = 1 # cap saturation at 1 
 #             c=0
 #             cond = np.zeros_like(sat)
 #             for j in np.unique(self.mesh.ptdf['zone'].values):
 #                 idx = self.mesh.ptdf['zone'].values == j 
 #                 cond[idx] = waxmanSmit(sat[idx],
 #                                   self.waxman['F'][c],
 #                                   self.waxman['sigma_w'][c],
 #                                   self.waxman['sigma_s'][c])
 #                 c+=1 # add one to loop iterator index 
             
 #             # add conductivity values to data 
 #             self.nodResultMulti[key]['step%i'%i]['Conductivity'] = cond.tolist() 
 #             self.nodResultMulti[key]['step%i'%i]['Resistivity'] = (1/cond).tolist() 
             
 #     return res 
 
 # def sat2mois(self):
 #     if self.resultNsteps is None:
 #         raise Exception('Must run a model and get results first before converting to moisture contents')
 #     if 'theta_sat' not in self.param.keys():
 #         raise Exception('Theta Sat must be defined!')
         
 #     mois = np.zeros(self.mesh.numnp) # array to hold conductivity values 
 #     vmc = np.zeros((self.mesh.numel,self.resultNsteps))
         
 #     for i in range(self.resultNsteps):
 #     # for i in tqdm(range(self.resultNsteps),ncols=100): # generally this bit of code is very quick 
 #         sat = np.array(self.nodResult['step%i'%i]['Saturation']) # pressure in kpa 
 #         sat[sat>1] = 1 # cap saturation at 1 
 #         c=0
 #         for j in np.unique(self.mesh.ptdf['zone'].values):
 #             mois = self.param['theta_sat'][c]*sat
 #             c+=1 # add one to loop iterator index 
         
 #         # add moisture values to data 
 #         self.nodResult['step%i'%i]['Theta'] = mois.tolist() 

 #         # and map to elements 
 #         vmc[:,i] = self.mesh.node2ElemAttr(mois,'theta%i'%i)
         
 #     return vmc 
 
 # def sat2res2phase(self,run_keys=None):
 #     if self.resultNsteps is None:
 #         raise Exception('Must run a model and get results first before converting to resistivity')

 #     if self.waxman['F'] is None:
 #         raise Exception('Formation factors not defined')
 #     if self.waxman['sigma_w0'] is None:
 #         raise Exception('Pore fluid conductivity 0 not defined')
 #     if self.waxman['sigma_w1'] is None:
 #         raise Exception('Pore fluid conductivity 1 not defined')
 #     if self.waxman['sat0'] is None:
 #         raise Exception('Need to define baseline saturation')
 #     if self.waxman['sigma_s'] is None:
 #         raise Exception('Grain surface conductivity not defined')
         
 #     if run_keys is None: 
 #         run_keys = self.nodResultMulti.keys() 
         
 #     cond = np.zeros(self.mesh.numnp) # array to hold conductivity values 
 #     res = np.zeros((self.mesh.numel,self.resultNsteps))
         
 #     for i in range(self.resultNsteps):
 #     # for i in tqdm(range(self.resultNsteps),ncols=100): # generally this bit of code is very quick 
 #         sat = np.array(self.nodResult['step%i'%i]['Saturation']) # pressure in kpa 
 #         sat[sat>1] = 1 # cap saturation at 1 
 #         c=0
 #         cond[:] = 0 
 #         for j in np.unique(self.mesh.ptdf['zone'].values):
 #             idx = self.mesh.ptdf['zone'].values == j 
 #             # sigma_w calculation (average of 2 phases)
 #             sigma_w0 = self.waxman['sigma_w0'][c]
 #             sigma_w1 = self.waxman['sigma_w1'][c]
 #             sat0 = self.waxman['sat0'][c]
 #             delta = sat[idx] - sat0 
 #             delta[delta<0] = 0 # cap delta 
 #             # sigma_w[delta == 0] = sigma_w0 
 #             sigma_w = ((delta/sat[idx])*sigma_w1)+((sat0/sat[idx])*sigma_w0)
 #             # compute conductivity 
 #             cond[idx] = waxmanSmit(sat[idx],
 #                               self.waxman['F'][c],
 #                               sigma_w,
 #                               self.waxman['sigma_s'][c])
 #             c+=1 # add one to loop iterator index 
         
 #         # add conductivity values to data 
 #         self.nodResult['step%i'%i]['Conductivity'] = cond.tolist() 
 #         self.nodResult['step%i'%i]['Resistivity'] = (1/cond).tolist() 
 #         # covert to resistivity and map to elements 
 #         res[:,i] = self.mesh.node2ElemAttr(1/cond,'res%i'%i)
         
 #         # catch mutliple runs results as well 
 #         for run in run_keys:
 #             sat = np.array(self.nodResultMulti[run]['step%i'%i]['Saturation']) # pressure in kpa 
 #             sat[sat>1] = 1 # cap saturation at 1 
 #             c=0
 #             cond = np.zeros_like(sat)
 #             for j in np.unique(self.mesh.ptdf['zone'].values):
 #                 idx = self.mesh.ptdf['zone'].values == j 
 #                 sigma_w0 = self.waxman['sigma_w0'][c]
 #                 sigma_w1 = self.waxman['sigma_w1'][c]
 #                 sat0 = self.waxman['sat0'][c]
 #                 delta = sat[idx] - sat0 
 #                 delta[delta<0] = 0 # cap delta 
 #                 sigma_w = ((delta/sat[idx])*sigma_w1)+((sat0/sat[idx])*sigma_w0) # compute sigma_w 
 #                 # compute conductivity 
 #                 cond[idx] = waxmanSmit(sat[idx],
 #                                   self.waxman['F'][c],
 #                                   sigma_w,
 #                                   self.waxman['sigma_s'][c])
 #                 c+=1 # add one to loop iterator index 
             
 #             # add conductivity values to data 
 #             self.nodResultMulti[run]['step%i'%i]['Conductivity'] = cond.tolist() 
 #             self.nodResultMulti[run]['step%i'%i]['Resistivity'] = (1/cond).tolist() 
         
 #     return res 
 
 # def res2sat(self,param='Resistivity'):
 #     """
 #     Convert resistivity back to saturation 

 #     Parameters
 #     ----------
 #     param : str, optional
 #         DESCRIPTION. The default is 'Resistivity'.

 #     Returns
 #     -------
 #     None.

 #     """
 #     if self.resultNsteps is None:
 #         raise Exception('Must run a model and get results first before converting to resistivity')

 #     if self.waxman['F'] is None:
 #         raise Exception('Formation factors not defined')
 #     if self.waxman['sigma_w'] is None:
 #         raise Exception('Pore fluid conductivity not defined')
 #     if self.waxman['sigma_s'] is None:
 #         raise Exception('Grain surface conductivity not defined')
         
 #     sw = np.zeros(self.mesh.numnp) # array to hold conductivity values 
 #     sat = np.zeros((self.mesh.numel,self.resultNsteps))
         
 #     for i in range(self.resultNsteps):
 #     # for i in tqdm(range(self.resultNsteps),ncols=100): # generally this bit of code is very quick 
 #         res = np.array(self.nodResult['step%i'%i][param]) # resisitivity?
 #         cond = 1/res 
 #         c=0
 #         for j in np.unique(self.mesh.ptdf['zone'].values):
 #             idx = np.argwhere(self.mesh.ptdf['zone'].values == j).flatten()
 #             for k in idx:
 #                 sw[k] = invWaxmanSmit(cond[k],
 #                                       self.waxman['F'][c],
 #                                       self.waxman['sigma_w'][c],
 #                                       self.waxman['sigma_s'][c])
 #             c+=1 # add one to loop iterator index 
         
 #         # add conductivity values to data 
 #         self.nodResult['step%i'%i]['Inverted_Saturation'] = sw.tolist() 
 #         # covert to resistivity and map to elements 
 #         sat[:,i] = self.mesh.node2ElemAttr(sw,'sat%i'%i)
         
 #     return sat