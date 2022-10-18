cimport cython  #import relevant modules 
import math as ma 
import numpy as np
cimport numpy as np

cdef double gmcFunc(double gmc, double Rt, double Rw,
                    double Pg, double Pw, double theta,
                    double cec, double FF, double n):
    
    cdef double phi, B, sigma_w # constants 
    cdef double iblock1,iblock2, oblock1, oblock2, oblock3 # blocks of equation 
    cdef double gmc_calc # function output 
    
    #compute constants
    phi = 1-theta # 1- porosity constant
    B = 4.6*(1-(0.6*ma.exp(-0.77/Rw)))
    sigma_w = 1/Rw # conductivity of pore fluid
    
    iblock1 = (phi*Pg*cec)/(100*theta) # inner block of equation 
    iblock2 = (theta*Pw)/(phi*Pg*gmc)
    oblock1 = sigma_w + (B*iblock1*iblock2) # outer block 1 of equation 
    oblock2 = ((FF/Rt)*(1/oblock1))**(1/n)
    oblock3 = (theta*Pw)/(phi*Pg)     

    gmc_calc = oblock3*oblock2    

    return gmc_calc      

cdef double rtFunc(double gmc, double Rw,
                    double Pg, double Pw, double theta,
                    double cec, double FF, double n):
    cdef double phi, B, sigma_w, Rt 
    cdef double iblock0, iblock1, iblock2, oblock0, oblock1 
    
    #compute constants
    phi = 1-theta # 1- porosity constant
    B = 4.6*(1-(0.6*ma.exp(-0.77/Rw)))
    sigma_w = 1/Rw # conductivity of pore fluid
    
    # solve equation 
    iblock0 = (phi*Pg*gmc)/(theta*Pw)
    oblock0 = FF*(iblock0**-n)
    iblock1 = (phi*Pg*cec)/(100*theta) # inner block of equation 
    iblock2 = (theta*Pw)/(phi*Pg*gmc)
    oblock1 = sigma_w + (B*iblock1*iblock2)
    
    Rt = oblock0 * (1/oblock1)
    
    return Rt 
    
def solveGmc(double[:] Rt, double Rw, double Pg, double Pw,
             double theta, double cec, double FF, double n):#working!
    """Convert true rock resistivity into a gravimetric moisture content 
    using modified waxman-smit model. After Ulhemann et al 2017. 
    model.
    
    Parameters
    ---------- 
    Rt: Array 
        total rock resistivity 
    Rw: double
        pore fluid resistivity 
    Pg: double
        grain density 
    Pw: double
        water density 
    theta:  double
        porosity (fractional)
    cec: double
        cation exchange capacity
    FF: double 
        formation factor 
    n: double
        Archie's saturation exponent
    
    Returns 
    ---------- 
    gmc: array double
        water (or conductive phase) gravimetric moisture content
    """
    cdef double trial_gmc, calc_gmc, delta 
    cdef int c = 0 
    cdef int i 
    cdef int nmeas = len(Rt) 
    cdef np.ndarray[double, ndim=1] gmc = np.zeros(nmeas, dtype=float)
    cdef double[:] gmca = np.array(gmc, dtype=float)
    
    for i in range(nmeas):
        c = 0 
        # set up trial gmc 
        trial_gmc = 0.01 
        # minimization scheme
        calc_gmc = gmcFunc(trial_gmc,Rt[i], Rw,Pg,Pw,theta,cec,FF,n)
        delta = calc_gmc-trial_gmc
        
        while abs(delta) > 0.0001: #
            calc_gmc = gmcFunc(trial_gmc,Rt[i], Rw,Pg,Pw,theta,cec,FF,n)
            delta = calc_gmc-trial_gmc
            trial_gmc = trial_gmc + delta
            c +=1
            if c > 1000:#fail safe 
                break 
        gmc[i] = trial_gmc 
    
    return gmc 


cdef float polyval(double[:] p, double x):
    
    cdef int i 
    cdef int nparam = len(p)
    cdef float n = nparam - 1.0 
    cdef float a = 0 
    
    for i in range(nparam): 
        a += p[i]*(x**n)
        n -= 1.0 
        
    return a 

cdef float porFunc(double [:] p, double x):
    cdef int i 
    cdef float e, er, es, alpha, n, m 

    if len(p) != 4: # fall back out if not expected 
        return -1.0 
    er = p[0]
    es = p[1]
    alpha = p[2]
    n = p[3]
    m = 1 - (1/n)
    numon = es - er 
    denom = 1 + ((x*alpha)/(es-x))**-n 
    e = er + numon/(denom**m)  
    return e 
    
def solveGmcWVP(double[:] Rt, double Rw, double Pg, double Pw,
             double[:] theta_param, double cec, double FF, double n):
    """Convert true rock resistivity into a gravimetric moisture content 
    using modified waxman-smit model. After Ulhemann et al 2017. 
    model. In this function a varying porosity is expected. 
    
    Parameters
    ---------- 
    Rt: Array 
        total rock resistivity 
    Rw: double
        pore fluid resistivity 
    Pg: double
        grain density 
    Pw: double
        water density 
    theta_param: array 
        Fitting parameters for theta 
    cec: double
        cation exchange capacity
    FF: double 
        formation factor 
    n: double
        Archie's saturation exponent
    shrink_limit: float, optional 
        Limit at which porosity changes, in which case porosity will be capped 
        at value for shrinkage limit gmc. 
        
    Returns 
    ---------- 
    gmc: array double
        water (or conductive phase) gravimetric moisture content
    """
    cdef double trial_gmc, calc_gmc, delta, theta  
    cdef int c = 0 # number of times while loop is run 
    cdef int i 
    cdef int nmeas = len(Rt) 
    cdef np.ndarray[double, ndim=1] gmc = np.zeros(nmeas, dtype=float)
    # cdef double[:] gmca = np.array(gmc, dtype=float)
    
    for i in range(nmeas):
        c = 0 
        # set up trial gmc 
        trial_gmc = 0.01 # propose a starting gmc value 
        
        # lookup a value of porosity to match the trial gmc value 
        theta = porFunc(theta_param, trial_gmc)
            
        # compute an estimated gmc value and difference 
        calc_gmc = gmcFunc(trial_gmc,Rt[i], Rw,Pg,Pw,theta,cec,FF,n)
        delta = calc_gmc-trial_gmc
        
        # minimization scheme
        while abs(delta) > 0.0001: #
            theta = porFunc(theta_param, trial_gmc)
            calc_gmc = gmcFunc(trial_gmc, Rt[i], Rw,Pg,Pw,theta,cec,FF,n)
            delta = calc_gmc-trial_gmc
            trial_gmc = trial_gmc + delta
            c +=1
            if c > 1000:#fail safe 
                calc_gmc = -1 
                break 
            
        gmc[i] = trial_gmc 
    
    return gmc 

def solveGmcWVPb(double[:] Rt, double Rw, double Pg, double Pw,
             double[:] theta_param, double cec, double FF, double n):
    """Convert true rock resistivity into a gravimetric moisture content 
    using modified waxman-smit model. After Ulhemann et al 2017. 
    model. In this function a varying porosity is expected. Solver quits when
    trial_gmc is within 10% of true value. 
    
    Parameters
    ---------- 
    Rt: Array 
        total rock resistivity 
    Rw: double
        pore fluid resistivity 
    Pg: double
        grain density 
    Pw: double
        water density 
    theta_param: array 
        Fitting parameters for theta 
    cec: double
        cation exchange capacity
    FF: double 
        formation factor 
    n: double
        Archie's saturation exponent
    shrink_limit: float, optional 
        Limit at which porosity changes, in which case porosity will be capped 
        at value for shrinkage limit gmc. 
        
    Returns 
    ---------- 
    gmc: array double
        water (or conductive phase) gravimetric moisture content
    """
    cdef double trial_gmc, calc_gmc, rt_trial, delta, theta  
    cdef int c = 0 # number of times while loop is run 
    cdef int i 
    cdef int nmeas = len(Rt) 
    cdef np.ndarray[double, ndim=1] gmc = np.zeros(nmeas, dtype=float)
    # cdef double[:] gmca = np.array(gmc, dtype=float)
    
    for i in range(nmeas):
        c = 0 
        # set up trial gmc 
        trial_gmc = 0.5 # propose a starting gmc value 
        
        # lookup a value of porosity to match the trial gmc value 
        theta = porFunc(theta_param, trial_gmc)
            
        # compute an estimated gmc value and difference 
        calc_gmc = gmcFunc(trial_gmc,Rt[i], Rw,Pg,Pw,theta,cec,FF,n)
        delta = calc_gmc-trial_gmc
        
        # minimization scheme
        while abs(delta) > 0.0001: #
            theta = porFunc(theta_param, trial_gmc)
            calc_gmc = gmcFunc(trial_gmc, Rt[i], Rw,Pg,Pw,theta,cec,FF,n)
            delta = calc_gmc-trial_gmc
            trial_gmc = trial_gmc + delta
            c +=1
            if c > 1000:#fail safe 
                calc_gmc = -1 
                break 
            rt_trial = rtFunc(trial_gmc, Rw, Pg, Pw, theta, cec, FF, n)
            if abs(rt_trial-Rt[i]) < Rt[i]*0.1:
                break 
            
        gmc[i] = trial_gmc 
    
    return gmc 

def solveRt(double[:] gmc, double Rw, double Pg, double Pw,
            double theta, double cec, double FF, double n):#working!
    """Convert true rock resistivity into a gravimetric moisture content 
    using modified waxman-smit model. After Ulhemann et al 2017. 
    model.
    
    Parameters
    ---------- 
    Gmc: Array 
        Gravimetric moisture content  
    Rw: double
        pore fluid resistivity 
    Pg: double
        grain density 
    Pw: double
        water density 
    theta:  double
        porosity (fractional)
    cec: double
        cation exchange capacity
    FF: double 
        formation factor 
    n: double
        Archie's saturation exponent
    
    Returns 
    ---------- 
    Rt: array double
        Total rock resistivity 
    """
    cdef int i 
    cdef int nmeas = len(gmc) 
    cdef np.ndarray[double, ndim=1] Rt = np.zeros(nmeas, dtype=float)
    # cdef double[:] Rta = np.array(Rt, dtype=float)
    
    for i in range(nmeas):
        Rt[i] = rtFunc(gmc[i], Rw, Pg, Pw, theta, cec, FF, n)
        
    return Rt 

def solveRtWVP(double[:] gmc, double Rw, double Pg, double Pw,
             double[:] theta_param, double cec, double FF, double n):
    """Convert true rock resistivity into a gravimetric moisture content 
    using modified waxman-smit model. After Ulhemann et al 2017. 
    model. In this function a varying porosity is expected. 
    
    Parameters
    ---------- 
    Gmc: Array 
        Gravimetric moisture content  
    Rw: double
        pore fluid resistivity 
    Pg: double
        grain density 
    Pw: double
        water density 
    theta_param: array 
        Fitting parameters for theta according to peng and hang,
        format is er, es, a, n 
    cec: double
        cation exchange capacity
    FF: double 
        formation factor 
    n: double
        Archie's saturation exponent
        
    Returns 
    ---------- 
    Rt: array double
        Total rock resistivity 
    """
    cdef double theta  
    cdef int i 
    cdef int nmeas = len(gmc) 
    cdef np.ndarray[double, ndim=1] Rt = np.zeros(nmeas, dtype=float)
    
    for i in range(nmeas):        
        # lookup a value of porosity to match the gmc value 
        theta = porFunc(theta_param, gmc[i])
        Rt[i] = rtFunc(gmc[i], Rw, Pg, Pw, theta, cec, FF, n)
        
    return Rt

def montaronWVP(double[:] gmc, double Rw, double Pg, double Pw,
                double[:] theta_param, double a, double chi, double mu):
    """Convert true rock resistivity into a gravimetric moisture content 
    using modified montaron(2009) relationship. 
    
    Parameters
    ---------- 
    Gmc: Array 
        Gravimetric moisture content  
    Rw: double
        pore fluid resistivity 
    Pg: double
        grain density 
    Pw: double
        water density 
    theta_param: array 
        Fitting parameters for theta according to peng and hang,
        format is er, es, a, n 
    a: double
        archie a, usually set to 1 
    FF: double 
        formation factor 
    n: double
        Archie's saturation exponent
        
    Returns 
    ---------- 
    Rt: array double
        Total rock resistivity 
    """
    cdef double theta, phi, Sw, denom, numon  
    cdef int i 
    cdef int nmeas = len(gmc) 
    cdef np.ndarray[double, ndim=1] Rt = np.zeros(nmeas, dtype=float)
    
    for i in range(nmeas):     
        # compute porosity using function 
        theta = porFunc(theta_param, gmc[i])        
        # convert gmc to Sw
        phi = 1.0 - theta
        Sw = (phi/theta)*(Pg/Pw)*gmc[i] 
        # compute Rt 
        numon = a*Rw*((1.0-chi)**mu)
        denom = ((Sw*theta)-phi)**mu
        # denom = (Sw*theta)**mu
        Rt[i] = numon/denom 
    return Rt 
