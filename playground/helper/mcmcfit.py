#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 09:07:17 2019

@author: yuhanyao
"""
import os
import numpy as np
import pandas as pd
import emcee
import time
import scipy.optimize as op
import astropy.constants as const
from multiprocessing import Pool


# =========================================================================== #
# define MCMC functions for blackbody fit

def planck_lambda(T, Rbb, lamb):
    '''
    T in the unit of K
    Rbb in the unit of Rsun
    lamb in the unit of AA
    '''
    ANGSTROM = 1.0e-8
    # convert to cm for planck equation
    lamb2 = lamb * ANGSTROM
    x = const.h.cgs.value * const.c.cgs.value / (const.k_B.cgs.value * T * lamb2)
    x = np.array(x)
    Blambda = (2. * const.h.cgs.value * const.c.cgs.value**2 ) /  (lamb2**5. ) / (np.exp(x) - 1. )
    # convert back to ANGSTROM   
    spec = Blambda*ANGSTROM # in units of erg/cm2/Ang/sr/s
    Rbb *= const.R_sun.cgs.value
    spec1 = spec * (4. * np.pi * Rbb**2) * np.pi # erg/AA/s
    # spec1 *= 1./ (4*np.pi*D**2) to correct for distance
    return spec1


def bb_lnlike(theta, wave, Llambda, Llambda_unc):
    Tbb, Rbb = theta
    model = planck_lambda(Tbb, Rbb, wave)
    
    chi2_term = -1/2*np.sum((Llambda - model)**2/Llambda_unc**2)
    error_term = np.sum(np.log(1/np.sqrt(2*np.pi*Llambda_unc**2)))
    ln_l = chi2_term + error_term
    return ln_l


bb_nll = lambda *args: -bb_lnlike(*args)


def bb_lnprior(theta):
    Tbb, Rbb = theta
    if (1e+3 < Tbb < 1e7 and 10 < Rbb < 1e+6):
        return 0.0
    return -np.inf


def bb_lnprob(theta, x, y, yerr):
    lp = bb_lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + bb_lnlike(theta, x, y, yerr)


def pool_bb_process(df, mydate):
    s = "_"
    filename = './19dge_mcmcresult/sampler_' + s.join(mydate.split(' '))+'.h5'
    
    if os.path.isfile(filename)==True:
        print ("already exist")
    
    else:
        subdf = df.iloc[np.where(df['date']==mydate)]
        x = subdf['wave'].values
        y = subdf['Llambda'].values
        yerr = subdf['Llambda_unc'].values
    
        result = op.minimize(bb_nll, [8e+3, 6e+3],
                         method='Powell', args=(x, y, yerr))
        ml_guess = result["x"]
        ndim = len(ml_guess)
        max_samples = 100000
        nwalkers = 100
        pos = [ml_guess + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]
        
        backend = emcee.backends.HDFBackend(filename)
        backend.reset(nwalkers, ndim)
        with Pool(5) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, bb_lnprob, 
                                        args=(x, y, yerr), pool=pool, backend=backend)
        

            index = 0
            autocorr = np.empty(max_samples)
            old_tau = np.inf
            for sample in sampler.sample(pos, iterations=max_samples, progress=True):
                if (sampler.iteration % 200):
                    #print (sampler.iteration)
                    continue
                tau = sampler.get_autocorr_time(tol=0)
                autocorr[sampler.iteration-1] = np.mean(tau)
                index += 1

                # Check convergence
                
                # Compute the autocorrelation time so far
                # Using tol=0 means that we'll always get an estimate even
                # if it isn't trustworthy
                converged = np.all(tau * 100 < sampler.iteration)
                converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
                if converged:
                    break
                old_tau = tau
        
        if np.isnan(tau[0])==True or np.isinf(tau[0])==True:
            print ('mydate = %s, not enough epochs to converge'%(mydate))
        else:
            print ('mydate = %s, done'%(mydate))


################ other helper functions
def mylinear_fit(x, y, yerr, npar = 2):
    '''
        Ref:
        1. Numerical Recipes, 3rd Edition, p745, 781 - 782
        2. http://web.ipac.caltech.edu/staff/fmasci/ztf/ztf_pipelines_deliverables.pdf, p38
        '''
    assert len(x) == len(y)
    assert len(y) == len(yerr)
    
    Sx = np.sum(x)
    Sy = np.sum(y)
    Sxy = np.sum(x * y)
    Sxx = np.sum(x**2)
    N = len(x)
    
    Sx_sigma = np.sum(x * yerr**2)
    Sxx_sigma = np.sum(x**2 * yerr**2)
    S_sigma = np.sum(yerr**2)
    
    if npar==1:
        Fpsf = Sxy / Sxx
        e_Fpsf = np.sqrt(Sxx_sigma) / Sxx
        a = 0
    elif npar==2:
        Fpsf = (N * Sxy - Sx * Sy) / (N * Sxx - Sx**2)
        a = (Sxx * Sy - Sx * Sxy) / (N * Sxx - Sx**2)
        e_Fpsf = np.sqrt(N**2*Sxx_sigma - 2*N*Sx*Sx_sigma + Sx**2*S_sigma) / (N * Sxx - Sx**2)
    # x_mean = np.mean(x)
    # y_mean = np.mean(y)
    # pearson_r = np.sum( (x - x_mean) * (y - y_mean) ) / np.sqrt(np.sum( (x - x_mean)**2 )) / np.sqrt(np.sum( (y - y_mean)**2 ))
    return Fpsf, e_Fpsf, a


if __name__ == "__main__": 
    dates1 = np.load('./19dge_dates.npy', allow_pickle=True)
    lcdet = pd.read_csv('./19dge_lcdet.csv')
    
    for i in [2,3]:
        mydate = dates1[i]
        pool_bb_process(lcdet, mydate)


