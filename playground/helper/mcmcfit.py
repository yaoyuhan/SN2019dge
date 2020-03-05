#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 09:07:17 2019

@author: yuhanyao
"""
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
    if (0 < Tbb < 1e6 and 0 < Rbb < 1e+6):
        return 0.0
    return -np.inf


def bb_lnprob(theta, x, y, yerr):
    lp = bb_lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + bb_lnlike(theta, x, y, yerr)


def pool_bb_process(df, mydate):
    s = "_"
    filename = '../data/analysis/mcmcresult/' + s.join(mydate.split(' '))+'.txt'
    '''
    if os.path.isfile(filename)==True:
        print ("already exist")
        return ""
    '''
    if 1+1==3:
        return ""
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
        nwalkers = 250
        pos = [ml_guess + 1e-3*np.random.randn(ndim) for i in range(nwalkers)]
        sampler = emcee.EnsembleSampler(nwalkers, ndim, bb_lnprob, 
                                    args=(x, y, yerr))

        index = 0
        autocorr = np.empty(max_samples)
        old_tau = np.inf
        for sample in sampler.sample(pos, iterations=max_samples):
            if ((sampler.iteration % 250) and 
                (sampler.iteration < 5000)):
                continue
            elif ((sampler.iteration % 1000) and 
                  (5000 <= sampler.iteration < 15000)):
                continue
            elif ((sampler.iteration % 2500) and 
                  (15000 <= sampler.iteration)):
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
            samples = sampler.get_chain(discard=1000, flat=True)
            print (tau)
        else:
            samples = sampler.get_chain(discard=int(10*tau[0]), flat=True)
        
        Lbbs = const.sigma_sb.cgs.value * samples[:,0] **4 * 4 * np.pi * (samples[:,1] * const.R_sun.cgs.value)**2
        
        Lbb_sigmas = np.percentile(Lbbs, (0.13, 2.27, 15.87, 50, 84.13, 97.73, 99.87))
        Tbb_sigmas = np.percentile(samples[:,0], (0.13, 2.27, 15.87, 50, 84.13, 97.73, 99.87))
        Rbb_sigmas = np.percentile(samples[:,1], (0.13, 2.27, 15.87, 50, 84.13, 97.73, 99.87))
    
        print ('mydate = %s, lgLbb_med = %.2f, Tbb_med = %.2f, Rbb_med = %.2f'%(mydate, np.log10(Lbb_sigmas[3]),Tbb_sigmas[3], Rbb_sigmas[3]))
        result = np.vstack([Lbb_sigmas, Tbb_sigmas, Rbb_sigmas])
        np.savetxt(filename, result)
        return result


def main_bbrun():
    dates1 = np.load('./helper/dates1.npy')
    lcdet = pd.read_csv('./helper/lcdet.csv')
    
    pool = Pool(10)
    tstart = time.time()
    results = [pool.apply_async(pool_bb_process, args=(lcdet, mydate,)) 
                for mydate in dates1]
    output = [p.get() for p in results]
    tend = time.time()
    print("Pool map took {:.4f} sec".format(tend-tstart))



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

