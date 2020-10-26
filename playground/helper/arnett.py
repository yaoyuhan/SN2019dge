#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 11:20:30 2020

@author: yuhanyao
"""
##### radioactivity: Arnett model
import os
import sys
sys.path.append("/scratch/yyao/AT2019dge/playground/")
sys.path.append("/Users/yuhanyao/Documents/GitHub/AT2019dge/playground/")
import time
import numpy as np
import scipy.integrate as integrate
from helper import phys
from helper.mcmcfit import planck_lambda
from multiprocessing import Pool
import emcee
import scipy.optimize as op
import matplotlib.pyplot as plt
import corner
fs = 14


def get_int_A(x, y, s):
    r = integrate.quad(lambda z: 2*z*np.exp(-2*z*y+z**2), 0, x)
    int_A = r[0]
    return int_A
    
    
def get_int_B(x, y, s):
    r = integrate.quad(lambda z: 2*z*np.exp(-2*z*y+2*z*s+z**2), 0, x)
    int_B = r[0]
    return int_B


def model_arnett_Ltph(ts_, taum_ = 3, Mni_ = 0.05):
    '''
    Calculate the flux of a radioactivity powered SN at photospheric phase
    
    ts is in the unit of day
    Mni_ is in the unit of Msun
    
    The euqation is from
    Valenti 2008 MNRAS 383 1485V, Appendix A
    '''
    ts = ts_ * 24*3600 #  in seconds
    Mni = Mni_ * phys.sm
    tau_m = taum_ * 24 * 3600.
    
    epsilon_ni = 3.9e+10 # erg / s / g
    epsilon_co = 6.78e+9 # erg / s / g
    tau_ni = 8.8 * 24 * 3600 # s
    tau_co = 111.3 * 24 * 3600 # s
    
    Ls = np.zeros(len(ts))
    for i in range(len(ts)):
        t = ts[i]
        x = t / tau_m
        y = tau_m / (2 * tau_ni)
        s = tau_m * (tau_co - tau_ni) / (2 * tau_co * tau_ni)
    
        int_A = get_int_A(x, y, s)
        int_B = get_int_B(x, y, s)
    
        L = Mni * np.exp(-x**2) * ( (epsilon_ni - epsilon_co) * int_A + epsilon_co * int_B )
        Ls[i] = L
    # plt.loglog(ts/24/3600, Ls)
    return Ls


def model_arnett_modified(ts_, taum_ = 3, Mni_ = 0.05, t0_ = 30, texp = 0):
    '''
    Calculate the flux of a radioactivity powered SN at photospheric phase
    
    ts is in the unit of day
    Mni_ is in the unit of Msun
    
    The euqation is from
    Valenti 2008 MNRAS 383 1485V, Appendix A
    '''
    ts_ = ts_ - texp
    ts = ts_ * 24*3600 #  in seconds
    Mni = Mni_ * phys.sm
    tau_m = taum_ * 24 * 3600.
    t0 = t0_ * 24 * 3600.
    
    epsilon_ni = 3.9e+10 # erg / s / g
    epsilon_co = 6.78e+9 # erg / s / g
    tau_ni = 8.8 * 24 * 3600 # s
    tau_co = 111.3 * 24 * 3600 # s
    
    Ls = np.zeros(len(ts))
    for i in range(len(ts)):
        t = ts[i]
        if t<=0:
            Ls[i] = 0
        else:
            x = t / tau_m
            y = tau_m / (2 * tau_ni)
            s = tau_m * (tau_co - tau_ni) / (2 * tau_co * tau_ni)
            
            int_A = get_int_A(x, y, s)
            int_B = get_int_B(x, y, s)
            
            L = Mni * np.exp(-x**2) * ( (epsilon_ni - epsilon_co) * int_A + epsilon_co * int_B )
            Ls[i] = L
    # plt.loglog(ts/24/3600, Ls)
    Ls_modified = np.zeros(len(Ls))
    ix = ts > 0
    Ls_modified[ix] = Ls[ix]* (1. - np.exp(-1*(t0/ts[ix])**2) )
    return Ls_modified


def arnett_lnlike(theta, t, Ldata, Ldata_unc):
    """
    taum_, Mni_, texp_ are in the unit of day, Msun, day
    """
    taum_, lgMni_, t0_, texp_ = theta
    Mni_ = 10**lgMni_
    model = model_arnett_modified(t, taum_, Mni_, t0_, texp_)
    lgmodel = np.log10(model)
    
    # not sure what is the reason why 
    # ValueError: Probability function returned NaN
    
    chi2_term = -1/2*np.sum((Ldata - lgmodel)**2/Ldata_unc**2)
    error_term = np.sum(np.log(1/np.sqrt(2*np.pi*Ldata_unc**2)))
    ln_l = chi2_term + error_term
    return ln_l


arnett_nll = lambda *args: -arnett_lnlike(*args)

def arnett_lnprior(theta):
    taum_, lgMni_, t0_, texp_ = theta
    if ((1 < taum_ < 20) and (-4 < lgMni_ < 0) and (20 < t0_ < 100) and (-2.931 < texp_ < -2.891)):
        return 0.
    return -np.inf

def arnett_lnprob(theta, x, y, yerr):
    lp = arnett_lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + arnett_lnlike(theta, x, y, yerr)


def plotChains(sampler, nburn, paramsNames, nplot):
    Nparams = len(paramsNames)
    fig, ax = plt.subplots(Nparams+1, 1, figsize = (8,2*(Nparams+1)), sharex = True)
    fig.subplots_adjust(hspace = 0)
    ax[0].set_title('Chains', fontsize=fs)
    xplot = np.arange(sampler.get_chain().shape[0])

    selected_walkers = np.random.choice(range(sampler.get_chain().shape[1]), nplot, replace=False)
    for i,p in enumerate(paramsNames):
        for w in selected_walkers:
            burn = ax[i].plot(xplot[:nburn], sampler.get_chain()[:nburn,w,i], 
                              alpha = 0.4, lw = 0.7, zorder = 1)
            ax[i].plot(xplot[nburn:], sampler.get_chain(discard=nburn)[:,w,i], 
                       color=burn[0].get_color(), alpha = 0.8, lw = 0.7, zorder = 1)
            
            ax[i].set_ylabel(p)
            if i==Nparams-1:
                ax[i+1].plot(xplot[:nburn], sampler.get_log_prob()[:nburn,w], 
                             color=burn[0].get_color(), alpha = 0.4, lw = 0.7, zorder = 1)
                ax[i+1].plot(xplot[nburn:], sampler.get_log_prob(discard=nburn)[:,w], 
                             color=burn[0].get_color(), alpha = 0.8, lw = 0.7, zorder = 1)
                ax[i+1].set_ylabel('ln P')
            
    return ax


def makeCornerArnett(sampler, nburn, paramsNames, quantiles=[0.16, 0.5, 0.84]):
    samples = sampler.get_chain(discard=nburn, flat=True)
    corner.corner(samples, labels = paramsNames, quantiles = quantiles, 
                  range = [0.999, 0.999, 0.999, 0.999],
                  show_titles=True, plot_datapoints=False, 
                  title_kwargs = {"fontsize": fs})



if __name__ == "__main__":
    xyey = np.loadtxt('./Lbb_p20subtracted.txt')
    tt = xyey[0]
    lgL = xyey[1]
    lgL_unc = xyey[2]
    
    tgrid = np.linspace(1, 65)
    taum_ = 6.35
    Mni_ = 0.0162
    t0_ = 24
    texp = -2.9112151494264173
    Lmodel2 = model_arnett_modified(tgrid, taum_ = taum_, Mni_ = Mni_, t0_ = t0_, texp = texp)
    lgLmodel2 = np.log10(Lmodel2)
    # 
    """
    plt.figure()
    plt.errorbar(tt, lgL, lgL_unc, fmt=".k")
    plt.plot(tgrid, lgLmodel2)
    """
    
    nwalkers = 100
    lgMni_ = np.log10(Mni_)
    ml_guess = np.array([taum_, lgMni_, t0_, texp])
    #initial position of walkers
    ndim = len(ml_guess)
    nfac = [1e-3]*ndim
    pos = [ml_guess + nfac * np.random.randn(ndim) for i in range(nwalkers)]
    
    max_samples = 10000
    check_tau = 200
    
    dirpath = "./arnettmodel/"
    filename = dirpath + "sampler.h5"
    if os.path.isfile(filename):
        os.remove(filename)
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)
    
    with Pool(20) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, arnett_lnprob, 
                                        args=(tt, lgL, lgL_unc),
                                        pool=pool, backend=backend)
        index = 0
        autocorr = np.empty(max_samples)
        old_tau = np.inf
        for sample in sampler.sample(pos, iterations=max_samples, progress=True):
            # Only check convergence every 30 steps
            if sampler.iteration % check_tau:
                continue
            
            tau = sampler.get_autocorr_time(tol=0)
            autocorr[index] = np.mean(tau[:3]) # only expect the first three parameters to converge
            index += 1
            # Check convergence
            converged = np.all(tau * 100 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                break
            old_tau = tau
            
    print ("")
    print ("****** converged? ******")
    print (converged)
    
    paramsNames=[r"$\tau_{\rm m}$", 
                 'lg' +r'$M_{\rm Ni}$', 
                 r"$t_0$",
                 r"$t_{\rm fl}$"]

    plotChains(sampler, 250, paramsNames, nplot=35)
    plt.tight_layout()
    plt.savefig(dirpath+"chains.pdf")
    
    makeCornerArnett(sampler, 20, paramsNames)
    plt.savefig(dirpath+"corner.pdf")
  
