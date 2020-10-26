#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 08:54:30 2020

@author: yuhanyao
"""
from copy import deepcopy
from helper import phys
import collections
import numpy as np
from helper.specread import gaplinelist
from helper.mcmcfit import mylinear_fit
from lmfit.models import LinearModel
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d


def get_vvyy(dt4, wv, binning = 1):
    v4 = (dt4['wave_rest'] - wv)/wv * phys.c /1e+5
    y4 = dt4['spec_obs0']
    if binning != 1:
        yy6 = deepcopy(y4)
        vv6 = deepcopy(v4)
        rest = len(yy6)%binning
        if rest!=0:
            vv6 = vv6[:(-1)*rest]
            yy6 = yy6[:(-1)*rest]
        nnew = int(len(yy6) / binning)
        yy6_new = yy6.reshape(nnew, binning)
        yy6_new = np.sum(yy6_new, axis=1)
        y4 = yy6_new / binning
        vv6_new = vv6.reshape(nnew, binning)
        vv6_new = np.sum(vv6_new, axis=1)
        v4 = vv6_new / binning
    yy4 = np.repeat(y4, 2, axis=0)
    v4diff = np.diff(v4)
    v4diff_left = np.hstack([v4diff[0], v4diff])
    v4diff_right = np.hstack([v4diff, v4diff[-1]])
    vv4 = np.repeat(v4, 2, axis=0)
    vv4[::2] -= v4diff_left/2
    vv4[1::2] += v4diff_right/2
    return vv4, yy4    


def add_tick(ax, wv, NIII_list, t1, t2):
    if type(NIII_list) != np.float64:
        vs = np.zeros(len(NIII_list))
        for i in range(len(NIII_list)):
            wvnew = NIII_list[i]
            v = (wvnew - wv)/wv * phys.c /1e+5
            vs[i] = v
            ax.plot([v,v], [t1, t2], 'k-', linewidth = 0.8, color = "k")
        #ax.plot([min(vs), max(vs)], [t2, t2], 'k-', linewidth = 0.8, color = "k")
    else:
        wvnew = NIII_list
        v = (wvnew - wv)/wv * phys.c /1e+5
        ax.plot([v,v], [t1, t2], 'k-', linewidth = 0.8, color = "k")
        
        
        
def plot_mask_gal_lines(ax2, wave, flux, plotfinal = False, returnfinal=False,
                        finalcolor= "k"):
    yaolist = gaplinelist(z=0)
    H_list = yaolist['H_list']
    HeI_list = yaolist['HeI_list']
    OIII_list = yaolist['OIII_list']
    OII_list = yaolist['OII_list']
    OI_list = yaolist['OI_list']
    SIII_list = yaolist['SIII_list']
    SII_list = yaolist['SII_list']
    NII_list = yaolist['NII_list']
    CaII_list = yaolist['CaII_list']
    NeIII_list = yaolist['NeIII_list']
    ArIII_list = yaolist["ArIII_list"]
    
    color = "mistyrose"
    ix_retain = np.ones(len(wave), dtype=bool)
    # ok
    for i in range(len(H_list)):
        wv = H_list[i]
        if i<3:
            ix = abs(wave - wv) < 10
        else:
            ix = abs(wave - wv) < 12
        ix_retain = ix_retain & (~ix)
        ax2.plot(wave[ix], flux[ix], color=color, zorder=5)
        
    for wv in OIII_list[1:]:
        ix = abs(wave - wv) < 14
        ix_retain = ix_retain & (~ix)
        ax2.plot(wave[ix], flux[ix], color=color, zorder=5)
        
    for wv in NII_list:
        ix = abs(wave - wv) < 10
        ix_retain = ix_retain & (~ix)
        ax2.plot(wave[ix], flux[ix], color=color, zorder=5)
        
    for wv in ArIII_list:
        ix = abs(wave - wv) < 10
        ix_retain = ix_retain & (~ix)
        ax2.plot(wave[ix], flux[ix], color=color, zorder=5)
        
    for i in range(len(SII_list)):
        wv = SII_list[i]
        if i==0:
            ix = abs(wave - wv) < 14
        else:
            ix = abs(wave - wv) < 10
        ix_retain = ix_retain & (~ix)
        ax2.plot(wave[ix], flux[ix], color=color, zorder=5)
    
    for wv in NeIII_list:
        ix = abs(wave - wv) < 10
        ix_retain = ix_retain & (~ix)
        ax2.plot(wave[ix], flux[ix], color=color, zorder=5)
    
    for wv in OII_list:
        ix = abs(wave - wv) < 12
        ix_retain = ix_retain & (~ix)
        ax2.plot(wave[ix], flux[ix], color=color, zorder=5)
    
    for i in range(len(SIII_list)):
        wv = SIII_list[i]
        if i==0:
            ix = abs(wave - wv) < 6
        else:
            ix = abs(wave - wv) < 16
        ix_retain = ix_retain & (~ix)
        ax2.plot(wave[ix], flux[ix], color=color, zorder=5)
        
    wave = wave[ix_retain]
    flux = flux[ix_retain]
    
    if  plotfinal==True:
        ax2.plot(wave, flux, color=finalcolor, zorder=6, linewidth = 0.9)
    if returnfinal==True:
        return wave, flux
    
    
def gaus(x, a, A, x0, sigma):
    return a + A * np.exp(-(x-x0)**2/(2*sigma**2))  


def parabola(x, a, A, x0):
    return a + A * (x-x0)**2
    
    
###### measure absorption minumum velocity
def measure_abs_velocity(wave,
                         flux,
                         line_info = None,
                         sigma_guess = 2000,
                         line_center = -6500,
                         line_bound_width = 1000,
                         plotfig=False):
    if line_info == None:
        # He I 5875
        line_info = {'line_shoulder_left': (-12600, -9800),
                     'line_shoulder_right': (-1300, 1800),
                     'line_fit': (-8000, -3500)}
    
    line_shoulder_left = line_info['line_shoulder_left']
    line_shoulder_right = line_info['line_shoulder_right']
    line_range = (line_shoulder_left[1], line_shoulder_right[0])
    line_fit = line_info["line_fit"]
    
    ind_shoulder = np.any([
            np.all([wave > line_shoulder_left[0],
                    wave < line_shoulder_left[1]], axis=0),
            np.all([wave > line_shoulder_right[0],
                    wave < line_shoulder_right[1]], axis=0)], axis=0)
    wave_shoulder = wave[ind_shoulder]
    flux_shoulder = flux[ind_shoulder]
    
    ind_range = np.logical_and(wave > line_range[0], wave < line_range[1])
    wave_range = wave[ind_range]
    flux_range = flux[ind_range]
    
    ind_fit = np.logical_and(wave > line_fit[0], wave < line_fit[1])
    wave_fit = wave[ind_fit]
    flux_fit = flux[ind_fit]
    
    mod_linear = LinearModel(prefix='mod_linear_')
    par_linear = mod_linear.guess(flux_shoulder, x=wave_shoulder)
    out_linear = mod_linear.fit(flux_shoulder,
                                    par_linear,
                                    x=wave_shoulder,
                                    method='leastsq')

    cont_shoulder = out_linear.best_fit
    noise_std = np.std(flux_shoulder / cont_shoulder)
    cont_range = mod_linear.eval(out_linear.params, x=wave_range)
    cont_fit = mod_linear.eval(out_linear.params, x=wave_fit)
    norm_fit = (flux_fit / cont_fit-1.)*(-1)
    
    a_fixed = 0.
    a_width = 0.05
    A_guess = max(norm_fit) - a_fixed
    bounds = ((a_fixed-a_width, 0.2*A_guess, line_center-line_bound_width*2, sigma_guess/5),
              (a_fixed+a_width, 5*A_guess, line_center+line_bound_width*2, sigma_guess*5))
    popt1, pcov1 = curve_fit(gaus, wave_fit, norm_fit, 
                             p0=[a_fixed, A_guess, line_center, sigma_guess],
                             bounds=bounds)
    print ("line width = %.2f +- %.2f km/s"%(popt1[-1], np.sqrt(pcov1[-1,-1])))
    print ("line center = %.2f +- %.2f km/s"%(popt1[2], np.sqrt(pcov1[2,2])))
    
    line_center = popt1[2]
    new_width = popt1[-1] * 4 # four times the sigma
    wvnew = np.linspace(line_center-new_width, line_center+new_width, 300)
    flnew = gaus(wvnew, *popt1)
    
    if plotfig == True:
        plt.figure(figsize = (6,6))
        ax1 = plt.subplot(211)
        ax1.plot(wave_shoulder, flux_shoulder, 'b-')
        ax1.plot(wave_range, cont_range, 'g-')
        ax1.plot(wave_range, flux_range, 'r-', alpha = 0.2)
        ax1.plot(wave_fit, flux_fit, 'r-')
        ax2 = plt.subplot(212)
        ax2.plot(wave_fit, norm_fit, 'k-')
        ax2.plot(wvnew, flnew)
        
    a_fixed = min(flux_fit)
    A_guess = (max(flux_fit) - min(flux_fit)) / 2000**2
    bounds = ((a_fixed-a_width, 0.2*A_guess, line_center-line_bound_width*2),
              (a_fixed+a_width, 5*A_guess, line_center+line_bound_width*2))
    popt1, pcov1 = curve_fit(parabola, wave_fit, flux_fit, 
                             p0=[a_fixed, A_guess, line_center],
                             bounds=bounds)
    print ("line center = %.2f +- %.2f km/s"%(popt1[2], np.sqrt(pcov1[2,2])))
    
    line_center = popt1[2]

    
    
###### measure the equivalent width
def measure_line_index(wave,
                       flux,
                       flux_err =None,
                       line_info=None,
                       num_refit_=100,
                       plotfig=False):
    if line_info == None:
        # He II 4686
        line_info = {'line_range': (4666, 4706),
                     'line_shoulder_left': (4545, 4620),
                     'line_shoulder_right': (4726, 4800)}
    try:
        # 0. do some input check
        # 0.1> check line_info
        line_info_keys = line_info.keys()
        assert 'line_range' in line_info_keys
        assert 'line_shoulder_left' in line_info_keys
        assert 'line_shoulder_right' in line_info_keys
        # 0.2> check line range/shoulder in spectral range
        assert np.min(wave) <= line_info['line_shoulder_left'][0]
        assert np.max(wave) >= line_info['line_shoulder_right'][0]

        # 1. get line information
        # line_center = line_info['line_center']  # not used
        line_range = line_info['line_range']
        line_shoulder_left = line_info['line_shoulder_left']
        line_shoulder_right = line_info['line_shoulder_right']

        # 2. data preparation
        wave = np.array(wave)
        flux = np.array(flux)
        if flux_err == None:
            flux_err = np.ones(wave.shape)

        # 3. estimate the local continuum
        # 3.1> shoulder wavelength range
        ind_shoulder = np.any([
            np.all([wave > line_shoulder_left[0],
                    wave < line_shoulder_left[1]], axis=0),
            np.all([wave > line_shoulder_right[0],
                    wave < line_shoulder_right[1]], axis=0)], axis=0)
        wave_shoulder = wave[ind_shoulder]
        flux_shoulder = flux[ind_shoulder]
        flux_err_shoulder = flux_err[ind_shoulder]

        # 3.2> integrated/fitted wavelength range
        ind_range = np.logical_and(wave > line_range[0], wave < line_range[1])
        wave_range = wave[ind_range]
        flux_range = flux[ind_range]
        # flux_err_range = flux_err[ind_range]  # not used
        # mask_shoulder = mask[ind_shoulder]    # not used

        # 4. linear model
        mod_linear = LinearModel(prefix='mod_linear_')
        par_linear = mod_linear.guess(flux_shoulder, x=wave_shoulder)
        # ############################################# #
        # to see the parameter names:                   #
        # model_linear.param_names                      #
        # {'linear_fun_intercept', 'linear_fun_slope'}  #
        # ############################################# #
        out_linear = mod_linear.fit(flux_shoulder,
                                    par_linear,
                                    x=wave_shoulder,
                                    method='leastsq')

        # 5. estimate continuum
        cont_shoulder = out_linear.best_fit
        noise_std = np.std(flux_shoulder / cont_shoulder)
        cont_range = mod_linear.eval(out_linear.params, x=wave_range)
        resi_range = 1 - flux_range / cont_range
        
        if plotfig==True:
            plt.figure(figsize = (6,4))
            ix = (wave > line_shoulder_left[0])&((wave < line_shoulder_right[1]))
            plt.plot(wave[ix], flux[ix], color="k", alpha=0.1)
            plt.plot(wave_shoulder, flux_shoulder, 'b-')
            plt.plot(wave_range, cont_range, 'g-')
            plt.plot(wave_range, flux_range, 'r-')
            

        # 6.1 Integrated EW (
        # estimate EW_int
        wave_diff = np.diff(wave_range)
        wave_step = np.mean(np.vstack([np.hstack([wave_diff[0], wave_diff]),
                                       np.hstack([wave_diff, wave_diff[-1]])]),
                            axis=0)
        EW_int = np.dot(resi_range, wave_step)

        # estimate EW_int_err
        if num_refit_ is not None and num_refit_>0:
            EW_int_err = np.std(np.dot(
                (resi_range.reshape(1, -1).repeat(num_refit_, axis=0) +
                 np.random.randn(num_refit_, resi_range.size) * noise_std),
                wave_step))

        # 6.2 Gaussian model
        # estimate EW_fit
        line_indx = collections.OrderedDict([
            ('SN_local_flux_err',        np.median(flux_shoulder / flux_err_shoulder)),
            ('SN_local_flux_std',        1. / noise_std),
            ('EW_int',                   EW_int),
            ('EW_int_err',               EW_int_err),
            ('mod_linear_slope',         out_linear.params[mod_linear.prefix + 'slope'].value),
            ('mod_linear_slope_err',     out_linear.params[mod_linear.prefix + 'slope'].stderr),
            ('mod_linear_intercept',     out_linear.params[mod_linear.prefix + 'intercept'].value),
            ('mod_linear_intercept_err', out_linear.params[mod_linear.prefix + 'intercept'].stderr)
            ])

        return line_indx
    except Exception:
        print ("Some error happened...?")


def host_subtraction(x1, y1, x2, y2, 
                     gal_regions, fixb = False, plotax = None):
    """
    x1, y1: object spectrum
    x2, y2: galaxy spectrum
    """
    ix1 = (x1 > min(x2))&(x1 < max(x2))
    x1 = x1[ix1]
    y1 = y1[ix1]
    nregions = gal_regions.shape[1]
    xknots = np.zeros(nregions)
    v1s = np.zeros(nregions)
    v2s = np.zeros(nregions)
    for i in range(nregions):
    
        xmin = gal_regions[0][i]
        xmax = gal_regions[1][i]
        ix1 = (x1 > xmin)&(x1 < xmax)
        xx1 = x1[ix1]
        yy1 = y1[ix1]
        ix2 = (x2 > xmin)&(x2 < xmax)
        xx2 = x2[ix2]
        yy2 = y2[ix2]
        v1s[i] = np.median(yy1)
        v2s[i] = np.median(yy2)
        xknots[i] = np.median(xx2)
    if fixb == False:
        npar=2
    else:
        npar = 1
    k, ek, b = mylinear_fit(v2s, v1s, np.ones(len(v1s))*np.median(v1s)/10, npar = npar)
    print (b)
    y2_ = k * y2 + b
    v2s_ = k* v2s + b
    factors = v1s/v2s_
    print (factors)
    fac_array = np.zeros(len(x2))
    xbounds = np.hstack([0, xknots, 100000])
    
    for i in range(nregions+1):
        xmin = xbounds[i]
        xmax = xbounds[i+1]
        ix2 = (x2 >= xmin)&(x2 <= xmax)
        if i==0:
            fac_array[ix2] = factors[i]
        elif i==nregions:
            fac_array[ix2] = factors[nregions-1]
        else:
            k1, ek1, b1 = mylinear_fit(np.array([xmin, xmax]), 
                                       factors[i-1:i+1], np.ones(2)/10, npar = 2)
            fac_array[ix2] = x2[ix2] * k1 + b1
    y2_matched = y2_*fac_array
    hostfunc = interp1d(x2, y2_matched)
    y1_host = hostfunc(x1)
    if plotax != None:
        plotax.plot(x1,y1)
        plotax.plot(x1,y1_host)
        for i in range(nregions):
            xmin = gal_regions[0][i]
            xmax = gal_regions[1][i]
            ix2 = (x2 > xmin)&(x2 < xmax)
            xx2 = x2[ix2]
            yy2_matched = y2_matched[ix2]
            plotax.plot(xx2, yy2_matched, color="grey")
        plotax.plot(x1,y1-y1_host, 'k')
    return x1, y1-y1_host
        
    
def measure_FWHM(dt0, wv, init_lim_left = -500, init_lim_right = 500, lim_cont = 300):
    vv0_, yy0 = get_vvyy(dt0, wv)
    ix = (vv0_<init_lim_right)&(vv0_>init_lim_left)
    vv0_ = vv0_[ix]
    yy0 = yy0[ix] * 1e+16
    
    # 0.0 redefine line center
    ix_max = np.where(yy0 == max(yy0))[0]
    vv_center = np.mean(vv0_[ix_max])
    vv0 = vv0_ - vv_center
    
    plt.figure(figsize= (9,4))
    ax1 = plt.subplot(121)
    ax1.plot(vv0_, yy0, color = "grey", linestyle = ":")
    ax1.plot(vv0, yy0, color = "k")
    
    # 1.0 fit a continuum
    ixc = abs(vv0)> lim_cont
    v_cont = vv0[ixc]
    f_cont = yy0[ixc]
    k, ek, intercept = mylinear_fit(v_cont, f_cont, np.ones(len(f_cont))*0.05, npar = 2)
    yy_cont = k * vv0 + intercept
    ax1.plot(vv0, yy_cont, color = "c")
    
    # ===============================
    # 2.0 method: direct measurement
    maxy= max(yy0)
    plt.plot([0,0], [intercept, maxy], color = 'salmon', linestyle = "--")
    ix_right = vv0 > 0
    vvright = vv0[ix_right]
    yyright = yy0[ix_right]
    vvleft = vv0[~ix_right]
    yyleft = yy0[~ix_right]
    
    yhm = 0.5*(maxy+intercept)
    
    # 2.1 right part find
    ix_right_below = yyright<(yhm)
    ax1.plot(vvright[ix_right_below], yyright[ix_right_below], 'r')
    
    # 2.2 left part find
    ix_left_below = yyleft<(yhm)
    ax1.plot(vvleft[ix_left_below], yyleft[ix_left_below], 'r')
    
    # 2.3 mark half maximum
    right_hm = vvright[ix_right_below][0]
    left_hm = vvleft[ix_left_below][-1]
    ax1.plot([left_hm, right_hm], [yhm, yhm], 'r:')
    
    fwhm = right_hm - left_hm
    ax1.set_title("FWHM = %d"%fwhm, fontsize  = 10)
    
    # ===============================
    # 3.0 fit a gaussian
    ax2 = plt.subplot(122)
    ax2.plot(vv0, yy0, color = "k")
    
    a_fixed = intercept
    a_width = max(yy_cont) - min(yy_cont)
    A_guess = maxy - intercept
    line_center = 0
    line_bound_width = 50
    sigma_guess = fwhm / np.sqrt(2 * np.log(2))
    bounds = ((a_fixed-a_width, 0.2*A_guess, line_center-line_bound_width*2, sigma_guess/5),
              (a_fixed+a_width, 5*A_guess, line_center+line_bound_width*2, sigma_guess*5))
    popt1, pcov1 = curve_fit(gaus, vv0, yy0, 
                             p0=[a_fixed, A_guess, line_center, sigma_guess],
                             bounds=bounds)
    fwhm_gaus = popt1[-1] * 2 * np.sqrt(2 * np.log(2))
    fwhm_gaus_unc = np.sqrt(pcov1[-1,-1]) * 2 * np.sqrt(2 * np.log(2))
    print ("line width = %.2f +- %.2f km/s"%(popt1[-1], np.sqrt(pcov1[-1,-1])))
    print ("line center = %.2f +- %.2f km/s"%(popt1[2], np.sqrt(pcov1[2,2])))
    print ("FWHM = %d +- %d km/s"%(fwhm_gaus, fwhm_gaus_unc))
    
    vvnew = np.linspace(min(vv0), max(vv0), 200)
    yy_fitted = gaus(vvnew, *popt1)
    ax2.plot(vvnew, yy_fitted)
    ax2.set_title(r"$\lambda%d$"%wv, fontsize = 10)
    
    #return fwhm