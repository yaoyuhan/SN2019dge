#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 16:04:35 2019

@author: yuhanyao
"""
import numpy as np
import pandas as pd
import extinction
import astropy.io.ascii as asci
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.size']=14


def get_ZTF_eff_wave(filename, return_type = 'R'):
    specg = np.loadtxt('../data/filters/P48/'+filename).T
    wv = specg[0]
    fg = specg[1]
    fg /= max(fg)

    wv_diff_ = wv[1:] - wv[:-1]
    wv_diff = 0.5 * (np.hstack([wv_diff_[0], wv_diff_]) + np.hstack([wv_diff_, wv_diff_[-1]]))

    g_eff = np.sum(wv_diff * fg * wv) / np.sum(wv_diff * fg)
    ebv = 1
    Rg = extinction.ccm89(np.array([g_eff]), 3.1*ebv, 3.1)[0]
    #print ("effective wavelength of %s is %f AA"%(filename, g_eff))
    if return_type == 'R':
        return Rg
    elif return_type == 'more':
        return g_eff, wv, fg


def get_LT_eff_wave(filename, return_type = 'R'):
    specg = asci.read('../data/filters/LT/'+filename)
    wv = specg['Wavelength(nm)'].data*10
    fg = specg['T%'].data
    fg /= max(fg)

    wv_diff_ = wv[1:] - wv[:-1]
    wv_diff = 0.5 * (np.hstack([wv_diff_[0], wv_diff_]) + np.hstack([wv_diff_, wv_diff_[-1]]))

    g_eff = np.sum(wv_diff * fg * wv) / np.sum(wv_diff * fg)
    ebv = 1
    Rg = extinction.ccm89(np.array([g_eff]), 3.1*ebv, 3.1)[0]
    
    #print ("effective wavelength of %s is %f AA"%(filename, g_eff))
    if return_type == 'R':
        return Rg
    elif return_type == 'more':
        return g_eff, wv, fg


def get_UVOT_eff_wave(filename, return_type = 'R'):
    specg = asci.read('../data/filters/UVOT/'+filename)
    wv = specg['col1'].data
    fg = specg['col2'].data 
    fg /= max(fg)

    wv_diff_ = wv[1:] - wv[:-1]
    wv_diff = 0.5 * (np.hstack([wv_diff_[0], wv_diff_]) + np.hstack([wv_diff_, wv_diff_[-1]]))

    g_eff = np.sum(wv_diff * fg * wv) / np.sum(wv_diff * fg)
    ebv = 1
    Rg = extinction.ccm89(np.array([g_eff]), 3.1*ebv, 3.1)[0]
    
    # print ("effective wavelength of %s is %f AA"%(filename, g_eff))
    if return_type == 'R':
        return Rg
    elif return_type == 'more':
        return g_eff, wv, fg
    
    
def get_P60_eff_wave(myfilter = "i'", return_type = 'R'):
    tb = pd.read_csv('../data/filters/P60/AstrodonSloanGen2Transmission.csv')
    if myfilter == "i'":
        wavecol = "Wavelength (nm).3"
    if myfilter == "r'":
        wavecol = "Wavelength (nm).2"
    wv = tb[wavecol].values * 10
    fg = tb[myfilter].values
    ix = (~np.isnan(wv))&(~np.isnan(fg))
    wv = wv[ix]
    fg = fg[ix]
    fg /= max(fg)
    
    wv_diff_ = wv[1:] - wv[:-1]
    wv_diff = 0.5 * (np.hstack([wv_diff_[0], wv_diff_]) + np.hstack([wv_diff_, wv_diff_[-1]]))

    g_eff = np.sum(wv_diff * fg * wv) / np.sum(wv_diff * fg)
    ebv = 1
    Rg = extinction.ccm89(np.array([g_eff]), 3.1*ebv, 3.1)[0]
    
    # print ("effective wavelength of %s is %f AA"%(myfilter, g_eff))
    if return_type == 'R':
        return Rg
    elif return_type == 'more':
        return g_eff, wv, fg
    


def see_filters():
    wvg, xg, yg = get_ZTF_eff_wave("P48_g.dat", return_type = 'more')
    wvr, xr, yr = get_ZTF_eff_wave("P48_R.dat", return_type = 'more')
    wvi, xi, yi = get_ZTF_eff_wave("P48_i.dat", return_type = 'more')
    
    wvg_lt, xg_lt, yg_lt = get_LT_eff_wave('IOO_SDSS-G.txt', return_type = 'more')
    wvr_lt, xr_lt, yr_lt = get_LT_eff_wave('IOO_SDSS-R.txt', return_type = 'more')
    wvi_lt, xi_lt, yi_lt = get_LT_eff_wave('IOO_SDSS-I.txt', return_type = 'more')
    wvz_lt, xz_lt, yz_lt = get_LT_eff_wave('IOO_SDSS-Z.txt', return_type = 'more')
    
    wvU, xU, yU = get_UVOT_eff_wave("Swift_UVOT.U.dat", return_type = 'more')
    wvB, xB, yB = get_UVOT_eff_wave("Swift_UVOT.B.dat", return_type = 'more')
    wvV, xV, yV = get_UVOT_eff_wave("Swift_UVOT.V.dat", return_type = 'more')
    
    wvUVW1, xUVW1, yUVW1 = get_UVOT_eff_wave("Swift_UVOT.UVW1.dat", return_type = 'more')
    wvUVW2, xUVW2, yUVW2 = get_UVOT_eff_wave("Swift_UVOT.UVW2.dat", return_type = 'more')
    wvUVM2, xUVM2, yUVM2 = get_UVOT_eff_wave("Swift_UVOT.UVM2.dat", return_type = 'more')
    
    plt.figure(figsize=(7,7))
    ax = plt.subplot(111)
    ax.plot(xg_lt, yg_lt, color='g', linestyle='--', label=r'$g_{\rm LT}$')
    ax.plot(xr_lt, yr_lt, color='red', linestyle='--', label=r'$r_{\rm LT}$')
    ax.plot(xi_lt, yi_lt, color='y', linestyle='--', label=r'$i_{\rm LT}$')
    ax.plot(xz_lt, yz_lt, color='pink', linestyle='--', label=r'$z_{\rm LT}$')
    
    ax.plot(xUVW2, yUVW2, color='k', linestyle='-.', label=r'$UVW2_{\rm UVOT}$')
    ax.plot(xUVM2, yUVM2, color='indigo', linestyle='-.', label=r'$UVM2_{\rm UVOT}$')
    ax.plot(xUVW1, yUVW1, color='b', linestyle='-.', label=r'$UVW1_{\rm UVOT}$')
    
    ax.plot(xU, yU, color='blueviolet', linestyle='-.', label=r'$U_{\rm UVOT}$')
    ax.plot(xB, yB, color='skyblue', linestyle='-.', label=r'$B_{\rm UVOT}$')
    ax.plot(xV, yV, color='lightgreen', linestyle='-.', label=r'$V_{\rm UVOT}$')
    
    ax.plot(xg, yg, color='darkcyan', label = r'$g_{\rm ZTF}$')
    ax.plot(xr, yr, color='crimson', label = r'$r_{\rm ZTF}$')
    ax.plot(xi, yi, color='gold', label = r'$i_{\rm ZTF}$')
    
    fs = 14
    ax.legend(loc = 'upper left', bbox_to_anchor=(-0.05, 1.4), ncol = 4, fontsize=fs)
    ax.set_xlabel(r'$\lambda$'+'('+r'$\AA$'+')', fontsize=fs+2)
    ax.set_ylabel('Transparentcy', fontsize=fs+2)
    plt.tight_layout()

