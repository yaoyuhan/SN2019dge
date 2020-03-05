#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 08:41:10 2019

@author: yuhanyao
"""
#import sys
#import os
from helper import phys
import numpy as np
import extinction
from astropy.time import Time
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70., Om0=0.275)


def deredden_df(tb, ebv):
    """
    perform extinction correction
    """
    if 'mag' in tb.columns:
        tb['mag0'] = tb['mag'] - extinction.ccm89(tb['wave'].values, 3.1*ebv, 3.1) # extinction in magnitude
    if "limmag" in tb.columns:
        tb['limmag0'] = tb["limmag"] - extinction.ccm89(tb['wave'].values, 3.1*ebv, 3.1) # extinction in magnitude
    return tb
    
    
def app2abs_df(tb, z, t_max):
    """
    convert apparent magnitude (already de-reddened) into absolute magnitude
    """
    D = cosmo.luminosity_distance([z])[0].value * 1e+6 # in pc
    dis_mod = 5*np.log10(D/10)
    tb['tmax_rf'] = (tb['mjd'] - t_max) / (1+z)
    tb['tmax_of'] = (tb['mjd'] - t_max)
    if 'mag0' in tb.columns:
        tb['mag0_abs'] = tb['mag0'] - dis_mod
    if "limmag0" in tb.columns:
        tb['limmag0_abs'] = tb['limmag0'] - dis_mod
    return tb


def add_physcol(tb, magcol = 'mag0_abs'):
    """
    tb is pandas dataframe
    
    columns that must exist: 
        wave: in angstrom
        magcol (e.e.g: mag0_abs): extinction corrected absolute magnitude
        emag: uncertainty in magnitude
        
    please avoid mag == 99 or any invalid values...
    """
    # zero point in AB magnitude: 3631 Jy 
    # 1 Jy = 1e-23 erg / s / Hz / cm^{-2}
    if "wave" in tb.columns:
        tb['freq'] = phys.c / (tb['wave'].values * 1e-8) # Hz
    elif "freq" in tb.columns:
        tb['wave'] = phys.c / tb['freq'].values * 1e8 # Hz

    tb['fratio'] = 10**(-0.4 * tb[magcol].values)
    tb['fratio_unc'] = np.log(10) / 2.5 * tb['emag'].values * tb['fratio'].values
    
    fnu0 = 3631e-23 # erg / s/ Hz / cm^2
    tb['fnu'] = tb['fratio'].values * fnu0 # erg / s/ Hz / cm^2
    tb['fnu_unc'] = tb['fratio_unc'].values * fnu0 # erg / s/ Hz / cm^2
    
    tb['nufnu'] = tb['fnu'].values * tb['freq'].values # erg / s / cm^2
    tb['nufnu_unc'] = tb['fnu_unc'].values * tb['freq'].values # erg / s / cm^2
    
    tb['flambda'] = tb['nufnu'].values / tb['wave'] # erg / s / cm^2 / A
    tb['flambda_unc'] = tb['nufnu_unc'].values / tb['wave'] # erg / s / cm^2 / A
    
    tb['Llambda'] = tb['flambda'].values * 4 * np.pi * (10*phys.pc)**2 # erg / s / A
    tb['Llambda_unc'] = tb['flambda_unc'].values * 4 * np.pi * (10*phys.pc)**2 # erg / s / A
    """
    ixno = tb['mag'].values==99
    tb['mag0'].values[ixno]=99
    tb['mag0_abs'].values[ixno]=99
    tb['flambda'].values[ixno]=99
    tb['flambda_unc'].values[ixno]=99
    tb['Llambda'].values[ixno]=99
    tb['Llambda_unc'].values[ixno]=99
    """
    return tb


def add_datecol(tb):
    """
    tb is pandas dataframe
    
    columns that must exist: mjd
    """
    t = Time(tb["mjd"].values, format='mjd')
    tb['datetime64'] = np.array(t.datetime64, dtype=str)
    space = " "
    date = [space.join(x.split('T')[0].split('-')) for x in tb['datetime64'].values]
    tb['date'] = date
    
    tb = tb.sort_values(by = "mjd")
    return tb


def get_date_span(tb):
    mjdstart = tb['mjd'].values[0]
    mjdend = tb['mjd'].values[-1]
    mjds = np.arange(mjdstart, mjdend+0.9)
    t = Time(mjds, format = "mjd")
    datetime64 = np.array(t.datetime64, dtype=str)
    space = " "
    dates = [space.join(x.split('T')[0].split('-')) for x in datetime64]
    dates = np.array(dates)
    return dates



 