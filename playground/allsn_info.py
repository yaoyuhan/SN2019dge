#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 08:41:10 2019

@author: yuhanyao
"""
import numpy as np
import pandas as pd
import extinction
from copy import deepcopy
from astropy.io import fits
import astropy.io.ascii as asci
from astropy.table import Table, vstack
from astropy.cosmology import FlatLambdaCDM
cosmo = FlatLambdaCDM(H0=70., Om0=0.275)
from collections import OrderedDict as odict
from helper.app2abs import get_date_span, add_datecol

def get_at2019dge(colorplt=False):
    t_max = 58583.2
    z = 0.0213
    ebv = 0.022
    tspecs = np.array([58583.59659, # Keck spec JD at midpoint
                       58597.46300334, # DBSP spec
                       58582.146159,
                       58583.129278,
                       58595.213889,
                       58668.492577])
    
    # LT, SEDM, P48
    tb = pd.read_csv('../data/otherSN/Yao2020/lc_at2019dge.csv')
    result = odict([('z', z),
                    ('ebv', ebv),
                    ('t_max', t_max),
                    ('tspecs', tspecs),
                    ("tb", tb)])
    tb = tb[tb.instrument!="P60+SEDM"]
    
    if colorplt==False:
        return result
    else:
        ix = np.any([tb["instrument"].values == "P48",
                     tb["instrument"].values == "LT+IOO"], axis=0)
        tb = tb[ix]
        ix = np.in1d(tb["filter"].values, np.array(['g', 'r', 'i']))
        tb = tb[ix]
        
        dates = get_date_span(tb)
        datesave = []
        for i in range(len(dates)):
            x = dates[i]
            ix = tb["date"].values == x
            tbsub = tb[ix]
            if len(tbsub)!=0:
                flts = tbsub['filter'].values
                if "r" in flts and np.sum(np.unique(flts))!=1:
                    datesave.append(x)
        datesave = np.array(datesave)
        
        mcolor = []
        mcolor_unc = []
        mjds = []
        colorname = []
        for i in range(len(datesave)):
            rmag = 99
            gmag = 99
            imag = 99
            x = datesave[i]
            ix = tb["date"].values == x
            tbsub = tb[ix]
            gtb = tbsub[tbsub["filter"].values=="g"]
            rtb = tbsub[tbsub["filter"].values=="r"]
            itb = tbsub[tbsub["filter"].values=="i"]
            if len(gtb)!=0:
                gmjds = gtb["mjd"].values
                gmags = gtb["mag0"].values
                gemags = gtb["emag"].values
                gwtgs = 1/gemags**2
                gmag = np.sum(gmags * gwtgs) / np.sum(gwtgs)
                gmjd = np.sum(gmjds * gwtgs) / np.sum(gwtgs)
                gemag = 1/ np.sqrt(np.sum(gwtgs))
            if len(rtb)!=0:
                rmjds = rtb["mjd"].values
                rmags = rtb["mag0"].values
                remags = rtb["emag"].values
                rwtgs = 1/remags**2
                rmag = np.sum(rmags * rwtgs) / np.sum(rwtgs)
                rmjd = np.sum(rmjds * rwtgs) / np.sum(rwtgs)
                remag = 1/ np.sqrt(np.sum(rwtgs))
            if len(itb)!=0:
                imjds = itb["mjd"].values
                imags = itb["mag0"].values
                iemags = itb["emag"].values
                iwtgs = 1/iemags**2
                imag = np.sum(imags * iwtgs) / np.sum(iwtgs)
                imjd = np.sum(imjds * iwtgs) / np.sum(iwtgs)
                iemag = 1/ np.sqrt(np.sum(iwtgs))
            if len(gtb)!=0 and len(rtb)!=0:
                mcolor.append(gmag - rmag)
                mjds.append( 0.5 * (gmjd + rmjd) )
                mcolor_unc.append( np.sqrt(gemag**2 + remag**2) )
                colorname.append("gmr")
            if len(rtb)!=0 and len(itb)!=0:
                mcolor.append(rmag - imag)
                mjds.append( 0.5 * (rmjd + imjd) )
                mcolor_unc.append( np.sqrt(remag**2 + iemag**2) )
                colorname.append("rmi")
            
        ctb = Table(data = [mjds, mcolor, mcolor_unc, colorname],
                    names = ["mjd", "c", "ec", "cname"])
        
        ctb['tmax_rf'] = (ctb['mjd'] - t_max) / (1+z)
        ctb = ctb.to_pandas()
        
        result.update({"ctb": ctb})
        return result



def get_iPTF14gqr(colorplt=False):
    """
    De+18, Table S1, already corrected for extinction
    """
    z = 0.063
    # ebv = 0.082
    D = cosmo.luminosity_distance([z])[0].value * 1e+6 # in pc
    dis_mod = 5*np.log10(D / 10)
    t_exp = 56943.74 # 
    t_max = 56950.26 # g band max light
    
    tb = Table(fits.open('../data/otherSN/De2018/tables1.fit')[1].data)
    tb = tb[~np.isnan(tb['e_mag'])]
    tb.rename_column('Filt' , 'filter')
    tb.rename_column('MJD' , 'mjd')
    tb.rename_column('mag' , 'mag0')
    
    tb['texp_rf'] = (tb['mjd'] - t_exp) / (1+z)
    tb['tmax_rf'] = (tb['mjd'] - t_max) / (1+z)
    ixg = tb['filter']=="g   "
    ixB = tb['filter']=="B   "
    ixV = tb['filter']=="V   "
    ixr = tb['filter']=="r   "
    ixi = tb['filter']=="i   "
    ixUVW1 = tb['filter']=="UVW1"
    ixUVW2 = tb['filter']=="UVW2"
    
    tb['wave'] = np.zeros(len(tb))
    tb['wave'][ixUVW2] = 2079
    tb['wave'][ixUVW1] = 2614
    tb['wave'][ixB] = 4359
    tb['wave'][ixg] = 4814
    tb['wave'][ixV] = 5430
    tb['wave'][ixr] = 6422
    tb['wave'][ixi] = 7883
    
    tb['mag0_abs'] = tb['mag0'] - dis_mod
    
    tb = tb.to_pandas()
    tb["texp_rf"] = tb["Phase"]
    tb = tb.drop(columns=["recno", "Phase", "l_mag"])
    
    ix = np.any([tb['Tel'].values=="P60 ",
                 tb["filter"].values=='g   '], axis=0)
    tb = tb[ix]
    if colorplt==False:
        return tb
    else:
        tb = add_datecol(tb)
        ix = np.in1d(tb["filter"].values, np.array(['g   ', 'r   ', 'i   ']))
        tb = tb[ix]

        dates = get_date_span(tb)
        datesave = []
        for i in range(len(dates)):
            x = dates[i]
            ix = tb["date"].values == x
            tbsub = tb[ix]
            if len(tbsub)!=0:
                flts = tbsub['filter'].values
                if "r   " in flts and np.sum(np.unique(flts))!=1:
                    datesave.append(x)
        datesave = np.array(datesave)
        
        mcolor = []
        mcolor_unc = []
        mjds = []
        colorname = []
        for i in range(len(datesave)):
            x = datesave[i]
            ix = tb["date"].values == x
            tbsub = tb[ix]
            gtb = tbsub[tbsub["filter"].values=="g   "]
            rtb = tbsub[tbsub["filter"].values=="r   "]
            itb = tbsub[tbsub["filter"].values=="i   "]
            if len(gtb)!=0:
                gmjds = gtb["mjd"].values
                gmags = gtb["mag0"].values
                gemags = gtb["e_mag"].values
                gwtgs = 1/gemags**2
                gmag = np.sum(gmags * gwtgs) / np.sum(gwtgs)
                gmjd = np.sum(gmjds * gwtgs) / np.sum(gwtgs)
                gemag = 1/ np.sqrt(np.sum(gwtgs))
            if len(rtb)!=0:
                rmjds = rtb["mjd"].values
                rmags = rtb["mag0"].values
                remags = rtb["e_mag"].values
                rwtgs = 1/remags**2
                rmag = np.sum(rmags * rwtgs) / np.sum(rwtgs)
                rmjd = np.sum(rmjds * rwtgs) / np.sum(rwtgs)
                remag = 1/ np.sqrt(np.sum(rwtgs))
            if len(itb)!=0:
                imjds = itb["mjd"].values
                imags = itb["mag0"].values
                iemags = itb["e_mag"].values
                iwtgs = 1/iemags**2
                imag = np.sum(imags * iwtgs) / np.sum(iwtgs)
                imjd = np.sum(imjds * iwtgs) / np.sum(iwtgs)
                iemag = 1/ np.sqrt(np.sum(iwtgs))
            if len(gtb)!=0 and len(rtb)!=0:
                mcolor.append(gmag - rmag)
                mjds.append( 0.5 * (gmjd + rmjd) )
                mcolor_unc.append( np.sqrt(gemag**2 + remag**2) )
                colorname.append("gmr")
            if len(rtb)!=0 and len(itb)!=0:
                mcolor.append(rmag - imag)
                mjds.append( 0.5 * (rmjd + imjd) )
                mcolor_unc.append( np.sqrt(remag**2 + iemag**2) )
                colorname.append("rmi")
            
        ctb = Table(data = [mjds, mcolor, mcolor_unc, colorname],
                    names = ["mjd", "c", "ec", "cname"])
        
        ctb['tmax_rf'] = (ctb['mjd'] - t_max) / (1+z)
        ctb = ctb.to_pandas()
        return ctb

    
def get_sn2005ek(colorplt=False):
    """
    Drout+13, Table 1, not corrected for extinction
    """
    z = 0.016551
    ebv = 0.210
    D = cosmo.luminosity_distance([z])[0].value * 1e+6 # in pc
    dis_mod = 5*np.log10(D / 10)
    t_max = 53639.9
    print ("adopt r band t_max from Drout+13")
    
    # tb = pd.read_csv('/Users/yuhanyao/Desktop/ZTF18abfcmjw/data/Drout2013/table1', sep='\t')
    # tb = tb.drop(columns=["Unnamed: 6"])
    
    mjds = np.array([53639.3, 53640.3, 53641.3, 53642.2, 53643.2, 53645.3,
                     53646.5, 53648.0, 53649.2, 53650.4, 53651.3, 53652.5,
                     53654.2, 53655.2, 53656.2, 53657.2])
    
    Bmags = np.array([18.25, 18.38, 18.65, np.nan, 19.10, 19.71,
                      20.07, np.nan, 20.67, 20.90, 21.05, np.nan,
                      21.74, np.nan, np.nan, np.nan])
    
    Bmag_uncs = np.array([0.02, 0.03, 0.02, np.nan, 0.05, 0.07, 
                          0.07, np.nan, 0.04, 0.04, 0.04, np.nan,
                          0.12, np.nan, np.nan, np.nan])
    
    Vmags = np.array([17.83, 18.03, 17.92, np.nan, 18.24, 18.66,
                      18.93, 19.48, 19.63, 19.86, 19.98, 20.35,
                      20.60, 20.74, 20.88, 21.22])
    
    Vmag_uncs = np.array([0.02, 0.03, 0.01, np.nan, 0.02, 0.02,
                          0.02, 0.06, 0.03, 0.03, 0.04, 0.05, 
                          0.08, 0.10, 0.08, 0.13])
    
    Rmags = np.array([17.46, 17.41, 17.60, 17.69, 17.86, 18.18, 
                      np.nan, 18.83, 19.03, 19.26, 19.48, 19.75,
                      20.08, np.nan, 20.47, np.nan])
    
    Rmag_uncs = np.array([0.01, 0.02, 0.01, 0.02, 0.01, 0.01,
                          np.nan, 0.03, 0.02, 0.02, 0.02, 0.04,
                          0.05, np.nan, 0.08, np.nan])

    Imags = np.array([17.20, 17.13, 17.18, np.nan, 17.47, 17.71, 
                      np.nan, 18.13, 18.26, 18.51, 18.61, 18.74, 
                      19.01, np.nan, 19.47, np.nan])
    
    Imag_uncs = np.array([0.02, 0.04, 0.02, np.nan, 0.03, 0.02,
                          np.nan, 0.06, 0.02, 0.02, 0.02, 0.03,
                          0.05, np.nan, 0.06, np.nan])
    
    mymjds = np.hstack([mjds, mjds, mjds, mjds])
    mymags = np.hstack([Bmags, Vmags, Rmags, Imags])
    myemags = np.hstack([Bmag_uncs, Vmag_uncs, Rmag_uncs, Imag_uncs])
    myfilts = np.hstack([ np.repeat("B", len(Bmags)),
                          np.repeat("V", len(Bmags)),
                          np.repeat("R", len(Rmags)),
                          np.repeat("I", len(Imags)) ])
    ix = ~np.isnan(mymags)
    tb = pd.DataFrame({'mjd': mymjds[ix],
                       'mag': mymags[ix],
                       'emag': myemags[ix],
                       "filter": myfilts[ix]})
    
    ixB = tb['filter'].values=="B"
    ixV = tb['filter'].values=="V"
    ixR = tb['filter'].values=="R"
    ixI = tb['filter'].values=="I"
    
    tb['wave'] = np.zeros(len(tb))
    tb['wave'].values[ixB] = 4359
    tb['wave'].values[ixV] = 5430
    tb['wave'].values[ixR] = 6349
    tb['wave'].values[ixI] = 8797
    
    tb['mag0'] = tb['mag'] - extinction.ccm89(tb['wave'].values, 3.1*ebv, 3.1)
    tb['mag0_abs'] = tb['mag0'] - dis_mod
    tb['tmax_rf'] = (tb['mjd'] - t_max) / (1+z)
    if colorplt==False:
        return tb
    else:
        tb = add_datecol(tb)
        ix = np.in1d(tb["filter"].values, np.array(['B', 'R', 'I']))
        tb = tb[ix]

        dates = get_date_span(tb)
        datesave = []
        for i in range(len(dates)):
            x = dates[i]
            ix = tb["date"].values == x
            tbsub = tb[ix]
            if len(tbsub)!=0:
                flts = tbsub['filter'].values
                if "R" in flts and np.sum(np.unique(flts))!=1:
                    datesave.append(x)
        datesave = np.array(datesave)
        
        mcolor = []
        mcolor_unc = []
        mjds = []
        colorname = []
        for i in range(len(datesave)):
            x = datesave[i]
            ix = tb["date"].values == x
            tbsub = tb[ix]
            gtb = tbsub[tbsub["filter"].values=="B"]
            rtb = tbsub[tbsub["filter"].values=="R"]
            itb = tbsub[tbsub["filter"].values=="I"]
            if len(gtb)!=0:
                gmjds = gtb["mjd"].values
                gmags = gtb["mag0"].values
                gemags = gtb["emag"].values
                gwtgs = 1/gemags**2
                gmag = np.sum(gmags * gwtgs) / np.sum(gwtgs)
                gmjd = np.sum(gmjds * gwtgs) / np.sum(gwtgs)
                gemag = 1/ np.sqrt(np.sum(gwtgs))
            if len(rtb)!=0:
                rmjds = rtb["mjd"].values
                rmags = rtb["mag0"].values
                remags = rtb["emag"].values
                rwtgs = 1/remags**2
                rmag = np.sum(rmags * rwtgs) / np.sum(rwtgs)
                rmjd = np.sum(rmjds * rwtgs) / np.sum(rwtgs)
                remag = 1/ np.sqrt(np.sum(rwtgs))
            if len(itb)!=0:
                imjds = itb["mjd"].values
                imags = itb["mag0"].values
                iemags = itb["emag"].values
                iwtgs = 1/iemags**2
                imag = np.sum(imags * iwtgs) / np.sum(iwtgs)
                imjd = np.sum(imjds * iwtgs) / np.sum(iwtgs)
                iemag = 1/ np.sqrt(np.sum(iwtgs))
            if len(gtb)!=0 and len(rtb)!=0:
                mcolor.append(gmag - rmag)
                mjds.append( 0.5 * (gmjd + rmjd) )
                mcolor_unc.append( np.sqrt(gemag**2 + remag**2) )
                colorname.append("BmR")
            if len(rtb)!=0 and len(itb)!=0:
                mcolor.append(rmag - imag)
                mjds.append( 0.5 * (rmjd + imjd) )
                mcolor_unc.append( np.sqrt(remag**2 + iemag**2) )
                colorname.append("RmI")
            
        ctb = Table(data = [mjds, mcolor, mcolor_unc, colorname],
                    names = ["mjd", "c", "ec", "cname"])
        
        ctb['tmax_rf'] = (ctb['mjd'] - t_max) / (1+z)
        ctb = ctb.to_pandas()
        return ctb


def get_ptf10iuv(colorplt = False):
    """
    Kasliwal+12, Table 3, not corrected for extinction
    """
    z = 0.0251485
    ebv = 0.0371 # SFD
    D = cosmo.luminosity_distance([z])[0].value * 1e+6 # in pc
    dis_mod = 5*np.log10(D / 10)
    print ("adopt g band t_max estimated by myself")
    t_max = 55357.387 
    tb = pd.read_csv('../data/otherSN/Kasliwal2012/PTF10iuv', sep='\t')
    tb = tb.drop(columns=["Unnamed: 4"])
    tb = tb.rename(columns={'Filter' : 'filter',
                            'MJD': 'mjd'})
    tb = tb[~np.array([x[0]=='>' for x in tb['Mag'].values])]
    tb['mag'] = np.array([float(x.split(" +or-")[0]) for x in tb['Mag'].values])
    tb['emag'] = np.array([float(x.split(" +or-")[1]) for x in tb['Mag'].values])
    tb = tb.drop(columns=["Mag"])
    
    ixg = tb['filter'].values == "g"
    ixr = tb['filter'].values == "r"
    ixi = tb['filter'].values == "i"
    ixz = tb['filter'].values == "z"
    ixB = tb['filter'].values == "B"
    tb['wave'] = np.zeros(len(tb))
    tb['wave'].values[ixB] = 4359
    tb['wave'].values[ixg] = 4814
    tb['wave'].values[ixr] = 6422
    tb['wave'].values[ixi] = 7883
    tb['wave'].values[ixz] = 9670
    
    tb['mag0'] = tb['mag'] - extinction.ccm89(tb['wave'].values, 3.1*ebv, 3.1)
    tb['mag0_abs'] = tb['mag0'] - dis_mod
    tb['tmax_rf'] = (tb['mjd'] - t_max) / (1+z)
    tb = tb.sort_values(by = "mjd")
    if colorplt==False:
        return tb
    
    else:
        tb = add_datecol(tb)
        ix = np.in1d(tb["filter"].values, np.array(['g', 'r', 'i']))
        tb = tb[ix]
        tb = tb[tb.mjd > 55352.5]
        tb = tb[tb.mjd < 55593.5]
        
        dates = get_date_span(tb)
        datesave = []
        for i in range(len(dates)):
            x = dates[i]
            ix = tb["date"].values == x
            tbsub = tb[ix]
            if len(tbsub)!=0:
                flts = tbsub['filter'].values
                if "r" in flts and np.sum(np.unique(flts))!=1:
                    datesave.append(x)
        datesave = np.array(datesave)
        
        mcolor = []
        mcolor_unc = []
        mjds = []
        colorname = []
        for i in range(len(datesave)):
            x = datesave[i]
            ix = tb["date"].values == x
            tbsub = tb[ix]
            gtb = tbsub[tbsub["filter"].values=="g"]
            rtb = tbsub[tbsub["filter"].values=="r"]
            itb = tbsub[tbsub["filter"].values=="i"]
            if len(gtb)!=0:
                gmjds = gtb["mjd"].values
                gmags = gtb["mag0"].values
                gemags = gtb["emag"].values
                gwtgs = 1/gemags**2
                gmag = np.sum(gmags * gwtgs) / np.sum(gwtgs)
                gmjd = np.sum(gmjds * gwtgs) / np.sum(gwtgs)
                gemag = 1/ np.sqrt(np.sum(gwtgs))
            if len(rtb)!=0:
                rmjds = rtb["mjd"].values
                rmags = rtb["mag0"].values
                remags = rtb["emag"].values
                rwtgs = 1/remags**2
                rmag = np.sum(rmags * rwtgs) / np.sum(rwtgs)
                rmjd = np.sum(rmjds * rwtgs) / np.sum(rwtgs)
                remag = 1/ np.sqrt(np.sum(rwtgs))
            if len(itb)!=0:
                imjds = itb["mjd"].values
                imags = itb["mag0"].values
                iemags = itb["emag"].values
                iwtgs = 1/iemags**2
                imag = np.sum(imags * iwtgs) / np.sum(iwtgs)
                imjd = np.sum(imjds * iwtgs) / np.sum(iwtgs)
                iemag = 1/ np.sqrt(np.sum(iwtgs))
            if len(gtb)!=0 and len(rtb)!=0:
                mcolor.append(gmag - rmag)
                mjds.append( 0.5 * (gmjd + rmjd) )
                mcolor_unc.append( np.sqrt(gemag**2 + remag**2) )
                colorname.append("gmr")
            if len(rtb)!=0 and len(itb)!=0:
                mcolor.append(rmag - imag)
                mjds.append( 0.5 * (rmjd + imjd) )
                mcolor_unc.append( np.sqrt(remag**2 + iemag**2) )
                colorname.append("rmi")
            
        ctb = Table(data = [mjds, mcolor, mcolor_unc, colorname],
                    names = ["mjd", "c", "ec", "cname"])
        
        ctb['tmax_rf'] = (ctb['mjd'] - t_max) / (1+z)
        ctb = ctb.to_pandas()
        return ctb
    
            

def get_sn1010X(colorplt = False):
    """
    Kasliwal+10
    """
    ebv = 0.1249 # SFD
    z = 0.015
    D = cosmo.luminosity_distance([z])[0].value * 1e+6 # in pc
    dis_mod = 5*np.log10(D / 10)
    t_max = 55239.2
    
    tb = pd.read_csv("../data/otherSN/Kasliwal2010/photometry.csv")
    tb = tb.drop(columns=["source", "event", "instrument"])
    tb = tb[tb.upperlimit=="F"]
    tb = tb.drop(columns=["upperlimit"])
    tb = tb.rename(columns={'magnitude' : 'mag',
                            'e_magnitude': 'emag',
                            'band': 'filter',
                            'time': 'mjd'})
    
    ixr = tb['filter'].values == "r"
    ixg = tb['filter'].values == "g"
    ixi = tb['filter'].values == "i"
    tb['wave'] = np.zeros(len(tb))
    tb['wave'].values[ixg] = 4814
    tb['wave'].values[ixr] = 6422
    tb['wave'].values[ixi] = 7883
    
    tb['mag0'] = tb['mag'] - extinction.ccm89(tb['wave'].values, 3.1*ebv, 3.1)
    tb['mag0_abs'] = tb['mag0'] - dis_mod
    tb['tmax_rf'] = (tb['mjd'] - t_max) / (1+z)
    if colorplt==False:
        return tb
    
    else:
        tb = add_datecol(tb)
        dates = get_date_span(tb)
        datesave = []
        for i in range(len(dates)):
            x = dates[i]
            ix = tb["date"].values == x
            tbsub = tb[ix]
            if len(tbsub)!=0:
                flts = tbsub['filter'].values
                if "r" in flts and np.sum(np.unique(flts))!=1:
                    datesave.append(x)
        datesave = np.array(datesave)
        
        mcolor = []
        mcolor_unc = []
        mjds = []
        colorname = []
        for i in range(len(datesave)):
            x = datesave[i]
            ix = tb["date"].values == x
            tbsub = tb[ix]
            gtb = tbsub[tbsub["filter"].values=="g"]
            rtb = tbsub[tbsub["filter"].values=="r"]
            itb = tbsub[tbsub["filter"].values=="i"]
            if len(gtb)!=0:
                gmjds = gtb["mjd"].values
                gmags = gtb["mag"].values
                gemags = gtb["emag"].values
                gwtgs = 1/gemags**2
                gmag = np.sum(gmags * gwtgs) / np.sum(gwtgs)
                gmjd = np.sum(gmjds * gwtgs) / np.sum(gwtgs)
                gemag = 1/ np.sqrt(np.sum(gwtgs))
            else:
                gmag = 0
            if len(rtb)!=0:
                rmjds = rtb["mjd"].values
                rmags = rtb["mag"].values
                remags = rtb["emag"].values
                rwtgs = 1/remags**2
                rmag = np.sum(rmags * rwtgs) / np.sum(rwtgs)
                rmjd = np.sum(rmjds * rwtgs) / np.sum(rwtgs)
                remag = 1/ np.sqrt(np.sum(rwtgs))
            else:
                rmag = 0
            if len(itb)!=0:
                imjds = itb["mjd"].values
                imags = itb["mag"].values
                iemags = itb["emag"].values
                iwtgs = 1/iemags**2
                imag = np.sum(imags * iwtgs) / np.sum(iwtgs)
                imjd = np.sum(imjds * iwtgs) / np.sum(iwtgs)
                iemag = 1/ np.sqrt(np.sum(iwtgs))
            else:
                imag = 0
            if gmag and rmag:
                mcolor.append(gmag - rmag)
                mjds.append( 0.5 * (gmjd + rmjd) )
                mcolor_unc.append( np.sqrt(gemag**2 + remag**2) )
                colorname.append("gmr")
            if rmag and imag:
                mcolor.append(rmag - imag)
                mjds.append( 0.5 * (rmjd + imjd) )
                mcolor_unc.append( np.sqrt(remag**2 + iemag**2) )
                colorname.append("rmi")
            
        ctb = Table(data = [mjds, mcolor, mcolor_unc, colorname],
                    names = ["mjd", "c", "ec", "cname"])
        
        ctb['tmax_rf'] = (ctb['mjd'] - t_max) / (1+z)
        ctb = ctb.to_pandas()
        return ctb
    
    
def digital_latex(mjds, phases, rs, insts):
    ix = np.array(["\\pm" in x for x in rs])
    mjds = np.array(mjds)[ix]
    phases = np.array(phases)[ix]
    insts = np.array(insts)[ix]
    rs = np.array(rs)[ix]
    mags = np.array([float(x.split("\\pm")[0]) for x in rs])
    emags = np.array([float(x.split("\\pm")[1][:5]) for x in rs])
    
    ix1 = np.array([x.split(" ")[1]!="LCOGT" for x in insts])
    ix2 = np.array([x.split(" ")[1]!="P60" for x in insts])
    ix3 = emags<0.5
    ix = ix1&ix2&ix3
    return mjds[ix], phases[ix], mags[ix], emags[ix]
    

def get_sn2018kzr(colorplt = False):
    """
    Owen R. Mcbrien 2019
    """
    ebv = 0.113/3.1
    z = 0.053
    D = cosmo.luminosity_distance([z])[0].value * 1e+6 # in pc
    dis_mod = 5*np.log10(D / 10)
    t_max = 58480.422+0.1
    
    f = open('../data/otherSN/Mcbrien2019/table1.tex')
    lines = f.readlines()
    f.close()
    lines = lines[:-4]
    
    dates = [x.split("&")[0] for x in lines]
    mjds = [float(x.split("&")[1]) for x in lines]
    phases = [float(x.split("&")[2].replace('$', '').replace('\t', '')) for x in lines]
    gs = [x.split("&")[3].replace('$', '') for x in lines]
    rs = [x.split("&")[4].replace('$', '') for x in lines]
    iis = [x.split("&")[5].replace('$', '') for x in lines]
    zs = [x.split("&")[6].replace('$', '') for x in lines]
    insts = [x.split("&")[7] for x in lines]
    
    dtg = digital_latex(mjds, phases, gs, insts)
    dtr = digital_latex(mjds, phases, rs, insts)
    dti = digital_latex(mjds, phases, iis, insts)
    
    filt = np.hstack([np.repeat("g", len(dtg[0])),
                      np.repeat("r", len(dtr[0])),
                      np.repeat("i", len(dti[0]))])
    phase = np.hstack([dtg[1], dtr[1], dti[1]])
    mag = np.hstack([dtg[2], dtr[2], dti[2]])
    emag = np.hstack([dtg[3], dtr[3], dti[3]])
    mjd = np.hstack([dtg[0], dtr[0], dti[0]])
    
    tb = Table(data = [(mjd - t_max) / (1+z), mag, emag, filt],
               names = ['tmax_rf', 'mag', 'emag', 'filter'])
    
    ixr = tb['filter'] == "r"
    ixg = tb['filter'] == "g"
    ixi = tb['filter'] == "i"
    tb['wave'] = np.zeros(len(tb))
    tb['wave'][ixg] = 4814
    tb['wave'][ixr] = 6422
    tb['wave'][ixi] = 7883
    tb['mag0'] = tb['mag'] - extinction.ccm89(tb['wave'], 3.1*ebv, 3.1)
    tb['mag0_abs'] = tb['mag0'] - dis_mod
    tb = tb.to_pandas()
    return tb
    

def _read_2019bkc_band():
    tb = pd.read_csv('../data/otherSN/Chen2020/table2', sep='\t')
    tb = tb.drop(columns = ['JD -', 'Telescope', 'Unnamed: 13', 'J', 'H'])
    tb = tb.rename(columns={'#Date': 'Date'})
    colpool = ['B', 'V', 'R', 'I', 'g', 'r', 'i']
    for magcol in colpool:
        tb1 = tb[tb[magcol].values!='cdots']
        tb1.insert(2, "filter", np.repeat(magcol,len(tb1)))
        mags= tb1[magcol]
        xx = [float(x.split("(")[0]) for x in mags]
        exx = [float(x.split("(")[1].split(")")[0])/100 for x in mags]
        tb1.insert(2, "mag", xx)
        tb1.insert(2, "emag", exx)
        tb1 = tb1.drop(columns = colpool)
        if magcol == "B":
            tb1['wave'] = np.ones(len(tb1))* 4450
            tb2 = deepcopy(tb1)
        else:
            if magcol == "r" or magcol == "R":
                tb1['wave'] = np.ones(len(tb1))* 6422
            elif magcol == "i" or magcol == "I":
                tb1['wave'] = np.ones(len(tb1))* 7500
            elif magcol == "g":
                df = pd.DataFrame({"Date":['2019 Feb 28'], 
                                   "Phase":[-5.5],
                                   "emag":[0.03],
                                   "mag": [18.7],
                                   "filter": ["g"]}) 
                tb1 = tb1.append(df)
                tb1['wave'] = np.ones(len(tb1))* 4810
            elif magcol == "V":
                tb1['wave'] = np.ones(len(tb1))* 5510
            tb2 = pd.concat([tb2, tb1])
    return tb2


def get_sn2019bkc(colorplt = False):
    """
    Chen 2019, Figure 5
    """
    ebv = 0.06 # SFD2011
    z = 0.020
    D = cosmo.luminosity_distance([z])[0].value * 1e+6 # in pc
    dis_mod = 5*np.log10(D / 10)
    
    tb = _read_2019bkc_band()
    
    tb['mag0'] = tb['mag'].values- extinction.ccm89(tb['wave'].values, 3.1*ebv, 3.1)
    tb['mag0_abs'] = tb['mag0'].values - dis_mod
    
    tb['tmax_rf'] = tb['Phase'].values / (1+z)

    if colorplt==False:
        return tb
    
    else:
        #tb = add_datecol(tb)
        tb['date'] = np.floor(tb['tmax_rf'].values)
        datesave = np.array(tb['date'].values)
        
        mcolor = []
        mcolor_unc = []
        mjds = []
        colorname = []
        for i in range(len(datesave)):
            x = datesave[i]
            ix = tb["date"].values == x
            tbsub = tb[ix]
            gtb = tbsub[tbsub["filter"].values=="g"]
            rtb = tbsub[tbsub["filter"].values=="r"]
            itb = tbsub[tbsub["filter"].values=="i"]
            if len(gtb)!=0:
                gmjds = gtb["tmax_rf"].values
                gmags = gtb["mag0"].values
                gemags = np.ones(len(gtb)) * 0.1
                gwtgs = 1/gemags**2
                gmag = np.sum(gmags * gwtgs) / np.sum(gwtgs)
                gmjd = np.sum(gmjds * gwtgs) / np.sum(gwtgs)
                gemag = 1/ np.sqrt(np.sum(gwtgs))
            else:
                gmag = 0
            if len(rtb)!=0:
                rmjds = rtb["tmax_rf"].values
                rmags = rtb["mag0"].values
                remags = np.ones(len(rtb)) * 0.1
                rwtgs = 1/remags**2
                rmag = np.sum(rmags * rwtgs) / np.sum(rwtgs)
                rmjd = np.sum(rmjds * rwtgs) / np.sum(rwtgs)
                remag = 1/ np.sqrt(np.sum(rwtgs))
            else:
                rmag = 0
            if len(itb)!=0:
                imjds = itb["tmax_rf"].values
                imags = itb["mag0"].values
                iemags = np.ones(len(itb)) * 0.1
                iwtgs = 1/iemags**2
                imag = np.sum(imags * iwtgs) / np.sum(iwtgs)
                imjd = np.sum(imjds * iwtgs) / np.sum(iwtgs)
                iemag = 1/ np.sqrt(np.sum(iwtgs))
            else:
                imag = 0
            if gmag and rmag:
                mcolor.append(gmag - rmag)
                mjds.append( 0.5 * (gmjd + rmjd) )
                mcolor_unc.append( np.sqrt(gemag**2 + remag**2) )
                colorname.append("gmr")
            if rmag and imag:
                mcolor.append(rmag - imag)
                mjds.append( 0.5 * (rmjd + imjd) )
                mcolor_unc.append( np.sqrt(remag**2 + iemag**2) )
                colorname.append("rmi")
            
        ctb = Table(data = [mjds, mcolor, mcolor_unc, colorname],
                    names = ["tmax_rf", "c", "ec", "cname"])
        
        ctb = ctb.to_pandas()
        return ctb
    

def get_ogle13sn079(colorplt=False):
    ebv = 0.02 # SFD2011
    z = 0.07
    D = cosmo.luminosity_distance([z])[0].value * 1e+6 # in pc
    dis_mod = 5*np.log10(D / 10)
    t_max = 56565.19  # I band maximum
    print ("adopt I band t_max estimated by myself")
    
    tb = pd.read_csv("../data/otherSN/Inserra2015/table1", sep='\t')
    tb = tb.drop(columns=["Unnamed: 9", "Date"])
    tb = tb.rename(columns={'MJD': 'mjd'})
    
    mjds = []
    mags = []
    emags = []
    filts = []
    for i in range(len(tb)):
        tdt = tb['mjd'].values[i]
        gdt = tb["g"].values[i]
        rdt = tb["r"].values[i]
        idt = tb["I"].values[i]
        if type(gdt)==str:
            if not gdt[0]=='>':
                mags.append( float(gdt.split(" (")[0]) )
                emags.append( float(gdt.split(" (")[1][:-1]) )
                mjds.append(tdt)
                filts.append("g")
        if type(rdt)==str:
            if not rdt[0]=='>':
                mags.append( float(rdt.split(" (")[0]) )
                emags.append( float(rdt.split(" (")[1][:-1]) )
                mjds.append(tdt)
                filts.append("r")
        if type(idt)==str:
            if not idt[0]=='>':
                mags.append( float(idt.split(" (")[0]) )
                emags.append( float(idt.split(" (")[1][:-1]) )
                mjds.append(tdt)
                filts.append("i")
    
    tb = Table(data = [mjds, mags, emags, filts],
               names = ["mjd", "mag", "emag", "filter"])
    ixr = tb['filter'] == "r"
    ixg = tb['filter'] == "g"
    ixi = tb['filter'] == "i"
    tb['wave'] = np.zeros(len(tb))
    tb['wave'][ixg] = 4814
    tb['wave'][ixr] = 6422
    tb['wave'][ixi] = 7883
    
    tb['mag0'] = tb['mag'] - extinction.ccm89(tb['wave'], 3.1*ebv, 3.1)
    tb['mag0_abs'] = tb['mag0'] - dis_mod
    tb['tmax_rf'] = (tb['mjd'] - t_max) / (1+z)
    tb = tb.to_pandas()
    if colorplt==False:
        return tb
    
    else:
        tb = add_datecol(tb)
        dates = get_date_span(tb)
        datesave = []
        for i in range(len(dates)):
            x = dates[i]
            ix = tb["date"].values == x
            tbsub = tb[ix]
            if len(tbsub)!=0:
                flts = tbsub['filter'].values
                if "r" in flts and np.sum(np.unique(flts))!=1:
                    datesave.append(x)
        datesave = np.array(datesave)
        
        mcolor = []
        mcolor_unc = []
        mjds = []
        colorname = []
        for i in range(len(datesave)):
            x = datesave[i]
            ix = tb["date"].values == x
            tbsub = tb[ix]
            gtb = tbsub[tbsub["filter"].values=="g"]
            rtb = tbsub[tbsub["filter"].values=="r"]
            itb = tbsub[tbsub["filter"].values=="i"]
            if len(gtb)!=0:
                gmjds = gtb["mjd"].values
                gmags = gtb["mag0"].values
                gemags = gtb["emag"].values
                gwtgs = 1/gemags**2
                gmag = np.sum(gmags * gwtgs) / np.sum(gwtgs)
                gmjd = np.sum(gmjds * gwtgs) / np.sum(gwtgs)
                gemag = 1/ np.sqrt(np.sum(gwtgs))
            else:
                gmag=0
            if len(rtb)!=0:
                rmjds = rtb["mjd"].values
                rmags = rtb["mag0"].values
                remags = rtb["emag"].values
                rwtgs = 1/remags**2
                rmag = np.sum(rmags * rwtgs) / np.sum(rwtgs)
                rmjd = np.sum(rmjds * rwtgs) / np.sum(rwtgs)
                remag = 1/ np.sqrt(np.sum(rwtgs))
            else:
                rmag = 0
            if len(itb)!=0:
                imjds = itb["mjd"].values
                imags = itb["mag0"].values
                iemags = itb["emag"].values
                iwtgs = 1/iemags**2
                imag = np.sum(imags * iwtgs) / np.sum(iwtgs)
                imjd = np.sum(imjds * iwtgs) / np.sum(iwtgs)
                iemag = 1/ np.sqrt(np.sum(iwtgs))
            else:
                imag = 0
            if gmag and rmag:
                mcolor.append(gmag - rmag)
                mjds.append( 0.5 * (gmjd + rmjd) )
                mcolor_unc.append( np.sqrt(gemag**2 + remag**2) )
                colorname.append("gmr")
            if rmag and imag:
                mcolor.append(rmag - imag)
                mjds.append( 0.5 * (rmjd + imjd) )
                mcolor_unc.append( np.sqrt(remag**2 + iemag**2) )
                colorname.append("rmi")
            
        ctb = Table(data = [mjds, mcolor, mcolor_unc, colorname],
                    names = ["mjd", "c", "ec", "cname"])
        
        ctb['tmax_rf'] = (ctb['mjd'] - t_max) / (1+z)
        ctb = ctb.to_pandas()
        return ctb
    

    
def get_ias(name = "ZTF18abclfee", cut=27):
    aam_df = pd.read_csv('/Users/yuhanyao/Desktop/earlyIa/2018/table/merged_table.csv')
    mylc = pd.read_csv('/Users/yuhanyao/Desktop/earlyIa/2018/data/binned_lc/'+name+'.csv')
    mylc = mylc[mylc.mag!=99]
    
    ix = np.where(aam_df['name'].values == name)[0][0]
    t_max = float(aam_df['t0_g_adopted'][aam_df['name'] == name].values)
    z = aam_df['z_adopt'].values[ix]
    ebv = aam_df['E_B_V_SandF'].values[ix]
    D = cosmo.luminosity_distance([z])[0].value * 1e+6 # in pc
    dis_mod = 5*np.log10(D / 10)
    
    phase = (mylc['jd'].values - t_max) / (1+z)
    mag = mylc['mag'].values
    
    emag = mylc['mag_unc'].values
    
    filt = np.repeat("g", len(mag))
    ix = mylc['filterid'].values==2
    filt[ix] = "r"
    
    tb = Table(data = [phase / (1+z), mag, emag, filt],
               names = ['tmax_rf', 'mag', 'emag', 'filter'])
    
    ixr = tb['filter'] == "r"
    ixg = tb['filter'] == "g"
    tb['wave'] = np.zeros(len(tb))
    tb['wave'][ixg] = 4814
    tb['wave'][ixr] = 6422
    
    tb['mag0'] = tb['mag'] - extinction.ccm89(tb['wave'], 3.1*ebv, 3.1)
    tb['mag0_abs'] = tb['mag0'] - dis_mod
    tb = tb.to_pandas()
    
    tb = tb[tb.tmax_rf < cut]
    return tb
    

def get_at2018erx():
    tb = asci.read('/Users/yuhanyao/Desktop/ZTF18abfcmjw/data/De2020/ztf18abkmbpy')
    tb = tb[tb['filter']=='r']
    tb = tb[tb['mag']!=99]
    tb = tb[tb['emag']<1]
    ebv = 0.016 # SFD2011
    z = 0.02938
    D = cosmo.luminosity_distance([z])[0].value * 1e+6 # in pc
    dis_mod = 5*np.log10(D / 10)
    t_max = 2458333.83  # r band maximum
    
    ixr = tb['filter'] == "r"
    tb['wave'] = np.zeros(len(tb))
    tb['wave'][ixr] = 6422
    
    tb['mag0'] = tb['mag'] - extinction.ccm89(tb['wave'], 3.1*ebv, 3.1)
    tb['mag0_abs'] = tb['mag0'] - dis_mod
    tb['tmax_rf'] = (tb['jdobs'] - t_max) / (1+z)
    tb = tb.to_pandas()
    return  tb
    

def get_ptf09dav():
    tb = asci.read('../data/otherSN/Sullivan2011/ptf09dav')
    tb.rename_column('band', 'filter')
    tb.rename_column('magnitude', 'mag')
    tb.rename_column('e_magnitude', 'emag')
    tb.remove_column("instrument")
    tb = tb[tb['mag']>19.7]
    ix = np.any([tb['filter']=='r', tb['filter']=='R'], axis=0)
    tb = tb[ix]
    tb['filter']=='r'
    z = 0.0359
    D = cosmo.luminosity_distance([z])[0].value * 1e+6 # in pc
    ebv = 0.044
    dis_mod = 5*np.log10(D / 10)
    t_max = 55054  # r band maximum
    
    tb['wave'] = np.ones(len(tb))* 6422
    tb['mag0'] = tb['mag'] - extinction.ccm89(tb['wave'], 3.1*ebv, 3.1)
    tb['mag0_abs'] = tb['mag0'] - dis_mod
    tb['tmax_rf'] = (tb['time'] - t_max) / (1+z)
    tb['emag'] = np.ones(len(tb))*0.1
    tb.remove_row(2)
    tb['mag0_abs'][1] = -15.4
    tb = tb.to_pandas()
    return  tb


def _read_2016hnk_band():
    tb = Table(fits.open('../data/otherSN/Galbany2019/tablec3.fit')[1].data)
    #tb.remove_columns(['Bmag', 'e_Bmag', 'zmag', 'e_zmag', 'imag', 'e_imag'])
    #tb.remove_columns(['Vmag', 'e_Vmag', 'gmag', 'e_gmag', 'umag', 'l_umag'])
    colpool = ["Bmag", "zmag", "imag", "rmag", "Vmag", "gmag"]
    for magcol in colpool:
        emagcol = "e_"+magcol
        tb1 = tb[~np.isnan(tb[magcol])]
        tb1["filter"] = magcol[0]
        tb1.rename_column(magcol, 'mag')
        tb1.rename_column(emagcol, 'emag')
        if magcol == "Bmag":
            tb1['wave'] = np.ones(len(tb1))* 4450
            tb2 = tb1
        else:
            if magcol == "rmag":
                tb1['wave'] = np.ones(len(tb1))* 6422
            elif magcol == "zmag":
                tb1['wave'] = np.ones(len(tb1))* 8500
            elif magcol == "imag":
                tb1['wave'] = np.ones(len(tb1))* 7500
            elif magcol == "gmag":
                tb1['wave'] = np.ones(len(tb1))* 4810
            elif magcol == "Vmag":
                tb1['wave'] = np.ones(len(tb1))* 5510
            tb2 = vstack([tb2, tb1])
    tb2.remove_columns(colpool)
    tb2.remove_columns(["umag", "l_umag", "Inst"])
    for magcol in colpool:
        emagcol = "e_"+magcol
        tb2.remove_columns([emagcol])
    tb2 = tb2[~np.isnan(tb2["emag"])]
    return tb2


def _read_2016hnk_ATLAS():
    tb = Table(fits.open('../data/otherSN/Galbany2019/table2.fit')[1].data)
    colpool = ["cmag", "omag"]
    for magcol in colpool:
        emagcol = "e_"+magcol
        tb1 = tb[~np.isnan(tb[magcol])]
        tb1["filter"] = magcol[0]
        tb1.rename_column(magcol, 'mag')
        tb1.rename_column(emagcol, 'emag')
        if magcol == "cmag":
            tb1['wave'] = np.ones(len(tb1))* 5300
            tb2 = tb1
        else:
            if magcol == "omag":
                tb1['wave'] = np.ones(len(tb1))* 6900
            tb2 = vstack([tb2, tb1])
    tb2.remove_columns(colpool)
    tb2.remove_columns(["l_omag"])
    for magcol in colpool:
        emagcol = "e_"+magcol
        tb2.remove_columns([emagcol])
    tb2 = tb2[~np.isnan(tb2["emag"])]
    return tb2
            

def get_sn2016hnk():
    z = 0.016
    D = cosmo.luminosity_distance([z])[0].value * 1e+6 # in pc
    ebv = 0.0224
    dis_mod = 5*np.log10(D / 10)
    tb1 = _read_2016hnk_band()
    tb2 = _read_2016hnk_ATLAS()
    
    tb = vstack([tb1, tb2])
    
    tb['tmax_rf'] = (tb['Epoch']*(1+z)-3.5)/(1+z)
    
    tb['mag0'] = tb['mag'] - extinction.ccm89(tb['wave'], 3.1*ebv, 3.1)
    tb['mag0_abs'] = tb['mag0'] - dis_mod
    tb = tb.to_pandas()
    return  tb
    
    
def get_sn2007ax():
    z = 0.006878 
    D = cosmo.luminosity_distance([z])[0].value * 1e+6 # in pc
    ebv = 0.0446
    dis_mod = 5*np.log10(D / 10)
    
    tb = asci.read('/Users/yuhanyao/Desktop/ZTF18abfcmjw/data/Kasliwal2008/sn2007ax')
    
    
    tb.rename_column('band', 'filter')
    tb.rename_column('magnitude', 'mag')
    tb.rename_column('e_magnitude', 'emag')
    #tb = tb[tb['mag']>19.7]
    ix = np.any([tb['filter']=='r', tb['filter']=='R'], axis=0)
    tb = tb[ix]

    t_max = 2454187-2400000.5+3  # r band maximum
    
    tb['wave'] = np.ones(len(tb))* 6422
    tb['mag0'] = tb['mag'] - extinction.ccm89(tb['wave'], 3.1*ebv, 3.1)
    tb['mag0_abs'] = tb['mag0'] - dis_mod
    tb['tmax_rf'] = (tb['time'] - t_max) / (1+z)
    tb = tb.to_pandas()
    return  tb


def get_sn2002bj(colorplt = False):
    z = 0.012029
    D = cosmo.luminosity_distance([z])[0].value * 1e+6 # in pc
    ebv = 0.0787
    dis_mod = 5*np.log10(D / 10)
    
    tb = asci.read('../data/otherSN/Poznanski2010/sn2002bj')
    tb.remove_columns(['source', 'upperlimit', 'event', 'instrument'])
    
    tb.rename_column('band', 'filter')
    tb.rename_column('magnitude', 'mag')
    tb.rename_column('e_magnitude', 'emag')
    tb.rename_column('time', 'mjd')
    ix = tb['filter']=='R'
    tb["filter"][ix] = "r"
    ix = np.any([tb['filter']=='r', tb['filter']=='B', tb['filter']=='I'], axis=0)
    tb = tb[ix]
    tb = tb[~tb['emag'].mask]

    t_max = 52335.79-2  # r band maximum
    
    ixr = tb["filter"] == "r"
    ixB = tb["filter"] == "B"
    ixI = tb["filter"] == "I"
    tb['wave'] = np.ones(len(tb))
    tb['wave'][ixr] = 6422
    tb['wave'][ixB] = 4450
    tb['wave'][ixI] = 8060
    tb['mag0'] = tb['mag'] - extinction.ccm89(tb['wave'], 3.1*ebv, 3.1)
    tb['mag0_abs'] = tb['mag0'] - dis_mod
    tb['tmax_rf'] = (tb['mjd'] - t_max) / (1+z)
    tb = tb.to_pandas()
    if colorplt == False:
        return  tb
    else:
        tb = add_datecol(tb)
        dates = get_date_span(tb)
        datesave = []
        for i in range(len(dates)):
            x = dates[i]
            ix = tb["date"].values == x
            tbsub = tb[ix]
            if len(tbsub)!=0:
                flts = tbsub['filter'].values
                if "r" in flts and np.sum(np.unique(flts))!=1:
                    datesave.append(x)
        datesave = np.array(datesave)
        
        mcolor = []
        mcolor_unc = []
        mjds = []
        colorname = []
        for i in range(len(datesave)):
            x = datesave[i]
            ix = tb["date"].values == x
            tbsub = tb[ix]
            gtb = tbsub[tbsub["filter"].values=="B"]
            rtb = tbsub[tbsub["filter"].values=="r"]
            itb = tbsub[tbsub["filter"].values=="I"]
            if len(gtb)!=0:
                gmjds = gtb["mjd"].values
                gmags = gtb["mag0"].values
                gemags = gtb["emag"].values
                gwtgs = 1/gemags**2
                gmag = np.sum(gmags * gwtgs) / np.sum(gwtgs)
                gmjd = np.sum(gmjds * gwtgs) / np.sum(gwtgs)
                gemag = 1/ np.sqrt(np.sum(gwtgs))
            else:
                gmag=0
            if len(rtb)!=0:
                rmjds = rtb["mjd"].values
                rmags = rtb["mag0"].values
                remags = rtb["emag"].values
                rwtgs = 1/remags**2
                rmag = np.sum(rmags * rwtgs) / np.sum(rwtgs)
                rmjd = np.sum(rmjds * rwtgs) / np.sum(rwtgs)
                remag = 1/ np.sqrt(np.sum(rwtgs))
            else:
                rmag = 0
            if len(itb)!=0:
                imjds = itb["mjd"].values
                imags = itb["mag0"].values
                iemags = itb["emag"].values
                iwtgs = 1/iemags**2
                imag = np.sum(imags * iwtgs) / np.sum(iwtgs)
                imjd = np.sum(imjds * iwtgs) / np.sum(iwtgs)
                iemag = 1/ np.sqrt(np.sum(iwtgs))
            else:
                imag = 0
            if gmag and rmag:
                mcolor.append(gmag - rmag)
                mjds.append( 0.5 * (gmjd + rmjd) )
                mcolor_unc.append( np.sqrt(gemag**2 + remag**2) )
                colorname.append("BmR")
            if rmag and imag:
                mcolor.append(rmag - imag)
                mjds.append( 0.5 * (rmjd + imjd) )
                mcolor_unc.append( np.sqrt(remag**2 + iemag**2) )
                colorname.append("RmI")
            
        ctb = Table(data = [mjds, mcolor, mcolor_unc, colorname],
                    names = ["mjd", "c", "ec", "cname"])
        
        ctb['tmax_rf'] = (ctb['mjd'] - t_max) / (1+z)
        ctb = ctb.to_pandas()
        return ctb
    
    
