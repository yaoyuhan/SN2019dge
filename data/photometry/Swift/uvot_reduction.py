#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 14:07:29 2019

@author: yuhanyao
"""
import glob
import numpy as np
import pandas as pd
from astropy.io import fits
import csv
from astropy.time import Time
#import astropy.io.ascii as asci

# w3browse-270382.tar, obsid: 00011286001; startdate: 2019-04-09 19:44:00
# w3browse-293532.tar, obsid: 00011286002; startdate: 2019-04-10 13:41:55
# calibration file: 00011286003


"""
quote from Brad Cenko:
    5 arcsec is the standard UVOT aperture (i.e., what the photometric system is defined for)
    3 arcsec generally works better for faint sources or those in high background regions
    
Thus for AT2019dge, I choose 3 arcsec aperutre
"""

def getuvotdt(dirname):
    filesd = glob.glob(dirname+'uvot/image/*.dat')
    filesf = glob.glob(dirname+'uvot/image/*.fits')
    filesd = np.array(filesd)
    filesf = np.array(filesf)
    arg = np.argsort(filesd)
    filesd = filesd[arg]
    arg = np.argsort(filesf)
    filesf = filesf[arg]
    
    filters = []
    exptime = []
    mags = []
    magerrs = []
    limmags = []
    jds = []
    dates = []

    for i in range(6):
        hd = fits.open(filesf[i])[0].header
        filters.append(hd['FILTER'])
        date = hd['DATE-OBS']
        t = Time(date, format='isot', scale='utc')
        dates.append(date)
        jds.append(t.jd)
        
        f = open(filesd[i])
        lst = []
        reader = csv.reader(f, delimiter="\t")
        for line in reader:
            lst += [line]
            
        exptime.append(lst[5][0].split('Exposure: ')[-1].split(' s')[0])
        magstr = lst[15][0].split('Source: ')[-1]
        limstr = lst[17][0]
        
        try:
            mags.append(float(magstr.split(' +/- ')[0]))
            magerrs.append(float(magstr.split(' +/- ')[1][:-7]))
        except:
            mags.append(99)
            magerrs.append(99)
        limmags.append(float(limstr.split('Background-limit: ')[-1]))
    
    filters = np.array(filters)
    exptime = np.array(exptime)
    mags = np.array(mags)
    magerrs = np.array(magerrs)
    limmags = np.array(limmags)
    jds = np.array(jds)
    s = " "
    date = [s.join(x.split('T')[0].split('-')) for x in dates]
    '''
    coof = glob.glob(dirname+'uvot/image/src.reg')[0]
    ra = float(asci.read(coof)['col1'].data[0].split('(')[-1])
    dec = asci.read(coof)['col2'].data[0]

    f = open(dirname + 'result.txt', mode='w')
    for i in range(6):
        f.write(str(ra)+' '+str(dec)+' ' +str(jds[i])+' '+str(exptime[i])+' '+\
                filters[i]+' '+str(mags[i])+' '+str(magerrs[i])+' '+\
                str(limmags[i])+'\n')
        
    f.close()
    '''   
    df = pd.DataFrame({'filter': filters,
                       'exptime': exptime,
                       'mag': mags,
                       'emag': magerrs,
                       'limmag': limmags,
                       'jdobs': jds,
                       'date': date})
    df = df.sort_values(by=['filter'])
    return df



def cal_transient_mag(df1, df3):
    filters = df1['filter'].values
    
    dfcopy = df1
    for i in range(len(filters)):
        
        m12 = df1['mag'].values[i] # transient
        m12_unc = df1['emag'].values[i]
        m1 = df3['mag'].values[i] # galaxy + transient
        m1_unc = df3['emag'].values[i] 
        m2 = -2.5 * np.log10(10**(-0.4 * m12) - 10**(-0.4 * m1))
        m2_unc = np.hypot(10**(-0.4 * m12)*m12_unc, 10**(-0.4 * m1)*m1_unc) / (10**(-0.4 * m12) - 10**(-0.4 * m1))
        
        dfcopy['mag'].values[i] = m2
        dfcopy['emag'].values[i] = m2_unc
        
    return dfcopy


def get_lc_uvot():
    '''
    ra = 264.1947913000000199
    dec = 50.54782575000000122
    
    rab = 264.2
    decb = 50.542
    '''
    # bkg.reg: fk5;circle(264.2,50.542,10")
    # src.reg: fk5;circle(2.641947913000000199e+02,+5.054782575000000122e+01,7")
    
    dirname1 = "./data_00011286001/00011286001/"
    dirname2 = "./data_00011286002/00011286002/"
    dirname3 = "./data_00011286003/00011286003/"
    
    df1 = getuvotdt(dirname1)
    df2 = getuvotdt(dirname2)
    df3 = getuvotdt(dirname3)
    
    date1 = "_".join(df1["date"].values[0].split(" "))
    date2 = "_".join(df2["date"].values[0].split(" "))
    date3 = "_".join(df3["date"].values[0].split(" "))
    
    df1['mjd'] = df1['jdobs'].values - 2400000.5
    df2['mjd'] = df2['jdobs'].values - 2400000.5
    df3['mjd'] = df3['jdobs'].values - 2400000.5
    df1 = df1.drop(columns = ["exptime", "date", "jdobs"])
    df2 = df2.drop(columns = ["exptime", "date", "jdobs"])
    df3 = df3.drop(columns = ["exptime", "date", "jdobs"])
    
    df1.to_csv(date1+".csv", index = False)
    df2.to_csv(date2+".csv", index = False)
    df3.to_csv(date3+".csv", index = False)
    
    dfgap1 = cal_transient_mag(df1, df3)
    dfgap2 = cal_transient_mag(df2, df3)
    
    
    lcuvot = pd.concat([dfgap1, dfgap2])
    lcuvot['instrument'] = "Swift"
    return lcuvot
    
    
    
lcuvot = get_lc_uvot()
lcuvot.to_csv("phot.csv", index = False)
    
    
    
    
    
    
    
    
    
    
    
    
    