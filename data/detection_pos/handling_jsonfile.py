#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 22:25:54 2020

@author: yuhanyao
"""
import numpy as np
import pandas as pd
from astropy.table import Table
from astropy.coordinates import SkyCoord


df = pd.read_json("kowalski_query_result.json")
result = df['ZTF_alerts']['(264_194758, 50_547826)']
ndet = len(result)
print ("In total: %d detections from ZTF alerts" %ndet)

ras = np.zeros(ndet)
decs = np.zeros(ndet)
jds = np.zeros(ndet)
fids = np.zeros(ndet)
magpsfs = np.zeros(ndet)
diffimgs = []
for i in range(ndet):
    data = result[i]
    ras[i] = data["candidate"]["ra"]
    decs[i] = data["candidate"]["dec"]
    jds[i] = data["candidate"]["jd"]
    fids[i] = data["candidate"]["fid"]
    diffimgs.append(data["candidate"]['pdiffimfilename'])
    magpsfs[i] = data["candidate"]["magpsf"]
diffimgs = np.array(diffimgs)

ix = jds>2458484.50000 # 2019 Jan 1
jds = jds[ix]
decs = decs[ix]
ras = ras[ix]
fids = fids[ix]
diffimgs = diffimgs[ix]
magpsfs = magpsfs[ix]

ndet = len(ras)
print ("In 2019: %d detections from ZTF alerts" %ndet)


tb = Table([ras, decs, fids, jds, diffimgs, magpsfs], names = ['ra', 'dec', 'fid', 'jd', 'diffimg', 'mag_alert'])
tb.write('alert_coo.csv', overwrite=True)
        
ra = np.median(ras)
dec = np.median(decs)
    
c2 = SkyCoord(ra = ra, dec = dec, unit = 'degree')
np.savetxt('coo_kowalski.reg', [ra, dec])
print (c2.ra.hms)
print (c2.dec.dms)

"""   
plt.plot(ras, decs, ".")
plt.scatter(ras, decs, c = jds-min(jds))
plt.xlim(min(ras)-0.00001, max(ras)+0.00001)
plt.ylim(min(decs)-0.00001, max(decs)+0.00001)
plt.colorbar()
""" 

tb = tb.to_pandas()
tb = tb.sort_values(by ='jd')

from astropy.time import Time
jd_det = tb["jd"].values[0]
t_det = Time(jd_det, format = "jd")
print (t_det.datetime64)



