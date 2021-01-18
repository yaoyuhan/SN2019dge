#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 16:15:25 2020

@author: yuhanyao
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
fs= 14
matplotlib.rcParams['font.size']=fs


f = open("avro_packets.json")
lines = f.readlines()
f.close()

newind = []
nline = len(lines)
for i in range(nline):
    line = lines[i]
    if line == '{\n':
        newind.append(i)
        
newind = np.array(newind)
nobs = len(newind)
jds = np.zeros(nobs)
fids = np.zeros(nobs)
mags = np.zeros(nobs)
emags = np.zeros(nobs)
ras = np.zeros(nobs)
decs = np.zeros(nobs)


for i in range(nobs):
    startind = newind[i]
    if i==(nobs-1):
        endind = -1
    else:
        endind = newind[i+1]
    sublines= lines[startind:endind]
    for line in sublines:
        if line[:9] == '    "jd":':
            jds[i] = float(line[10:-2])
        if line[:9] == '    "ra":':
            ras[i] = float(line[10:-2])
        if line[:10] == '    "fid":':
            fids[i] = int(line[11:-2])
        if line[:10] == '    "dec":':
            decs[i] = float(line[11:-2])
        if line[:12] == '    "magpsf"':
            mags[i] = float(line[14:-2])
        if line[:14] == '    "sigmapsf"':
            emags[i] = float(line[16:-2])
            
ix = ras!=0
plt.figure()
plt.plot(ras[ix], decs[ix], '.')

ra_kow = np.median(ras[ix])
dec_kow = np.median(decs[ix])

np.savetxt("kowalski_coo.txt", np.array([ra_kow, dec_kow]))