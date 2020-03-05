#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 11:57:41 2020

@author: yuhanyao
"""

import numpy as np
import numpy.polynomial.polynomial as mypoly
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

"""
x = np.array([-2.65915989e+00, -1.73230197e+00, -1.02300989e+00, -7.32791540e-01,
       -3.47596201e-02,  1.01165182e+00,  1.17585430e+00,  2.17281896e+00,
        3.14315089e+00,  4.13414276e+00,  5.12092431e+00,  6.02065994e+00,
        6.68363850e+00,  7.70439636e+00,  8.08038774e+00,  8.66268481e+00,
        8.99637717e+00,  9.70420053e+00,  1.00181142e+01,  1.06873592e+01,
        1.16454519e+01,  1.26689513e+01,  1.29177519e+01,  1.36606286e+01,
        1.39087438e+01,  1.46105943e+01,  1.48448056e+01,  1.59449721e+01,
        1.69112895e+01,  2.17696074e+01,  2.33879369e+01,  2.36664056e+01,
        2.46776657e+01,  2.64343484e+01,  2.73916577e+01,  2.86447665e+01,
        3.25190444e+01,  3.34260257e+01,  4.32694605e+01,  4.41440321e+01,
        5.10641339e+01,  5.34870263e+01,  5.78790757e+01])
y = np.array([-14.00392524, -15.70392524, -16.05642807, -16.13392524,
       -16.26642807, -16.21642807, -16.20392524, -16.14392524,
       -15.97392524, -15.87392524, -15.79392524, -15.90392524,
       -15.73642807, -15.58642807, -15.58392524, -15.63642807,
       -15.43392524, -15.48642807, -15.45392524, -15.39642807,
       -15.29642807, -15.14642807, -15.21392524, -15.12642807,
       -15.04392524, -15.00642807, -14.93392524, -14.92392524,
       -14.81392524, -14.16392524, -13.67642807, -14.20392524,
       -14.25392524, -13.53642807, -13.37642807, -13.66392524,
       -13.95392524, -13.87392524, -13.31392524, -13.68469888,
       -12.94392524, -12.62392524, -12.49392524])
ey = np.array([0.14, 0.05, 0.02, 0.02, 0.01, 0.02, 0.02, 0.03, 0.03, 0.05, 0.03,
       0.2 , 0.04, 0.09, 0.05, 0.12, 0.13, 0.06, 0.06, 0.08, 0.07, 0.06,
       0.05, 0.05, 0.06, 0.06, 0.09, 0.06, 0.1 , 0.08, 0.11, 0.09, 0.13,
       0.21, 0.19, 0.11, 0.09, 0.17, 0.21, 0.23, 0.19, 0.16, 0.17])
"""


def poly_fit_timescales(x, y, ey, name=None):
    """
    assert a1 > 0
    """
    thre = 1.5
    if name == "AT2019dge":
        a1 = 4
        a2 = 0
        order1=4
        order2=4
    elif name == "iPTF14gqr":
        a1 = 2
        a2 = -3
        order1=2
        order2=3
        thre = 2
    elif name == "SN2005ek":
        a2 = -2
        order2=2
        thre = 2.
    elif name == "SN2016hnk":
        a1 = 2
        a2 = -2
        order2=3
        order1 = 2
        ey[0] = 0.01
    elif name =="SN2010X":
        a1 = 2
        a2 = -3
        order1=1
        order2=2
        thre = 2
    elif name == "SN2019bkc":
        a1 = 2
        a2 = -3
        order2=2
        thre = 3.5
    elif name == "OGLE13-079":
        a1 = 2
        a2 = -2
        order2=2
        thre = 2.5
    elif name == "SN2018kzr":
        a2 = -2
        order2=2
        thre = 3
    elif name == "PTF09dav":
        a1 = 3
        a2 = -3
        order2=2
        thre = 2
        ix = x<25
        x = x[ix]
        y = y[ix]
        ey = ey[ix]
    elif name == "SN2002bj":
        a1 = 2
        a2 = -3
        order1=1
        order2=2
        thre = 2.5
    elif name == "PTF10iuv":
        a1 = 2
        a2 = -3
        order1=2
        order2=2
        thre = 1.8
    else:
        a1 = 2
        a2 = -3
        order1=1
        order2=2
    
    b = min(y)
    y = y - b
    NSAMPLES = 100
    
    # cut data points not useful
    #plt.errorbar(x, y, ey, fmt=".k")
    ix = np.any([x<=0, (x>0)&(y<(min(y)+thre))],axis=0)
    x = x[ix]
    y = y[ix]
    ey = ey[ix]
    
    if name not in ["SN2005ek", "SN2010X", "SN2019bkc", "OGLE13-079", 
                    "SN2018kzr", "PTF09dav", "SN2002bj"]:
        # adjust peak epoch
        ix1 = x<=a1
        x1 = x[ix1]
        y1 = y[ix1]
        ey1 = ey[ix1]
        coefs1 = mypoly.polyfit(x1, y1, order1, w=1/ey1**2)
        xnew1 = np.linspace(x1[0]-0.5, x1[-1],1000)
        ynew1 = mypoly.polyval(xnew1, coefs1)
        id_peak = np.argsort(ynew1)[0]
        tpeak = xnew1[id_peak]
        x = x-tpeak
    
    plt.figure(figsize=(6,4))
    plt.errorbar(x, y, ey, fmt=".k")
    
    if name not in ["SN2005ek", "SN2010X", "SN2019bkc", "OGLE13-079", 
                    "SN2018kzr", "PTF09dav", "SN2002bj"]:
        # peak timescale
        ix1 = x<=a1
        x1 = x[ix1]
        y1 = y[ix1]
        ey1 = ey[ix1]
        p1, C_p1 = np.polyfit(x1, y1, order1, w=1/ey1**2, cov=True)
        xnew1 = np.linspace(x1[0], x1[-1],1000)
        ynew1 = np.polyval(p1, xnew1)
        ymax = min(ynew1)
        N1 = len(p1)
        L1 = np.linalg.cholesky(C_p1)
        # you should find np.dot(L1, L1.T) == C_p1
        zprep = np.zeros((NSAMPLES, N1))
        t1s = np.zeros(NSAMPLES)
        limflag1 = 0
        for i in range(NSAMPLES):
            try:
                p1_ = p1 + np.dot(L1, zprep[i])
                ynew1 = np.polyval(p1_, xnew1)
                plt.plot(xnew1, ynew1, color = "r", linewidth=0.5)
                ydiff1 = abs(ynew1 - (ymax+1))
                id_rise = np.argsort(ydiff1)[0]
                if limflag1==0:
                    if ydiff1[id_rise]>0.01:
                        riselim = True
                    else:
                        riselim = False
                    limflag1=1
                t1s[i] = xnew1[id_rise] * (-1)     
            except Exception:
                print (i)
        tau_rise = np.median(t1s)
        tau_rise_unc = np.std(t1s)
        plt.plot(tau_rise*(-1), 1, 'ro')
    elif name in ["SN2019bkc", "OGLE13-079", "PTF09dav"]:
        ymax = 0
        ix1 = x<=a1
        x1 = x[ix1]
        y1 = y[ix1]
        func1 = interp1d(x1, y1)
        xnew1 = np.linspace(x1[0]+0.01, x1[-1]-0.01,1000)
        ynew1 = func1(xnew1)
        plt.plot(xnew1, ynew1, color = "r", linewidth=0.5)
        ydiff1 = abs(ynew1 - (ymax+1))
        id_rise = np.argsort(ydiff1)[0]
        tau_rise = xnew1[id_rise] * (-1)   
        tau_rise_unc = -99        
        riselim = False
        plt.plot(tau_rise*(-1), 1, 'ro')
    else:
        tau_rise = -99
        tau_rise_unc = -99
        riselim = True
        ymax = 0
    
    # decline timescale
    ix2 = x>=a2
    x2 = x[ix2]
    y2 = y[ix2]
    ey2 = ey[ix2]
    p2, C_p2 = np.polyfit(x2, y2, order2, w=1/ey2**2, cov=True)
    xnew2 = np.linspace(x2[0], x2[-1],1000)
    ynew2 = np.polyval(p2, xnew2)
    N2 = len(p2)
    L2 = np.linalg.cholesky(C_p2)
    # you should find np.dot(L1, L1.T) == C_p1
    zprep = np.zeros((NSAMPLES, N2))
    t2s = np.zeros(NSAMPLES)
    limflag2 = 0
    for i in range(NSAMPLES):
        try:
            p2_ = p2 + np.dot(L2, zprep[i])
            ynew2 = np.polyval(p2_, xnew2)
            plt.plot(xnew2, ynew2, color = "b", linewidth=0.5)
            ydiff2 = abs(ynew2 - (ymax+1))
            id_decay = np.argsort(ydiff2)[0]
            if limflag2==0:
                if ydiff2[id_decay]>0.01:
                    decaylim = True
                else:
                    decaylim = False
                limflag2 = 1  
            t2s[i] = xnew2[id_decay]     
        except Exception:
            print (i)
    
    tau_decay = np.median(t2s)
    tau_decay_unc = np.std(t2s)
    
    plt.plot(tau_decay, 1, 'bo')
    
    result = {"name": name,
              "Mpeak": ymax + b,
              "tau_rise": tau_rise,
              "tau_rise_unc": tau_rise_unc,
              "tau_rise_lim": riselim,
              "tau_decay": tau_decay,
              "tau_decay_unc": tau_decay_unc,
              "tau_decay_lim": decaylim}
    
    return result
    
    # maximum light magnitude
    
    