#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 22:00:48 2019

@author: yuhanyao
"""

import astropy.constants as const

ggrav = const.G.cgs.value
sm = const.M_sun.cgs.value
sr = const.R_sun.cgs.value
h = const.h.cgs.value
k = const.k_B.cgs.value
c = const.c.cgs.value
sigma = const.sigma_sb.cgs.value
eV = 1.60218e-12 # erg
Mpc = 1e+6 * const.pc.cgs.value
pc = const.pc.cgs.value