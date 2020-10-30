# SN2019dge

This repository contains everything about SN2019dge, a helium-rich ultra-stripped envelope supernova. 
See publication [Yao et al. 2020, ApJ, 900, 46](https://iopscience.iop.org/article/10.3847/1538-4357/abaa3d)

- Figure 1: playground/discovery_image.ipynb
- Figure 2: playground/make_lightcurve.ipynb
- Figure 3: playground/compare_timescale.ipynb
- Figure 4: playground/blackbody_plot.ipynb   
- Figure 5: playground/compare_color.ipynb 
- Figure 6, 7, and 10: playground/spectral_part1.ipynb
- Figure 8 and 9: plaground/compare_spec.ipynb
- Figure 11: playground/subtract_spec.ipynb
- Figure 12, B3, B4: playground/Cooling_model-Piro2020
- Figure 13, 15: playground/rates/rate_shock_cooling

- Figure B2: playground/bbfit_19dge.ipynb 
- Figure B5: playground/radioactivity.ipynb

Essential Models:

- Shock Cooling model from [Piro et al. 2020](https://arxiv.org/pdf/2007.08543.pdf). Code at playground/helper/models_piro2020.py <br>
In the main function, replace `tt` (day), `wv` (angstrom), `lgL` (log 10 of erg/s), and `lgL_unc` (uncertainty of lgL) with your data. Change `tcut`. The modeling will be performed on data with `tt<tcut`. **Remember** to change the prior distribution in function `piro20_lnprior`, especially `texp` is the expected explosion epoch. Otherwise the MCMC will not converge.  

- Arnett model of radioactive decay. Code at playgorund/helper/arnett.py. <br>
Again, need to double-check prior definition `arnett_lnprior` before running the model.

Both models call the blackbody function defined at playground/helper/mcmcfit.py 
