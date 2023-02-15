import os, sys, importlib
sys.path.append('../../')
sys.path.append('../../configs/')
import numpy as np
import scipy.constants as sc
from structure_functions import *
from astropy.visualization import (SqrtStretch, LogStretch, ImageNormalize)
from scipy.interpolate import interp1d

# style setups
import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmasher as cmr
from matplotlib import cm, font_manager
plt.style.use(['default', '/home/sandrews/mpl_styles/nice_line.mplstyle'])
font_dirs = ['/home/sandrews/extra_fonts/']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
plt.rcParams['font.family'] = 'Helvetica'


cfg = 'modeld'
rtest = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
cols = ['C0', 'C1']


# 2-D grids (in meters)
r_, z_ = np.linspace(0.5, 600.5, 601), np.linspace(0., 500.5, 801)
r_ = np.logspace(0, 3, 601)
z_ = np.concatenate((np.array([0]), np.logspace(-3, 2.5, 800)))
rr, zz = np.meshgrid(r_ * sc.au, z_ * sc.au)

# Load configuration file as dictionary
inp = importlib.import_module('gen_sg_'+cfg)

# compute the pressure support velocity residuals
om_k = omega_kep(rr, zz, inp)
om_prs2 = eps_P(rr, zz, inp, nselfgrav=True)
om_tot = np.sqrt(om_k**2 + om_prs2)
dvp = (om_tot - om_k) * rr

Hp = H_pressure(np.array(rtest) * sc.au, inp) / sc.au

# loop through test radii
for i in range(len(rtest)):

    # index, value of radius grid closest to rtest
    idr = np.abs(r_ - rtest[i]).argmin()
    rx = r_[idr]

    # profile interpolator
    fint = interp1d(dvp[:,idr], z_, fill_value='extrapolate')

    # stationary altitude
    zcross = fint(0.)

    # print stationary aspect ratio in z/r and H units
    print('%.3f, %.1f' % (zcross/rx, zcross/Hp[i]))

    
sys.exit()


# plot the vertical profiles
fig, ax = plt.subplots()

for i in range(len(idr)):
    ax.plot(z_, dvp[:,idr[i]], '.'+cols[i])

ax.axhline(y=0, linestyle=':', color='k')
ax.set_xlim([0, 0.4*np.array(rtest).max()])
ax.set_ylim([-40, 100])
plt.show()
