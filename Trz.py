import os, sys, importlib
sys.path.append('../../configs/')
import numpy as np
import scipy.constants as sc
from structure_functions import *
from astropy.visualization import (SqrtStretch, LogStretch, ImageNormalize)

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



# Plot configuration 
fig, ax = plt.subplots(figsize=(3.5, 2.0))
left, right, bottom, top = 0.13, 0.83, 0.19, 0.98

# axes configurations
xlims = [0, 300]
ylims = [0, 105.]

# colormaps
T_ra = [5, 160]
cmap_T = cm.get_cmap('cmr.sepia_r', 21)


# configuration files
cfgs = 'gen_sg_modeld'


# 2-D grids (in meters)
r_, z_ = np.linspace(0.5, 300.5, 301), np.linspace(0.5, 300.5, 301)
rr, zz = np.meshgrid(r_ * sc.au, z_ * sc.au)
z__ = np.logspace(-1.5, 2, 501)
rr_, zz_ = np.meshgrid(r_ * sc.au, z__ * sc.au)
ext = (r_.min(), r_.max(), z_.min(), z_.max())



# Load configuration file as dictionary
inp = importlib.import_module(cfgs)

# compute the gas temperatures
Tgas = temperature(rr, zz, inp)

# plot the gas temperatures
norm = ImageNormalize(vmin=T_ra[0], vmax=T_ra[1], stretch=SqrtStretch())
imT = ax.imshow(Tgas, origin='lower', cmap=cmap_T, extent=ext, 
                aspect='auto', norm=norm)

ax.plot(r_, 0.1 * r_, ':', color='gray', lw=1)
ax.plot(r_, 0.2 * r_, ':', color='gray', lw=1)
ax.plot(r_, 0.3 * r_, ':', color='gray', lw=1)
ax.plot(r_, 0.4 * r_, ':', color='gray', lw=1)
Hp_mid = H_pressure(r_ * sc.au, inp)
ax.plot(r_, Hp_mid / sc.au, '--', color='gray', lw=1.3)

# limits
ax.set_xlim(xlims)
ax.set_ylim(ylims)
ax.set_xlabel('$r$  (au)') 
ax.set_ylabel('$z$  (au)', labelpad=2)

# plot adjustments
fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)

# colorbars
Xcb = 0.9
cbax_T = fig.add_axes([right+0.02, bottom+0.5*(1-Xcb)*(top-bottom), 
                       0.025, Xcb*(top-bottom)])
cb_T = Colorbar(ax=cbax_T, mappable=imT, orientation='vertical', 
                ticklocation='right', ticks=[5, 10, 20, 40, 80, 160])
cb_T.set_label('$T$  (K)', fontsize=8, rotation=270, labelpad=9)

# save figure to output
fig.savefig('figs/Trz.pdf')
fig.clf()
