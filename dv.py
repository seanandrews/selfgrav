import os, sys, importlib
sys.path.append('../../')
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
fig, axs = plt.subplots(nrows=4, ncols=3, figsize=(7.5, 6.5))
left, right, bottom, top = 0.06, 0.905, 0.06, 0.995
hspace, wspace = 0.28, 0.05

# axes configurations
xlims = [0, 300]
ylims = [0, 105.]

# colormaps
v_ra = [-200, 200]
cmap_v = cm.get_cmap('RdBu_r', 35)


# configuration files
cfgs = ['taper1hi', 'taper2hi', 'taper3hi', 'modeld']
lbls = ['$\mathsf{model \,\, a}$', '$\mathsf{model \,\, b}$',
        '$\mathsf{model \,\, c}$', '$\mathsf{model \,\, d}$']


# 2-D grids (in meters)
r_, z_ = np.linspace(0.5, 600.5, 501), np.linspace(0., 500.5, 501)
rr, zz = np.meshgrid(r_ * sc.au, z_ * sc.au)
ext = (r_.min(), r_.max(), z_.min(), z_.max())


for i in range(len(cfgs)):

    # Load configuration file as dictionary
    inp = importlib.import_module('gen_sg_'+cfgs[i])

    # compute gas density cutoff
    ngas = numberdensity(rr, zz, inp, selfgrav=True)
    above = (ngas <= 100.)

    # compute the pressure support velocity residuals
    om_k = omega_kep(rr, zz, inp)
    om_prs2 = eps_P(rr, zz, inp, nselfgrav=True)
    om_tot = np.sqrt(om_k**2 + om_prs2)
    dvp = (om_tot - om_k) * rr
    dvp[above] = np.nan

    # plot the pressure support velocity residuals
    imvp = axs[i, 0].imshow(dvp, origin='lower', cmap=cmap_v, extent=ext,
                            aspect='auto', vmin=v_ra[0], vmax=v_ra[1])

    # compute the self-gravity velocity residuals
    om_sg2 = eps_g(rr, zz, inp)
    om_tot = np.sqrt(om_k**2 + om_sg2)
    dvs = (om_tot - om_k) * rr
    dvs[above] = np.nan

    # plot the pressure support velocity residuals
    imvs = axs[i, 1].imshow(dvs, origin='lower', cmap=cmap_v, extent=ext,
                            aspect='auto', vmin=v_ra[0], vmax=v_ra[1])

    # compute the total velocity residuals
    om_tot = np.sqrt(om_k**2 + om_prs2 + om_sg2)
    dv_ = (om_tot - om_k) * rr
    dv_[above] = np.nan

    # plot the total velocity residuals
    imv_ = axs[i, 2].imshow(dv_, origin='lower', cmap=cmap_v, extent=ext,
                            aspect='auto', vmin=v_ra[0], vmax=v_ra[1])

    # show the CO emission surface
    Xco = abundance(rr, zz, inp, selfgrav=True)
    Hp_mid = H_pressure(r_ * sc.au, inp)
    for j in range(3):
        axs[i, j].plot(r_, 0.1 * r_, ':', color='gray', lw=1)
        axs[i, j].plot(r_, 0.2 * r_, ':', color='gray', lw=1)
        axs[i, j].plot(r_, 0.3 * r_, ':', color='gray', lw=1)
        axs[i, j].plot(r_, 0.4 * r_, ':', color='gray', lw=1)
        axs[i, j].plot(r_, Hp_mid / sc.au, '--', color='gray', lw=1.3)
        axs[i, j].contour(r_, z_, Xco, levels=[inp.xmol], colors='k',
                          linewidths=1.)

    # limits
    [axs[i, j].set_xlim(xlims) for j in range(3)]
    [axs[i, j].set_ylim(ylims) for j in range(3)]

    # labeling
    if i == 3:
        axs[i,0].set_xlabel('$r$  (au)')
    axs[i,0].set_ylabel('$z$  (au)', labelpad=1)
    axs[i,1].set_xticklabels([])
    axs[i,1].set_yticklabels([])
    axs[i,2].set_xticklabels([])
    axs[i,2].set_yticklabels([])
    axs[i,0].text(0.03, 0.85, lbls[i]+'\n$\\delta v_P$', va='center', 
                  ha='left', transform=axs[i,0].transAxes, fontsize=8)
    axs[i,1].text(0.03, 0.85, lbls[i]+'\n$\\delta v_g$', va='center',
                  ha='left', transform=axs[i,1].transAxes, fontsize=8)
    axs[i,2].text(0.03, 0.85, lbls[i]+'\n$\\delta v$', va='center',
                  ha='left', transform=axs[i,2].transAxes, fontsize=8)

# plot adjustments
fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top,
                    hspace=hspace, wspace=wspace)

# colorbars
for i in range(len(cfgs)):
    pos_ = axs[i,2].get_position()
    Xcb = 0.9
    cbax = fig.add_axes([right+0.01, pos_.y0+0.5*(1-Xcb)*(pos_.y1-pos_.y0),
                         0.012, Xcb*(pos_.y1-pos_.y0)])
    cb = Colorbar(ax=cbax, mappable=imv_, orientation='vertical',
                  ticklocation='right')
    cb.set_label('residual $v_\\phi$  (m/s)', fontsize=8, rotation=270,
                 labelpad=12)


#pos_vs = axs[3, 2].get_position()
#cbax_vs = fig.add_axes([right+0.015, pos_vs.y0, 0.015, pos_vs.y1-pos_vs.y0])
#cb_vs = Colorbar(ax=cbax_vs, mappable=imvs, orientation='vertical',
#                 ticklocation='right')
#cb_vs.set_label('$\delta v_{\phi, g}$  (m/s)')

#pos_v_ = axs[4, 2].get_position()
#cbax_v_ = fig.add_axes([right+0.015, pos_v_.y0, 0.015, pos_v_.y1-pos_v_.y0])
#cb_v_ = Colorbar(ax=cbax_v_, mappable=imv_, orientation='vertical',
#                 ticklocation='right')
#cb_v_.set_label('$\delta v_{\phi}$  (m/s)')



# save figure to output
fig.savefig('figs/dv.pdf')
fig.clf()
