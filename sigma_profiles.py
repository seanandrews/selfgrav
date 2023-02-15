import os, sys, importlib
sys.path.append('../../')
sys.path.append('../../configs/')
import numpy as np
import scipy.constants as sc
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
from structure_functions import *
from astropy.visualization import (SqrtStretch, ImageNormalize)

# style setups
import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm
import matplotlib as mpl
import cmasher as cmr
from matplotlib import font_manager
plt.style.use(['default', '/home/sandrews/mpl_styles/nice_line.mplstyle'])
font_dirs = ['/home/sandrews/extra_fonts/']
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font_file in font_files:
    font_manager.fontManager.addfont(font_file)
plt.rcParams['font.family'] = 'Helvetica'

# Plot configuration
fig, axs = plt.subplots(nrows=3, figsize=(3.5, 5.2))
left, right, bottom, top, hspace = 0.15, 0.85, 0.075, 0.99, 0.12

# axes
xlims = [0, 350]
ylims = [0, 3]

# radius grid (au)
r_ = np.logspace(-1, 3, 2048)

# models
cfgs = ['taper1hi', 'taper2hi', 'taper3hi', 'modeld']
lbls = ['$\\sf{model \,\, a}$', '$\\sf{model \,\ b}$', '$\\sf{model \,\, c}$',
        '$\\sf{model \,\, d}$']
cols = ['C0', 'C3', 'C1', 'purple']


# Loop through models
for i in range(len(cfgs)):

    # Load configuration file as dictionary
    inp = importlib.import_module('gen_sg_'+cfgs[i])

    # Compute the surface densities
    sigma = sigmagas(r_ * sc.au, inp)

    # Plot the surface densities
    axs[0].plot(r_, sigma, '-', color=cols[i], label=lbls[i])

    # Compute the encircled mass profile
    m_in_r = cumtrapz(2 * np.pi * sigma * r_ * sc.au, r_ * sc.au, initial=0)
    m_in_r /= 1.98847e30
    print(cfgs[i], m_in_r[-1])

    # Plot the encircled mass profile
    axs[1].plot(r_, m_in_r, '-', color=cols[i], label=lbls[i])

    # Toomre Q value
    omega_k = omega_kep(r_ * sc.au, np.zeros_like(r_), inp)
    c_s = c_sound(r_ * sc.au, np.zeros_like(r_), inp)
    Qtoomre = c_s * omega_k / (np.pi * sc.G * sigma)
    axs[2].plot(r_, Qtoomre, '-', color=cols[i], label=lbls[i])


axs[0].set_xlim([2, 500])
axs[0].set_ylim([0.2, 5000])
axs[0].set_yscale('log')
axs[0].set_xscale('log')
axs[0].set_ylabel('${\\Sigma}$  (g cm$^{-2}$)', labelpad=2)
axs[0].set_yticks([1, 10, 100, 1000])
axs[0].set_yticklabels(['1', '10', '100', '1000'])
axs[0].set_xticks([10, 100])
axs[0].set_xticklabels([])	#['10', '100'])

axs[0].legend(prop={'size': 8})

axs[1].set_xlim([2, 500])
axs[1].set_ylim([0.0, 0.11])
axs[1].set_xscale('log')
axs[1].set_ylabel('$M_{\\rm d} \, (< r)$  ($M_\odot$)')
axs[1].set_xticks([10, 100])
axs[1].set_xticklabels([])	#['10', '100'])
axs[1].set_yticks([0, 0.02, 0.04, 0.06, 0.08, 0.10])
axs[1].set_yticklabels(['0.00', '0.02', '0.04', '0.06', '0.08', '0.10'])

axs[2].set_xlim([2, 500])
axs[2].set_ylim([0.8, 30])
axs[2].set_xscale('log')
axs[2].set_yscale('log')
axs[2].set_xlabel('$r$  (au)')
axs[2].set_ylabel('Toomre  $Q$', labelpad=11)
axs[2].set_xticks([10, 100])
axs[2].set_xticklabels(['10', '100'])
axs[2].set_yticks([1, 10])
axs[2].set_yticklabels(['1', '10'])


fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top,
                    hspace=hspace)
fig.savefig('figs/sigma_profiles.pdf')
fig.clf()


