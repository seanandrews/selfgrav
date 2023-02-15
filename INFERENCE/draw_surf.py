import os, sys, time, importlib
sys.path.append('configs/')
import numpy as np
import scipy.constants as sc
from surface_tools import *

# user selections
mdl = 'taper2hi'
dtyp = 'noisy'
ndraws = 2000



#########################

# Load the configuration file
inp = importlib.import_module('gen_sg_'+mdl)

# surface extraction parameter assignments
vra = [-1.7e3, 1.7e3]
chans = [(0, 18), (24, 42)]
rms_T = 4.4
iter_SNRs = [1, 2, 3]
nbeams = [0, 2, 1]
zr_bounds = [0.05, 1.00]

# cube filename
cdir = inp.reduced_dir+inp.basename+'/images/'
cfile = cdir+inp.basename+'_'+dtyp+'.DATA.image.fits'

### Draw geometric parameters (and save them away in case you want them later)
np.random.seed(23)
x0 = np.random.normal(inp.dx, 0.02, ndraws)
y0 = np.random.normal(inp.dy, 0.02, ndraws)
inc = np.random.normal(inp.incl, 0.7, ndraws)
PA = np.random.normal(inp.PA, 0.7, ndraws)
vlsr = np.random.normal(inp.Vsys, 4, ndraws)
np.savez('HIER_DRAWS/geom/'+mdl+'_'+dtyp+'.geom_draws.npz',
         x0=x0, y0=y0, inc=inc, PA=PA, vlsr=vlsr)


### Loop over geometric parameter draws
for i in range(1500, 2000):

    # assign the indexed output file for the surface
    outsurf = 'HIER_DRAWS/surf/'+\
              mdl+'_'+dtyp+'.draw'+str(i).zfill(len(str(ndraws)))+'.npz'

    # extract the surface
    t0 = time.time()
    r, z, dz = extract_surface(cfile, outfile=outsurf, vra=vra, chans=chans, 
                               x0=x0[i], y0=y0[i], inc=inc[i], PA=PA[i],
                               vsys=vlsr[i], zr_bounds=zr_bounds, 
                               rms_T=rms_T, nbeams=nbeams, iter_SNRs=iter_SNRs)
    t1 = time.time()
    
    # progress monitoring
    print('Surface '+str(i+1).zfill(len(str(ndraws)))+' / '+str(ndraws)+\
          ' in %.1f seconds\n' % (t1 - t0))
