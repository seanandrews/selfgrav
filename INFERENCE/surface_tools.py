import os, sys, importlib, time
import numpy as np
import scipy.constants as sc
from disksurf import observation
import emcee
from multiprocessing import Pool

"""
   This is a simplified wrapper for disksurf extractions for this problem.
"""
def extract_surface(cubefile, outfile=None, FOV=6.0, nu0=230.538e9,
                    vra=[-1.7e3, 1.7e3], chans=[(0, 18), (24, 42)], 
                    x0=0, y0=0, vsys=0e3, inc=30, PA=130, zr_bounds=[0, 1], 
                    rms_T=4.0, smooth0=None, 
                    nbeams=[1, 1, 1, 1], iter_SNRs=[2.5, 1.2, 2.5, 3.75]):

    # load the cube into disksurf format
    cube = observation(cubefile, FOV=FOV, velocity_range=vra)

    # convert the target RMS (provided in K) into cube units
    targ_rms = 1e26 * cube.beamarea_str * 2 * nu0**2 * sc.k * rms_T / sc.c**2

    # convert target SNRs to proper clips
    SNR_clips = [j * targ_rms / cube.rms for j in iter_SNRs]

    # get an initial set of emission surface coordinates
    surf = cube.get_emission_surface(x0=x0, y0=y0, vlsr=vsys, inc=inc, PA=PA, 
                                     chans=chans, smooth=smooth0, 
                                     min_SNR=SNR_clips[0])

    # mask that surface in z/r
    surf.mask_surface(side='both', reflect=True,
                      min_zr=zr_bounds[0], max_zr=zr_bounds[1])

    # iterations on the initial surface (if requested)
    if len(SNR_clips) > 1:
        surf = cube.get_emission_surface_iterative(surf, N=len(SNR_clips)-1, 
                                                   nbeams=nbeams[1:], 
                                                   min_SNR=SNR_clips[1:])

    # final z/r and SNR masking
    surf.mask_surface(side='both', reflect=True, min_SNR=SNR_clips[-1],
                      min_zr=zr_bounds[0], max_zr=zr_bounds[1])

    # get the front surface coordinates
    rsurf = surf.r(side='front', masked=True)
    zsurf = surf.z(side='front', masked=True)
    dzsurf = np.sqrt(cube.bmaj * cube.bmin) / \
             surf.SNR(side='front', masked=True)

    # save the surface coordinates (if requested)
    if outfile is not None:
        if outfile[-4:] != '.npz': 
            outfile += '.npz'
        print('\nWriting surface coordinates to '+outfile)
        np.savez(outfile, rsurf=rsurf, zsurf=zsurf, dzsurf=dzsurf,
                 incl=inc, PA=PA, dx=x0, dy=y0, vsys=vsys)

    return rsurf, zsurf, dzsurf



""" 
    This is a simplified wrapper for MCMC fitting surfaces.
"""
# likelihood function
def lnL(pars, r, z, dz):

    # compute model surface
    z_m = pars[0] * r**pars[1] * np.exp(-(r / pars[2])**pars[3])

    # variance
    sigma2 = dz**2 + z_m**2 * np.exp(2 * pars[-1])

    # return log-likelihood
    return -0.5 * np.sum((z - z_m)**2 / sigma2 + np.log(sigma2))

# prior function
def lnT(pars):

    # uniform priors
    if (0.0 < pars[0] < 1.0) and (0.0 < pars[1] < 5.0) and \
       (0.0 < pars[2] < 5.0) and (0.0 < pars[3] < 15.0) and \
       (-10. < pars[4] < 1.0):
        return 0.0
    else:
        return -np.inf

# posterior function
def lnP(pars, r, z, dz):

    # check prior
    lp = lnT(pars)
    if not np.isfinite(lp):
        return -np.inf

    return lp + lnL(pars, r, z, dz)

def fit_surface(data, outfile=None,
                prior_types=None, prior_pars=None, 
                nwalk=64, nthread=6, nsteps=10500, nburn=None, ninit=200):

    if nthread > 1: os.environ["OMP_NUM_THREADS"] = "1"

    # MCMC setups
    ndim = 5

    # initialization
    p0 = np.array([0.2, 1, 1.5, 3, -2]) + 0.1 * np.random.randn(nwalk, ndim)

    # set a timer
    #t0 = time.time()

    # parse the data
    r, z, dz = data
    nan_mask = np.isfinite(r) & np.isfinite(z) & np.isfinite(dz)
    r, z, dz = r[nan_mask], z[nan_mask], dz[nan_mask]
    idx = np.argsort(r)
    r, z, dz = r[idx], z[idx], dz[idx]

    # initialization sampling
    with Pool(processes=nthread) as pool:
        isampler = emcee.EnsembleSampler(nwalk, ndim, lnP, pool=pool,
                                         args=(r, z, dz))
        isampler.run_mcmc(p0, ninit)

    # reset initialization to deal with any stray walkers
    isamples = isampler.get_chain()
    lop0 = np.quantile(isamples[-1,:,:], 0.25, axis=0)
    hip0 = np.quantile(isamples[-1,:,:], 0.75, axis=0)
    p00 = [np.random.uniform(lop0, hip0, ndim) for iw in range(nwalk)]
    #print('\nChains now properly initialized...\n')

    # run the MCMC
    with Pool(processes=nthread) as pool:
        sampler = emcee.EnsembleSampler(nwalk, ndim, lnP, pool=pool,
                                        args=(r, z, dz))
        sampler.run_mcmc(p00, nsteps, progress=True)

    # mark time
    #t1 = time.time()
    #print('\nMCMC took %.1f seconds\n' % (t1-t0))

    # compute the autocorrelation times
    tau = sampler.get_autocorr_time(quiet=True)
    #print('tau: ', tau)

    # flattened chain after burn-in
    if nburn is not None:
        chain = sampler.get_chain(discard=nburn, flat=True)
    else:
        chain = sampler.get_chain()

    # save the output posteriors
    if outfile is not None:
        if outfile[-4:] != '.npz': 
            outfile += '.npz'
        np.savez(outfile, chain=chain, tau=tau)

    return chain
