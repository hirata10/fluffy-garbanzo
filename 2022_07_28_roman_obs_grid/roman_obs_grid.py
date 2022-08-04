import numpy as np
import os,sys
import pyimcom_interface
import time
import re
from astropy.io import fits
import fitsio as fio
import galsim
import yaml

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def _make_grid_image(nx_tiles, ny_tiles, stamp_xsize, stamp_ysize, config, psf, save_image=False):
    """Returns an image with a grid of postage stamps.
    
    Parameters
    ----------
    nx_tiles: Number of postage stamps in x-axis
    ny_tiles: Number of postage stamps in y-axis
    stamp_xsize: Size of a postage stamp in x-axis
    stamp_ysize: Size of a postage stamp in y-axis
    config: configuration info
    psf: Input PSF that you can convolve with input galaxy stamps. 

    Returns
    -------
    
    """

    random_seed = config['rngseed']
    pixel_scale = config['s_in']
    # Let's do point sources first. 
    # gal = galsim.Exponential(flux=1., half_light_radius=0.1)
    st_model = galsim.DeltaFunction(flux=1.)
    sky_level = 1.e6
    
    # Make an image where postage stamps fall in.
    gal_image = galsim.ImageF(stamp_xsize * nx_tiles-1 , stamp_ysize * ny_tiles-1,
                              scale=pixel_scale)
    psf_image = galsim.ImageF(stamp_xsize * nx_tiles-1 , stamp_ysize * ny_tiles-1,
                              scale=pixel_scale)
    
    rng = np.random.default_rng(random_seed)
    f = np.zeros((2,))
    f[0] = rng.random()
    f[1] = rng.random()
    offset = (f[0],f[1])
    k = 0
    for iy in range(ny_tiles):
        for ix in range(nx_tiles):
            ud = galsim.UniformDeviate(random_seed+k+1)
            gal_signal_to_noise = 200 * ud()

            b = galsim.BoundsI(ix*stamp_xsize+1 , (ix+1)*stamp_xsize-1,
                               iy*stamp_ysize+1 , (iy+1)*stamp_ysize-1)
            sub_gal_image = gal_image[b]
            sub_psf_image = psf_image[b]

            # OPERATION ON gal? - shear, rotation, noise etc.
            final_gal = galsim.Convolve([psf, st_model])
            final_gal.drawImage(sub_gal_image, offset=offset)

            # Any noise realization?
            # sky_level_pixel = sky_level * pixel_scale**2
            # noise = galsim.PoissonNoise(ud, sky_level=sky_level_pixel)
            # sub_gal_image.addNoiseSNR(noise, gal_signal_to_noise)

            k = k+1

    if save_image:
        image_fname = os.path.join(config['OUT'], 'star_image_grid.fits')
        gal_image.write(image_fname)

    return gal_image, offset

def _compute_T(config, InPSF, posoffset, outpsf='simple'): 
    """Returns a transformation matrix T.
    
    Parameters
    ----------
    config: configuration info
    InPSF: Input PSF array
    outpsf: Target output PSF

    Returns
    -------
    
    """
    n_in = config['n_in']
    n1 = config['n1']
    nps = config['nps']
    s_in = config['s_in']
    s_out = config['s_out']
    ld = config['lambda']/2.37e6*206265./s_in
    rng = np.random.default_rng(config['rngseed'])
    extbdy = config['extbdy']

    # Leakage
    uctarget = config['uctarget']
    flat_penalty = config['flat_penalty']


    # ROLL, SHEAR, MAGNIFY
    roll = np.fromstring(config['roll'], dtype=int, sep=' ')
    shear = np.fromstring(config['shear'], dtype=int, sep=' ')
    magnify = np.fromstring(config['magnify'], dtype=int, sep=' ')
    sigout = config['sigout']
    badfrac = config['badfrac']
    nx_in, ny_in = np.fromstring(config['insize'], dtype=int, sep=' ')
    nx_out, ny_out = np.fromstring(config['outsize'], dtype=int, sep=' ')

    mlist = []
    posoffset = []
    for k in range(n_in):
        mlist += [pyimcom_interface.rotmatrix(roll[k])@pyimcom_interface.shearmatrix(shear[2*k],shear[2*k+1])/(1.+magnify[k])]
        
        # positional offsets
        # f = np.zeros((2,))
        # f[0] = rng.random()
        # f[1] = rng.random()
        # Mf = s_in*mlist[k]@f
        # posoffset += [(Mf[0],Mf[1])]

    if outpsf == 'simple':
        OutPSF = [ pyimcom_interface.psf_simple_airy(n1,nps*ld,tophat_conv=0.,sigma=nps*sigout) ]

    # Compute PSF overlap (lookup tables)
    P = pyimcom_interface.PSF_Overlap(InPSF, OutPSF, .5, 2*n1-1, s_in, distort_matrices=mlist)

    # mask
    inmask=None
    if badfrac>0:
        inmask = np.where(rng.random(size=(n_in,ny_in,nx_in))>badfrac,True,False)
        print('number good', np.count_nonzero(inmask), 'of', n_in*ny_in*nx_in)
        print(np.shape(inmask))

    # Compute coadd matrix. 
    ims = pyimcom_interface.get_coadd_matrix(P, float(nps), [uctarget**2], posoffset, mlist, s_in, (ny_in,nx_in), s_out, (ny_out,nx_out), inmask, extbdy, smax=1./n_in, flat_penalty=flat_penalty)

    return ims['T']

def main(argv):

    # INPUT
    with open(sys.argv[1], "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    
    # READ-IN Roman PSFs.
    # roman_psf = fio.FITS('/hpc/group/cosmology/masaya/imcom_phase1/input_1x1arcmin/psf/dc2_psf_100659.fits.gz')[1].read()
    roman_psf = galsim.fits.read('/hpc/group/cosmology/masaya/imcom_phase1/input_1x1arcmin/psf/dc2_psf_100659.fits.gz', hdu=1)
    roman_psf_gsobj = galsim.InterpolatedImage(roman_psf, x_interpolant='lanczos15')
    InPSF = [roman_psf_gsobj for n in range(6)]

    Nimage = []
    pos_offset = []
    nx_tiles = 10
    ny_tiles = 10
    stamp_xsize = 64
    stamp_ysize = 64
    print('making an input image...')
    for n in range(config['n_in']):
        image, off = _make_grid_image(nx_tiles, ny_tiles, stamp_xsize, stamp_ysize, config, InPSF[n], save_image=False)
        Nimage.append(image)
        pos_offset.append(off)

    # sys.exit()
    T = _compute_T(config, InPSF, outpsf='simple')
    print(T)

    # Not sure about the step after. 
    # Just apply the matrix to the input image?


if __name__ == "__main__":
    main(sys.argv)