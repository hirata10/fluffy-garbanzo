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

def _make_grid_image(nx_tiles, ny_tiles, stamp_xsize, stamp_ysize, config, psf, offset, save_image=False):
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

def _compute_T(config, InPSF, outpsf='simple'): 
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
    uctarget = float(config['uctarget'])
    flat_penalty = float(config['flat_penalty'])
    cd = .3 # charge diffusion, rms per axis in pixels
    sigout       = np.sqrt(1./12.+cd**2)

    # ROLL, SHEAR, MAGNIFY
    roll = np.fromstring(config['roll'], dtype=int, sep=' ')
    shear = np.fromstring(config['shear'], dtype=int, sep=' ')
    magnify = np.fromstring(config['magnify'], dtype=int, sep=' ')
    badfrac = config['badfrac']
    nx_in, ny_in = np.fromstring(config['insize'], dtype=int, sep=' ')
    nx_out, ny_out = np.fromstring(config['outsize'], dtype=int, sep=' ')

    mlist = []
    ctrpos_offset = [] # The center of the postage stamp in image coordinates of the output image (which is centered at (0,0)), meaning that this is the dither offset relative to the center of the image.
    for k in range(n_in):
        mlist += [pyimcom_interface.rotmatrix(roll[k])@pyimcom_interface.shearmatrix(shear[2*k],shear[2*k+1])/(1.+magnify[k])]

        f = np.zeros((2,))
        f[0] = rng.random()
        f[1] = rng.random()
        Mf = s_in*mlist[k]@f
        ctrpos_offset += [(Mf[0],Mf[1])]

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
    ims = pyimcom_interface.get_coadd_matrix(P, float(nps), [uctarget**2], ctrpos_offset, mlist, s_in, (ny_in,nx_in), s_out, (ny_out,nx_out), inmask, extbdy, smax=1./n_in, flat_penalty=flat_penalty)

    return ims['T'], OutPSF, ctrpos_offset, mlist

def _coadd_image(T, psf_in_list, psf_out_list, image_in_list, config):

    (n_out,ny_out,nx_out,n_in,ny_in,nx_in) = np.shape(T)

    # center pixel of input stamps (non-galsim coordinates)
    xctr = (nx_in-1)/2.; yctr = (ny_in-1)/2.

    # What needs to be done to set up input image array? 
    # Step 1. Make an input image array, instead of input stamp array. (image_in_list[n].array.shape)
    # Step 2. 
    # Do I have to use the oversampled PSF? 
    # Do I have to apply the same distortion matrix to the input image? 

    """
    # make input stamp array
    in_array = np.zeros((n_in,ny_in,nx_in))

    p = 5 # pad length

    # make the input stamps
    """
    # --- end construction of the input images ---

    # What needs to be done to set up output+output_target image array? 
    # Step 1. 
    # Step 2. 



    return (in_array,out_array,out_array-target_out_array)

def main(argv):

    # INPUT
    with open(sys.argv[1], "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    
    # READ-IN Roman PSFs.
    # roman_psf = fio.FITS('/hpc/group/cosmology/masaya/imcom_phase1/input_1x1arcmin/psf/dc2_psf_100659.fits.gz')[1].read()
    roman_psf = galsim.fits.read('/hpc/group/cosmology/masaya/imcom_phase1/input_1x1arcmin/psf/dc2_psf_100659.fits.gz', hdu=1)
    roman_psf_gsobj = galsim.InterpolatedImage(roman_psf, x_interpolant='lanczos3')
    interpolated_psf = [roman_psf_gsobj for n in range(config['n_in'])]
    ImInPSF = [roman_psf_gsobj.image.array for n in range(config['n_in'])]

    # Same transformation matrix as testdither.py
    # posoffset[k] is the position of the centroid of the k-th input stamp in the coadd coordinates in absolute (arcsec) units.
    T, ImOutPSF, ctrpos, mlist = _compute_T(config, ImInPSF, outpsf='simple')
    np.save('/hpc/group/cosmology/masaya/imcom_phase1/fluffy-garbanzo/2022_07_28_roman_obs_grid/ctrpos.npy', np.array(ctrpos))
    np.save('/hpc/group/cosmology/masaya/imcom_phase1/fluffy-garbanzo/2022_07_28_roman_obs_grid/mlist.npy', np.array(mlist))
    print(T)
    sys.exit()

    # Similar steps to test_psf_inject() but with a grid. 
    # (a) make a list of locations for the grid of point sources in global coordinates (coadd coordinates)
    nx_out, ny_out = np.fromstring(config['outsize'], dtype=int, sep=' ')
    steps = 10 # number of stamps in a row
    xy_min = -steps/2. * nx_out
    xy_max = steps/2. * nx_out
    out_stamp_ctr = (nx_out-1)/2.
    grid = np.linspace(xy_min+out_stamp_ctr, xy_max-out_stamp_ctr, steps)
    x_mesh, y_mesh = np.meshgrid(grid, grid)

    # (b) map these to (x,y) in the input images
    nx_in, ny_in = np.fromstring(config['insize'], dtype=int, sep=' ')
    n_in = config['n_in']
    n1 = config['n1']
    nps = config['nps']
    s_in = config['s_in']
    s_out = config['s_out']
    xctr = (nx_in-1)/2.; yctr = (ny_in-1)/2.
    # make input stamp array
    in_array = np.zeros((n_in,ny_in,nx_in))
    p = 5 # pad length

    for ipsf in range(n_in):
        M_inv = np.linalg.inv(s_in*mlist[ipsf])
        # locations of point sources in i-th input frame
        px = M_inv[0,0]*(x_mesh - ctrpos[ipsf][0]) + M_inv[0,1]*(y_mesh - ctrpos[ipsf][1]) + xctr
        py = M_inv[1,0]*(x_mesh - ctrpos[ipsf][0]) + M_inv[1,1]*(y_mesh - ctrpos[ipsf][1]) + yctr

        # (c) draw stars with unit flux at those locations
        im_size = nx_in * steps
        gal_image = galsim.ImageF(im_size, im_size, scale=s_in)
        print('making an ', ipsf,'-th input image...')
        for n in range(config['n_in']):
            for x_loc, y_loc in zip(px, py):
                b = galsim.BoundsI(ix*stamp_xsize+1 , (ix+1)*stamp_xsize-1,
                                iy*stamp_ysize+1 , (iy+1)*stamp_ysize-1)
                sub_gal_image = gal_image[b]
                st_model = galsim.DeltaFunction(flux=1.)
                final_gal = galsim.Convolve([interpolated_psf[ipsf], st_model])
                final_gal.drawImage(sub_gal_image)

    # (d) feed those images to the image co-addition
    print('coadding images...')
    in_array, out_array, error_array = _coadd_image(T, ImInPSF, ImOutPSF, Nimage, config)

if __name__ == "__main__":
    main(sys.argv)