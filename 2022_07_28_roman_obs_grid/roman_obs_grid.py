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
    n_out = config['n_out']
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

    hdu = fits.PrimaryHDU(ims['T'].reshape((n_out,ny_out*nx_out, n_in*ny_in*nx_in,))); hdu.writeto(os.path.join(config['OUT'], 'T.fits'), overwrite=True)
    hdu = fits.PrimaryHDU(np.sqrt(ims['UC'])); hdu.writeto(os.path.join(config['OUT'], 'sqUC.fits'), overwrite=True)

    return ims['T'], OutPSF, ctrpos_offset, mlist, inmask


def main(argv):

    # INPUT
    with open(sys.argv[1], "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    save_image = sys.argv[1]
    
    # READ-IN Roman PSFs.
    # roman_psf = fio.FITS('/hpc/group/cosmology/masaya/imcom_phase1/input_1x1arcmin/psf/dc2_psf_100659.fits.gz')[1].read()
    roman_psf = galsim.fits.read('/hpc/group/cosmology/masaya/imcom_phase1/input_1x1arcmin/psf/dc2_psf_100659.fits.gz', hdu=1)
    roman_psf_gsobj = galsim.InterpolatedImage(roman_psf, x_interpolant='lanczos50')
    # re-draw roman PSF with top-hat convolution
    psf_image = galsim.ImageF(512, 512, scale=0.11)
    roman_psf_gsobj.drawImage(psf_image, method='no_pixel')
    interpolated_psf = galsim.InterpolatedImage(psf_image, x_interpolant='lanczos50')
    psfs = [interpolated_psf for n in range(config['n_in'])]
    ImInPSF = [psf_image.array for n in range(config['n_in'])]

    # Same transformation matrix as testdither.py
    # posoffset[k] is the position of the centroid of the k-th input stamp in the coadd coordinates in absolute (arcsec) units.
    T, ImOutPSF, ctrpos, mlist, inmask = _compute_T(config, ImInPSF, outpsf='simple')

    # input and output image config
    nx_in, ny_in = np.fromstring(config['insize'], dtype=int, sep=' ')
    nx_out, ny_out = np.fromstring(config['outsize'], dtype=int, sep=' ')
    n_in = config['n_in']
    n_out = config['n_out']
    n1 = config['n1']
    nps = config['nps']
    s_in = config['s_in']
    s_out = config['s_out']
    steps = config['grid_step']

    # Similar steps to test_psf_inject() but with a grid. 
    # (a) make a list of locations of point sources in input frame
    nx_in_stamp = nx_in // steps; ny_in_stamp = ny_in // steps
    nx_out_stamp = nx_out // steps; ny_out_stamp = ny_out // steps

    stamp_ctr = (nx_in_stamp-1)/2. + 1 # galsim ver.
    in_ctr = (nx_in-1)/2. + 1 # galsim ver.
    grid = [stamp_ctr + i*nx_in_stamp for i in range(steps)]
    x_mesh, y_mesh = np.meshgrid(grid, grid)
    positions = np.vstack([x_mesh.ravel(), y_mesh.ravel()])
    # dx = 48; dy = 48
    # dx = nx_in/4.; dy = ny_in/4. # how much overlap of the stamp bounds do we want? 

    # (b) map these to (x,y) to world coordinates
    input_wcs = galsim.PixelScale(s_in)
    world_pos = input_wcs.toWorld(positions[0,:], positions[1,:])

    # (c) map these back to the locations in output frame
    out_ctr = (nx_out-1)/2. + 1 # galsim ver
    output_wcs = galsim.PixelScale(s_out)
    srcpos = output_wcs.toImage(world_pos[0], world_pos[1])

    # input image array
    in_array = np.zeros((n_in, ny_in, nx_in))
    image_list = []
    # (d) make images using the locations in the input frame + distortion matrix + position offset
    for ipsf in range(n_in):
        posx = positions[0,:]-ctrpos[ipsf][0]/s_in
        posy = positions[1,:]-ctrpos[ipsf][1]/s_in

        gal_image = galsim.ImageF(nx_in, ny_in, scale=s_in)
        print('making an ', ipsf,'-th input image...')
        for n in range(len(posx)):
            xy = galsim.PositionD(posx[n], posy[n])
            xyI = xy.round()
            draw_offset = xy - xyI
            b = galsim.BoundsI( xmin=xyI.x-int(nx_in_stamp/2)+1,
                                ymin=xyI.y-int(ny_in_stamp/2)+1,
                                xmax=xyI.x+int(nx_in_stamp/2),
                                ymax=xyI.y+int(ny_in_stamp/2))

            sub_gal_image = gal_image[b]
            st_model = galsim.DeltaFunction(flux=1.)
            final_gal = galsim.Convolve([psfs[ipsf], st_model])
            final_gal.drawImage(sub_gal_image, offset=draw_offset, add_to_image=True)
            image_list.append(gal_image)
        in_array[ipsf,:,:] = gal_image.array
        if save_image:
            image_fname = os.path.join(config['OUT'], 'star_image_grid_'+str(ipsf)+'.fits')
            gal_image.write(image_fname)

    # (d) feed those images to the image co-addition
    print('coadding images...')
    out_array = (T.reshape(n_out*ny_out*nx_out,n_in*ny_in*nx_in)@in_array.flatten()).reshape(n_out,ny_out,nx_out)
    hdu = fits.PrimaryHDU(out_array); hdu.writeto(os.path.join(config['OUT'], 'grid_src_out.fits'), overwrite=True)
    # ----- end of coaddition -----

    # ----- start of making target array -----
    print(ImOutPSF)
    target_out_array = np.zeros((n_out,ny_out,nx_out))
    for ipsf in range(n_out):
        # get position of source in stamp coordinates
        xpos = srcpos[0] # out_srcpos_x / s_out + out_ctr
        ypos = srcpos[1] # out_srcpos_y / s_out + out_ctr

        target_image = galsim.ImageF(nx_out, ny_out, scale=s_out)
        for n in range(len(xpos)):
            xy = galsim.PositionD(xpos[n], ypos[n])
            xyI = xy.round()
            draw_offset = xy - xyI
            b = galsim.BoundsI( xmin=xyI.x-int(nx_out_stamp/2),
                                ymin=xyI.y-int(ny_out_stamp/2),
                                xmax=xyI.x+int(nx_out_stamp/2)-1,
                                ymax=xyI.y+int(ny_out_stamp/2)-1)
            sub_gal_image = target_image[b]
            st_model = galsim.DeltaFunction(flux=1.)
            outpsf = galsim.InterpolatedImage(ImOutPSF[ipsf], x_interpolant='lanczos50')
            final_gal = galsim.Convolve([outpsf, st_model])
            final_gal.drawImage(sub_gal_image, offset=draw_offset)

        target_out_array[ipsf,:,:] = target_image.array
        if save_image:
            image_fname = os.path.join(config['OUT'], 'star_image_target_'+str(ipsf)+'.fits')
            target_image.write(image_fname)

    err = out_array - target_out_array
    if save_image:
        image_fname = os.path.join(config['OUT'], 'error_target_output.fits')
        err.write(image_fname)

    print('done')
    

if __name__ == "__main__":
    main(sys.argv)