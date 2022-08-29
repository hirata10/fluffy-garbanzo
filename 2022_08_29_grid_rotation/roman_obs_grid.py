import numpy as np
import os,sys
import pyimcom_interface
import time
import re
from astropy.io import fits
import fitsio as fio
import galsim
import yaml
from galsim import roman

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')

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
    sigout = np.sqrt(sigout**2+float(config['extrasmooth'])**2)

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
        hdu = fits.PrimaryHDU(OutPSF[0]); hdu.writeto(os.path.join(config['OUT'], 'OutPSF.fits'), overwrite=True)

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

    # hdu = fits.PrimaryHDU(ims['T'].reshape((n_out,ny_out*nx_out, n_in*ny_in*nx_in,))); hdu.writeto(os.path.join(config['OUT'], 'T.fits'), overwrite=True)
    hdu = fits.PrimaryHDU(np.sqrt(ims['UC'])); hdu.writeto(os.path.join(config['OUT'], 'sqUC.fits'), overwrite=True)

    return ims['T'], OutPSF, ctrpos_offset, mlist, inmask

def get_roman_psfs(n_in, roll_angles, sca):
    # from astropy.time import Time
    ra = galsim.Angle.from_hms('16:01:41.01257')
    dec = galsim.Angle.from_dms('66:48:10.1312')
    bpass=roman.getBandpasses(AB_zeropoint=True)['J129']

    optical_psf = roman.getPSF(sca, 'J129', wavelength=bpass, n_waves=10, pupil_bin=8)
    epsf = galsim.Convolve(optical_psf, galsim.Pixel(0.11))
    psf_image = galsim.ImageF(256, 256, scale=0.11/8.)
    epsf.drawImage(psf_image, method='no_pixel')
    ImInPSF = [epsf.array for n in range(n_in)]

    InputPSF = []
    for n in range(n_in):
        # d=fio.FITS('observing_sequence_hlsonly_5yr.fits')[-1][100659]
        # date   = Time(d['date'],format='mjd').datetime
        ra = galsim.Angle.from_hms('16:01:41.01257')
        dec = galsim.Angle.from_dms('66:48:10.1312')
        wcs = roman.getWCS(world_pos  = galsim.CelestialCoord(ra=ra, dec=dec), 
                           PA          = roll_angles[n]*galsim.radians, 
                           # date        = date,
                           SCAs        = sca,
                           PA_is_FPA   = True)[sca]
        optical_rotated_psf = roman.getPSF(sca, 'J129', wavelength=bpass, n_waves=10, wcs=wcs, pupil_bin=8)
        effective_rotated_psf = galsim.Convolve(optical_rotated_psf, galsim.Pixel(0.11))
        InputPSF.append(effective_rotated_psf)

    return ImInPSF, InputPSF

def main(argv):

    # INPUT
    with open(sys.argv[1], "r") as f:
        config = yaml.load(f, Loader=yaml.Loader)
    save_image = sys.argv[1]
    
    # Create Roman PSFs.
    roll_angles = np.fromstring(config['roll'], dtype=int, sep=' ')
    ImInPSF, psfs = get_roman_psfs(config['n_in'], roll_angles, 1)

    # Same transformation matrix as testdither.py
    # posoffset[k] is the position of the centroid of the k-th input stamp in the coadd coordinates in absolute (arcsec) units.
    t0 = time.time()
    T, ImOutPSF, ctrpos, mlist, inmask = _compute_T(config, ImInPSF, outpsf='simple')
    print('time it took to compute T is ', time.time()-t0)
    outpsf = []
    for ipsf in range(config['n_out']):
        outpsf_image = galsim.Image(ImOutPSF[ipsf], scale=0.11/8.)
        outpsf.append(galsim.InterpolatedImage(outpsf_image, x_interpolant='lanczos50'))
    
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
        M = np.linalg.inv(s_in*mlist[ipsf])
        posx = positions[0,:]-ctrpos[ipsf][0]/s_in # how does the distortion matrices come in with this setting. I suppose rotation would not do anything?
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
    target_out_array = np.zeros((n_out,ny_out,nx_out))
    for ipsf in range(n_out):
        # get position of source in stamp coordinates
        xpos = srcpos[0] - 0.5 # out_srcpos_x / s_out + out_ctr
        ypos = srcpos[1] - 0.5 # out_srcpos_y / s_out + out_ctr

        target_image = galsim.ImageF(nx_out, ny_out, scale=s_out)
        for n in range(len(xpos)):
            xy = galsim.PositionD(np.float32(xpos[n]), np.float32(ypos[n]))
            xyI = xy.round()
            draw_offset = xy - xyI
            b = galsim.BoundsI( xmin=xyI.x-int(nx_out_stamp/2)+1,
                                ymin=xyI.y-int(ny_out_stamp/2)+1,
                                xmax=xyI.x+int(nx_out_stamp/2),
                                ymax=xyI.y+int(ny_out_stamp/2))
            sub_gal_image = target_image[b]
            st_model = galsim.DeltaFunction(flux=1.*(s_in/s_out)**2) 
            final_gal = galsim.Convolve([outpsf[ipsf], st_model])
            final_gal.drawImage(sub_gal_image, offset=draw_offset)

        target_out_array[ipsf,:,:] = target_image.array
        if save_image:
            image_fname = os.path.join(config['OUT'], 'star_image_target_'+str(ipsf)+'.fits')
            target_image.write(image_fname)

    err = out_array - target_out_array
    if save_image:
        image_fname = os.path.join(config['OUT'], 'error_target_output.fits')
        fio.write(image_fname, err)

    print('done')
    

if __name__ == "__main__":
    main(sys.argv)