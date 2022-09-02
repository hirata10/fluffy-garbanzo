# This file will contain routines to make an input image of injected objects using GalSim.
#
#

import numpy as np
from astropy import wcs
import galsim
import healpy as hp

SCAFov = np.asarray([
  [-0.071,-0.037], [-0.071, 0.109], [-0.070, 0.240], [-0.206,-0.064], [-0.206, 0.083], [-0.206, 0.213], [-0.341,-0.129], [-0.341, 0.018], [-0.342, 0.147],
  [ 0.071,-0.037], [ 0.071, 0.109], [ 0.070, 0.240], [ 0.206,-0.064], [ 0.206, 0.083], [ 0.206, 0.213], [ 0.341,-0.129], [ 0.341, 0.018], [ 0.342, 0.147]
])

# Import the PSF function
#
from psf_utils import get_psf_pos
#
# To get the PSF, you want to call:
# get_psf_pos(inpsf, idsca, obsdata, pos, extraargs=None)
# which returns a numpy array of the PSF for:
#   inpsf --> input PSF format
#   idsca --> ordered pair of (observation ID, SCA)
#   obsdata --> observation data table
#   pos --> (x,y) on the SCA
#   extraargs --> for future compatibility
#
# The oversampling of the PSF is inpsf['oversamp']

# Example of a function used here that we can call from coadd_utils.get_all_data:
#
# Inputs:
#   res = HEALPix resolution (nside = 2**res)
#   mywcs = WCS object (astropy.wcs format)
#   inpsf, idsca, obsdata = PSF information to pass to get_psf_pos
#   sca_nside = side length of the SCA (4088 for Roman)
#   extraargs = for future compatibility
#
#  Output: [when complete]
#   nside x nside SCA with a grid of stars with unit flux
#
def galsim_star_grid(res, mywcs, inpsf, idsca, obsdata, sca_nside, extraargs=None):

    ra_cent, dec_cent = mywcs.all_pix2world((sca_nside-1)/2, (sca_nside-1)/2, 0)

    search_radius = (sca_nside * 0.11)/3600*(np.pi/180.)*np.sqrt(2)
    vec = hp.ang2vec(ra_cent, dec_cent, lonlat=True)
    qp = hp.query_disc(2**res, vec, search_radius, nest=True)
    ra_hpix, dec_hpix = hp.pix2ang(2**res, qp, nest=True, lonlat=True)

    # convert to SCA coordinates
    x_sca, y_sca = mywcs.all_world2pix(ra_hpix, dec_hpix, 0)
    msk_sca = ((x_sca >= 0) & (x_sca <= 4088) & (y_sca >= 0) & (y_sca <= 4088))
    x_sca = x_sca[msk_sca]; y_sca = y_sca[msk_sca]
    num_obj = len(x_sca)

    n_in_stamp = 128
    pad = n_in_stamp
    sca_image = galsim.ImageF(sca_nside+pad, sca_nside+pad, scale=0.11)
    for n in range(num_obj):
      
        psf = get_psf_pos(inpsf, idsca, obsdata, (x_sca[n], y_sca[n]), extraargs=None)
        psf_image = galsim.Image(psf, scale=0.11/inpsf['oversamp'])
        interp_psf = galsim.InterpolatedImage(psf_image, x_interpolant='lanczos50')
        
        xy = galsim.PositionD(x_sca[n], y_sca[n])
        xyI = xy.round()
        draw_offset = (xy - xyI) + galsim.PositionD(0.5, 0.5)
        b = galsim.BoundsI( xmin=xyI.x-n_in_stamp//2+pad//2+1,
                            ymin=xyI.y-n_in_stamp//2+pad//2+1,
                            xmax=xyI.x+n_in_stamp//2+pad//2,
                            ymax=xyI.y+n_in_stamp//2+pad//2)

        sub_image = sca_image[b]
        st_model = galsim.DeltaFunction(flux=1.)
        source = galsim.Convolve([interp_psf, st_model])
        source.drawImage(sub_image, offset=draw_offset, add_to_image=True, method='no_pixel')
    
    sca_image.write('/hpc/group/cosmology/masaya/imcom_phase1/fluffy-garbanzo/out/test_'+str(idsca[0])+'_'+str(idsca[1])+'.fits')
    return sca_image.array[pad//2:-pad//2,pad//2:-pad//2]
