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

    npix = hp.nside2npix(2**res)
    m = np.arange(npix)
    ra_hpix, dec_hpix = hp.pix2ang(2**res, m, nest=True, lonlat=True)

    ra_cent, dec_cent = mywcs.all_pix2world(SCAFov[idsca[0]-1, 0], SCAFov[idsca[0]-1, 1], 0)
    side = (sca_nside * 0.11)/3600
    ra_min = ra_cent - side; ra_max = ra_cent + side
    dec_min = dec_cent - side; dec_max = dec_cent + side

    msk = ((ra_hpix > ra_min) & (ra_hpix < ra_max) & (dec_hpix > dec_min) & (dec_hpix < dec_max))
    ra_hpix = ra_hpix[msk]
    dec_hpix = dec_hpix[msk]

    # convert to SCA coordinates
    x_sca, y_sca = mywcs.all_world2pix(ra_hpix, dec_hpix, 0)
    num_obj = len(x_sca)

    pad = 10
    n_in_stamp = 128
    sca_image = galsim.ImageF(sca_nside+pad, sca_nside+pad, scale=0.11)
    for n in range(num_obj):
      
        psf = get_psf_pos(inpsf, idsca, obsdata, (x_sca[n], y_sca[n]), extraargs=None)
        interp_psf = galsim.InterpolatedImage(psf, x_interpolant='lanczos50')
        
        xy = galsim.PositionD(x_sca[n], y_sca[n])
        xyI = xy.round()
        draw_offset = xy - xyI
        b = galsim.BoundsI( xmin=xyI.x-int(n_in_stamp/2)+1,
                            ymin=xyI.y-int(n_in_stamp/2)+1,
                            xmax=xyI.x+int(n_in_stamp/2),
                            ymax=xyI.y+int(n_in_stamp/2))

        sub_image = sca_image[b]
        st_model = galsim.DeltaFunction(flux=1.)
        source = galsim.Convolve([interp_psf, st_model])
        source.drawImage(sub_image, offset=draw_offset, add_to_image=True)
    
    sca_image.write('/hpc/group/cosmology/masaya/imcom_phase1/fluffy-garbanzo/test.fits')
    return sca_image.array
