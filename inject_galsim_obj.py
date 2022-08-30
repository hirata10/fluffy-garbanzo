# This file will contain routines to make an input image of injected objects using GalSim.
#
#

import numpy
from astropy import wcs
import galsim
import healpy

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

  return numpy.ones((sca_nside,sca_nside)) # right now a bunch of 1's -- placeholder.
