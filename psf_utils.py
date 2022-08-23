import numpy
import scipy
import fitsio
from fitsio import FITS,FITSHDR
from astropy.io import fits
from os.path import exists

### Routines for handling the PSF ###

# Utility to smear a PSF with a tophat and a Gaussian
# tophat --> width, Gaussian --> sigma in units of the pixels given (not native pixel)
#
def smooth_and_pad(inArray, tophatwidth=0., gaussiansigma=0.):
  npad = int(numpy.ceil(tophatwidth + 6*gaussiansigma + 1))
  npad += (4-npad)%4 # make a multiple of 4
  (ny,nx) = numpy.shape(inArray)
  nyy = ny+npad*2; nxx = nx+npad*2
  outArray = numpy.zeros((nyy,nxx))
  outArray[npad:-npad,npad:-npad]=inArray
  outArrayFT = numpy.fft.fft2(outArray)

  # convolution
  uy = numpy.linspace(0,nyy-1,nyy)/nyy; uy = numpy.where(uy>.5,uy-1,uy)
  ux = numpy.linspace(0,nxx-1,nxx)/nxx; ux = numpy.where(ux>.5,ux-1,ux)
  outArrayFT *= numpy.sinc(ux[None,:]*tophatwidth)*numpy.sinc(uy[:,None]*tophatwidth)*numpy.exp(-2.*numpy.pi**2*gaussiansigma**2*(ux[None,:]**2+uy[:,None]**2))

  outArray = numpy.real(numpy.fft.ifft2(outArrayFT))
  return outArray

# Get PSF information
#
# Inputs:
#   inpsf = PSF dictionary
#   idsca = tuple (obsid, sca) (sca in 1..18)
#   obsdata = observation data table (information needed for some formats)
#   pos = (x,y) tuple or list containing the position where the PSF is to be interpolated
#   extraargs = for future compatibility
#
# Returns the PSF at that position
# 'effective' PSF is returned, some formats may include extra smoothing
#
# Returns None if can't find the file.
#
def get_psf_pos(inpsf, idsca, obsdata, pos, extraargs=None):

  if inpsf['format']=='dc2_imsim':
    fname = inpsf['path'] + '/dc2_psf_{:d}.fits'.format(idsca[0])
    if not exists(fname): return None

    # fitsio version
    fileh = fitsio.FITS(fname)
    this_psf = smooth_and_pad(fileh[idsca[1]+1][:,:], tophatwidth=inpsf['oversamp'])
    fileh.close()

    # old astropy.io fits input
    #f = fits.open(fname)
    #this_psf = smooth_and_pad(f[idsca[1]+1].data, tophatwidth=inpsf['oversamp'])
    #f.close()

  return this_psf

# get PSF oversampling factor relative to native pixel scale
# In: PSF dictionary
# Out: integer oversampling factor
# Returns None if unknown type
def get_psf_oversamp(inpsf):
  if inpsf['format']=='dc2_imsim':
    return 8
  return None
