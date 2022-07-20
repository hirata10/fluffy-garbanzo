import numpy
from astropy.io import fits
from os.path import exists

### This file contains assorted utilities and Roman WFI data needed for the coadd code. ###

degree = numpy.pi/180.
arcmin = degree/60.
arcsec = arcmin/60.

# filter list
RomanFilters  = ['W146', 'F184', 'H158', 'J129', 'Y106', 'Z087', 'R062', 'PRSM', 'DARK', 'GRSM', 'K213']
QFilterNative = [ 1.155,  1.456,  1.250,  1.021,  0.834,  0.689,  0.491,  1.009,  0.000,  1.159,  1.685]

# SCA parameters
pixscale_native = 0.11*arcsec
sca_nside = 4096 # includes reference pixels
sca_ctrpix = (sca_nside-1)/2
sca_sidelength = sca_nside * pixscale_native

### Routines to check observation table and get a list of input images

# SCA field of view centers
# SCAFov[i,j] = position of SCA #[i+1] (i=0..17) in j coordinate (j=0 for X, j=1 for Y)
# these are in 'WFI local' field angles, in degrees
# just for checking coverage since only 3 decimal places
SCAFov = numpy.asarray([
  [-0.071,-0.037], [-0.071, 0.109], [-0.070, 0.240], [-0.206,-0.064], [-0.206, 0.083], [-0.206, 0.213], [-0.341,-0.129], [-0.341, 0.018], [-0.342, 0.147],
  [ 0.071,-0.037], [ 0.071, 0.109], [ 0.070, 0.240], [ 0.206,-0.064], [ 0.206, 0.083], [ 0.206, 0.213], [ 0.341,-0.129], [ 0.341, 0.018], [ 0.342, 0.147]
])

# helper function for sorting
def obsOrder(pair):
  return pair[0]+0.01*pair[1]

# select observation/SCA pairs from a observation table
#
# Inputs:
#   obsdata (the observation table)
#   ra (in degrees)
#   dec (in degrees)
#   radius (in degrees)
#   filter (integer, 0..10)
#
# Outputs:
#   list of (observation, sca) that overlaps this target
#   [note SCA numbers are 1..18 for ease of comparison with the Project notation]
#
def get_obs_cover(obsdata, ra, dec, radius, filter):

  obslist = []
  n_obs_tot = len(obsdata.field(0))

  # rotate this observation to the (X,Y) of the local FoV for each observation
  # first rotate the RA direction
  x1 = numpy.cos(dec*degree)*numpy.cos((ra-obsdata['ra'])*degree)
  y1 = numpy.cos(dec*degree)*numpy.sin((ra-obsdata['ra'])*degree)
  z1 = numpy.sin(dec*degree)*numpy.ones((n_obs_tot,))
  # then rotate the Dec direction
  x2 =  numpy.sin(obsdata['dec']*degree)*x1 - numpy.cos(obsdata['dec']*degree)*z1
  y2 =  y1
  z2 =  numpy.cos(obsdata['dec']*degree)*x1 + numpy.sin(obsdata['dec']*degree)*z1
  # and finally the PA direction
  X = (-numpy.sin(obsdata['pa']*degree)*x2 - numpy.cos(obsdata['pa']*degree)*y2) / degree
  Y = (-numpy.cos(obsdata['pa']*degree)*x2 + numpy.sin(obsdata['pa']*degree)*y2) / degree
  #
  # throw away points in wrong hemisphere -- important since in orthographic projection, can have (X,Y)=0 for antipodal point
  X = numpy.where(z2>0, X, 1e49)

  for isca in range(18):
    obsgood = numpy.where(numpy.logical_and(numpy.sqrt((X-SCAFov[isca][0])**2 + (Y-SCAFov[isca][1])**2)<radius, obsdata['filter']==filter))
    for k in range(len(obsgood[0])): obslist.append((obsgood[0][k],isca+1))

  obslist.sort(key=obsOrder)
  return obslist

# Input file name (can add formats as needed)
#
# path = directory for the files
# idsca = tuple (obsid, sca) (sca in 1..18)
# obsdata = observation data table (information needed for some formats)
# format = string describing type of file name
# extraargs = for future compatibility
#
# returns None if unrecognized format
def get_sca_imagefile(path, idsca, obsdata, format, extraargs=None):

  out = path+'/'

  if format=='dc2_imsim':
    out += 'dc2_{:s}_{:d}_{:d}.fits'.format(RomanFilters[obsdata['filter'][idsca[0]]], idsca[0], idsca[1])
    return out

  return None

# make a window of specified radius around a specified point, within an SCA
# (x,y) and insize in pixels
# also takes in input and output wcs
#
# intended to be used in [ymin:ymax,xmin:xmax] windows
#
# return None if failed
def genWindow(x,y,insize,wcsin,wcsout):

  # window coordinates
  xmin = int(numpy.floor(x))-insize//2+1
  ymin = int(numpy.floor(y))-insize//2+1
  xmax = xmin + insize
  ymax = ymin + insize
  xc = (xmin+xmax-1)/2.
  yc = (ymin+ymax-1)/2.

  # location in output system
  p = wcsin.wcs_pix2world(numpy.array([[xc,yc]]) ,0)[0]
  wcspos = wcsout.wcs_world2pix(numpy.array([p]),0)[0]

  return {'xmin':xmin, 'xmax':xmax, 'ymin':ymin, 'ymax':ymax, 'xc':xc, 'yc':yc, 'outx':wcspos[0], 'outy':wcspos[1]}

### Routines for handling the PSF ###

# Utility to smear a PSF with a tophat and a Gaussian
#
#
def smooth_and_pad(inArray, tophatwidth=0., gaussiansigma=0.):
  npad = int(numpy.ceil(tophatwidth + 6*gaussiansigma + 1))
  npad += (4-npad)%4 # make a multiple of 4
  (ny,nx) = numpy.shape(inArray)
  nyy = ny+npad*2; nxx = nx+npad*2
  outArray = numpy.zeros((nyy,nxx))
  outArray[npad:-npad,npad:-npad]=inArray
  outArrayFT = numpy.fft.fft2(outArray)

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
def get_psf_pos(inpsf, idsca, obsdata, pos, extraargs=None):

  if inpsf['format']=='dc2_imsim':
    fname = inpsf['path'] + '/dc2_psf_{:d}.fits'.format(idsca[0])
    if not exists(fname): return None
    with fits.open(fname) as f: this_psf = smooth_and_pad(f[idsca[1]+1].data, tophatwidth=inpsf['oversamp'])

  return this_psf

# get PSF oversampling factor relative to native pixel scale
# In: PSF dictionary
# Out: integer oversampling factor
# Returns None if unknown type
def get_psf_oversamp(inpsf):
  if inpsf['format']=='dc2_imsim':
    return 8
  return None
