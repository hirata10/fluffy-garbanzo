import numpy
import scipy
from scipy.ndimage import convolve
import re
import fitsio
from fitsio import FITS,FITSHDR
from astropy.io import fits
from astropy import wcs
from os.path import exists

# may need to be able to read a PSF
import psf_utils

# interface for injecting stars using pyimcom_croutines functions
import grid_inject
# interface to injecting GalSim objects
import inject_galsim_obj
# and more general noise fields
import inject_complex_noise

### This file contains assorted utilities and Roman WFI data needed for the coadd code. ###

degree = numpy.pi/180.
arcmin = degree/60.
arcsec = arcmin/60.

# filter list
RomanFilters  = ['W146', 'F184', 'H158', 'J129', 'Y106', 'Z087', 'R062', 'PRSM', 'DARK', 'GRSM', 'K213']
QFilterNative = [ 1.155,  1.456,  1.250,  1.021,  0.834,  0.689,  0.491,  1.009,  0.000,  1.159,  1.685]

# linear obscuration of the telescope
obsc = 0.31

# SCA parameters
pixscale_native = 0.11*arcsec
sca_nside = 4088 # excludes reference pixels
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
# extraargs = dictionary of extra arguments
#
# returns None if unrecognized format
def get_sca_imagefile(path, idsca, obsdata, format, extraargs=None):

  # right now this is the only type defined
  if format=='dc2_imsim':
    out = path+'/simple/dc2_{:s}_{:d}_{:d}.fits'.format(RomanFilters[obsdata['filter'][idsca[0]]], idsca[0], idsca[1])
    if extraargs is not None:
      if 'type' in extraargs.keys():
        if extraargs['type']=='truth':
          out = path+'/truth/dc2_{:s}_{:d}_{:d}.fits'.format(RomanFilters[obsdata['filter'][idsca[0]]], idsca[0], idsca[1])
        if extraargs['type']=='labnoise':
          out = path+'/labnoise/slope_{:d}_{:d}.fits'.format(idsca[0], idsca[1])
        if extraargs['type']=='skyerr':
          out = path+'/simple/dc2_{:s}_{:d}_{:d}.fits'.format(RomanFilters[obsdata['filter'][idsca[0]]], idsca[0], idsca[1])

    return out

  return None

# makes a 4D array of the image data
#  axes of the output = [input type (e.g., 0=sci or sim), exposure index, y, x]
#
# Inputs:
#   n_inframe = number of input frames
#   obslist = which observations to use (list of tupes (obsid, sca) (sca in 1..18))
#   obsdata = observation data table (information needed for some formats)
#   path = directory for the files
#   format = string describing type of file name
#   inwcs = input WCS list (same length as obslist)
#   inpsf = input PSF information (dictionary; to be passed to psfutils routines if we draw objects)
#   extrainput = make multiple maps (list of types, first should be None, rest strings)
#   extraargs = dictionary; may have the following (or more for future compatibility)
#
#      (*) tempfile => for memmap'ed arrays
#
def get_all_data(n_inframe, obslist, obsdata, path, format, inwcs, inpsf, extrainput, extraargs=None):

  # start by allocating the memory ...
  if extraargs is not None:
    if 'tempfile' in extraargs.keys():
      hypercube = numpy.memmap(extraargs['tempfile'], mode='w+', dtype=numpy.float32, shape=(n_inframe, len(obslist), sca_nside, sca_nside))
    else:
      hypercube = numpy.zeros((n_inframe, len(obslist), sca_nside, sca_nside), dtype=numpy.float32)
  else:
    hypercube = numpy.zeros((n_inframe, len(obslist), sca_nside, sca_nside), dtype=numpy.float32)

  # now fill in each slice in the observation
  # (missing files are blank)
  for j in range(len(obslist)):
    filename = get_sca_imagefile(path, obslist[j], obsdata, format)
    if exists(filename):
      if format=='dc2_imsim':
        with fits.open(filename) as f: hypercube[0,j,:,:] = f['SCI'].data - float(f['SCI'].header['SKY_MEAN'])
    else:
      continue
    #
    # now for the extra inputs
    if n_inframe>1:
      for i in range(1,n_inframe):
        # truth image (no noise)
        if extrainput[i].casefold() == 'truth'.casefold():
          filename = get_sca_imagefile(path, obslist[j], obsdata, format, extraargs = {'type':'truth'})
          if exists(filename):
            with fits.open(filename) as f: hypercube[i,j,:,:] = f['SCI'].data
        # white noise frames (generated from RNG, not file)
        m = re.search(r'^whitenoise(\d+)$', extrainput[i], re.IGNORECASE)
        if m:
          q = int(m.group(1))
          seed = 1000000*(18*q+obslist[j][1]) + obslist[j][0]
          print('noise rng: frame_q={:d}, seed={:d}'.format(q,seed))
          rng = numpy.random.default_rng(seed)
          hypercube[i,j,:,:] = rng.normal(loc=0., scale=1., size=(sca_nside,sca_nside))
          del rng
        # 1/f noise frames (generated from RNG, not file)
        m = re.search(r'^1fnoise(\d+)$', extrainput[i], re.IGNORECASE)
        if m:
          q = int(m.group(1))
          seed = 1000000*(18*q+obslist[j][1]) + obslist[j][0]
          print('noise rng: frame_q={:d}, seed={:d}'.format(q,seed), '--> 1/f')
          hypercube[i,j,:,:] = inject_complex_noise.noise_1f_frame(seed)
        # lab noise frames (if applicable)
        if extrainput[i].casefold() == 'labnoise'.casefold():
          filename = get_sca_imagefile(path, obslist[j], obsdata, format, extraargs = {'type':'labnoise'})
          print('lab noise: searching for ' + filename)
          if exists(filename):
            with fits.open(filename) as f: hypercube[i,j,:,:] = f[0].data
          else:
            print('Warning: labnoise file not found, skipping ...')
        # skyerr
        if extrainput[i].casefold() == 'skyerr'.casefold():
          filename = get_sca_imagefile(path, obslist[j], obsdata, format, extraargs = {'type':'skyerr'})
          if exists(filename):
            with fits.open(filename) as f: hypercube[i,j,:,:] = f['ERR'].data - float(f['SCI'].header['SKY_MEAN'])
        # C routine star grid
        m = re.search(r'^cstar(\d+)$', extrainput[i], re.IGNORECASE)
        if m:
          res = int(m.group(1))
          print('making grid using C routines: ', res, obslist[j])
          hypercube[i,j,:,:] = grid_inject.make_image_from_grid(res, inpsf, obslist[j], obsdata, inwcs[j], sca_nside)
        # galsim star grid
        m = re.search(r'^gsstar(\d+)$', extrainput[i], re.IGNORECASE)
        if m:
          res = int(m.group(1))
          print('making grid using GalSim: ', res, obslist[j])
          hypercube[i,j,:,:] = inject_galsim_obj.galsim_star_grid(res, inwcs[j], inpsf, obslist[j], obsdata, sca_nside)

  return hypercube

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
  p = wcsin.all_pix2world(numpy.array([[xc,yc]]) ,0)[0]
  wcspos = wcsout.all_world2pix(numpy.array([p]),0)[0]

  # get active pixels
  activepix = numpy.ones((insize,insize), dtype=bool)
  if xmin<0: activepix[:,:-xmin] = False
  if xmax>sca_nside: activepix[:,sca_nside-xmax:] = False
  if ymin<0: activepix[:-ymin,:] = False
  if ymax>sca_nside: activepix[sca_nside-ymax:,:] = False

  return {'xmin':xmin, 'xmax':xmax, 'ymin':ymin, 'ymax':ymax, 'xc':xc, 'yc':yc, 'outx':wcspos[0], 'outy':wcspos[1], 'active':activepix}

# make a cutout of an nside x nside array corresponding to a given window
#
# Input:
#   window: the window structure from genWindow
#   readarray: the array to read from
#   fillvalue: what to pad material off the array
#
def win_cutout(window, readarray, fillvalue):
  p = numpy.amax([0, -window['xmin'], -window['ymin'], window['xmax']-sca_nside, window['ymax']-sca_nside]) # amount to pad
  return(numpy.pad(readarray,p,mode='constant', constant_values = fillvalue)[p+window['ymin']:p+window['ymax'],p+window['xmin']:p+window['xmax']])

# makes a snapshot of the frame given by the [ymin:ymax,xmin:xmax] of the window, possibly overlapping the edge
def snap_instamp(image,window):
  d = max(window['xmax']-window['xmin'], window['ymax']-window['ymin'])
  return numpy.pad(image, ((0,0), (d,d), (d,d)))[:,d+window['ymin']:d+window['ymax'],d+window['xmin']:d+window['xmax']]

# makes a psuedorandom mask that randomly removes groups of pixels (intended for CR simulation)
#
# Input:
#   idsca = tuple (obsid, sca)
#   pcut = probability that a pixel is hit
#   hitinfo = dictionary (use None for default)
#
def randmask(idsca, pcut, hitinfo=None):

  seed = 100000000 + idsca[0]
  rng = numpy.random.default_rng(seed)
  pad = 10
  g = rng.uniform(size=(18,2*pad+sca_nside,2*pad+sca_nside))[idsca[1]-1,:,:]
  crhits = numpy.where(g<pcut,1.,0.) # hit mask

  # different ways of making a mask
  if hitinfo is None:
    return(numpy.where(convolve(crhits,numpy.ones((3,3)),mode='constant')[pad:-pad,pad:-pad]<.5,True,False))

### Fade Kernel methods ###

# array with a trapezoid filter of width n+2*fade_kernel on each side
def trapezoid(n,fade_kernel,use_trunc_sinc=True):
  ar = numpy.ones((n+2*fade_kernel,n+2*fade_kernel))
  if n<=2*fade_kernel:
    print('Fatal error in coadd_utils.trapezoid: insufficient patch size, n=', n, 'fade_kernel=', fade_kernel)
    exit()
  for i in range(2*fade_kernel):
    s = (i+1)/(2*fade_kernel+1)
    if use_trunc_sinc: s = s-numpy.sin(2*numpy.pi*s)/(2*numpy.pi)
    ar[   i,   :] *= s
    ar[-1-i,   :] *= s
    ar[   :,   i] *= s
    ar[   :,-1-i] *= s
  return ar
