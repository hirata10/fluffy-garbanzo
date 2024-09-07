# This file will contain routines to make an input image of injected objects using GalSim.
#
#

import numpy
import scipy
import scipy.linalg
import re
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

    ra_cent, dec_cent = mywcs.all_pix2world((sca_nside-1)/2, (sca_nside-1)/2, 0)

    search_radius = (sca_nside * 0.11)/3600*(numpy.pi/180.)*numpy.sqrt(2)
    vec = healpy.ang2vec(ra_cent, dec_cent, lonlat=True)
    qp = healpy.query_disc(2**res, vec, search_radius, nest=True)
    ra_hpix, dec_hpix = healpy.pix2ang(2**res, qp, nest=True, lonlat=True)

    # convert to SCA coordinates
    x_sca, y_sca = mywcs.all_world2pix(ra_hpix, dec_hpix, 0)
    d = 16
    msk_sca = ((x_sca >= -d) & (x_sca <= 4087+d) & (y_sca >= -d) & (y_sca <= 4087+d))
    x_sca = x_sca[msk_sca]; y_sca = y_sca[msk_sca]
    num_obj = len(x_sca)

    n_in_stamp = 280
    pad = n_in_stamp+2*(d+1)
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

    return sca_image.array[pad//2:-pad//2,pad//2:-pad//2]

import numpy

# auxiliary function to skip n numbers in rng
def auxgen(rng, n):
  block = 262144
  for i in range(n//block): dump = rng.uniform(size=block)
  if n%block>0: dump = rng.uniform(size=n%block)

# generates the next lenpix numbers from the random number generator rng,
# and reports R[subpix[0]] .. R[subpix[-1]]
#
# designed to work even when lenpix is too large for memory
# assumes no repeated entries in subpix (but also doesn't have to be sorted)
# 
def subgen(rngX, lenpix, subpix):
  N = numpy.size(subpix)
  out_temp = numpy.zeros(N)
  k = numpy.argsort(subpix)
  subpix_sort = subpix[k]
  nskip = subpix_sort-1
  nskip[1:] -= subpix_sort[:-1]
  nskip[0] += 1
  for i in range(N):
    rngX.advance(nskip[i])
    out_temp[i] = numpy.random.Generator(rngX).uniform()
  rngX.advance(lenpix-subpix_sort[-1]-1)
  out = numpy.zeros(N)
  for i in range(N): out[k[i]] = out_temp[i]
  return out

# like subgen except makes P rows
#
def subgen_multirow(rngX, lenpix, subpix, P):
  out = numpy.zeros((P, numpy.size(subpix)))
  for j in range(P): out[j,:] = subgen(rngX,lenpix,subpix)
  return out

# generates object parameters for a list of galaxies at pixels subpix (array)
# seed = random number generator seed
# lenpix = number of pixels ( = 12 * 4**nside)
#
# galstring = string containing the type of galaxy ('type')
# possible types:
#   'exp1' -> exponential profile, random shear (up to 0.5), log distrib radius in .125 .. .5 arcsec
#
# returns a dictionary with a bunch of arrays of galaxy information
#
def genobj(lenpix, subpix, galstring, seed):

  nobj = numpy.size(subpix)
  rngX = numpy.random.PCG64(seed=seed)

  # now consider each type of object
  if galstring=='exp1':
    data = subgen_multirow(rngX, lenpix, subpix, 3)
    g1 = .5*numpy.sqrt(data[1,:])*numpy.cos(2*numpy.pi*data[2,:])
    g2 = .5*numpy.sqrt(data[1,:])*numpy.sin(2*numpy.pi*data[2,:])
    mydict = {'sersic': {'n': 1., 'r': .5/4**data[0,:], 't__r': 8.},
              'g': numpy.stack((g1,g2))}
  else:
    mydict = {}

  return mydict

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
#   to apply shear, include
#   'g': must have galtype['g'] as length 2 array giving g1 and g2.
#   (conserves area)
#
def galsim_extobj_grid(res, mywcs, inpsf, idsca, obsdata, sca_nside, extraargs=[]):

    # default parameters
    seed = 4096
    rot = None
    shear = None
    # unpack extraargs
    for arg in extraargs:
      m = re.search(r'^seed=(\d+)$', arg, re.IGNORECASE)
      if m: seed = int(m.group(1))
      m = re.search(r'^rot=(\S+)$', arg, re.IGNORECASE)
      if m: rot = float(m.group(1))
      m = re.search(r'^shear=([^ \:]+)\:([^ \:]+)$', arg, re.IGNORECASE)
      if m: shear = [float(m.group(1)), float(m.group(2))]
    print('rng seed =', seed, '  transform: rot=', rot, 'shear=', shear)

    refscale = 0.11 # reference pixel size in arcsec
    ra_cent, dec_cent = mywcs.all_pix2world((sca_nside-1)/2, (sca_nside-1)/2, 0)

    search_radius = (sca_nside * 0.11)/3600*(numpy.pi/180.)*numpy.sqrt(2)
    vec = healpy.ang2vec(ra_cent, dec_cent, lonlat=True)
    qp = healpy.query_disc(2**res, vec, search_radius, nest=True)
    ra_hpix, dec_hpix = healpy.pix2ang(2**res, qp, nest=True, lonlat=True)

    # convert to SCA coordinates
    x_sca, y_sca = mywcs.all_world2pix(ra_hpix, dec_hpix, 0)
    d = 128
    msk_sca = ((x_sca >= -d) & (x_sca <= 4087+d) & (y_sca >= -d) & (y_sca <= 4087+d))
    x_sca = x_sca[msk_sca]; y_sca = y_sca[msk_sca]
    ipix = qp[msk_sca] # pixel index of the objects within the SCA
    my_ra = ra_hpix[msk_sca]; my_dec = dec_hpix[msk_sca]
    num_obj = len(x_sca)

    # generate object parameters
    galstring = 'exp1'
    galtype = genobj(12*4**res, ipix, galstring, seed)
    print(galtype)

    n_in_stamp = 280
    pad = n_in_stamp+2*(d+1)
    sca_image = galsim.ImageF(sca_nside+pad, sca_nside+pad, scale=refscale)
    for n in range(num_obj):
        # PSF
        psf = get_psf_pos(inpsf, idsca, obsdata, (x_sca[n], y_sca[n]), extraargs=None)
        psf_image = galsim.Image(psf, scale=0.11/inpsf['oversamp'])
        interp_psf = galsim.InterpolatedImage(psf_image, x_interpolant='lanczos50')

        # Jacobian
        Jac = wcs.utils.local_partial_pixel_derivatives(mywcs, x_sca[n], y_sca[n])
        Jac[0,:] *= -numpy.cos(my_dec[n]*numpy.pi/180.)
        # convert to reference pixel size
        Jac /= refscale/ 3600.
        # now we have d(X,Y)|_{zweibein: X=E,Y=N} / d(X,Y)|_{pixel coords}

        xy = galsim.PositionD(x_sca[n], y_sca[n])
        xyI = xy.round()
        draw_offset = (xy - xyI) + galsim.PositionD(0.5, 0.5)
        b = galsim.BoundsI( xmin=xyI.x-n_in_stamp//2+pad//2+1,
                            ymin=xyI.y-n_in_stamp//2+pad//2+1,
                            xmax=xyI.x+n_in_stamp//2+pad//2,
                            ymax=xyI.y+n_in_stamp//2+pad//2)

        sub_image = sca_image[b]
        st_model_round = galsim.DeltaFunction(flux=1.)
        # now consider the possible non-point profiles
        #
        if 'sersic' in galtype:
          st_model_round = galsim.Sersic(galtype['sersic']['n'], half_light_radius=galtype['sersic']['r'][n], flux=1.0,
            trunc=galtype['sersic']['t__r']*galtype['sersic']['r'][n])
        #
        # now transform the round object if desired
        if 'g' in galtype:
          jshear = numpy.asarray([[ 1+galtype['g'][0,n], galtype['g'][1,n] ],[ galtype['g'][1,n], 1-galtype['g'][0,n] ]])\
            /numpy.sqrt(1.-galtype['g'][0,n]**2-galtype['g'][1,n]**2)
          st_model_undist = galsim.Transformation(st_model_round, jac=jshear, offset=(0.,0.), flux_ratio=1)
        else:
          st_model_undist = st_model_round
        # rotate, if desired
        if rot is not None:
          theta = rot * numpy.pi/180. # convert to radians
          jrot = numpy.asarray([[ numpy.cos(theta), -numpy.sin(theta)], [numpy.sin(theta), numpy.cos(theta)]])
          st_model_undist = galsim.Transformation(st_model_undist, jac=jrot, offset=(0.,0.), flux_ratio=1)
        # applied shear, if desired
        if shear is not None:
          jsh = scipy.linalg.expm(numpy.asarray([[shear[0], shear[1]], [shear[1], -shear[0]]]))
          st_model_undist = galsim.Transformation(st_model_undist, jac=jsh, offset=(0.,0.), flux_ratio=1)

        # and convert to image coordinates
        st_model = galsim.Transformation(st_model_undist, jac=numpy.linalg.inv(Jac), offset=(0.,0.), flux_ratio=numpy.abs(numpy.linalg.det(Jac)))

        source = galsim.Convolve([interp_psf, st_model])
        source.drawImage(sub_image, offset=draw_offset, add_to_image=True, method='no_pixel')

    return sca_image.array[pad//2:-pad//2,pad//2:-pad//2]
