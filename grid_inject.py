import numpy
import healpy
import pyimcom_interface, pyimcom_croutines
from astropy import wcs
from astropy.io import fits

import psf_utils

# Get Healpix pixels at resolution res that are within the given radius of (ra, dec)
# ra, dec, radius all in radians.
#
# note: nside = 2**res
#
# Output is a dictionary of:
#   'npix' => number of pixels used
#   'ipix' => numpy array of pixel indices
#   'rapix' => ra of pixels (numpy array)
#   'decpix' => dec of pixel  (numpy array)
#
def make_sph_grid(res, ra, dec, radius):

  # get healpix nside
  nside = 2**res

  # get bounding range of pixels
  radext = radius + 3/nside # extended radius, overlap by 2 rings so there is no clipping later
  dmin = max(dec-radext,-numpy.pi/2.)
  dmax = min(dec+radext, numpy.pi/2.)
  pmin = healpy.pixelfunc.ang2pix(nside,numpy.pi/2.-dmax,ra,nest=False,lonlat=False)
  pmax = healpy.pixelfunc.ang2pix(nside,numpy.pi/2.-dmin,ra,nest=False,lonlat=False)

  # and now the pixel values and their positions
  pvec = numpy.asarray(range(pmin,pmax+1)).astype(numpy.int64)
  theta,phi = healpy.pixelfunc.pix2ang(nside,pvec,nest=False, lonlat=False)
  thetac = numpy.pi/2.-theta

  mu = numpy.sin(thetac)*numpy.sin(dec) + numpy.cos(thetac)*numpy.cos(dec)*numpy.cos(ra-phi)
  good = numpy.where(mu>=numpy.cos(radius))

  ipix = pvec[good]
  rapix = phi[good]
  decpix = thetac[good]
  npix = numpy.size(ipix)

  return {'res':res, 'nside':nside, 'npix':npix, 'ipix':ipix, 'rapix':rapix, 'decpix':decpix}

# Makes a grid of positions to inject simulated sources into an SCA image
#
# Inputs:
#   res = HEALPix resolution
#   wcs = WCS structure
#   scapar = dictionary of sca_nside and pixel scale in arcsec
#
# Outputs: ipix,xsca,ysca
#   ipix = array of HEALPix indices
#   xsca = array of x positions on the SCA
#   ysca = array of y positions on the SCA
def generate_star_grid(res, myWCS, scapar={'nside':4088, 'pix_arcsec':0.11}):

  # SCA side length in radians
  degree = numpy.pi/180
  sidelength = scapar['nside']*scapar['pix_arcsec']/3600*degree
  radius = sidelength

  # and center position
  cpos_local = (scapar['nside']-1)/2
  cpos_world = myWCS.all_pix2world([[cpos_local,cpos_local]], 0)[0]
  ra_ctr = cpos_world[0]*degree
  dec_ctr = cpos_world[1]*degree

  # stars
  stargrid = make_sph_grid(res, ra_ctr, dec_ctr, radius)
  # and positions in the SCA image
  px,py = myWCS.all_world2pix(stargrid['rapix']/degree, stargrid['decpix']/degree, 0)

  return(stargrid['ipix'], px, py)

# Make an SCA image with this grid of stars with a PSF from a specified file with unit flux
#
# Inputs:
#   res = HEALPix resolution
#   inpsf = PSF dictionary
#   idsca = tuple (obsid, sca) (sca in 1..18)
#   obsdata = observation table (needed for some data format)
#   mywcs = the WCS solution for this SCA
#   nside_sca = side length of SCA
#
def make_image_from_grid(res, inpsf, idsca, obsdata, mywcs, nside_sca):

  thisimage = numpy.zeros((nside_sca, nside_sca))
  ipix,xsca,ysca = generate_star_grid(res,mywcs)
  p = 6 # padding for interpolation (n/2+1 for nxn interpolation kernel)
  d = 64 # region to draw

  for istar in range(len(ipix)):
    thispsf = psf_utils.get_psf_pos(inpsf, idsca, obsdata, (xsca[istar],ysca[istar]))
    this_xmax = min(nside_sca, int(xsca[istar])+d)
    this_xmin = max(0, int(xsca[istar])-d)
    this_ymax = min(nside_sca, int(ysca[istar])+d)
    this_ymin = max(0, int(ysca[istar])-d)
    pnx = this_xmax - this_xmin
    pny = this_ymax - this_ymin
    if pnx<1 or pny<1: continue

    # draw at this location
    inX = numpy.zeros((pny,pnx))
    inY = numpy.zeros((pny,pnx))
    inX[:,:] = (numpy.array(range(this_xmin,this_xmax))-xsca[istar])[None,:]
    inY[:,:] = (numpy.array(range(this_ymin,this_ymax))-ysca[istar])[:,None]
    interp_array = numpy.zeros((1,pny*pnx))
    (ny,nx) = numpy.shape(thispsf)
    pyimcom_croutines.iD5512C(numpy.pad(thispsf,p).reshape((1,ny+2*p,nx+2*p)),
      inpsf['oversamp']*inX.flatten()+(nx-1)/2.+p, inpsf['oversamp']*inY.flatten()+(ny-1)/2.+p, interp_array)
    thisimage[this_ymin:this_ymax,this_xmin:this_xmax] = (thisimage[this_ymin:this_ymax,this_xmin:this_xmax]
      + interp_array.reshape((pny,pnx)) * inpsf['oversamp']**2)

  return(thisimage)

if __name__=='__main__':
  degree = numpy.pi/180.

  f = fits.open('/Users/christopherhirata/sampledata/simple/dc2_J129_100659_15.fits')
  mywcs = wcs.WCS(f['SCI'].header)
  ipix,xsca,ysca = generate_star_grid(14,mywcs)

  thisimage = make_image_from_grid(14, {'format':'dc2_imsim', 'path':'/Users/christopherhirata/sampledata/psf', 'oversamp':8},
    (100659,15), None, mywcs, 4088)

  for j in range(len(ipix)):
     theta,phi = healpy.pixelfunc.pix2ang(2**14, ipix[j], nest=False, lonlat=False)
     print(ipix[j], (phi/degree, 90-theta/degree), (xsca[j],ysca[j]))

  hdu_list = fits.HDUList(fits.PrimaryHDU(thisimage.astype(numpy.float32), header=mywcs.to_header(relax=True)))
  hdu_list.writeto('test-grid-1x.fits', overwrite=True)
