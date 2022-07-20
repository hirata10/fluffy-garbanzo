import numpy
import re
import os
import sys
import time
from os.path import exists
from astropy.io import fits
from astropy import wcs

# furry-parakeet tools
import pyimcom_interface

# other scripts in this directory
import coadd_utils

class EmptyClass:
  pass
outcoords = EmptyClass() # will fill this in later

# some default settings
sigmatarget = 1.5/2.355 # FWHM Gaussian smoothing divided by 2.355 to be a sigma
npixpsf = 64 # size of PSF postage stamp in native pixels
instamp_pad = 1.*coadd_utils.arcsec # input stamp size padding
imcom_smax = 0.2
imcom_flat_penalty = 1e-8
n_inframe = 1 # number of input images to stack at once

# Read in information
config_file = sys.argv[1]
with open(config_file) as myf: content = myf.read().splitlines()
obsfile = None
nblock = 1
this_sub = 0
n_out = 1
use_filter = 0
hdu_with_wcs = 'SCI' # which header in the input file contains the WCS information
outcoords.postage_pad = 0 # pad this many IMCOM postage stamps around the edge

for line in content:
  m = re.search(r'^OBSFILE\:\s*(\S+)', line)
  if m: obsfile = m.group(1)

  # tile center in degrees RA, DEC
  m = re.search(r'^CTR\:\s*(\S+)\s+(\S+)', line)
  if m:
    outcoords.ra = float(m.group(1)); outcoords.dec = float(m.group(2))
  #
  # and output size: n1 (number of IMCOM blocks), n2 (size of single run), dtheta (arcsec)
  # output array size will be (n1 x n2 x dtheta) on a side
  # with padding, it is (n1 + 2*postage_pad) n2 x dtheta on a side
  #
  m = re.search(r'^OUTSIZE\:\s*(\S+)\s+(\S+)\s+(\S+)', line)
  if m:
    outcoords.n1 = int(m.group(1))
    outcoords.n2 = int(m.group(2))
    outcoords.dtheta = float(m.group(3))/3600. # convert to degrees
    outcoords.Nside = outcoords.n1*outcoords.n2
  #
  # if we are doing a nblock x nblock array on the same projection
  m = re.search('^BLOCK\:\s*(\S+)', line)
  if m: nblock=int(m.group(1))

  m = re.search('^PAD\:\s*(\S+)', line)
  if m: outcoords.postage_pad = int(m.group(1))

  # number of output PSFs
  m = re.search('^NOUT\:\s*(\S+)', line)
  if m: n_out = int(m.group(1))

  # which filter to make coadd
  m = re.search('^FILTER\:\s*(\S+)', line)
  if m: use_filter = int(m.group(1))

  # input files
  m = re.search('^INDATA\:\s*(\S+)\s+(\S+)', line)
  if m:
    inpath = m.group(1)
    informat = m.group(2)

  # input files
  m = re.search('^INPSF\:\s*(\S+)\s+(\S+)', line)
  if m:
    inpsf = { 'path': m.group(1), 'format': m.group(2) }
    inpsf['oversamp'] = coadd_utils.get_psf_oversamp(inpsf)

  # output stem
  m = re.search('^OUT\:\s*(\S+)', line)
  if m: outstem = m.group(1)

# --- end configuration file ---

# Build/check output coordinate information
outcoords.NsideP = outcoords.Nside + outcoords.postage_pad*outcoords.n2*2
outcoords.n1P = outcoords.n1 + outcoords.postage_pad*2
if outcoords.n1%2!=0:
  print('Error: n1 must be even since PSF computations are in 2x2 groups')
  exit()
#
# search radius for input pixels
rpix_search_in = int(numpy.ceil((outcoords.n2*outcoords.dtheta*coadd_utils.degree/numpy.sqrt(2.) + instamp_pad)/coadd_utils.pixscale_native + 1))
insize = 2*rpix_search_in
print('input stamp radius -->', rpix_search_in, 'native pixels   stamp={:3d}x{:3d}'.format(insize,insize))
print('')

# Get observation table
if obsfile is not None:
  print('Getting observations from {:s}'.format(obsfile))
  with fits.open(obsfile) as myf:
    obsdata = myf[1].data
    obscols = myf[1].columns
  n_obs_tot = len(obsdata.field(0))
  print('Retrieved columns:', obscols.names, ' {:d} rows'.format(n_obs_tot))
else:
  print('Error: no obsfile found')
  exit()

# subregion information
if len(sys.argv)>2:
  this_sub = int(sys.argv[2])

# display output information
print('Output information: ctr at RA={:10.6f},DEC={:10.6f}'.format(outcoords.ra, outcoords.dec))
print('pixel scale={:8.6f} arcsec or {:11.5E} degree'.format(outcoords.dtheta*3600, outcoords.dtheta))
print('output array size = {:d} ({:d} blocks of {:d})'.format(outcoords.Nside, outcoords.n1, outcoords.n2))
print('')
#
# block information
# prime number to not do all the blocks next to each other first
p = 1567
if nblock%p==0: p=281
j = (this_sub*p)%(nblock**2)
outcoords.ibx = j//nblock; outcoords.iby = j%nblock
print('sub-block {:4d} <{:2d},{:2d}> of {:2d}x{:2d}={:2d}'.format(this_sub, outcoords.ibx, outcoords.iby, nblock, nblock, nblock**2))
#
# make the WCS
wcsout = wcs.WCS(naxis=2)
wcsout.wcs.crpix = [(outcoords.NsideP+1)/2. - outcoords.Nside*(outcoords.ibx-(nblock-1)/2.),
               (outcoords.Nside+1)/2. - outcoords.Nside*(outcoords.iby-(nblock-1)/2.)]
wcsout.wcs.cdelt = [-outcoords.dtheta, outcoords.dtheta]
wcsout.wcs.ctype = ["RA---STG", "DEC--STG"]
wcsout.wcs.crval = [outcoords.ra, outcoords.dec]
#
# print the corners of the square and the center, ordering:
#   2   3
#     4
#   0   1
cornerx = [-.5,outcoords.NsideP-.5,-.5,outcoords.NsideP-.5,(outcoords.NsideP-1)/2.]
cornery = [-.5,-.5,outcoords.NsideP-.5,outcoords.NsideP-.5,(outcoords.NsideP-1)/2.]
for i in range(5): print(i, wcsout.wcs_pix2world(numpy.array([[cornerx[i],cornery[i]]]), 0))
outcoords.centerpos = wcsout.wcs_pix2world(numpy.array([[cornerx[-1],cornery[-1]]]), 0)[0] # [ra,dec] array in degrees

# make basic output array
out_map = numpy.zeros((n_out, n_inframe, outcoords.NsideP, outcoords.NsideP), dtype=numpy.float32)

# and the output PSFs and target leakages
#
# (this will have to be modified for multiple outputs)
OutputPSF = [pyimcom_interface.psf_simple_airy(npixpsf*inpsf['oversamp'],coadd_utils.QFilterNative[use_filter]*inpsf['oversamp'],
  tophat_conv=0.,sigma=sigmatarget*inpsf['oversamp'])]
uctarget = [1e-6]

### Now figure out which observations we need ###

search_radius = coadd_utils.sca_sidelength/numpy.sqrt(2.)/coadd_utils.degree + outcoords.NsideP*outcoords.dtheta/numpy.sqrt(2.)
obslist = coadd_utils.get_obs_cover(obsdata, outcoords.centerpos[0], outcoords.centerpos[1], search_radius, use_filter)
print(len(obslist), 'observations within range ({:7.5f} deg)'.format(search_radius), 'filter =', use_filter, '({:s})'.format(coadd_utils.RomanFilters[use_filter]))
infiles = []; infile_exists = []; infile_chars = []; inwcs = []
for j in range(len(obslist)):
  infiles += [coadd_utils.get_sca_imagefile(inpath, obslist[j], obsdata, informat)]
  infile_exists += [exists(infiles[j])]
  if infile_exists[j]:
    infile_chars += [' ']
    with fits.open(infiles[j]) as f: inwcs += [wcs.WCS(f[hdu_with_wcs].header)]
  else:
    infile_chars += ['x']
    inwcs += [None]
print('The observations -->')
print('  OBSID SCA  RAWFI    DECWFI   PA     RASCA   DECSCA       FILE (x=missing)')
for j in range(len(obslist)):
  cpos = '                 '
  if infile_exists[j]:
    cpos_coord = inwcs[j].wcs_pix2world([[coadd_utils.sca_ctrpix,coadd_utils.sca_ctrpix]], 0)[0]
    cpos = '{:8.4f} {:8.4f}'.format(cpos_coord[0], cpos_coord[1])
  print('{:7d} {:2d} {:8.4f} {:8.4f} {:6.2f} {:s} {:s} {:s}'.format(obslist[j][0], obslist[j][1], obsdata['ra'][obslist[j][0]], obsdata['dec'][obslist[j][0]],
    obsdata['pa'][obslist[j][0]], cpos, infile_chars[j], infiles[j]))
print('')
if len(obslist)==0:
  print('No candidate observations found to stack. Exiting now.')
  exit()
sys.stdout.write('Reading input data ... ')
in_data = coadd_utils.get_all_data(n_inframe, obslist, obsdata, inpath, informat)
sys.stdout.write('done.\n')
sys.stdout.flush()
print('Size = {:6.1f} MB, shape ='.format(in_data.size*in_data.itemsize/1e6), numpy.shape(in_data))
print('')

### Begin loop over all the postage stamps we want to create ###

nrun = outcoords.n1P**2
nrun = 312 # <-- for testing only, to do the first patches
for ipostage in range(nrun):
  ipostageX = 2 * ((ipostage//4)% (outcoords.n1P//2) )
  ipostageY = 2 * ((ipostage//4)//(outcoords.n1P//2) )
  if ipostage%2==1: ipostageX += 1
  if ipostage%4>=2: ipostageY += 1
  print('postage stamp {:2d},{:2d}  {:6.3f}% t= {:9.2f} s'.format(ipostageX, ipostageY, 100*ipostage/outcoords.n1P**2, time.perf_counter()))

  # In these cases, we need to compute the PSF matrices and decide which inputs to use
  # (to save time, only done every 4th IMCOM run)
  if ipostage%4==0:
    # get where to compute the PSF and the camera distortion matrix
    psf_compute_point = wcsout.wcs_pix2world(numpy.array([[(ipostageX+1)*outcoords.n2-.5,(ipostageY+1)*outcoords.n2-.5]]) ,0)[0]
    dWdp_out = wcs.utils.local_partial_pixel_derivatives(wcsout,(ipostageX+1)*outcoords.n2-.5,(ipostageY+1)*outcoords.n2-.5)
    print('INPUT/PSF computation at RA={:8.4f}, Dec={:8.4f}'.format(psf_compute_point[0], psf_compute_point[1]))
    useInput = infile_exists.copy()
    psfs = [None for l in range(len(obslist))]
    pixloc = [None for l in range(len(obslist))]
    for j in range(len(obslist)):
      if infile_exists[j]:
        pos = inwcs[j].wcs_world2pix(numpy.array([psf_compute_point.tolist()]), 0)[0]
        pixloc[j] = pos
        # check if this point is on the SCA
        if pos[0]<0 or pos[0]>=coadd_utils.sca_nside or pos[1]<0 or pos[1]>=coadd_utils.sca_nside:
          useInput[j] = False
          continue
        psfs[j] = coadd_utils.get_psf_pos(inpsf, obslist[j], obsdata, pos)
        if psfs[j] is None:
          useInput[j] = False
          continue
        #
        # window location
        win = coadd_utils.genWindow(pos[0],pos[1],rpix_search_in,inwcs[j],wcsout)
        print('PSF info -->', j,pos,inpsf['oversamp'], numpy.shape(psfs[j]), numpy.sum(psfs[j]))

    # tabulate which psfs we are using, and get the distortion matrices d[(X,Y)perfect]/d[(X,Y)native]
    # Note that rotations and magnifications are included in the distortion matrix, as well as shear
    # Also the distortion is relative to the output grid, not to the tangent plane to the celestial sphere
    # (although we really don't want the difference to be large ...)
    #
    useList = []
    InputPSF = []
    distort_matrices = []
    for j in range(len(obslist)):
      if useInput[j]:
        useList.append(j)
        InputPSF.append(psfs[j])
        distort_matrices.append(numpy.linalg.inv(dWdp_out) @ wcs.utils.local_partial_pixel_derivatives(inwcs[j],pixloc[j][0],pixloc[j][1])
          * outcoords.dtheta*coadd_utils.degree/coadd_utils.pixscale_native)
    print('using input exposures:', useList)

    psfOverlap = pyimcom_interface.PSF_Overlap(InputPSF, OutputPSF, .5, 2*npixpsf*inpsf['oversamp']-1, coadd_utils.pixscale_native/coadd_utils.arcsec,
        distort_matrices=distort_matrices)

    # this output was for testing. might put it back later
    #hdu = fits.PrimaryHDU(psfOverlap.psf_array)
    #hdu.writeto(outstem+'_testpsf.fits', overwrite=True)

  # -- end PSF matrices --

  n_in = len(useList)
  if n_in==0:
    print('empty input list ... skipping')
    continue
  inmask = numpy.zeros((n_in,insize,insize), dtype=bool)

  # get windows for the input images
  windows = []
  posoffset = []
  xtile = ipostageX*outcoords.n2 + (outcoords.n2-1)/2.
  ytile = ipostageY*outcoords.n2 + (outcoords.n2-1)/2.
  ctr = wcsout.wcs_pix2world(numpy.array([[xtile,ytile]]) ,0)[0]
  incube = numpy.zeros((n_inframe, n_in, insize, insize))
  for i_in in range(n_in):
    j = useList[i_in]
    pos = inwcs[j].wcs_world2pix(numpy.array([ctr.tolist()]), 0)[0]
    windows.append( coadd_utils.genWindow(pos[0],pos[1],insize,inwcs[j],wcsout) )
    posoffset.append(((windows[i_in]['outx']-xtile)*outcoords.dtheta*3600., (windows[i_in]['outy']-ytile)*outcoords.dtheta*3600.))
    inmask[i_in,:,:] = windows[i_in]['active']
    incube[0,i_in,:,:] = coadd_utils.snap_instamp(in_data[:,j,:,:], windows[i_in])

  imcomSysSolve = pyimcom_interface.get_coadd_matrix(psfOverlap, float(inpsf['oversamp']), uctarget, posoffset, distort_matrices,
    coadd_utils.pixscale_native/coadd_utils.arcsec, (insize,insize), outcoords.dtheta*3600, (outcoords.n2,outcoords.n2),
    inmask, instamp_pad/coadd_utils.arcsec, imcom_smax, flat_penalty=imcom_flat_penalty)
  print('  n input pix =',numpy.sum(numpy.where(imcomSysSolve['full_mask'],1,0)))
  sumstats = '  sqUC,sqSig %iles |'
  for i in [50,90,98,99]:
    sumstats += ' {:2d}% {:8.2E} {:8.2E} |'.format(i, numpy.percentile(numpy.sqrt(imcomSysSolve['UC']),i), numpy.percentile(numpy.sqrt(imcomSysSolve['Sigma']),i))
  print(sumstats)

  hdu = fits.PrimaryHDU(imcomSysSolve['A']); hdu.writeto(outstem+'A.fits', overwrite=True)
  hdu = fits.PrimaryHDU(imcomSysSolve['T'].reshape((n_out,outcoords.n2**2, n_in*insize**2))); hdu.writeto(outstem+'T.fits', overwrite=True)

  # the actual multiplication
  for v in range(n_inframe):
    out_map[:,v,ipostageY*outcoords.n2:(ipostageY+1)*outcoords.n2,ipostageX*outcoords.n2:(ipostageX+1)*outcoords.n2] =\
      (imcomSysSolve['T'].reshape(n_out*outcoords.n2**2,n_in*insize**2)@incube[v,:,:,:].flatten()).reshape(n_out,outcoords.n2,outcoords.n2)

  sys.stdout.flush()

### End for ipostage loop ###

### Output array ###

maphdu = fits.PrimaryHDU(out_map, header=wcsout.to_header())
hdu_list = fits.HDUList([maphdu])
hdu_list.writeto(outstem+'_map.fits', overwrite=True)

print('')
print('finished at t =', time.perf_counter(), 's')
print('')

