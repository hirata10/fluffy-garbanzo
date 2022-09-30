import sys
import numpy
import healpy
from astropy.io import fits
from astropy import wcs
from os.path import exists
import galsim

SNR = 10.

n = 2600
bd = 50
bd2 = 10
res = 14
rs = 1./numpy.sqrt(2.)/60.*numpy.pi/180*1.08333
sigma = 20.

nblock = 48
ncol = 26
nstart = 0
WNuse_slice = 4
use_slice = 2
use_sliceB = 3

area = {'Y': 7.06, 'J': 8.60, 'H': 10.96, 'F': 15.28}

filter = sys.argv[1]; nblockuse = int(sys.argv[2])

if filter=='Y': filtername='Y106'
if filter=='J': filtername='J129'
if filter=='H': filtername='H158'
if filter=='F': filtername='F184'

pos = numpy.zeros((1,ncol)) # ra, dec, ibx, iby, x, y, xi, yi, dx, dy, [Axysg1g2]G, [Axysg1g2]C, fid, ct
image = numpy.zeros((1,bd*2-1,bd*2-1))
imageB = numpy.zeros((1,bd*2-1,bd*2-1))

outfile_g = 'StarCat_galsim_{:s}.fits'.format(filter)
outfile_c = 'StarCat_croutines_{:s}.fits'.format(filter)

fhist = numpy.zeros((61,),dtype=numpy.uint32)

for iblock in range(nstart,nstart+nblockuse):

  p = 1567
  if nblock%p==0: p=281
  j = (iblock*p)%2304
  ibx = j//nblock; iby = j%nblock

  infile = r'/Volumes/Seagate Backup Plus Drive/Roman_IMCOM_Project/block_output_files/{:s}/test3{:s}_{:02d}_{:02d}_map.fits'.format(filtername,filter,ibx,iby)
  if not exists(infile): continue
  with fits.open(infile) as f:
    mywcs = wcs.WCS(f[0].header)
    WNmap = f[0].data[0,WNuse_slice,:,:]
    map = f[0].data[0,use_slice,:,:]
    wt = numpy.rint(1./numpy.amax(f['INWEIGHT'].data[0,:,:,:]+1e-6, axis=0))
    mapB = f[0].data[0,use_sliceB,:,:]
    fmap = f['FIDELITY'].data[0,:,:].astype(numpy.float32)
    for fy in range(61): fhist[fy] += numpy.count_nonzero(f['FIDELITY'].data[0,100:-100,100:-100]==fy)

  ra_cent, dec_cent = mywcs.all_pix2world([(n-1)/2], [(n-1)/2], [0.], [0.], 0, ra_dec_order=True)
  ra_cent = ra_cent[0]; dec_cent = dec_cent[0]
  vec = healpy.ang2vec(ra_cent, dec_cent, lonlat=True)
  qp = healpy.query_disc(2**res, vec, rs, nest=False)
  ra_hpix, dec_hpix = healpy.pix2ang(2**res, qp, nest=False, lonlat=True)
  npix = len(ra_hpix)
  x, y, z1, z2 = mywcs.all_world2pix(ra_hpix, dec_hpix, numpy.zeros((npix,)), numpy.zeros((npix,)), 0)
  xi = numpy.rint(x).astype(numpy.int16); yi = numpy.rint(y).astype(numpy.int16)
  grp = numpy.where(numpy.logical_and(numpy.logical_and(xi>=bd,xi<n-bd),numpy.logical_and(yi>=bd,yi<n-bd)))
  ra_hpix = ra_hpix[grp]
  dec_hpix = dec_hpix[grp]
  x = x[grp]
  y = y[grp]
  npix = len(x)

  newpos = numpy.zeros((npix,ncol))
  xi = numpy.rint(x).astype(numpy.int16)
  yi = numpy.rint(y).astype(numpy.int16)
  # position information
  newpos[:,0] = ra_hpix
  newpos[:,1] = dec_hpix
  newpos[:,2] = ibx
  newpos[:,3] = iby
  newpos[:,4] = x
  newpos[:,5] = y
  newpos[:,6] = xi
  newpos[:,7] = yi
  newpos[:,8] = dx = x-xi
  newpos[:,9] = dy = y-yi

  WNimage = numpy.zeros((npix,bd*2-1,bd*2-1))
  newimage = numpy.zeros((npix,bd*2-1,bd*2-1))
  newimageB = numpy.zeros((npix,bd*2-1,bd*2-1))
  print(iblock, infile, npix)
  for k in range(npix):
    thisnoiseimage = WNmap[yi[k]+1-bd:yi[k]+bd,xi[k]+1-bd:xi[k]+bd]/SNR/numpy.sqrt(area[filter])
    newimage[k,:,:] = map[yi[k]+1-bd:yi[k]+bd,xi[k]+1-bd:xi[k]+bd]
    newimageB[k,:,:] = mapB[yi[k]+1-bd:yi[k]+bd,xi[k]+1-bd:xi[k]+bd]

    # PSF shape
    moms = galsim.Image(newimage[k,:,:]).FindAdaptiveMom()
    newpos[k,10] = moms.moments_amp
    newpos[k,11] = moms.moments_centroid.x-bd-dx[k]
    newpos[k,12] = moms.moments_centroid.y-bd-dy[k]
    newpos[k,13] = moms.moments_sigma
    newpos[k,14] = moms.observed_shape.g1
    newpos[k,15] = moms.observed_shape.g2
    moms = galsim.Image(newimageB[k,:,:]).FindAdaptiveMom()
    newpos[k,16] = moms.moments_amp
    newpos[k,17] = moms.moments_centroid.x-bd-dx[k]
    newpos[k,18] = moms.moments_centroid.y-bd-dy[k]
    newpos[k,19] = moms.moments_sigma
    newpos[k,20] = moms.observed_shape.g1
    newpos[k,21] = moms.observed_shape.g2

    # fidelity
    newpos[k,22] = numpy.mean(fmap[yi[k]+1-bd2:yi[k]+bd2,xi[k]+1-bd2:xi[k]+bd2])

    # coverage
    newpos[k,23] = wt[yi[k]//bd,xi[k]//bd]

    # noise bias
    try:
      moms_noise_zero = galsim.Image(newimage[k,:,:]).FindAdaptiveMom()
      moms_noise_positive = galsim.Image(newimage[k,:,:] + thisnoiseimage).FindAdaptiveMom()
      moms_noise_negative = galsim.Image(newimage[k,:,:] - thisnoiseimage).FindAdaptiveMom()
      newpos[k,24] = (moms_noise_positive.observed_shape.g1 + moms_noise_negative.observed_shape.g1)/2. - moms_noise_zero.observed_shape.g1
      newpos[k,25] = (moms_noise_positive.observed_shape.g2 + moms_noise_negative.observed_shape.g2)/2. - moms_noise_zero.observed_shape.g2
    except:
      try:
        print('BACKUP-0.5    {:d},{:d}  coverage={:2d}'.format(iblock,k,int(newpos[k,23])))
        sc = numpy.sqrt(10.)
        moms_noise_zero = galsim.Image(newimage[k,:,:]).FindAdaptiveMom()
        moms_noise_positive = galsim.Image(newimage[k,:,:] + thisnoiseimage/sc).FindAdaptiveMom()
        moms_noise_negative = galsim.Image(newimage[k,:,:] - thisnoiseimage/sc).FindAdaptiveMom()
        newpos[k,24] = (moms_noise_positive.observed_shape.g1 + moms_noise_negative.observed_shape.g1)/2. - moms_noise_zero.observed_shape.g1
        newpos[k,25] = (moms_noise_positive.observed_shape.g2 + moms_noise_negative.observed_shape.g2)/2. - moms_noise_zero.observed_shape.g2
        newpos[k,24:26] *= sc**2
      except:
        try:
          print('BACKUP-1.0    {:d},{:d}  coverage={:2d}'.format(iblock,k,int(newpos[k,23])))
          sc = 10.
          moms_noise_zero = galsim.Image(newimage[k,:,:]).FindAdaptiveMom()
          moms_noise_positive = galsim.Image(newimage[k,:,:] + thisnoiseimage/sc).FindAdaptiveMom()
          moms_noise_negative = galsim.Image(newimage[k,:,:] - thisnoiseimage/sc).FindAdaptiveMom()
          newpos[k,24] = (moms_noise_positive.observed_shape.g1 + moms_noise_negative.observed_shape.g1)/2. - moms_noise_zero.observed_shape.g1
          newpos[k,25] = (moms_noise_positive.observed_shape.g2 + moms_noise_negative.observed_shape.g2)/2. - moms_noise_zero.observed_shape.g2
          newpos[k,24:26] *= sc**2
        except:
          print('ERROR {:d},{:d}  coverage={:2d}'.format(iblock,k,int(newpos[k,23])))
          pass
    newpos[k,24:26] *= SNR**2

    # end galaxy loop

  pos = numpy.concatenate((pos, newpos), axis=0)
  image = numpy.concatenate((image, newimage), axis=0)
  imageB = numpy.concatenate((image, newimageB), axis=0)

pos = pos[1:,:]
image = image[1:,:,:]
imageB = image[1:,:,:]

fits.HDUList([fits.PrimaryHDU(image.astype(numpy.float32))]).writeto(outfile_g, overwrite=True)
fits.HDUList([fits.PrimaryHDU(imageB.astype(numpy.float32))]).writeto(outfile_c, overwrite=True)

numpy.savetxt('StarCat_{:s}.txt'.format(filter), pos)

print(pos[:,-1])
#print(pos[:,-2])
#print(numpy.sqrt(pos[:,11]**2+pos[:,12]**2)/numpy.sqrt(2))
#print(numpy.sqrt(pos[:,14]**2+pos[:,15]**2)/numpy.sqrt(2))
#print(numpy.sqrt(pos[:,17]**2+pos[:,18]**2)/numpy.sqrt(2))
print(numpy.sqrt(pos[:,20]**2+pos[:,21]**2)/numpy.sqrt(2))

for fy in range(20,61):
  print('{:2d} {:8.6f} {:8.6f}'.format(fy, fhist[fy]/numpy.sum(fhist), numpy.sum(fhist[:fy+1])/numpy.sum(fhist)))
