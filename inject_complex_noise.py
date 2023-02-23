import numpy
from astropy.io import fits

def noise_1f_frame(seed):
  this_array = numpy.zeros((4096,4096))
  rng = numpy.random.default_rng(seed)
  len = 8192*128

  # get frequencies and amplitude ~ sqrt{power}
  freq = numpy.linspace(0,1-1./len,len)
  freq[len//2:] -= 1.
  amp = (1.e-99+numpy.abs(freq*len))**(-.5)
  amp[0] = 0.
  for ch in range(32):
    # get array
    ftsignal = numpy.zeros((len,),dtype=numpy.complex128)
    ftsignal[:] = rng.normal(loc=0., scale=1., size=(len,))
    ftsignal[:] += 1j*rng.normal(loc=0., scale=1., size=(len,))
    ftsignal *= amp
    block = numpy.fft.fft(ftsignal).real[:len//2]/numpy.sqrt(2.)
    block -= numpy.mean(block)

    xmin = ch*128; xmax = xmin+128
    # mapping into the image depends on L->R or R->L read order
    if ch%2==0:
      this_array[:,xmin:xmax] = block.reshape((4096,128))
    else:
      this_array[:,xmin:xmax] = block.reshape((4096,128))[:,::-1]

  return(this_array[4:4092,4:4092].astype(numpy.float32))

# test function
if __name__ == "__main__":
  noiseframe = noise_1f_frame(1000)
  noiseprimary = fits.PrimaryHDU(noiseframe)
  hdu_list = fits.HDUList([fits.PrimaryHDU(noiseframe)])
  hdu_list.writeto('test-noise-1000.fits', overwrite=True)
  print(numpy.std(noiseframe))
