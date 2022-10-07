import sys
import numpy

infile = sys.argv[1]
usetype = sys.argv[2]

data = numpy.loadtxt(infile)

if usetype=='gG':
  yd = numpy.sqrt(data[:,14]**2 + data[:,15]**2)
if usetype=='gC':
  yd = numpy.sqrt(data[:,20]**2 + data[:,21]**2)
if usetype=='aG':
  yd = numpy.sqrt(data[:,11]**2 + data[:,12]**2) * .025

for x in range(25,60):
  print('{:2d} {:16.9E} {:16.9E}'.format(x, numpy.sqrt(numpy.mean(yd[data[:,22]>=x]**2)),
    numpy.sqrt(numpy.mean(yd[numpy.logical_and(data[:,22]>=x,data[:,22]<x+1)]**2))
  ))
