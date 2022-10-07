# bands: Y, J, H, F
@bandnames = ('Y106', 'F184');

@infiles = ('StarCat_Y_sample221007.txt', 'StarCat_F_sample221007.txt');

open(G, "|tee script.txt | gnuplot");
print G qq^set term postscript enhanced 26 eps color\n^;
print G qq^set output "starstatfig.eps"\n^;
print G qq^set size 4,3.6\n^;
print G qq^set origin 0,0\n^;
print G qq^set multiplot\n^;
print G qq^set label "**DRAFT**" at 51,-2.2\n^;
print G qq^set size 1,1.2\n^;
for $iband (0 .. ((scalar @infiles) -1)) {
  print G qq^set origin $iband,2.4\n^;
  print "band $iband\n";
  print G qq^set xrange [25:61]; set ytics 5\n^;
  print G qq^set yrange [-5:-1]; set ytics 1\n^;
  print G qq^set grid\n^;
  print G qq^set style line 1 lt 1 lw 1 dt 1 pt 1 ps .5 lc rgb "#ff8080"\n^;
  print G qq^set style line 2 lt 1 lw 5 dt 3 pt 1 ps .5 lc rgb "#2000e0"\n^;
  print G qq^set style line 3 lt 1 lw 5 dt 1 pt 1 ps .5 lc rgb "#0080a0"\n^;
  print G qq^set xlabel "Fidelity -10 log_{10} U_{/Symbol a}/C"\n^;
  print G qq^set ylabel "log_{10} g"\n^;
  print G qq^set title "Ellipticity of stars: $bandnames[$iband], GalSim\n^;
  print G qq^plot "$infiles[$iband]" using 23:(log10(sqrt(\$15**2+\$16**2))) with points notitle ls 1, ^;
  print G qq^"-" using 1:(log10(\$2)) with lines notitle ls 2, ^;
  print G qq^"-" using (\$1+.5):(log10(\$3)) with lines notitle ls 3\n^;
  $c = `python rms_stat.py $infiles[$iband] gG`;
  print G $c;
  print G qq^e\n^;
  print G $c;
  print G qq^e\n^;

  # same thing, with croutines
  print G qq^set origin $iband,1.2\n^;
  print G qq^set title "Ellipticity of stars: $bandnames[$iband], croutines\n^;
  print G qq^plot "$infiles[$iband]" using 23:(log10(sqrt(\$21**2+\$22**2))) with points notitle ls 1, ^;
  print G qq^"-" using 1:(log10(\$2)) with lines notitle ls 2, ^;
  print G qq^"-" using (\$1+.5):(log10(\$3)) with lines notitle ls 3\n^;
  $c = `python rms_stat.py $infiles[$iband] gC`;
  print G $c;
  print G qq^e\n^;
  print G $c;
  print G qq^e\n^;

  # astrometry
  print G qq^set origin $iband,0.0\n^;
  print G qq^set title "Astrometric errors of stars: $bandnames[$iband], GalSim\n^;
  print G qq^set ylabel "log_{10} error [arcsec]"\n^;
  print G qq^set yrange [-6:-2]\n^;
  print G qq^plot "$infiles[$iband]" using 23:(log10(.025*sqrt(\$12**2+\$13**2))) with points notitle ls 1, ^;
  print G qq^"-" using 1:(log10(\$2)) with lines notitle ls 2, ^;
  print G qq^"-" using (\$1+.5):(log10(\$3)) with lines notitle ls 3\n^;
  $c = `python rms_stat.py $infiles[$iband] aG`;
  print G $c;
  print G qq^e\n^;
  print G $c;
  print G qq^e\n^;
}
print G qq^unset multiplot\n^;
close G;
