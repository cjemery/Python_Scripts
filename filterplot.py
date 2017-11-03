#!/user/bin/env python

import os, sys
from matplotlib import pyplot;
from pylab import genfromtxt;


data = genfromtxt(sys.argv[1]);
x=data[:,1]*data[:,2];
print(x);
pyplot.plot(x, data[:,2], label = "Data")


pyplot.legend();
pyplot.xlabel("Current")
pyplot.ylabel("Voltage")
#pyplot.savefig('FilterPlot.png')
pyplot.show();

