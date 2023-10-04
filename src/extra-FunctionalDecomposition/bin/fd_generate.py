#!/usr/bin/env python

import argparse, os, sys
import numpy             as np
import matplotlib.pyplot as plt

from   numpy.lib.format import open_memmap
from   math             import sqrt, pi

# Parameters for test PDF function
par    = [  9.50700e+04,
            4.58242e+01,
           -1.21268e+01,
           -1.51309e+00,
            2.38849e-01,

           -8.38068e+06,
            3.34980e+01,
            1.83062e+01,
         ]
eLambda = 250


def mkCache(base, name, shape, **kwargs):
    opath = os.path.join(base, "Data", name)
    try:
        os.makedirs(opath)
    except OSError:
        pass

    return { k : open_memmap(os.path.join(opath, k + ".npy"), dtype=t, mode='w+', shape=shape) for k, t in kwargs.items() }

# Gaussian PDF
def Gauss(x, u, s):
    return np.exp( -0.5*( (x-u)/s )**2 ) / sqrt(2*pi*s*s)

def Exp(x, l):
    return np.exp(-x/l) / l

# Background PDF
def G_dijet5Param(x, com, *p):
  x[x < 0]   = 0
  x[x > com] = com
  mX         = x / com
  e          = p[2] + p[3]*np.log(mX) + p[4]*np.log(x)**2

  return p[5]*Gauss(x,p[6],p[7]) + p[0] * (1-mX)**p[1] * mX**e

###### Okay, start it up.
ArgP    = argparse.ArgumentParser(description=' === Functional Decomposition Test Data Generator ===')
ArgP.add_argument('--base',    type=str,   default=".",      help="FD base directory.")
ArgP.add_argument('--setname', type=str,   default="Test",   help="Name of dataset to create.")
ArgP.add_argument('--varname', type=str,   default="Mass",   help="Variable name.")
ArgP.add_argument('--wgtname', type=str,   default="Wgt",    help="Variable name.")
ArgP.add_argument('--varcut',  type=float, default=60.0,     help="Minimum value for variable.")
ArgP.add_argument('--com',     type=float, default=13000,    help="Center-of-mass energy.")
ArgP.add_argument('--size',    type=int,   default=int(1e6), help="Number of events to create.")
ArgP.add_argument('--cksize',  type=int,   default=int(1e6), help="Event creation chunk size.")
ArgP.add_argument('--show',    action='store_true',          help="Show histogram of generated events before exit.")

ArgC    = ArgP.parse_args()

###### Initialize the variable and weight arrays.
types   = { ArgC.varname: np.float,
            ArgC.wgtname: np.float,
          }
outSets = mkCache(ArgC.base, ArgC.setname, (ArgC.size,), **types)
wgt     = outSets[ArgC.wgtname]
var     = outSets[ArgC.varname]

# get acceptance ratio
x       = np.linspace(ArgC.varcut, ArgC.com, 1e6)
M       = G_dijet5Param(x, ArgC.com, *par) / Exp(x - ArgC.varcut, eLambda)
M       = 1.01*M.max()

# generate the data.
n       = 0
ns      = 0

print
print "======> FD TEST DATA GENERATOR <======"
print

while n < ArgC.size:
    x    = ArgC.varcut + np.random.exponential(scale=eLambda, size=ArgC.cksize)
    u    = np.random.uniform(size=ArgC.cksize)
    v    = G_dijet5Param(x, ArgC.com, *par)

    keep = x[M * u < v / Exp(x - ArgC.varcut, eLambda) ]
    num  = min(ArgC.size - n, len(keep) )

    var[n:n+num] = keep[:num]
    wgt[n:n+num] = 1.0

    n   += num
    ns  += 1

    print "\rGenerating: % 14d / % 9d" % ( n, ArgC.size ), ; sys.stdout.flush()
print
print "PDF Ratio: % 27.3f" % M
print "Gen Efficiency: %22.3f" % ( float(ArgC.size) / float(ns*ArgC.cksize) )
print

if ArgC.show:
    plt.hist(var, 500, facecolor='g', alpha=0.75)
    plt.xlabel(ArgC.varname)
    plt.ylabel('Events')
    plt.title('Generated Events')
    plt.yscale('log')
    plt.xscale('log')
    plt.grid(True)
    plt.show()


