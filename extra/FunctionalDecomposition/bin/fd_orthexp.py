#!/usr/bin/python

import os,argparse, ConfigParser

import numpy             as np
import matplotlib.pyplot as plt

from   Tools.OrthExp     import ExpDecompFn

end    = 10        # Maximum value to evaluate
Npt    = int(1e6)  # Number of points to use
Nbasis = 101       # Number of basis elements to evaluate

toplot = {   1 : {'color': 'green'},
            25 : {'color': 'blue'},
             2 : {'color': 'black'},
            10 : {'color': 'orange'},
             0 : {'color': 'black', 'ls': '--'},
         }

np.set_printoptions(precision=3, linewidth=160)

######
# Parse command line parameters and config files
######
ArgP      = argparse.ArgumentParser(description='=== Functional Decomposition Orthonormal Exponential Plotter ===')
ArgP.add_argument('--base', type=str, default=".",                  help="FD base directory.")
ArgP.add_argument('--show', action="store_true",                    help="Display plots interactively.")
ArgP.add_argument('--logx', action="store_true",                    help="Set x axis to log.")
ArgP.add_argument('--save', type=str, default="Output/orthexp.pdf", help="Filename to save plot.")
ArgC      = ArgP.parse_args()

Config    = ConfigParser.ConfigParser()
Config.optionxform = str
Config.read( os.path.join(ArgC.base, "base.conf") )

PlotStyle = Config.get("General", "PlotStyle")
try:            plt.style.use( PlotStyle )
except IOError: plt.style.use( os.path.join(ArgC.base, PlotStyle) )


x      = np.linspace(0.001, end, Npt)
w      = np.ones((Npt,)) / Npt

Decomp = ExpDecompFn( x=x, w=w, Nbasis=max(toplot.keys())+1, Lambda=1, x0=0, Alpha=1.0 )

######
for D in Decomp:
    if D.N in toplot:
        plt.plot(x, Decomp.Values(), zorder=-D.N, label='$E_{%d}\\left(z\\right)$' % D.N, **toplot[D.N])
    print D.Values()
    print "%d: %f" % (D.N, Decomp.Moment())

plt.xlabel("z")
plt.ylabel("Arbitrary Units")
plt.xlim( 1e-2, end)
plt.ylim(-2, 2)

if ArgC.logx:
    plt.xscale('log')
plt.legend()
plt.tight_layout()

try:
    plt.savefig(ArgC.save)
except IOError:
    print "Directory for save does not exist or cannot be written to."
    
if ArgC.show:
    plt.show()
