#!/usr/bin/python

import numpy.random

import numpy             as np
import matplotlib.pyplot as plt

from   math            import sqrt, pi
from   Tools.Decomp    import DataSet
from   Tools.OrthExp   import ExpDecompFactory


###

end    = 12.0   # Maximum value to evaluate
Npt    = 3000   # Number of points to use

Nbasis = 128    # Number of basis elements to evaluate
Ntrial = 1      # Number of PEs to perform

Bsamp  = 1e7    # Number of bkg points per PE
Bmu    = 1.0    # bkg gaussian mean
Bsig   = 2.0    # bkg gaussian sigma

Lambda = 2.8    # 2.18   #1.5
Alpha  = 1.9    # 1.02

x      = np.linspace(0, end, Npt)
dx     = float(end)/Npt

###

def gaus(x, mu, sigma):
   x = np.array(x)
   return np.exp(-( (x-mu)/sigma)**2/2)/(sigma*sqrt(2*pi))

np.set_printoptions(precision=3, linewidth=160)
#np.show_config()

Factory = ExpDecompFactory(Nthread=4, Nbasis=Nbasis, Lambda=Lambda, x0=0.0, Alpha=Alpha, a=0)

for n in range(Ntrial):
    # Generate pseudodata
    t = numpy.random.normal(loc=Bmu, scale=Bsig, size=int(Bsamp))
    D = DataSet( t, Factory)

    print "PE: %d/%d; Nevt: %.1f" % (n+1, Ntrial, D.DatNint)

    '''
    Lbest = np.inf
    Pbest = ((0, 0))

    for a in np.linspace(1, 2, 11):
        for l in np.linspace(1, 3, 21):
            L = D.ObjFunc((a, l))
            if L < Lbest:
                Lbest = L
                Pbest = ((a, l))
    D.ObjFunc(Pbest)
    '''
    #D.Decompose()
    D.FitW()
    #D.ScanN(Nsearch=30)

    print "wgt:  ", Factory["a"]
    print "Nb:   ", D.Test.Nmax
    print "Mom:  ", D.Test.Mom

    plt.plot(x, D.Test( x ),   color='red', alpha=0.1)

Tf = (Bsamp/D.DatNint) * np.exp(-( (x-Bmu)/Bsig )**2/2)/(Bsig*sqrt(2*pi))

plt.plot(x, Tf,    lw=2.5, ls='--', color='black')

plt.yscale('log')
plt.show()

exit()

