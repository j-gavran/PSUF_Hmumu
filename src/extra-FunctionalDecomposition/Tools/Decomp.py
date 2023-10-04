import numpy               as np
import numexpr             as ne
import Tools.CacheMgr      as Cache

from   Tools.PrintMgr      import *
from   Tools.Base          import ParametricObject
from   scipy.optimize      import minimize
from   scipy.linalg        import solve
from   numpy.linalg        import slogdet, multi_dot
from   scipy.linalg        import cho_factor, cho_solve
from   math                import pi, e, sqrt
from   scipy.special       import erfinv, gammaln, gammainc, gammaincc
from   scipy.stats         import norm

import time

###
## Conduct a signal scan on a DataSet and store the results.
###
class SignalScan(ParametricObject):
    # p-values for [-2, -1, 0, 1, 2] sigma levels
    _siglevels = [0.022750, 0.158655, 0.5, 0.841345, 0.977250]

    # Get the background and signal variance contributions
    @Cache.Element("{self.Factory.CacheDir}", "var", "{self.Factory}", "{self.DataSet}", "{self.SigName}.npy")
    def _varCache(self, DataMom, SigMoms, P, Rs):
        n      = self.DataSet.N
        P1     = P[-1]

        Mb     = DataMom.copy()
        Mb[n:] = Rs[:-1].dot( SigMoms[:-1, n:] )
        Ms     = SigMoms[-1]
        Mp     = np.zeros_like(Mb)
        Mp[n:] = P1

        Mx     = self.Factory.MxTensor(Mb, Ms, Mp)

        return P1.dot( Mx[n:,n:] ).T

    # return the first three central moments assuming a signal contribution 't'.
    def _cmom(self, t):
        mu1    = t
        mu2    = self.B2 + mu1*self.S2 - mu1**2
        mu3    = self.B3 + mu1*self.S3 - 3*mu1*mu2 - mu1**3
        mu3    = np.maximum(mu2, mu3)

        return mu1, mu2, mu3

    # return the CPD parameters
    def _CPDpar(self, mu1, mu2, mu3, N):
        dN     = N - mu1 * self.DataSet.Nint
        a      = (mu2**3 / mu3**2) * self.DataSet.Nint
        b      = mu2 / mu3
        k      = a + b*dN + 0.5

        return a, k

    def CPDpdf(self, mu1, mu2, mu3, N):
        a, k   = self._CPDpar(mu1, mu2, mu3, N)
        return np.exp( (k-1)*np.log(a) - a - gammaln(k) )

    def CPDcdf(self, mu1, mu2, mu3, N):
        a, k   = self._CPDpar(mu1, mu2, mu3, N)
        return np.nan_to_num(gammaincc(k, a))

    def CPDcdf1m(self, mu1, mu2, mu3, N):
        a, k   = self._CPDpar(mu1, mu2, mu3, N)
        return np.nan_to_num(gammainc(k, a))

    # Return the upper limit for x (x in events)
    def UL(self, x):
        mu1, mu2, mu3  = self._cmom(self.t)
        
        if self.Lumi > 0:
            mu2  += self.LumiUnc * self.t**2

        pdf    = self.CPDpdf(mu1, mu2, mu3, x)
        c      = pdf.cumsum()
        idx    = np.searchsorted(c/c[-1], self.Confidence)
        ul     = self.t[idx] * self.DataSet.Nint

        return ul / self.Lumi if self.Lumi > 0 else ul

    # Return the moments equivalent to (-2, -1, 0, +1, +2)-sigma levels
    def _levels(self, Mu2, Mu3, npt=int(1e5), levels=(-2, -1, 0, 1, 2)):
        var    = Mu2 * self.DataSet.Nint
        r      = 4*sqrt(var)
        t      = np.linspace( -r, r, npt + 1)

        cdf    = self.CPDcdf(0, Mu2, Mu3, t)
        idx    = np.searchsorted(cdf, norm.cdf(levels) ) - 1

        return t[idx]

    # Implementation of iterator.
    def next(self):
        self.idx      += 1

        if self.idx >= len(self.Names):
            raise StopIteration

        Data           = self.DataSet
        DataMom        = getattr(Data, Data.attr)

        self.SigName   = self.Names[self.idx]
        SigMoms, P     = Data.NormSignalEstimators(self.SigName)   # Raw signals, estimators
        Rs             = P.dot(DataMom[Data.N:])                   # Estimated signal moments

        # Get the background and signal variance contributions.
        P1             = P[-1]
        PB, PS, P2     = self._varCache(DataMom, SigMoms, P, Rs)

        # Second central moments for b-part, s-part and b-only
        self.B2        = P1.dot( PB )
        self.S2        = P1.dot( PS )
        Mu2            = multi_dot((P1, self.MxC[Data.N:,Data.N:], P1))

        # Third central moments for b-part, s-part and b-only
        self.B3        = P2.dot( PB )
        self.S3        = P2.dot( PS )
        Mu3            = multi_dot((P1, self.MxC[Data.N:,Data.N:], P2))
        Mu3            = max(Mu2, Mu3)

        # Store the output values and calculate limits
        i              = self.idx
        self.var       = Mu2 * Data.Nint
        pts            = self._levels(Mu2, Mu3)

        self.Mass  [i] = Data[self.SigName].Mass
        self.Yield [i] = Rs[-1] * Data.Nint
        self.Unc   [i] = sqrt(self.var)
        self.ObsLim[i] =   self.UL( self.Yield[i] )
        self.ExpLim[i] = [ self.UL( v ) for v in pts ]
        self.PValue[i] = self.CPDcdf1m(0, Mu2, Mu3, self.Yield[i])
        self.Sig   [i] = norm.isf(self.PValue[i])

        return self.SigName, self.Yield[i], self.Unc[i], self.ObsLim[i], self.ExpLim[i]

    # Initialize covariances for shared signals.  Transposes copies
    #   are stored in memory, because this optimizes memory access
    #   and greatly speeds up the sums in _setShortCov.
    def __iter__(self):
        sl       = self.DataSet.GetActive()
        SigMoms  = sum( self.DataSet[s].Moment * self.DataSet[s].Sig for s in sl )

        Data     = self.DataSet
        n        = self.DataSet.N
        DataMom  = getattr(Data, Data.attr)

        Mom      = DataMom.copy()
        if len(self.DataSet.GetActive()) > 0:
            Mom[n:]  = SigMoms[n:]
        self.MxC = self.Factory.CovMatrix(Mom)

        return self

    def __init__(self, Factory, DataSet, *arg, **kwargs):
        ParametricObject.__init__(self)

        self.Factory    = Factory
        self.DataSet    = DataSet
        self.Lumi       = float(kwargs.get("Lumi",             0.0))
        self.LumiUnc    = float(kwargs.get("LumiUnc",          0.0))

        self.Nmax       = float(kwargs.get("Nmax",             1e5))
        self.Npt        = float(kwargs.get("Npt",              1e6))
        self.Confidence = float(kwargs.get("ConfidenceLevel", 0.95))

        self.Names      = arg
        self.Mass       = np.zeros( (len(arg),   ) )
        self.Yield      = np.zeros( (len(arg),   ) )
        self.Unc        = np.zeros( (len(arg),   ) )
        self.ExpLim     = np.zeros( (len(arg), 5 ) )
        self.ObsLim     = np.zeros( (len(arg),   ) )
        self.PValue     = np.zeros( (len(arg),   ) )
        self.Sig        = np.zeros( (len(arg),   ) )

        # Initial idx and test points
        self.idx        = -1
        self.t          = np.linspace(0, self.Nmax, self.Npt) / self.DataSet.Nint

###
## Optimize hyperparameters on a specified DataSet.
###
class Optimizer(ParametricObject):
    # Transform the dataset to the specified hyperparameters
    def UpdateXfrm(self, reduced=True, **kwargs):
        self.update( kwargs )
        self.PriorGen.update(self.ParamVal)
        self.Prior.FromIter(self.PriorGen)

        Act        = self.DataSet.GetActive() if reduced else self.DataSet.Signals

        inm        = [ self.DataSet[name].Mom for name in Act ]
        outm       = self.Xfrm(self.DataSet.Mom, *inm, **self.ParamVal)
        Mom, sigs  = outm[0], outm[1:]

        for name, m in zip(Act, sigs):
            self.DataSet[name].MomX[len(m):] = 0
            self.DataSet[name].MomX[:len(m)] = m

        self.DataSet.MomX[:] = Mom
        self.DataSet.Full.Set (Mom=Mom)

    # Scan for the optimal number of moments.
    @Cache.Element("{self.Factory.CacheDir}", "LLH", "{self.Factory}", "{self}", "{self.DataSet}.json")
    def ScanN(self, reduced=True, **kwargs):
        L  = np.full((self.Factory["Ncheck"],), np.inf)
        D  = self.DataSet
        a  = kwargs.get("attr", "MomX")
        h  = D.Full.Mom[1] * sqrt(2)

        self.UpdateXfrm(reduced=reduced)

        for j in range(2, self.Factory["Ncheck"]):
            try:
                D.SetN(j, attr=a)

                dof  = (j**2 + j) / 2
                Raw  = D.TestS.LogP(D.Full)
                Pen  = float(j) * np.log(D.Nint/(dof*h*e)) / 2

                pdot()
            except (np.linalg.linalg.LinAlgError, ValueError):
                pdot(pchar="x")
                continue
            L[j] = Raw + Pen

        j = np.nanargmin(L)

        return j, L[j]

    # Optimize the hyperparameters
    @pstage("Optimizing Hyperparameters")
    def FitW(self, initial_simplex=None):
        ini = [ self.Factory[p] for p in self.Factory._fitparam ]
        res = minimize(self.ObjFunc, ini, method='Nelder-Mead', options={'xatol': 1e-2, 'initial_simplex' : initial_simplex})
        par = dict(zip(self.Factory._fitparam, res.x))

        # Now set to the best parameters
        self.update( par )
        self.UpdateXfrm()
        
        N, L = self.ScanN()

        return N, L, par

    # Scan through a grid of hyperparameters
    @pstage("Scanning Hyperparameters")
    def ScanW(self, Ax, Lx, Ad, Ld):
        D        = self.DataSet
        LLH      = []
        P        = []

        for ax, lx, ad, ld in zip(Ax.ravel(), Lx.ravel(), Ad.ravel(), Ld.ravel()):
            if ad != self.Factory["Alpha"] or ld != self.Factory["Lambda"] or not hasattr(D, "Full"):
                self.Factory.update( {'Alpha': ad, 'Lambda': ld} )
            D.Decompose(xonly=True)

            LLH.append( self.ObjFunc((ax, lx)) )
            P.append  ( { 'Alpha' : ax, 'Lambda' : lx } )

        LLH = np.asarray(LLH)

        return LLH.reshape(Ax.shape), P[ np.nanargmin(LLH) ]

    def ObjFunc(self, arg):
        prst()
        pstr(str_nl % "Nint" + "%.2f" % self.DataSet.Nint)
        pstr(str_nl % "Neff" + "%.2f" % self.DataSet.Neff)
        pstr(str_nl % "Signals" + " ".join(self.DataSet.GetActive()))
        pstr(str(self))

        pini("MOMENT SCAN")

        self.update( dict( zip(self.Factory._fitparam, arg) ) )
        self.Xfrm.update(self.Factory.ParamVal)
        j, L       = self.ScanN()

        pstr(str_nl % "Nmom" + "%2d"  % j)
        pstr(str_nl % "LLH"  + "%.2f" % L)
        pstr(str_nl % "Nfev" +  "%d / %d"  % (self.Nfev, self.Nfex))

        self.Nfev += 1

        return L

    def __init__(self, Factory, DataSet, **kwargs):
        self.DataSet   = DataSet
        self.Factory   = Factory
        self.Nfev      = 0
        self.Nfex      = 0

        # Copy parameters from the Factory object.
        self._param    = Factory._fitparam
        self._fitparam = Factory._fitparam
        ParametricObject.__init__(self, **Factory.ParamVal)

        self.Prior     = TruncatedSeries(self.Factory, np.zeros((Factory["Nbasis"],)), DataSet.Neff/DataSet.Nint, Nmax=2 )
        self.PriorGen  = Factory.Pri()
        self.Xfrm      = self.Factory.Xfrm()

###
## An object to hold data to decompose along with signal objects.
###
class DataSet(ParametricObject):
    def AddParametricSignal(self, name, func, **kwargs):
        if name in self:
            return
        self[name] = ParametricSignal(self.Factory, name, func, **kwargs)
        self.Signals.append(name)

    def DelSignal(self, name):
        try:
            list.remove(name)
        except ValueError:
            pass
        try:
           del self[name]
        except ValueError:
           pass

    def GetActive(self, *args):  return [ n for n in self.Signals if self[n].Active or self[n].name in args ]

    # Decompose the dataset and active signals
    def Decompose(self, reduced=True, xonly=False, cksize=2**20):
        pini("Data Moments")
        N             = self.Factory["Nxfrm"]
        Nb            = self.Factory["Nxfrm"] if xonly else self.Factory["Nbasis"]

        self.Mom      = np.zeros((self.Factory["Nbasis"],))
        self.MomX     = np.zeros((self.Factory["Nxfrm"],))

        self.Mom[:Nb] = self.Factory.CachedDecompose(self.x, self.w, str(self.uid), cksize=cksize, Nbasis=Nb)

        self.Full     = TruncatedSeries(self.Factory, self.Mom, self.Neff, Nmax=N )
        self.TestS    = TruncatedSeries(self.Factory, self.Mom, self.Neff, Nmax=N )
        self.TestB    = TruncatedSeries(self.Factory, self.Mom, self.Neff, Nmax=N )
        pend()

        Act = self.GetActive() if reduced else self.Signals
        for name  in Act:
            pini("%s Moments" % name)
            self[name].Decompose(cksize=cksize, Nbasis=Nb)
            pend()

    # Solve for the raw signal estimators.  Use Cholesky decomposition,
    #   as it is much faster than the alternative solvers.
    @pstage("Preparing Signal Estimators")
    def PrepSignalEstimators(self, reduced=True, verbose=False):
        D     = getattr(self, self.attr).copy()
        n, N  = self.N, D.size
        Act   = self.GetActive() if reduced else self.Signals

        D[n:] = 0
        LCov  = self.Factory.CovMatrix(D)[n:N,n:N]
        reg   = np.diag(LCov).mean() / sqrt(self.Nint)
        Ch    = cho_factor( LCov + reg*np.eye(N-n))

        if verbose: pini("Solving")
        for name in Act:
            sig     = self[name]
            sig.Sig = getattr(sig, self.attr)                           # set the moments to use.
            sig.Res = sig.Sig[n:]                                       # sig res
            sig.Est = cho_solve(Ch, sig.Res.T)

            if verbose: pdot()
        if verbose: pend()

    # Solve for the normalized signal estimators
    def NormSignalEstimators(self, *extrasignals):
        sl   = self.GetActive(*extrasignals)

        Sig  = np.array([ self[s].Sig for s in sl ])                    # raw signals
        Res  = np.array([ self[s].Res for s in sl ])                    # sig residuals
        Est  = np.array([ self[s].Est for s in sl ])                    # sig raw estimators

        P    = solve ( Est.dot(Res.T), Est)                             # normalized estimators

        return Sig, P

    # Extract the active signals
    def ExtractSignals(self, Data, *extrasignals):
        N            = self.N
        Sig, P       = self.NormSignalEstimators(self, *extrasignals)
        R            = P.dot(Data[self.N:])

        self.FullSig = R.dot(Sig)
        self.P       = P

        for i, name in enumerate( self.GetActive(*extrasignals) ): 
            self[name].Moment = R[i]
            self[name].Yield  = R[i] * self.Nint

    # Return covariance matrix (in events)
    def Covariance(self):
        sigs = self.GetActive()
        N    = self.N

        if len(sigs) == 0:
            self.Cov  = [[]]
            self.Unc  = []
            self.Corr = [[]]
        else:
            Mf        = self.Factory.CovMatrix( getattr(self, self.attr) )
            self.Cov  = self.Nint * multi_dot (( self.P, Mf[N:,N:], self.P.T ))
            self.Unc  = np.sqrt(np.diag(self.Cov))
            self.Corr = self.Cov / np.outer(self.Unc, self.Unc)

        for n, name in enumerate(sigs):
            self[name].Unc = self.Unc[n]

    # Set the number of moments
    def SetN(self, N, attr="MomX"):
        self.N = N
        Mom    = getattr(self, attr)
        isSig  = np.arange(Mom.size) >= N
        dMom   = Mom * (~isSig)

        self.attr = attr

        if len(self.GetActive()) > 0:
            self.PrepSignalEstimators(verbose=False)
            self.ExtractSignals(Mom)
        else:
            self.FullSig = np.zeros_like(dMom)

        self.TestB.Set(Mom=dMom - self.FullSig * (~isSig), Nmax=N)
        self.TestS.Set(Mom=dMom + self.FullSig * ( isSig) )

    def __init__(self, x, Factory, **kwargs):
        w              = kwargs.get('w', np.ones(x.shape[-1]))
        sel            = (w != 0) * (x > Factory["x0"] )
        self.w         = np.compress(sel, w, axis=-1)
        self.x         = np.compress(sel, x, axis=-1)

        # Record some vital stats
        self.uid       = self.x.dot(self.w)  # Use the weighted sum as a datset identifier.
        self.Nint      = self.w.sum()
        self.Neff      = self.Nint**2 / np.dot(self.w, self.w)
        self.w        /= self.Nint

        self.Factory   = Factory
        self.Signals   = []

        ParametricObject.__init__(self, **Factory.ParamVal)

    # Format as a list of floats joined by '_'.  The
    #  str(float()) construction ensures that numpy
    #  singletons are p.copy()rinted consistently
    def __format__(self, fmt):
        id_str = [ str(self.uid) ] + self.GetActive()
        return "_".join( id_str )

###
## A parametric signal model.
###
class ParametricSignal(object):
    def Decompose(self, cksize=2**20, Nbasis=0, **kwargs):
        Mom                 = self.CachedDecompose(cksize, Nbasis=Nbasis, **kwargs)
        self.Mom[:len(Mom)] = Mom
        self.Mom[len(Mom):] = 0
        self.MomX[:]        = 0

    # Decompose the signal sample data.
    @Cache.Element("{self.Factory.CacheDir}", "Decompositions", "{self.Factory}", "{self.name}.npy")
    def CachedDecompose(self, cksize=2**20, **kwargs):
        Nb           = kwargs.pop("Nbasis", 0)
        Mom          = np.zeros((self.Factory["Nbasis"],))
        sumW         = 0.0
        sumW2        = 0.0
        Neff         = 0
        cksize       = min(cksize, self.Npt)
        Fn           = self.Factory.Fn(np.zeros((cksize,)), w=np.zeros((cksize,)),
                                       Nbasis=Nb if Nb > 0 else self.Factory["Nbasis"])

        while Neff < self.Npt:
            Fn['x']    = np.random.normal(loc=self.mu, scale=self.sigma, size=cksize)
            Fn['w']    = self.func(Fn['x']) / self._gauss(Fn['x'])

            k          = np.nonzero(Fn['x'] <= self.Factory["x0"])
            Fn['w'][k] = 0
            Fn['x'][k] = 2*self.Factory["x0"]

            sumW      += Fn['w'].sum()
            sumW2     += np.dot(Fn['w'], Fn['w'])
            Neff       = int(round( (sumW*sumW)/sumW2 ))

            for D in Fn: Mom[D.N] += D.Moment()
            pdot()

        return Mom / sumW

    # Gaussian PDF
    def _gauss(self, x):
        u, s = self.mu, self.sigma
        return np.exp( -0.5*( (x-u)/s )**2 ) / sqrt(2*pi*s*s)

    # Set a signal model and generate
    def __init__(self, Factory, name, func, mu, sigma, **kwargs):
        self.Factory = Factory
        self.name    = name
        self.func    = func
        self.mu      = mu
        self.sigma   = sigma
        self.Mass    = float(kwargs.get("Mass", self.mu)) 
        self.Npt     = int(  kwargs.get("NumPoints", 2**22))
        self.Active  = bool( kwargs.get("Active", False))

        self.Moment  = 0
        self.Yield   = 0

        self.Mom     = np.zeros((self.Factory["Nbasis"],))
        self.MomX    = np.zeros((self.Factory["Nxfrm"],))

###
## A truncated orthonormal series
###
class TruncatedSeries(object):
    # Evaluate series
    def __call__(self, x, trunc=False):
        Mom      = self.MomAct if trunc else self.MomU
        Val      = np.zeros_like(x)
        w        = np.ones_like(x)

        for D in self.Factory.Fn(x, w):
            if D.N >= self.Nmin:
                Val  += D.Values() * Mom[D.N]

        return Val

    # Get the common index range between self and other
    def _ci(self, othr):
        return ( max(self.Nmin, othr.Nmin),
                 min(self.Nmax, othr.Nmax))

    # Get the entropy of this TruncatedSeries.
    def Entropy(self):
       j     = kwargs.get('j', self.Nmin)
       k     = kwargs.get('k', self.Nmax)
       k     = min(k, self.Ncov/2)
       Cov   = self.Cov[j:k,j:k]

       return slogdet(2*pi*e*Cov)[1] / 2

    # Dkl(othr||self) --> prior.KL(posterior). If specified, scale the
    #  statistical precision of 'self' by Scale
    def KL(self, othr):
        j, k        = self._ci(othr)
        k           = min(k, self.Ncov/2, othr.Ncov/2)
        delta       = self.MomAct[j:k] - othr.MomAct[j:k]

        ChSelf      = cho_factor(self.Cov[j:k,j:k])
        h           = cho_solve(ChSelf, delta)
        r           = cho_solve(ChSelf, othr.Cov[j:k,j:k])

        return (np.trace(r) + delta.dot(h) - slogdet(r)[1] - k + j) / 2

    #Log-likelihood of othr with respect to self.
    def Chi2(self, othr):
        j, k        = self._ci(othr)
        k           = min(k, self.Ncov/2)
        delta       = self.MomAct[j:k] - othr.MomAct[j:k]

        Ch          = cho_factor(self.Cov[j:k,j:k])
        h           = cho_solve(Ch, delta)

        return delta.dot(h) / 2

    # Negative log-likelihood of othr with respect to self.
    def LogP(self, othr):
        j, k        = self._ci(othr)
        k           = min(k, self.Ncov/2)
        delta       = self.MomAct[j:k] - othr.MomAct[j:k]

        Ch          = cho_factor(self.Cov[j:k,j:k])
        h           = cho_solve(Ch, delta)
        l           = 2*np.log(np.diag(Ch[0])).sum()

        return (  (k-j)*np.log(2*pi) + l + delta.dot(h)) / 2

    # Set the number of active moments.
    def Set(self, **kwargs):
        self.Nmin   = kwargs.get('Nmin', self.Nmin)
        self.Nmax   = kwargs.get('Nmax', self.Nmax)
        self.Mom    = kwargs.get('Mom',  self.Mom).copy()
        Ncov        = self.Mom.size

        # Truncate or pad with zeros as necessary.  Keep a copy of the original.
        self.MomU = self.Mom.copy()
        self.Mom.resize( (self.Factory["Nbasis"],), )

        R           = np.arange(self.Mom.size, dtype=np.int)
        self.MomAct = self.Mom * (self.Nmin <= R) * (R < self.Nmax)

        # Build covariance matrix
        N           = self.Cov.shape[0]
        self.Mx     = self.Factory.MomMx( self.Mom, self.Nmin, self.Nmax, out=self.Mx)
        self.Cov    = (self.Mx - np.outer(self.MomAct[:N], self.MomAct[:N])) / self.StatPre
        self.Ncov   = Ncov

    # Set the moments from an iterator
    def FromIter(self, iter):
        Mom         = np.asarray( [ D.Moment() for D in iter ] )
        self.Set(Mom=Mom)

    # Initialize by taking the decomposition of the dataset `basis'.
    #   Store all moments.  The `active' moments are [self.Nmin, self.Nmax)
    def __init__(self, Factory, Moments, StatPre=1, **kwargs):
        self.Factory = Factory
        self.StatPre = StatPre
        self.Nmin    = kwargs.get("Nmin", 1)
        self.Nmax    = kwargs.get("Nmax", Factory["Nbasis"])

        Nbasis       = Factory["Nbasis"]
        Nxfrm        = Factory["Nxfrm"]

        self.Mx      = np.zeros( (Nxfrm, Nxfrm) )
        self.Cov     = np.zeros( (Nxfrm, Nxfrm) )     # Covariance (weighted)
        self.MomAct  = np.zeros( (Nbasis,) )          # Weighted, truncated moments
        self.MomU    = Moments.copy()
        self.Mom     = Moments.copy()

        self.Set()

