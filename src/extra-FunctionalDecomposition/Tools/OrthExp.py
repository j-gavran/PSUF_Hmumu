'''
An implementation of an orthonormal basis: the orthogonal exponentials. The
implementation relies on the machinery for three-term recurrence relations
implemented in Tools.Base.

The orthonormal exponentials are constructed by orthogonalizing the set of
functions:

    F(n) = exp( -n*z )
    z    = (x-x0)/Lambda) ** Alpha

(ignoring normalization constants) where

    n in a positive integer indexing the functions.
    x0 is a real number specifying the lower mass cut.
    Alpha is a positive, dimensionless real numbers.
    Lambda is positive real number with dimensions of energy.

It turns out that this set of functions is complete with respect to real-valued
functions who tend to zero as their argument goes to infinity.

Unfortunately, it is difficult to deal with this series directly: the functions
F(n) are nearly degenerate, and working with more than a small handful of terms
is numerically impossible.

Fortunately, they can be used to produce an orthonormal basis which does not
have these numerical issues and is extraordinarily convenient for modeling the
falling spectra frequently seen in high-energy physics.

This orthonormal basis, the orthonormal exponentials, can be written using
three-term recurrence relations:

    E(n+1) = a*exp( -n*z )*E(n) + b*E(n) + c*E(n-1)

The recurrence relations are fast and numerically stable, and can readily be
used to evaluate n ~ 10000 and higher - the upper limit has not been explored.
They are particularly advantageous when all functions from n=1 to n=N must be
evaluated, as each F_n must naturally be computed along the way to F_N!

Hyperparameters
---------------

x0, Alpha and Lambda are hyperparameters that specify a coordinate
transformation that converts the dimensioned coordinate x into a nondimensional
coordinate z.  The hyperparameters are arbitrary - every choice of Alpha and
Lambda results in a complete basis.  Some choices are more effective than
others, though, and so Tools.Decomp provides machinery for making some (semi-)
automated optimizations of these.

The OrthExp Module
------------------

This module implements the orthonormal exponentials and transformations between
different choices of hyperparameters.  It relies on the base classes defined
in Tools.Base.  The following classes are provided:

    ExpDecompFn        : Evaluate the orthonormal exponentials at a list of
                         points. Also compute the series coefficients of this
                         dataset in the orthonormal exponentials.
    ExpDecompMxForm    : Use the recursion relations to take a 1-dimensional
                         moment vector and convert it into its corresponding
                         matrix form.
    ExpPrior           : Return the moments of the first orthonormal
                         exponential, i.e. (0, 1, 0, 0, 0, ...) with a proper
                         normalization.
    ExpDecompTransform : Transform a moment vector created with one set of
                         hyperparameters into the moment vector with a
                         different set of hyperparameters.
    ExpDecompFactory   : Implementation of Tools.Base.Factory to store the
                         hyperparameters and produce instantiations of the
                         above objects having those hyperparameters.

Users should normally instantiate an ExpDecompFactory and use that to produce
the other objects.

'''
import numpy               as np
import numexpr             as ne
import Tools.Base          as Base

from   Tools.PrintMgr      import *
from   numpy.core.umath    import euler_gamma
from   scipy.special       import exprel
from   fractions           import Fraction
from   math                import log, sqrt, ceil

#### Orthogonal exponentials as functions
class ExpDecompFn (Base.Basis):
    '''
    The orthonormal exponentials as functions.
    
    This class is instantiated as

        En = ExpDecompFn(x, w, **kwargs)

    where x and w are one-dimensional, same-sized ndarrays containing the
    positions and their weights, respectively.  The hyperparameters
    (x0, Lambda, and Alpha) must be supplied as keyword arguments, as must
    Nbasis (the maximum N to evaluate).

    If ExpDecompFn is produced by an ExpDecompFactory object (this is generally
    the correct apprach), then the hyperparameters stored in the factory object
    will automatically be supplied.

    ExpDecompFn acts as an iterator, so the values [ E1(x), E2(x), E3(x), ... ]
    can be stepped through in order like this:
    
        for x in En:
            print x.N, x.Values()

    x.Values() returns an array with the same shape as x, with the values of
    the N'th orthonormal exponential at each x.  If x is a dataset, its moments
    can be obtained using the Moment() method:

        for x in En:
            print x.N, x.Moment()
    '''
    _param = Base.Basis._param + ( "x0", "Lambda", "Alpha" )

    # User-facing functions.
    def Values(self):           return self['t'][ self.N % 2 ] * np.sqrt(self.NormSq(self.N)) * self['xf']

    def Zeros (self, shape=()): return np.zeros(shape + self['x'].shape)

    def NormSq(self, N):        return 2. / N if N > 0 else 0
    def Base0 (self, out):      out[:] = 0
    def Base1 (self, out):      out[:] = self['rz']
    def Raise (self, out):      return ne.evaluate( "exp( -(x-x0) * xf / Alpha)", self.ParamVal, out=out)
    def Xfrm  (self, out):      return ne.evaluate( "(Alpha/Lambda) * ((x-x0)/Lambda)**(Alpha-1)", self.ParamVal, out=out)

    # The recurrence relation itself.
    def Recur(self, N, En, Ep, out):
        A      = float(4*N + 2) / N
        B      = float(4*N)     / (2*N - 1)
        C      = float(2*N + 1) / (2*N - 1)

        ne.evaluate( "A*rz*En - B*En - C*Ep", global_dict = self, out=out)

    def Moment(self):           return np.dot(self['t'][ self.N % 2 ], self['w']) * sqrt(self.NormSq(self.N))

    def __init__(self, x, w, **kwargs):
        Base.Basis.__init__(self, **kwargs)

        self['x'] = x
        self['w'] = w

    def Reinit(self, x, w):
        self['x'] = x
        self['w'] = w

#### Class to elevate a decomposition to it's matrix form
class ExpDecompMxForm (Base.Basis):
    '''
    A class to elevate a decomposition from its vector form to its matrix form.

    This class is initialized with a moment vector <orig>.  It implements the
    iterator interface, and each step of the iteration returns the  moment
    vectors of <E_N * orig>.  That is, if <orig> is the decomposition of some
    function F(x), then each iteration returns the decomposition of
    F(x) * E_n(x).

    The primary utility is that this allows easy construction of the matrix
    representation of <orig>, and hence provides a convenient mechanism to
    construct the covariance matrix between the elements of the <orig> vector.

    Instantiate this as

        MxForm = ExpDecompMxForm(*mom, Nbasis=N)

    where N is the maximum n to evaluate.  If multiple original moment vectors
    are supplied, then they are iterated through simultaneously, e.g.

        MxForm = ExpDecompMxForm(origA, origB, origC, Nbasis=N)

        for x in MxForm:
            V = x.Values()

            print "Decomposition of A*E_%d:" % x.N, V[0]
            print "Decomposition of B*E_%d:" % x.N, V[1]
            print "Decomposition of C*E_%d:" % x.N, V[2]
    '''
    def Values(self):           return self['t'][ self.N % 2 ][:,1:self["Nbasis"]+1]
    def Zeros (self, shape=()): return np.zeros(shape + self['x'].shape)

    def Base0 (self, out):      out[:] = 0
    def Base1 (self, out):      out[:] = 0; self.Recur(0, self['x'], self['t'][0], out)
    def Raise (self, out):
        n           = np.arange(self["Nrow"]-2, dtype=np.float)

        out         = np.zeros((2, self["Nrow"]))
        out[0,1:-1] = np.sqrt(n*(n+1)) / (      2*(2*n+1) ) #sub/sup diagonal
        out[1,1:-1] =           2*n**2 / ((2*n-1)*(2*n+1) ) #diagonal
        out[1,-2]   = 0

        return out
    def Xfrm  (self, out):      return None

    # The recurrence relation itself.
    def Recur(self, N, En, Ep, out):
        N = float(N)
        if N > 0:
            e  = sqrt(     N / (N+1) )
            f  = sqrt( (N-1) / (N+1) )

            A  = e * (4*N + 2) / N
            B  = e * (4*N)     / (2*N - 1)
            C  = f * (2*N + 1) / (2*N - 1)
        else:
            A,B,C = 2.0,0.0,0.0

        dg  = self['rz'][1,1:-1]
        us  = self['rz'][0,1:-1]
        ds  = self['rz'][0,0:-2]

        x   = En[:,1:-1]
        u   = En[:,2:  ]
        d   = En[:, :-2]
        p   = Ep[:,1:-1]

        return ne.evaluate("A*(dg*x + us*u + ds*d) - B*x - C*p ", out=out[:,1:-1])

    def __init__(self, *mom, **kwargs):
        Base.Basis.__init__(self, **kwargs)

        Nmom          = max([x.size for x in mom])
        Nbasis        = self["Nbasis"]
        Nrow          = Nmom + Nbasis + 2

        self["Nrow"]  = Nrow
        self['x']     = np.zeros((len(mom), Nrow))

        for n,x in enumerate(mom):
            self['x'][n,1:len(x)+1] = x

#### Decomposition of a simple exponential e**(-x/Lambda) 
class ExpPrior (Base.Basis):
    '''
    Return the moments of a distribution consisting of only the first
    orthonormal exponential, properly normalized.
    '''
    _param = Base.Basis._param + ( "Alpha", "Lambda" )

    def Moment(self):
        if self.N != 1:
            return 0
        return 1./sqrt(2)

#### Transformation matrix generator for orthogonal exponentials.
class ExpDecompTransform(Base.Transform):
    '''
    Transformation object for the orthonormal exponentials.

    ExpDecompTransform implements transformations on Alpha and Lambda for the
    orthonormal exponentials (x0 cannot be transformed).

    Mandatory argument on initialization are Alpha, Lambda, and Nxfrm. Nxfrm
    specifies the size of the transformation matrices (i.e. the maximum number
    of moments that can be transformed). Alpha and Lambda specify the
    hyperparameters for the starting point of the transformation.

    Initialize and use it like this:

        Xfrm   = ExpDecompTransform(Alpha=AlphaIni, Lambda=LambdaIni, Nxfrm=N)
        NewMom = Xfrm(OldMom, Alpha=AlphaNew, Lambda=LambdaNew)

    The remaining methods in ExpDecompTransform exist to compute the
    transformation matrices, and are not intended for direct use. If AlphaNew
    is unspecified, the transformation assumes that AlphaNew = AlphaOld, and
    likewise for Lambda.
    '''
    _param = Base.Transform._param + ("Lambda", "Alpha")

    # Return a high-accuracy rational approximation of log(n) for integer n
    # Use recursion and cache results to enhance performance.
    def _log(self, n, numIni=96, numMin=48):
        if n <= 1: return 0
        if n not in self:
            num     = max( numIni - (n - 2), numMin )
            s       = [ Fraction(2, 2*k+1) * Fraction(1, 2*n-1) ** (2*k+1) for k in range(num) ]
            self[n] = self._log(n-1) + sum( s )
        return self[n]

    # Recursion relations for series coefficients
    def CoeffOrth(self, n):
        r = [ 0, Fraction((-1)**(n+1) * n) ]

        for m in range(1, n):
            r.append(  Fraction(m**2 - n**2, m*(m+1)) * r[-1]  )
        return r

    def CoeffDeriv(self, n):
        r = [ 0, Fraction((-1)**n * n) ]

        for m in range(1, n):
            r.append(  Fraction(m**2 - n**2, m**2) * r[-1]  )
        return r

    def Norm (self, *arg):   return sqrt(np.prod([ 2*a for a in arg]))

    # Infinitesimal transformations
    @Base.Transform.OrthogonalizeHankel
    def Alpha    (self, n):   return (1 - Fraction(euler_gamma) - self._log(n)) / n**2 if n > 0 else 0
    @Base.Transform.OrthogonalizeHankel
    def Lambda   (self, n):   return Fraction(1, n**2)    if n > 0 else 0

    # Infinitesimal transformation parameters
    # 'fin' is a dict with the final value for all parameter;
    # 'ini' is a dict with the initial value for all parameters
    def XfPar (self, fin, ini):
        r  = log( fin["Lambda"] / ini["Lambda"] )
        yc = log( fin["Alpha"]  / ini["Alpha"] )
        ys = -fin["Alpha"] * r / exprel(yc)

        return { "Alpha": yc, "Lambda": ys }


#### A decomposer factory using the orthonormal exponentials
class ExpDecompFactory ( Base.DecompFactory ):
    '''
    A factory object for the orthonormal exponentials.

    The ExpDecompFactory is the preferred way to store hyperparameter values
    and to create the ExpDecompFn, ExpDecompMxForm, and ExpDecompTransform
    objects.
    
    A simple example, assuming that 'x' and 'w' are a list of points and the
    corresponding weights, respectively:

        Factory = ExpDecompFactory(Alpha=AlphaOld, Lambda=LambdaOld, x0=x0
                                   Nbasis=Nbasis, Nxfrm=Nxfrm, Ncheck=Ncheck)

        Mom       = [ r.Moment() for r in Factory.Fn(x, w) ]
        MomCov    = Factory.CovMatrix( Mom )

        Xfrm      = Factory.Xfrm()
        MomNew    = Xfrm(Mom, Alpha=AlphaNew, Lambda=LambdaNew)
        MomNewCov = Factory.CovMatrix(MomNew)

    This

        1. Creates a factory object with hyperparameters (AlphaOld, LambdaOld).
        2. Computes the moments (i.e. the decomposition) of the dataset x.
        3. Computes the covariance of those moments.
        4. Transforms the moments to the hyperparameters (AlphaNew, LambdaNew).
        5. Computes the covariance matrix in the transformed basis.

    The ExpDecompFactory is the preferred top-level interface for using all of
    the implementation of the orthonormal exponentials.
    '''
    _param    = tuple(set( Base.DecompFactory._param
                          + ExpDecompFn._param
                          + ExpDecompMxForm._param
                          + ExpPrior._param))
    _fitparam = ("Alpha", "Lambda")

    # Methods to create the function, matrix and weight objects with correct parameters.
    def Fn    (self, x, w, **kw): return ExpDecompFn        (x, w, **self._arg(kw) )
    def MxForm(self, *x,   **kw): return ExpDecompMxForm    (*x,   **self._arg(kw) )
    def Pri   (self,       **kw): return ExpPrior           (      **self._arg(kw) )
    def Xfrm  (self,       **kw): return ExpDecompTransform (      **self._arg(kw) )
 
