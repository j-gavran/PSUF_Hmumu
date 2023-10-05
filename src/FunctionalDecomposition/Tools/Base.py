'''
Base classes for implementing an orthonormal basis. These classes are intended
to enable a convenient and uniform implementation of any number of orthonormal
bases in terms of their recursion relations.  At present, the only basis
actually implemented is the orthonormal exponentials.  However, this structure
will make it convenient to implement, e.g. Legendre polynomials for an angular
analysis, or any other othonormal function set that can be defined in terms
of its recursion relations.

This module consists of four classes:

  ParametricObject: A base class that implements named parameters that can be
                    marked as fittable or fixed. Used as a base class for many
                    FD classes.

  DecompFactory:    Store all parameters for the basis and provide methods to
                    create various basis objects with the correct parameters.
                    Also provides several convenience methods for performing
                    decompositions and construction covariance matrices.

  Basis:            An abstract iterator object used by child classes to
                    implement three-term recursion relations.

  Transform:        An abstract base class to implement operations to transform
                    between different sets of basis hyperparameters.
'''

import numpy               as np
import numexpr             as ne
import Tools.CacheMgr      as Cache

from   Tools.PrintMgr      import *
from   scipy.sparse.linalg import expm_multiply
from   scipy.linalg        import solve
from   multiprocessing     import Pool
from   abc                 import abstractmethod
from   os                  import environ

class ParametricObject(object):
    '''
    A base class that implements a collection of named parameters.
    ParametricObject is _NOT_ intended for direct usage, rather it provides
    shared methods for its subclasses.  It uses the __getitem__ / __setitem__
    interface, so access properties as if it were a dict.
    
    The valid parameters are defined in two class variables:

        _param    : a list of all named parameters
        _fitparam : a list of which parameters are fittable
    
    These must be overridden by the subclass.  When instantiating the subclass,
    initial values must be specified by name for ALL parameters in _param, e.g.

        class MyClass(ParametricObject):
          _param = ( "Prop1", "Prop2", "ThirdProp" )
          ...

    must be initialized like

        Instance = MyClass(Prop1="val1", Prop2="val2", ThirdProp="val3")

    , or a KeyError will be thrown.
    '''

    _param    = ( )
    _fitparam = ( )
      
    # Update the parameters with the values from the dict 'ind'.
    def update(self, ind):  return  self.ParamVal.update(ind)
    # Return all parameter / value pairs.
    def items (self):       return  self.ParamVal.items()

    # Get parameter 'k' by name.
    def __getitem__ (self, k):       return self.ParamVal[k]
    # Set parameter 'k' by name
    def __setitem__ (self, k, v):    self.ParamVal[k] = v
    # Check if paramameter 'k' exists.
    def __contains__(self, k):       return k in self.ParamVal

    # Get the argument dict, updated by the contents of 'kw'
    def _arg(self, kw):
        d = self.ParamVal.copy()
        d.update(kw)
        return d

    # kwargs must include ALL parameters listed in self._param.
    def __init__(self, **kwargs):
        self.ParamVal = { p : kwargs[p] for p in self._param }

    # Pretty-formatted string representation.
    def __str__(self):
        s  = [ str_nl % "className" + type(self).__name__ ]
        s += [ str_nl % k + str(self[k]) for k in self._param if k not in self._fitparam ]
        s += [ str_el % k + "%.3f" % self[k] for k in self._fitparam ]

        return '\n'.join(s)

    # Format as a list of floats joined by '_'.
    def __format__(self, fmt):
        # The str(float()) construction ensures that numpy
        # singletons are printed nicely.

        s = []
        s += [ str(float(self[k])) for k in self._fitparam ]
        s += [ str(self[k])        for k in self._param if k not in self._fitparam ]

        return "_".join(s)

class DecompFactory(ParametricObject):
    '''
    DecompFactory is a class to store the hyperparameters for an orthonormal
    function set and conveniently create various decomposition instances
    with those parameters. It is an abstract class - it should be extended
    by a concrete subclass that defines the correct hyperparameters and methods
    for the orthonormal function set.

    Three parameters are defined in _param:
        Nbasis : The total number of basis functions to consider
        Nxfrm  : The number of basis functions to use for fast transformations
                 during the hyperparamer scan.
        Ncheck : The maximum number of basis functions to consider for
                 inclusion in the background-only set.
    Subclasses should extend _param with any additional parameters that they
    require, and set _fitparam as appropriate.

    Furthermore, four abstract factory methods must be implemented by the subclass:
        Fn     (self, x, w, **kw)
        MxForm (self,       **kw)   
        Pri    (self,       **kw)   
        Xfrm   (self,       **kw)   
    See the respective docstrings of each for more detail.

    For an example implementation, see Tools.OrthExp.ExpDecompFactory.

    DecompFactory additionally provides some convenience methods for decomposing datasets
    and calculating covariance matrices.
    '''
    _param    = ("Nbasis", "Nxfrm", "Ncheck")

    @abstractmethod
    def Fn     (self, x, w, **kw):
        '''
        Return a subclass of Basis that implements an orthonormal basis function.
        'x' is an ndarray with the x-values at which to evaluate the function.
        'w' is an ndarray with the weights corresponding to each 'x'.
        These must have the same shape; if the weights are not used, just pass
        np.ones_like(x) as a weight.
        
        The kwargs can be used to override the hyperparameter values that are
        set in this DecompFactory.
        '''
        return

    @abstractmethod
    def MxForm (self, *mom, **kw):
        '''
        Return a subclass of Basis that takes in a list of moment vectors and
        returns the corresponding moment matrices row-by-row.  Each positional
        argument is a single moment-vector.

        The kwargs can be used to override the hyperparameter values that are
        set in this DecompFactory.
        '''
        return

    @abstractmethod
    def Pri    (self,       **kw):
        '''
        Return the moments of the prior distribution moment-by-moment.

        The kwargs can be used to override the hyperparameter values that are
        set in this DecompFactory.
        '''
        return

    @abstractmethod
    def Xfrm   (self,       **kw):
        '''
        Return a transformation object.  The transformation object takes a
        moment vector with the current choice of hyperparameter values and
        transforms it into a vector with an alternate choice of hyperparameter
        values.  See Tools.Base.Transform for a more detailed description
        of the transformation object.

        The kwargs can be used to override the hyperparameter values that are
        set in this DecompFactory.
        '''
        return

    def MomMx  (self, Mom, i, j, **kwargs):
        '''
        Return the matrix form of Mom using the fast matrix cache stored in
        this DecompFactory object. This method is only able to use 'Nxfrm'
        moments. It is intended to be used only during the hyperparameter scan.

        The parameters are:
          Mom:     The moment vector to convert to matrix form.
          [i, j) : The range of moments to include. 'i' is included; 'j' is not

        Optional kwargs are:
          N:       size of the matrix to return (N by N)
          out:     ndarray to store the output.  Otherwise, return a new allocation.

        '''
        out = kwargs.get("out", None)
        N   = kwargs.get("N", self["Nxfrm"])
        j   = min(j, self.TDot.shape[-1])
        return np.dot( self.TDot[:N,:N,i:j], Mom[i:j], out=out)

    def MxTensor(self, *Mom):
        '''
        Return the matrix form of each moment vector listed in the positional
        arguments.  This uses the full MxForm machinery, so it can work on
        moment vectors of any length.  The size of the returned covariance matrices
        is N by N, where N is the length of the first positional argument.
        '''
        Nb  = Mom[0].size
        Ct  = np.zeros((Nb, Nb, len(Mom)))
        for r in self.MxForm(*Mom, Nbasis=Nb):
            Ct[r.N]  += r.Values().T
        return Ct

    def CovMatrix(self, Mom):
        '''
        Return the covariance matrix association with the moment vector Mom.
        This method works on moment vectors of any length.  The returned
        covariance matrix is N by N, where N is the length of Mom.
        '''
        Nb      = Mom.size
        Ct      = -np.outer(Mom[:Nb], Mom[:Nb])
        Ct[0,0] = 1
        for r in self.MxForm(Mom, Nbasis=Nb):
            Ct[r.N]  += r.Values()[0]
        return Ct

    def OpMatrix(self, N, **kwargs):
        '''
        Create and return the fast matrix cache.  This calculates the N by N
        matrix form of each basis function, and returns the result in a form
        such that
          Mx.dot ( MomVec )
        results in the matrix form of MomVec. Option kwargs:
          M : the number of basis elements for which to calculate the N by N
              matrix form.

        Note that generally this function should not need to be used directly.
        One of CovMatrix, MomMx, or MxTensor probably do what you want.
        '''
        M = kwargs.get("M", N)
        v = np.zeros((N,N,M))
        F = self.MxForm( *[x for x in np.eye(N)[:M]], Nbasis=N )

        for x in F:
           v[x.N] = x.Values().T
        return v

    def Decompose(self, x, w, cksize=2**20, **kw):
        '''
        Given a dataset 'x' of values with weights 'w', compute the moments of
        that dataset.  Both 'x' and 'w' should be one-dimensional ndarrays
        with the same length.

        The kwargs can be used to override the hyperparameter values that are
        set in this DecompFactory.

        This function recomputes the decomposition from scratch each time it is
        called.  Normally, you want to use CachedDecompose instead - this will
        store the results on disk and load the stored results if that can
        avoid recomputing a decomposition.
        '''
        Nb   = kw.pop("Nbasis", 0)
        Fn   = self.Fn(x[:cksize+1], w[:cksize+1], Nbasis=Nb if Nb > 0 else self["Nbasis"], **kw)
        Mom  = np.zeros( (Fn["Nbasis"],) )

        for i in range(0, x.size, cksize): 
            pdot()
            Fn.Reinit( x[i:i+cksize+1 ], w[i:i+cksize+1] )
            for D in Fn: Mom[D.N] += D.Moment()

        return Mom

    @Cache.Element("{self.CacheDir}", "Decompositions", "{self}", "{2:s}-{Nbasis}.npy")
    def CachedDecompose(self, x, w, name, cksize=2**20, Nbasis=0, **kwargs):
        '''
        Given a dataset 'x' of values with weights 'w', compute the moments of
        that dataset.  Both 'x' and 'w' should be one-dimensional ndarrays
        with the same length.

        The kwargs can be used to override the hyperparameter values that are
        set in this DecompFactory.

        This function stores the results on disk, and loads the stored result
        if the same decomposition is requested a second time.  This avoids
        unnecessary re-decompositions, particularly if the same code is run
        repeatedly.
        '''
        return self.Decompose(x, w, cksize, Nbasis=Nbasis, **kwargs)

    def __init__(self, **kwargs):
        '''
        All parameters listed in _params are mandatory.  In addition, optional
        arguments are:
            Nthread:  the number of threads to use in numexpr computations.
            CacheDir: the directory in which to store cahced files.
        '''
        self.Nthread    = kwargs.get('Nthread',  1)
        self.CacheDir   = kwargs.get('CacheDir', "tmp")
        self.FDDir      = environ.get('FD_DIR', ".")

        ParametricObject.__init__(self, **kwargs)
        ne.set_num_threads(self.Nthread)

        self.TDot       = self.OpMatrix( self["Nxfrm"] )

        self['Factory'] = self

class Basis(ParametricObject):
    '''
    A base class for orthonormal bases in terms of their three-term recursion
    relations.  It is not intended for direct use, but instead provides several
    common methods for its subclasses.

    'Basis' exposes an interator interface.  Subclasses implement the recursion
    relations, and the Basis object applies them to step through the resulting
    values one-by-one. Three-term recursion relations take the general form:

        V_(n+1) =  a*x*V(n) + b*V(n) + c*V(n-1)

    Because each term depends on the _two_ preceding terms, two base values
    are required.  These are implemented in the 'Base0' and 'Base1' methods.
    'x' is a raising operator, implemented in 'Raise'.  The recursion relation
    itself is implemented in 'Recur'.

    An optional constant of integration can be included by overriding 'Xfrm'.
    If used, this must explicitly be incorporated in the subclass's
    implementation.

    See Tools.OrthExp for several examples of different subclasses that utilize
    the 'Basis' object.
    '''
    _param      = ("Nbasis", )

    # User-facing functions.
    #def Values(self):                 return self.t[ self.N % 2 ] * np.sqrt(self.NormSq(self.N))

    # Return a zero ndarray of the appropriate shape and type.
    def Zeros (self, shape=()):       return np.zeros(shape + (1,))

    # The zero'th value of the basis.
    @abstractmethod
    def Base0 (self, out):            return None

    # The first value of the basis.
    @abstractmethod
    def Base1 (self, out):            return None

    # The raising operator. This is computed on initialization and stored
    #  in self['rz'].
    @abstractmethod
    def Raise (self, out):            return None

    # The recursion relation.  This should set 'out' to V(N+1)
    # given En=V(N) and Ep=V(N-1).
    @abstractmethod
    def Recur (self, N, En, Ep, out): return

    # (Optional)  constant of integration. This is computed on initialization
    #  and stored in self['xf'].
    def Xfrm  (self, out):            return self.Zeros()

    ##
    # Implementation of iterator interface
    ##

    def __iter__(self, **kwargs):
        self.N       = -1

        for k, v in kwargs.items():
            self[k] = v

        if 'xf' in self.ParamVal and 'x' in self.ParamVal and self['x'].size == self['xf'].size:
            self.Xfrm  (self['xf'])
            self.Raise (self['rz'])
        else:
            self['xf'] = self.Xfrm(None)
            self['rz'] = self.Raise(None)
            self['t']  = self.Zeros((2,))
        self.Base0 (self['t'][0])
        self.Base1 (self['t'][1])
      
        return self

    def next(self):
        next    = self['t'][ (self.N + 1) % 2 ]
        this    = self['t'][ (self.N    ) % 2 ]
        prev    = self['t'][ (self.N - 1) % 2 ]

        if   self.N < 0:                   self.Base0(next)
        elif self.N == 0:                  self.Base1(next)
        elif self.N < self['Nbasis']:      self.Recur(self.N, this, prev, next)
        if   self.N == self['Nbasis'] - 1: raise StopIteration

        self.N += 1
        return self

    def __str__(self):
        s  = [ str_nl % "className" + type(self).__name__ ]
        s += [ str_nl % k           + str(self[k]) for k in self._param ]

        return '\n'.join(s)


# These go with the Transform object.  Must be globals for Multiprocessing
gK = {}
def gEle(n, m): return gK['obj'].Ele(n, m, **gK)

class Transform(ParametricObject):
    '''
    A base class for implementing transformations between different choices of
    hyperparameters.  This assumes that a transformation on each parameter can
    be written in the following way:

      x' = exp( a_p * T_p ) . x

    where

      x_p is the initial decomposition vector
      T_p is the infinitesimal transformation matrix for parameter p
      a_p is the amount to transform parameter p

    Not all transformations can be written in this way! However, those than can
    be are convenient and tractable for numerical computation.  Since T_p is
    presumed to be a constant, independent of the choice of hyperparameters,
    it can be computed exactly once and then cached on disk.  Since this is
    typically an expensive computation, this is ideal.

    The 'Transform' object provides some framework to calculate T for each
    hyperparameter and cache the result.  It also supplies methods to use
    these transformation matrices on decomposition vectors to transform them
    from one set of hyperparameters to another.

    The _param class member must specify each of the hyperparameters (by name)
    for which a transformation matrix will be computed. Each specified
    hyperparameter must have a corresponding method of the same name that
    return T_p.  An example implementation would look something like:

        _param = ("HyperA", "HyperB" )

        def HyperA(self):
            ...
        def HyperB(self):
            ...

    In most cases, the matrix is computed more easily in some non-orthonormal
    basis and can then be transformed into the orthonormal basis.  In this
    case,

        T_p = O . U_p . D

    where O is the orthonormal functions expressed in the non-orthonormal
    basis and D is the derivatives of the orthonormal functions expressed in
    the non-orthonormal basis.

    The @OrthogonalizeHankel decorator is provided to automate this in the
    case that U_p is a Hankel matrix (i.e. all antidiagonals are constant). To
    make this work, the subclass must implement two functions

        def CoeffOrth(self, n):  return the n by n matrix O[n,m] / O[n,m-1]
        def CoeffDeriv(self, n): return the n by n matrix D[n,m] / D[m,m-1]

    and implement the the transformations like:

        @Base.Transform.OrthogonalizeHankel
        def HyperA(self, n):  return the value of U_p[n, n]

    CoeffOrth and CoeffDeriv return _ratios_ of the columns in O and D, as this
    generally has a much simplier expression than the columns directly.  This
    in turns greatly increases evaluation speed when the underlying type is
    Fraction.

    Then OrthogonalizeHankel will handle the matrix multiplications using a
    multiprocessing pool.  This is overkill if the O, D and U_p are simple
    types like an int or a float.  However, it is sometimes necessary for
    numerical reasons to use slow but accurate types like Fraction, and when
    the matrices are large this can become very slow.  In this case, the
    multiprocessing provides a substantial performance improvement.

    The results of OrthogonalizeHankel are cached on disk, so in any case these
    computations need only be performed once.  This cache is located in

        <FD_DIR>/data
    
    See Tools.OrthExp.ExpDecompTransform for an example implementation.
    '''
    _param = ( )

    def __call__(self, *vec, **kwargs):
        '''
        Transform a list of moments vectors from the current hyperparameters
        into an alternate choice of hyperparameters.  The current parameters
        are specified by the values stored in this object, and the new
        parameters are specified in kwargs.  Each positional argument
        should be an ndarray moment vector.
        '''
        for p in self._param:
            if p not in kwargs:
                kwargs[p] = self.ParamVal[p]

        inv = kwargs.get("inv", False)

        ini = kwargs        if inv else self.ParamVal
        fin = self.ParamVal if inv else kwargs
        ret = np.stack(vec, axis=1)[:self["Nxfrm"]]

        arg = self.XfPar(fin, ini)
        mx  = sum( arg[p] * self.KMx[p].T for p in self._param )
        ret = expm_multiply( mx, ret )

        return tuple(ret.T)

    @Cache.AtomicElement("{self.Factory.FDDir}", "data", "ele-cache-{name:s}", "{0:d}-{1:d}.json")
    def Ele(self, n, m, **kwargs):
        '''
        Compute element (n, m) of self.O . H . self.D where H is a Hankel matrix.
        H is specified by passing it's diagonal as the kwarg 'h'.
        '''
        h   = kwargs.get("h")
        v   = np.outer(self.O[n][::-1], self.D[m])
        s   = [ np.trace(v, offset=k) for k in range(-v.shape[0]+1, v.shape[1]) ]

        return np.dot(h[:len(s)], s) * self.Norm(n, m)

    @staticmethod
    def OrthogonalizeHankel(func):
        '''
        Decorator to create N x N infinitesimal transformation matrices from a
        wrapped Hankel matrix.  This function takes two parameters:
            name : the name of the corresponding hyperparameter.  Used to cache
                   the matrix in a unique file.
            N    : the size of the tranformation matrix.

        The wrapped function must take a single argument, like this:

            @Base.Transform.OrthogonalizeHankel
            def TheWrappedFunction(self, n):

        TheWrappedFunction should take a number n and return the value of the
        n'th diagonal element (i.e. H[n, n]).

        See Tools.OrthExp.ExpDecompTransform for an example.
        '''
        @Cache.Element("{self.Factory.FDDir}", "data", "xfrm-cache-{0:s}-{1:d}.npy")
        def wrap(self, name, N):
            pini(name + " xfrm moment")
            global gK
            gK = {
               'h'   : [ pdot(func(self, k)) for k in range(2*N-1) ],
               'name': name,
               'obj' : self
            }
            pend()

            pini(name + " xfrm matrix", interval=N)
            p  = Pool( self.Factory.Nthread )
            r  = [ p.apply_async(gEle, (n, m), callback=pdot) for n in range(N) for m in range(N) ]
            p.close()
            r  = np.array([ x.get() for x in r]).reshape((N, N))
            pend()

            return r
        return wrap

    # Calculate infinitesimal transformations
    def __init__(self, **kwargs):
        '''
        All parameters listed in _params are mandatory. Additional mandatory
        parameters are:
            Nxfrm:   The size of the transformation matrices (Nxfrm x Nxfrm)
            Factory: The parent Factory object.
        '''
        ParametricObject.__init__(self, **kwargs)

        self["Nxfrm"]   = kwargs["Nxfrm"]
        self.Factory    = kwargs["Factory"]
        self.O          = [ self.CoeffOrth (n) for n in range(self["Nxfrm"]) ]
        self.D          = [ self.CoeffDeriv(n) for n in range(self["Nxfrm"]) ]
        self.KMx        = { n : getattr(self, n)(n, self["Nxfrm"]) for n in self._param }

