import numpy as np
from sklearn.gaussian_process.kernels import Hyperparameter, Kernel, StationaryKernelMixin
from scipy.spatial.distance import squareform


def _convert_to_double(X):
    return np.ascontiguousarray(X, dtype=np.double)


class Gibbs(Kernel):
    """Gibbs kernel
    sqrt( (2*l(x)l(x')) / (l(x)**2 + l(x')**2 ) * exp( -(x-x')**2 / (l(x)**2 + l(x')**2) )
    where, l(x) = b*x + c
    This is the RBF kernel where the length scale is allowed to vary by some
    function, in this case a linear function.
    """

    def __init__(self, l0=1.0, l0_bounds=(1e-6, 1e5), l_slope=1.0, l_slope_bounds=(1e-6, 1e5)):
        self.l0 = l0
        self.l0_bounds = l0_bounds
        self.l_slope = l_slope
        self.l_slope_bounds = l_slope_bounds

    @property
    def hyperparameter_l0(self):
        return(Hyperparameter("l0", "numeric", self.l0_bounds))

    @property
    def hyperparameter_l_slope(self):
        return(Hyperparameter("l_slope", "numeric", self.l_slope_bounds))

    def __call__(self, X, Y=None, eval_gradient=False):
        """Retrn K(X,Y)"""

        def l(x):  # Helper function
            return self.l0 + self.l_slope * (x)

        X = np.atleast_2d(X)

        s = X.shape
        if len(s) != 2:
            raise ValueError('A 2-dimensional array must be passed.')

        if Y is None:
            m, n = s
            K = np.zeros((m * (m - 1)) // 2, dtype=np.double)
            X = _convert_to_double(X)
            X = X - X[0]  # note: this step just shifts the x-range to make the hyper-pars a bit more intuitive
            t = 0
            for i in range(0, m - 1):
                for j in range(i + 1, m):
                    xi_xj = X[i] - X[j]
                    li = l(X[i])  # b*x+c ( b = l_slope, c = l0 )
                    lj = l(X[j])
                    li2_lj2 = np.dot(li, li) + np.dot(lj, lj)  # l(x)^2 + l(x')^2
                    coeff = np.sqrt(2 * li * lj / (li2_lj2))
                    K[t] = coeff * np.exp(-1 * (xi_xj * xi_xj) / li2_lj2)
                    t = t + 1

            K = squareform(K)
            np.fill_diagonal(K, 1)

            if eval_gradient:
                # approximate gradient numerically
                def f(theta):  # helper function
                    return self.clone_with_theta(theta)(X, Y)
                return K, _approx_fprime(self.theta, f, 1e-10)
            else:
                return K

        else:
            mx, nx = s
            sy = Y.shape
            my, ny = sy
            K = np.zeros((mx, my), dtype=np.double)
            X = _convert_to_double(X)
            Y = _convert_to_double(X)

            X = X - X[0]  # note: this step just shifts the x-range to make the hyper-pars a bit more intuitive
            Y = Y - Y[0]

            # import pdb; pdb.set_trace()
            t = 0
            for i in range(0, mx):
                for j in range(0, my):
                    xi_yj = X[i] - Y[j]
                    li = l(X[i])  # b*x+c ( b = l_slope, c = l0 )
                    lj = l(Y[j])
                    li2_lj2 = li * li + lj * lj  # l(x)^2 + l(x')^2
                    coeff = np.sqrt(2 * li * lj / (li2_lj2))
                    K[i][j] = coeff * np.exp(-1 * (xi_yj * xi_yj) / li2_lj2)
                    t = t + 1

            # K = squareform(K)
            # np.fill_diagonal(K,1)

            if eval_gradient:
                # approximate gradient numerically
                def f(theta):  # helper function
                    return self.clone_with_theta(theta)(X, Y)
                return K, _approx_fprime(self.theta, f, 1e-10)
            else:
                return K

        pass  # __call__

    def diag(self, X):
        return np.diag(self(X))

    def is_stationary(self):
        return False

    def __repr__(self):
        return "{0}(l0={1:.3g}, l_slope={2:.3g})".format(self.__class__.__name__, self.l0, self.l_slope)


class FallExp(Kernel):
    """Falling exponential kernel
    exp( (d - (x+x'))/(2*a) )
    """
    def __init__(self, d=1.0, d_bounds=(1e-5, 1e5), a=1.0, a_bounds=(1e-5, 1e5)):
        self.d = d
        self.d_bounds = d_bounds
        self.a = a
        self.a_bounds = a_bounds

    @property
    def hyperparameter_d(self):
        return(Hyperparameter("d", "numeric", self.d_bounds))

    @property
    def hyperparameter_a(self):
        return(Hyperparameter("a", "numeric", self.a_bounds))

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return K(X,Y)
        """
        X = np.atleast_2d(X)

        s = X.shape
        if len(s) != 2:
            raise ValueError('A 2-dimensional array must be passed.')

        if Y is None:
            m, n = s
            K = np.zeros((m * (m - 1)) // 2, dtype=np.double)
            X = _convert_to_double(X)
            t = 0
            for i in range(0, m - 1):
                for j in range(i + 1, m):
                    xi_xj = np.dtype('d')
                    xi_xj = X[i] + X[j]
                    K[t] = np.exp((self.d - xi_xj) / (2 * self.a))
                    t = t + 1

            K = squareform(K)
            np.fill_diagonal(K, 1)

            if eval_gradient:
                # approximate gradient numerically
                def f(theta):  # helper function
                    return self.clone_with_theta(theta)(X, Y)
                return K, _approx_fprime(self.theta, f, 1e-10)
                return K, None
            else:
                return K

        else:
            mx, nx = s
            sy = Y.shape
            my, ny = sy
            K = np.zeros((mx, my), dtype=np.double)
            X = _convert_to_double(X)
            Y = _convert_to_double(Y)
            for i in range(0, mx):
                for j in range(0, my):
                    xi_yj = np.dtype('d')
                    xi_yj = X[i] + Y[j]
                    K[i][j] = np.exp((self.d - xi_yj) / (2 * self.a))

            # K = squareform(K)

            if eval_gradient:
                # approximate gradient numerically
                def f(theta):  # helper function
                    return self.clone_with_theta(theta)(X, Y)
                return K, _approx_fprime(self.theta, f, 1e-10)
            else:
                return K

        pass  # __call__

    def diag(self, X):
        return np.diag(self(X))

    def is_stationary(self):
        return False

    def __repr__(self):
        return "{0}(d={1:.3g}, a={2:.3g})".format(self.__class__.__name__, self.d, self.a)
    pass  # fallExp


class LinearNoiseKernel(StationaryKernelMixin, Kernel):
    """White kernel.
    The main use-case of this kernel is as part of a sum-kernel where it
    explains the noise-component of the signal. Tuning its parameter
    corresponds to estimating the noise-level.
    k(x_1, x_2) = noise_level(x_1) if x_1 == x_2 else 0
    Parameters
    ----------
    noise_level : array, shape (n_samples_X, n_features)
        Parameter controlling the noise level (vector of same shape as X)
    """
    def __init__(self, noise_level=1.0, noise_level_bounds=(1e-5, 1e5), b=1.0, b_bounds=(1e-5, 1e5)):
        self.noise_level = noise_level
        self.noise_level_bounds = noise_level_bounds
        self.b = b
        self.b_bounds = b_bounds

    @property
    def hyperparameter_noise_level(self):
        return Hyperparameter("noise_level", "numeric", self.noise_level_bounds)

    @property
    def hyperparameter_b(self):
        return(Hyperparameter("b", "numeric", self.b_bounds))

    def __call__(self, X, Y=None, eval_gradient=False):
        """Return the kernel k(X, Y) and optionally its gradient.
        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)
        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
            Right argument of the returned kernel k(X, Y). If None, k(X, X)
            if evaluated instead.
        eval_gradient : bool (optional, default=False)
            Determines whether the gradient with respect to the kernel
            hyperparameter is determined. Only supported when Y is None.
        Returns
        -------
        K : array, shape (n_samples_X, n_samples_Y)
            Kernel k(X, Y)
        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
            The gradient of the kernel k(X, X) with respect to the
            hyperparameter of the kernel. Only returned when eval_gradient
            is True.
        """

        X = np.atleast_2d(X)

        if Y is not None and eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")

        if Y is None:
            K = np.zeros((X.shape[0], X.shape[0]))
            for i in range(0, X.shape[0]):
                K[i, i] = self.noise_level + (-1.0) * self.b * (X[i][0] - 99.0)
                if(K[i, i] < 0): K[i, i] = 0.0

            if eval_gradient:
                def f(theta):
                    return self.clone_with_theta(theta)(X, Y)
                return K, _approx_fprime(self.theta, f, 1e-10)
                return K, None
            else: return K

        else: return np.zeros((X.shape[0], Y.shape[0]))

    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).
        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.
        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)
        Returns
        -------
        K_diag : array, shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """

        K_diag = np.ones(X.shape[0])
        for i in range(0, X.shape[0]):
            K_diag[i] = self.noise_level + (-1.0) * self.b * (X[i][0] - 99.0)

        return K_diag

    def __repr__(self):
        return "{0}(noise_level={1},b={2})".format(self.__class__.__name__, self.noise_level, self.b)


# adapted from scipy/optimize/optimize.py for functions with 2d output
def _approx_fprime(xk, f, epsilon, args=()):
    f0 = f(*((xk,) + args))
    grad = np.zeros((f0.shape[0], f0.shape[1], len(xk)), float)
    ei = np.zeros((len(xk), ), float)
    for k in range(len(xk)):
        ei[k] = 1.0
        d = epsilon * ei
        grad[:, :, k] = (f(*((xk + d,) + args)) - f0) / d[k]
        ei[k] = 0.0
    return grad
