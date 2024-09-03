'''
CacheMgr is a collection of functions to handle the automatic result caching
for FD.  It handles primitive Python types as well as Numpy ndarrays, and has
facilities for caching both in the numpy ".npy" format as well as JSON.

There are three user-facing functions:
  CacheMgr.Element       : Decorator; cache the result of the decorated function
  CacheMgr.AtomicElement : Decorator; atomically cache the result of the
                           decorated function
  CacheMgr.SymArray      : Decorator; cache the result of the deorated function
                           assuming that the result is a symmetric ndarray and
                           save only unique elements to disk.
'''

import itertools, json, os
import numpy           as np

from  numpy.lib.format import open_memmap
from  Tools.PrintMgr   import *

# Format path elements from pathTpl, create the directory chain,
#  return the full path, filename and extension.
def _mkPath(pathTpl, *args, **kwargs):
    path     = [x.format(*args, **kwargs) for x in pathTpl]
    FullPath = os.path.join(*path)
    Ext      = os.path.splitext(path[-1])[1].lower()

    for n in range(1, len(path)):
        try:
            os.mkdir( os.path.join(*path[:n]) )
        except OSError:
            pass

    return FullPath, path[-1], Ext

# Atomic file creation method
def _acreate(filename):
    fd = os.open(filename, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0644)
    return os.fdopen(fd, 'wb')

def _fcreate(filename):
    fd = os.open(filename, os.O_CREAT |             os.O_WRONLY, 0644)
    return os.fdopen(fd, 'wb')

# Save a sparse symmetric array.
def _saveSparseSym(Mx, file):
    idx   = np.ndindex(*Mx.shape)
    sdict = { tuple(sorted(i)) : Mx[i] for i in idx if Mx[i] != 0 }

    keys  = zip(*sdict.keys())
    vals  = sdict.values()

    json.dump( (Mx.shape, keys, vals), file)

# Load a spare symmetric array.
def _loadSparseSym(file):
    shp, keys, vals = json.load( file )
    Mx              = np.zeros(shp)

    for x in itertools.permutations(keys):
        Mx[x] = vals

    return Mx

# Cache / calculate a symmetric, multi-dimensional numpy array.
def SymArray(*pathTpl):
    '''
    Cache a symmetric, multi-dimensional ndarray.
    
    This decorator should wrap a function that returns a symmetric,
    n-dimensional ndarray.  The decorator's positional arguments should specify
    the path for the cache file (one path component per argument).  The ndarray
    is saved as a sparse JSON file, that is, if the wrapped function returns a
    3-axis ndarray, then only a single element is written to disk for the six
    components
        (1,2,3), (2,1,3), (1,3,2), (2,3,1), (3,2,1), (3,1,2)
    which, by symmetric, are presumed to be equal.

    When the cache is loaded, the saved file is expanded to a full non-sparse
    ndarray.
    '''
    def owrap(func):
        def wrap(self, *args, **kwargs):
            FullPath, Name, Ext = _mkPath(pathTpl, *args, self=self, **kwargs)
            desc                = kwargs.get("desc", "")

            try:
                with open(FullPath, 'rb') as file:
                    pini("Loading %s (%s)" % (desc, Name) )
                    M = _loadSparseSym(file)
                    pend("Success")
                    return M
            except IOError:
                pass

            pini("Calculating %s" % desc)
            M = func(self, *args)
            pend("Done")

            with open(FullPath, 'wb') as file:
                pini("Saving %s (%s)" % (desc, Name))
                _saveSparseSym(M, file)
                pend("Success")

            return M
        return wrap
    return owrap

# Cache / calculate a generic numpy array or JSON-serializable object.
def _fsave(file, Ext, res):
    if Ext == ".npy":
        np.save  (file, res)
    elif Ext== ".json":
        json.dump(res, file)
    else:
        print "Unknown file extension '%s' from file '%s'." % (Ext, Name)
        
def _fload(Ext, FullPath):
    if Ext == ".npy":
        return np.load(FullPath, mmap_mode='r')
    elif Ext==".json":
        with open(FullPath, 'rb') as file:
            return json.load(file)
    else:
        print "Unknown file extension '%s' from file '%s'." % (Ext, Name)
        raise ValueError

def Element(*pathTpl):
    '''
    Cache a function result.

    The wrapped function can return an ndarray or any JSON-serializable Python
    structure.  The decorator's positional arguments should specify the path
    for the cache file (one path component per argument).  The result is saved
    as a JSON or npy file depending on the extension of the last path element.

    If the file already exists but cannot be deserialized, Element re-computes
    the wrapped function.
    '''
    def owrap(func):
        def wrap(self, *args, **kwargs):
            FullPath, Name, Ext = _mkPath(pathTpl, *args, self=self, **kwargs)

            # Try to load the file
            try:
                return _fload(Ext, FullPath)
            except (ValueError, IOError):
                pass

            # Otherwise, re-create the file and re-compute contents.
            with _fcreate(FullPath) as file:
                res = func(self, *args, **kwargs)
                _fsave(file, Ext, res)
            return _fload(Ext, FullPath)
        return wrap
    return owrap

def AtomicElement(*pathTpl):
    '''
    Cache a function result.

    The wrapped function can return an ndarray or any JSON-serializable Python
    structure.  The decorator's positional arguments should specify the path
    for the cache file (one path component per argument).  The result is saved
    as a JSON or npy file depending on the extension of the last path element.

    AtomicElement uses an atomic file creation to check if the result cache is
    either existing or in progress.  If the file exists and can be loaded, it
    is loaded.  If it exists and cannot be loaded, AtomicElement returns a
    value of zero.
    '''
    def owrap(func):
        def wrap(self, *args, **kwargs):
            FullPath, Name, Ext = _mkPath(pathTpl, *args, self=self, **kwargs)

            # Try an atomic file creation; if successful, run and save wrapped calculation.
            try:
                with _acreate(FullPath) as file:
                    res = func(self, *args, **kwargs)
                    _fsave(file, Ext, res)
            except OSError:
                pass

            # Otherwise, the file exists, so just load it.
            try:
                return _fload(Ext, FullPath)
            except (ValueError, IOError):
                return 0.0
        return wrap
    return owrap
