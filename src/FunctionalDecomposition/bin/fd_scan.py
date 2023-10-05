#!/usr/bin/env python

import os, sys, signal, argparse, ConfigParser
import numpy                           as np
import matplotlib.pyplot               as plt
import Plots.Plots                     as Plots

from   Tools.Decomp                    import DataSet, Optimizer, SignalScan
from   Tools.OrthExp                   import ExpDecompFactory
from   Tools.ConfigIter                import ConfigIter
from   Tools.PrintMgr                  import *
from   Plots.Plots                     import *

from   matplotlib.backends.backend_pdf import PdfPages

# Load a specified variable from base dir
def loadVar(varName, DSList, Dec):
    Path = [ os.path.join(ArgC.base, "Data", n, varName + ".npy") for n in DSList ]
    Ar   = [ np.load(p, mmap_mode = 'r')[::int(d)] for p, d in zip(Path, Dec) ]

    return np.concatenate(Ar)

# Test a condition on on cached variables
def testVar(cond, DSList, Dec):
    Keep = []

    for n, d in zip(DSList, Dec):
        Path  = os.path.join ( ArgC.base, "Data", n )
        Names = [ os.path.splitext(k)[0] for k in os.listdir ( Path ) ]
        Full  = { k : os.path.join(Path, k + ".npy") for k in Names }
        Vars  = { k : np.load(Full[k], mmap_mode = 'r')[::int(d)] for k in Names }

        Keep.append(eval(cond, Vars))
    return np.concatenate(Keep)

# Evaluate a signal function string.
def _fmtSignalFunc(x, **kwargs):
    try:
        return eval(x.format(**kwargs))
    except AttributeError:
        return x

def _addSignalFunc(D, func, name, M, **kw):
    Active = kw.get("Active", (M is None))
    arg    = { k : _fmtSignalFunc(v, Mass=M, **kw) for k, v in c.items() }
    ffunc  = _fmtSignalFunc(func, **arg)

    D.AddParametricSignal(name, ffunc, Active=Active, **arg)

########################################
########################################

# Stop stack trace on ctrl-c
def nice_quit(signal, frame):
    sys.exit(0)
signal.signal(signal.SIGINT, nice_quit)

# Lower default NP print precision.
np.set_printoptions(precision=4)

######
# Parse command line parameters and config files
######
ArgP      = argparse.ArgumentParser(description='=== Functional Decomposition Fitter ===')
ArgP.add_argument('--base', type=str, default=".", help="FD base directory.")
ArgP.add_argument('--show', action="store_true",   help="Display plots interactively.")
ArgC      = ArgP.parse_args()

Config    = ConfigParser.ConfigParser()
Config.optionxform = str
Config.read( os.path.join(ArgC.base, "base.conf") )

PlotStyle = Config.get("General", "PlotStyle")
try:            plt.style.use( PlotStyle )
except IOError: plt.style.use( os.path.join(ArgC.base, PlotStyle) )

# Reduce numpy default print precision to make logs a bit neater.
np.set_printoptions(precision=2)





######
# Initialize the factory and signal models.
######

# Get hyperparameter scan range
ranges     = { p : r.rstrip("j") for p, r in Config.items("HyperParameterScan") }
rcomp      = [ ranges[p].split(':')for p in ExpDecompFactory._fitparam ]

rlist_xfrm = []  # The hyperparameter scan range for transformations
rlist_dcmp = []  # The scan range for direct decompositions

for param, r in zip(ExpDecompFactory._fitparam, rcomp) :
    start  = min( float(r[0]), float(r[1]))
    stop   = max( float(r[0]), float(r[1]))
    step_x = float(r[2])
    step_d = float(r[3]) if len(r) > 3 else 1

    dcmp   = np.linspace (start,     stop, step_d)
    xidx   = np.linspace (    0, step_d-1, step_x)
    xfrm_r = np.linspace (start,     stop, step_x)

    if param == 'Alpha':
        dcmp_idx = np.round(xidx-0.5, 0).astype(int)
    elif param == 'Lambda':
        dcmp_idx = np.round(xidx+0.5, 0).astype(int)

    dcmp_idx = np.clip(dcmp_idx, 0, dcmp.size-1)
    dcmp_r   = dcmp[ dcmp_idx ]

    rlist_xfrm.append( xfrm_r )
    rlist_dcmp.append( dcmp_r )

Ax, Lx    = np.meshgrid(*rlist_xfrm)
Ad, Ld    = np.meshgrid(*rlist_dcmp)

# Create Factory configuration
fConf     = { k: eval(v) for k, v in Config.items('ExpDecompFactory') }
for p, r in zip(ExpDecompFactory._fitparam, rlist_dcmp):
    fConf[p] = r[0]
fConf['CacheDir'] = os.path.join(ArgC.base, "Cache")

# Read input variables.
SetList   = Config.items("InputFiles")

General   = { p: r for p, r in Config.items("General") }
varName   = General.get("Variable")
wgtName   = General.get("Weight")
Scale     = General.get("Scale")
Lumi      = General.get("Lumi",      0.0)
LumiUnc   = General.get("LumiUnc",   0.0)
XSecUnits = General.get("XSecUnits") if Lumi > 0 else "Events"
NumOpt    = General.get("NumOptSteps", 1)

print
print
print "Input files:", " ".join( [n for n, d in SetList] )

x       = loadVar(varName, *zip(*SetList))
w       = loadVar(wgtName, *zip(*SetList)) * float(Scale)
cutflow = [ ("Initial", w.sum()) ]

for name, cond in Config.items("Cuts"):
    w *= testVar(cond, *zip(*SetList))
    cutflow.append((name, w.sum()))

# Create objects
Factory = ExpDecompFactory( **fConf )
Tr      = Factory.Xfrm()
D       = DataSet(x, Factory, w=w)
FOpt    = Optimizer(Factory, D)

Names   = {}
Scans   = {}

# Initialize signal peaks
for c, name in ConfigIter(Config, "ParametricSignal", "ParametricSignalDefault"):
   func = "lambda x:" + c.pop("func")
   Scan = c.pop("Scan", None)

   if Scan is None:
       _addSignalFunc(D, func, name, None, **c)
   else:
       Names[name] = [ name + "%.3f" % M for M in Scan ]

       for M, fname in zip( Scan, Names[name] ):
           _addSignalFunc(D, func, fname, M, **c)

print

######
# Decompose
######
LLH, PBest = FOpt.ScanW(Ax, Lx, Ad, Ld)
dA         = Ax[0,1] - Ax[0,0]
dL         = Lx[1,0] - Lx[0,0]

for _ in range(1):
    Ab     = PBest['Alpha']
    Lb     = PBest['Lambda']
    ini    = np.asarray( [ (Ab-dA, Lb-dL), (Ab-dA, Lb+dL), (Ab+dA, Lb) ] )

    Factory.update( PBest )
    D.Decompose(xonly=True)

    NBest, LBest, PBest = FOpt.FitW( initial_simplex = ini)

### Re-decompose with the full expansion and extract signals.
Factory.update( PBest)
D.Decompose(reduced=False)
FOpt.UpdateXfrm(**PBest)

### attr="Mom": use the full expansion from here on out
D.SetN(N=NBest, attr="Mom")
D.Covariance()
if len(D.Signals) > 0:
    D.PrepSignalEstimators(reduced=False, verbose=True)

### Calculate yields, uncertainties and CLs
fmt = {
  "wht": "\x1b[37m %8.1f \x1b[0m",
  "grn": "\x1b[32m %8.1f \x1b[0m",
  "yel": "\x1b[33m %8.1f \x1b[0m",
  "red": "\x1b[31m %8.1f \x1b[0m",
}
lfmt = [ "red", "yel", "grn", "wht", "grn", "yel", "red" ]

print
print
print "=====> YIELD AND LIMITS <====="
print
print "%-16s: %8s +- %8s (%5s) [ %9s  ] [ %9s  %9s  %9s  %9s  %9s  ]" % (
       "Signal", "Yield", "Unc", "Sig.", "Obs. CL95",
       "-2sigma", "-1sigma", "Exp. CL95", "+1sigma", "+2sigma")

for scan_name, sig_names in Names.items():
    Scans[scan_name] = SignalScan(Factory, D, *sig_names, Lumi=Lumi, LumiUnc=LumiUnc)

    t1 = time.time()
    for name, yld, unc, obs, exp in Scans[scan_name]:
        t2   = time.time()
        sig  = yld / unc
        isig = 3 + np.clip(int(sig) + (1 if sig > 0 else -1), -3, 3)

        print "%-16s: %8.1f +- % 8.1f (% 4.2f)" % (name, yld, unc, sig),
        print "[", fmt[lfmt[ isig ]] % obs, "] [",
        for l, e in zip(lfmt[1:-1], exp):
            print fmt[l] % e,
        print "] (%4.2fs)" % (t2-t1)

        t1 = time.time()

######
# Output fit results
######
Nxfrm = Factory["Nxfrm"]
print
print
print "=====> CUTFLOW <====="
for c in cutflow:
    print "% 12s: %.2f" % c
print
print
print "=====> SIGNAL RESULTS <====="
for name in D.GetActive():
    s = D[name]
    print "% 12s: %.2f +- %.2f" % (name, s.Yield, s.Unc)
print
print
print "=====> COVARIANCE < ====="
print D.Cov
print
print
print "=====> CORRELATION < ====="
print D.Corr
print
print
print "=====> RAW MOMENTS <====="
for n in range(33):
  print "% 3d: %+.3e   " % (n, D.Mom[n]),
  if n % 4 == 3: print
print
print
print "=====> BACKGROUND COEFFICIENTS <====="
for n in range(D.N):
  print "% 3d: %+.3e   " % (n, D.TestB.Mom[n]),
  if n % 4 == 3: print
print
print

######
# Plotting
######
print
print
print "=====> PLOTTING < ====="
def op(*x):
    return os.path.join(ArgC.base, "Output", *x)

try:
    os.mkdir(op())
except OSError:
    pass

pdf = PdfPages(op('all.pdf'))
ini =   fConf["Lambda"],   fConf["Alpha"]
fin = Factory["Lambda"], Factory["Alpha"]

Plots.cutflow (cutflow, pdf=pdf, fname=op('cutflow.pdf'))

if LLH.size > 3:
    Plots.scan (Lx, Ax, LLH, LBest, Ld, Ad, fin, pdf=pdf, fname=op('hyperparameter_scan_400.pdf'))
    Plots.scan (Lx, Ax, LLH, LBest, Ld, Ad, fin, pdf=pdf, fname=op('hyperparameter_scan_25.pdf'), maxZ=25)


Plots.summary_table(D,                   pdf=pdf, fname=op('signal_summary.pdf'))

for p, file in ConfigIter(Config, "Plot", "PlotDefault"):
    Pull = p.pop("DrawPull", True)

    try:
        Type = p.pop("Type")
    except KeyError:
        print "ERROR: Must specify 'Type' key in config for %s.  Valid types are:" % file
        print "   Fit  (requires keys: 'Bins')"
        print "   Scan (requires keys: 'Scans')"
        continue

    if Type == "Fit":
        h, res = Plots.fit (D,      pdf=pdf, fname=op(file), **p)
        if Pull:
            Plots.pull     (h, res, pdf=pdf, fname=op("pull-" + file) )
    elif Type == "Scan":
        Plots.mass_scan    (Scans,  pdf=pdf, fname=op(file), Units=XSecUnits, **p)
    elif Type == "Estimators":
        Plots.estimators   (D,      pdf=pdf, fname=op(file), **p)
    elif Type == "Moments":
        Plots.moments      (D,      pdf=pdf, fname=op(file), **p)

pdf.close()

