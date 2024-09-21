# ################################ #
# EXAMPLE: GPR with custom kernels #
# ################################ #


import matplotlib.pyplot as plt
import numpy as np
from custom_kernels import Gibbs, LinearNoiseKernel
from scipy.optimize import curve_fit
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel

########### User Defined Options ##########
inFileName = "data/original_histograms/mass_mm_higgs_Background.npz"
with np.load(inFileName) as data:
    bin_edges = data["bin_edges"]
    bin_centers = data["bin_centers"]
    bin_values = data["bin_values"]
    bin_errors = data["bin_errors"]


# Whether or not to calculate systematic error contribution by varying the length scale
calculateSystematics = True

# How much (in fractional form) to vary the length scale when doing systematic GP fits
# i.e. to vary by 20%, set as 0.2
systVarFrac = 0.02

# The hyper-parameter ranges to be used, if not using the outputs of the hyper-par optimization
Gibbs_l0_bounds = [5, 25]
Gibbs_l_slope_bounds = [10e-5, 0.05]

# How to determine the errors while doing the GPR fit and in making the resulting smooth template.
# Options are:
#   1: Use the linear error kernel to flexibly approximate Poisson errors in the fit
#   2: Use the original errors (non-flexible) from the input template, and assign these to the output
#   3: Use the original errors (non-flexible) from the input template, but calculate Poisson errors for the output
whichErrorTreatment = 2

useGPRLinearError = False
useOriginalTemplateError = False
usePoissonError = False

if whichErrorTreatment == 1:
    useGPRLinearError = True
elif whichErrorTreatment == 2:
    useOriginalTemplateError = True
elif whichErrorTreatment == 3:
    usePoissonError = True


def mySimpleExponential(x, a, b):
    """Basic Exponential for GPR Prior"""
    return a * np.exp(b * x)


###########################################
########### The Important Stuff ###########
nEvts = np.sum(bin_values)
nBins = bin_values.size
xMin = bin_edges[0]
xMax = bin_edges[-1]

# Define the GPR Prior (should be reasonably similar to the input template shape)
# this makes the baseline (histogram/array) by a coarse fit to the input array, the final normalization should match the input

fitfun = mySimpleExponential
binshift = bin_centers - bin_centers[0]
popt, pcov = curve_fit(fitfun, binshift, bin_values, p0=[bin_values[0], -0.05])
perr = np.sqrt(np.diag(pcov))
a, b = popt
print("a,b", a, b)
myBaseLine = np.array(fitfun(binshift, a, b))
myBaseLine = (nEvts / np.sum(myBaseLine)) * myBaseLine

# print("mya",nEvts,bin_values[1:5])
# print("myb",np.sum(myBaseLine),myBaseLine[1:5])
# xerrs = 0.5*(bin_edges[1:]-bin_edges[:-1]) # or None
# plt.errorbar(bin_centers, bin_values, bin_errors, xerrs, fmt="none",color='b',ecolor='b',label='Original histogram') #fmt='.k'
# plt.plot(bin_centers, fitfun(binshift, *popt), 'r-',label='fit: a=%5.3f, b=%5.3f' % tuple(popt))
# plt.plot(bin_centers,myBaseLine,'g-', label='fit baseline')
# plt.legend()
# plt.savefig("m_mumu_fit.pdf",format="pdf")

# Get estimates of different hyper-parameter bounds from the input template
# Set bounds on the scaling of the GPR template (basically a maximum y-range)
const_low = 10.0**-5
const_hi = nEvts * 10.0

Gibbs_l0 = Gibbs_l0_bounds[0] + 0.9 * (Gibbs_l0_bounds[1] - Gibbs_l0_bounds[0])
Gibbs_l_slope = Gibbs_l_slope_bounds[0] + 0.9 * (Gibbs_l_slope_bounds[1] - Gibbs_l_slope_bounds[0])

# Set bounds on the magnitude of the GPR error bars

errConst0 = np.max(bin_errors)
errConst_low = 0.90 * errConst0
errConst_hi = 1.10 * errConst0

# Estimate the (decreasing) slope of the magnitude of the GPR error bars

slope_est0 = -1.0 * (bin_errors[0] - bin_errors[-1]) / (bin_centers[0] - bin_centers[-1])
if slope_est0 < 10**-5:
    slope_est0 = 10**-4
slope_est_low = 0.5 * slope_est0
slope_est_hi = 1.05 * slope_est0


# Define the GPR Kernel
if useGPRLinearError:
    kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(const_low, const_hi)) * Gibbs(
        l0=Gibbs_l0,
        l0_bounds=(Gibbs_l0_bounds[0], Gibbs_l0_bounds[1]),
        l_slope=Gibbs_l_slope,
        l_slope_bounds=(Gibbs_l_slope_bounds[0], Gibbs_l_slope_bounds[1]),
    ) + ConstantKernel(constant_value=1.0, constant_value_bounds=(1.0, 1.0)) * LinearNoiseKernel(
        noise_level=errConst0,
        noise_level_bounds=(errConst_low, errConst_hi),
        b=slope_est0,
        b_bounds=(slope_est_low, slope_est_hi),
    )
else:
    kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(const_low, const_hi)) * Gibbs(
        l0=Gibbs_l0,
        l0_bounds=(Gibbs_l0_bounds[0], Gibbs_l0_bounds[1]),
        l_slope=Gibbs_l_slope,
        l_slope_bounds=(Gibbs_l_slope_bounds[0], Gibbs_l_slope_bounds[1]),
    )

# Fit the GPR Kernel to the input template
gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, alpha=bin_errors**2)
gpr.fit(bin_centers.reshape(len(bin_centers), 1), (bin_values - myBaseLine).ravel())

# Check to see if GPR Prediction is effectively the same as the exponential prior
# If so, re-do the GPR fitting, but with a flat prior instead of exponential prior
gprPrediction, gprCov = gpr.predict(bin_centers.reshape(len(bin_centers), 1), return_cov=True)

if np.sqrt((gprPrediction * gprPrediction).sum()) / bin_values.sum() < 0.0001:
    myBaseLine = np.zeros(len(bin_values))
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, normalize_y=True)
    gpr.fit(bin_centers.reshape(len(bin_centers), 1), (bin_values - myBaseLine).ravel())
    gprPrediction, gprCov = gpr.predict(bin_centers.reshape(len(bin_centers), 1), return_cov=True)

gprError = np.copy(np.diagonal(gprCov))
gprPrediction = gprPrediction + myBaseLine


# Determine the error bars based on requested error treatment
outputErrorBars = np.zeros(len(gprPrediction), "d")

# Use the GPR template's linear error kernel for errors
if useGPRLinearError:
    outputErrorBars = gprError.copy()

# Use the original template's errors
if useOriginalTemplateError:
    outputErrorBars = bin_errors.copy()

# Calculate Poisson errors
if usePoissonError:
    for i in range(len(gprPrediction)):
        outputErrorBars[i] = np.sqrt(gprPrediction[i])

# Calculate Systematic Errors by Varying the Length Scale
if calculateSystematics:
    print("\n Peforming systematics variations of the lengths scales\n")

    # Define the systematic kernels (vary the length scale by some fraction, keeping the other parameters constant
    theta_nom = gpr.kernel_.theta
    theta_up = []
    theta_down = []
    for ipar, hyperpar in enumerate(gpr.kernel_.hyperparameters):
        theta_up.append(theta_nom[ipar])
        theta_down.append(theta_nom[ipar])
        if ("l0" in hyperpar.name) or ("length_scale" in hyperpar.name):
            theta_up[ipar] += np.log(1.0 + systVarFrac)
            theta_down[ipar] += np.log(1.0 - systVarFrac)

    kernel_systUp = gpr.kernel_.clone_with_theta(theta_up)
    kernel_systDown = gpr.kernel_.clone_with_theta(theta_down)

    # Perform the fits using the systematic kernels
    gpr_systUp = GaussianProcessRegressor(kernel=kernel_systUp, n_restarts_optimizer=3, alpha=bin_errors**2)
    gpr_systDown = GaussianProcessRegressor(kernel=kernel_systDown, n_restarts_optimizer=3, alpha=bin_errors**2)

    gpr_systUp.fit(bin_centers.reshape(len(bin_centers), 1), (bin_values - myBaseLine).ravel())
    gpr_systDown.fit(bin_centers.reshape(len(bin_centers), 1), (bin_values - myBaseLine).ravel())

    gprPrediction_systUp = gpr_systUp.predict(bin_centers.reshape(len(bin_centers), 1))
    gprPrediction_systUp = gprPrediction_systUp + myBaseLine

    gprPrediction_systDown = gpr_systDown.predict(bin_centers.reshape(len(bin_centers), 1))
    gprPrediction_systDown = gprPrediction_systDown + myBaseLine

    # Calculate the error bars (maximum of the two variations)
    gprHiErrors = np.zeros(len(gprPrediction), "d")
    gprLowErrors = np.zeros(len(gprPrediction), "d")
    gprSystErrors = np.zeros(len(gprPrediction), "d")
    for i in range(len(gprPrediction)):
        gprHiErrors[i] = np.abs(gprPrediction[i] - np.max([gprPrediction_systUp[i], gprPrediction_systDown[i]]))
        gprLowErrors[i] = np.abs(gprPrediction[i] - np.min([gprPrediction_systUp[i], gprPrediction_systDown[i]]))
        gprSystErrors[i] += np.max([gprHiErrors[i], gprLowErrors[i]])
        outputErrorBars[i] += gprSystErrors[i]

    gprHiErrors = np.array(gprHiErrors)
    gprLowErrors = np.array(gprLowErrors)
    gprSystErrors = np.array(gprSystErrors)


##########################################################################################################
# PLOT
##########################################################################################################
f, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(6, 6), gridspec_kw={"height_ratios": [3, 1]})


col = "g"
cols = "r"
logy = False
ymax = 1.2 * 10**5
yloc = 0.95

xwidths = bin_edges[1:] - bin_edges[:-1]
xerrs = 0.4 * xwidths

ys = bin_values
yerrs = bin_errors
ax1.errorbar(bin_centers, ys, yerrs, xerrs, fmt=".", color="k", ecolor="k")  # fmt='.k'

gys = gprPrediction
gerrs = outputErrorBars
ax1.errorbar(bin_centers, gys, gerrs, xerrs, fmt="none", color=col, ecolor=col)  # fmt='.k'


if calculateSystematics:
    yval = gprPrediction
    yvalm = gprPrediction_systDown
    yvalp = gprPrediction_systUp
    yerr = gprSystErrors
    ax1.fill_between(bin_centers, yvalm, yvalp, color=col, alpha=0.2)

import matplotlib.lines as mlines

gprlab = mlines.Line2D([], [], color=col, markersize=10, label="Smoothed GPR")
orglab = mlines.Line2D([], [], color="k", marker=".", lw=0, markersize=10, label="Original histogram")

ax1.legend(handles=[orglab, gprlab], loc="upper right", frameon=0, bbox_to_anchor=(0.95, yloc))

ax1.set_ylabel("Events/bin")
ax1.set_xlabel(r"$m_{\mu\mu}$ [GeV]")

ax1.set_xlim([bin_edges[0], bin_edges[-1]])
ax1.set_ylim([0.01, ymax])

# prettyfy #1
from matplotlib import ticker

formatter = ticker.ScalarFormatter(useMathText=True)
formatter.set_scientific(True)
formatter.set_powerlimits((-1, 1))
ax1.yaxis.set_major_formatter(formatter)

ax1.tick_params(labeltop=False, labelright=False)
plt.xlabel(ax1.get_xlabel(), horizontalalignment="right", x=1.0)
plt.ylabel(ax1.get_ylabel(), horizontalalignment="right", y=1.0)
from matplotlib.ticker import AutoMinorLocator

ax1.xaxis.set_minor_locator(AutoMinorLocator())
ax1.yaxis.set_minor_locator(AutoMinorLocator())
ax1.tick_params(axis="y", direction="in", left=1, right=1, which="both")
ax1.tick_params(axis="x", direction="in", bottom=1, top=1, which="both")
ax1.tick_params(axis="both", which="major", length=12)
ax1.tick_params(axis="both", which="minor", length=6)

if logy:
    from matplotlib.ticker import LogLocator, NullFormatter

    ax1.set_yscale("log")
    locmin = LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8, 9, 10))
    locmin = LogLocator(base=10.0, subs=(2, 4, 6, 8, 10))
    ax1.yaxis.set_minor_locator(locmin)
    ax1.yaxis.set_minor_formatter(NullFormatter())
    finelab = ax1.yaxis.get_ticklabels()
    finelab[1].set_visible(False)

# ratio plot
zvals = ys / gys
ry = yerrs / ys
rb = gerrs / gys
zerrs = zvals * np.sqrt(ry * ry + rb * rb)
ax2.errorbar(bin_centers, zvals, xerr=xerrs, yerr=zerrs, fmt="none", color=col, markersize=10)
ax2.set_ylabel("Org/Smooth")


# fine y-label control for overlap
finelab = ax2.yaxis.get_ticklabels()
finelab[0].set_visible(False)
if not logy:
    finelab[-1].set_visible(False)


ax2.set_xlabel(r"$m_{\mu\mu}$ [GeV]")
ax2.set_xlim([bin_edges[0], bin_edges[-1]])
ax2.set_ylim([0.8, 1.2])
ax2.axhline(1, color="k", lw=1)

# prettyfy #2
ax2.tick_params(labeltop=False, labelright=False)
plt.xlabel(ax2.get_xlabel(), horizontalalignment="right", x=1.0)
plt.ylabel(ax2.get_ylabel(), horizontalalignment="center", y=0.5)
from matplotlib.ticker import AutoMinorLocator

ax2.xaxis.set_minor_locator(AutoMinorLocator())
ax2.yaxis.set_minor_locator(AutoMinorLocator())
ax2.tick_params(axis="y", direction="in", left=1, right=1, which="both")
ax2.tick_params(axis="x", direction="in", bottom=1, top=1, which="both")
ax2.tick_params(axis="both", which="major", length=12)
ax2.tick_params(axis="both", which="minor", length=6)

f.subplots_adjust(hspace=0)
plt.savefig("helpers/plots/GPR_smooth.pdf", format="pdf")
plt.show()
plt.close()
