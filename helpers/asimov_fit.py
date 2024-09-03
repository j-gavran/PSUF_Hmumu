# ################### #
# EXAMPLE: Asimov fit #
# ################### #

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from atlas_fit_function import atlas_invMass_mumu

data_path = "src/DATA/generated_histograms"
histo_name = "hist_range_110-160_nbin-25_Asimov"
labels = ["Background", "Signal", "Data"]
save_figs = False

########################################################################################################################
# Load data

pldict = {}
for label in labels:
    fileName = os.path.join(data_path, histo_name + label + ".npz")
    with np.load(fileName, "rb") as data:
        bin_centers = data["bin_centers"]
        bin_edges = data["bin_edges"]
        bin_values = data["bin_values"]
        bin_errors = data["bin_errors"]

    pldict[label] = [bin_centers, bin_edges, bin_values, bin_errors]

xs = np.linspace(bin_edges[0], bin_edges[-1], 301)
width = (np.max(xs) - np.min(xs)) / len(bin_centers)


########################################################################################################################
# Helper fit functions


def BackgroundFit(x, a, b, c, d, e):
    """Some random function as a weight to the atlas_fit_function"""
    return atlas_invMass_mumu(np.exp(a * x) + b * x ** 3 + c * x ** 2 + d * x + e, x)


def CrystalBall(x, A, aL, aR, nL, nR, mCB, sCB):
    """Double-Sided Crystal Ball Function, see e.g. arXiv:2009.04363v2"""
    np.seterr(all="ignore")  # Turn off some annoying warnings
    return np.piecewise(
        x,
        [(x - mCB) / sCB <= -aL, (x - mCB) / sCB >= aR],
        [
            lambda x: A
            * (nL / np.abs(aL)) ** nL
            * np.exp(-(aL ** 2) / 2)
            * (nL / np.abs(aL) - np.abs(aL) - (x - mCB) / sCB) ** (-nL),
            lambda x: A
            * (nR / np.abs(aR)) ** nR
            * np.exp(-(aR ** 2) / 2)
            * (nR / np.abs(aR) - np.abs(aR) + (x - mCB) / sCB) ** (-nR),
            lambda x: A * np.exp(-((x - mCB) ** 2) / (2 * sCB ** 2)),
        ],
    )


########################################################################################################################
# Fit background from data

bin_centers = pldict["Data"][0]
bin_edges = pldict["Data"][1]
bin_values = pldict["Data"][2]
bin_errors = pldict["Data"][3]
xerrs = 0.5 * (bin_edges[1:] - bin_edges[:-1])

# Blind signal region
signal_region = (117, 135)
mask = (bin_centers < signal_region[0]) | (bin_centers > signal_region[1])
bin_centers_masked = bin_centers[mask]
bin_values_masked = bin_values[mask]
bin_errors_masked = bin_errors[mask]
xerrs_masked = xerrs[mask]

popt_masked, pcov = curve_fit(
    BackgroundFit,
    bin_centers_masked,
    bin_values_masked,
    sigma=bin_errors_masked,
    p0=[0.23, -3.5e10, 1.2e13, -1.5e15, 6e16],
)

f, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(16, 9), gridspec_kw={"height_ratios": [3, 1]})
f.suptitle("Fitting background from data", fontsize=22)

ax1.errorbar(
    bin_centers, bin_values, bin_errors, xerrs, marker="o", markersize=5, color="k", ecolor="k", ls="", label="Data"
)
ax1.plot(xs, BackgroundFit(xs, *popt_masked), "r-", label="Blinded Data fit")
ax1.axvspan(signal_region[0], signal_region[1], color="gainsboro", lw=2, ls="--", ec="k")
ax1.set_ylabel("Number of events", fontsize=20)
ax1.text(123, 2 * 10 ** 5, "Blinded area", size=20)
ax1.tick_params(axis="both", which="major", labelsize=20)
ax1.legend(fontsize=20)
# ax1.set_yscale('log')

ax2.bar(
    bin_centers_masked, bin_values_masked / BackgroundFit(bin_centers_masked, *popt_masked) - 1, width=width, color="k"
)
ax2.axhline(0, color="k", ls="--", alpha=0.7)
ax2.set_xlabel(r"$m_{\mu \mu}$", fontsize=20)
ax2.set_ylabel("(Data-Pred.)/Pred.", fontsize=20)
ax2.set_xticks(bin_edges[::2])
ax2.tick_params(axis="both", which="major", labelsize=20)
ax2.grid(True)

f.tight_layout()
if save_figs:
    plt.savefig("DataBkgFit.pdf")


########################################################################################################################
# Fit simiulated signal with the crystal ball function

bin_centers = pldict["Signal"][0]
bin_edges = pldict["Signal"][1]
bin_values = pldict["Signal"][2]
bin_errors = pldict["Signal"][3]

popt_CB, pcov = curve_fit(
    CrystalBall, bin_centers, bin_values, sigma=bin_errors, p0=[2.6 * 10 ** 4.0, 1.5, 1.5, 5.0, 5.0, 124.6, 2.5]
)

f, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(16, 12), gridspec_kw={"height_ratios": [3, 1]})
f.suptitle("Fitting inflated simulated signal", fontsize=22)

ax1.scatter(bin_centers, bin_values, color="k", label="Simulated 100x signal")
ax1.plot(xs, CrystalBall(xs, *popt_CB), color="r", label="Fitted signal")
ax1.axvline(popt_CB[-2], color="k", ls="--", alpha=0.3)
ax1.annotate(
    r"$m_{{Higgs}}^{{fit}}= {:.2f}$ GeV".format(popt_CB[-2])
    + "\n"
    + r"$N_{{Higgs}}^{{exp}} = {:d}$".format(int(popt_CB[0])),
    (popt_CB[-2], 10000),
    xytext=(148, 2000),
    fontsize=20,
    arrowprops=dict(facecolor="black", shrink=0.05),
    size=20,
    bbox=dict(facecolor="w", edgecolor="gray", boxstyle="round,pad=0.5"),
)
string = "CB parameters after fit:\n"
for par, pop in zip(
    ["A", r"$\alpha_L$", r"$\alpha_R$", r"$n_L$", r"$n_R$", r"$\overline{m}_{CB}$", r"$\sigma_{CB}$"], popt_CB
):
    string += par + f": {pop:.3f}\n"
ax1.text(148, 10 ** 4, string[:-1], size=20, bbox=dict(facecolor="w", edgecolor="gray", boxstyle="round,pad=0.5"))
ax1.legend(loc="upper right", fontsize=20)
ax1.set_ylabel("Number of events", fontsize=20)
ax1.set_xticks(bin_edges[::2])
ax1.tick_params(axis="both", which="major", labelsize=20)
ax1.grid(True)

ax2.bar(bin_centers, bin_values / CrystalBall(bin_centers, *popt_CB) - 1, width=width, color="k")
ax2.axhline(0, color="k", ls="--", alpha=0.7)
ax2.set_xlabel(r"$m_{\mu \mu}$", fontsize=20)
ax2.set_ylabel("(Data-Pred.)/Pred.", fontsize=20)
ax2.set_xticks(bin_edges[::2])
ax2.tick_params(axis="both", which="major", labelsize=20)
ax2.grid(True)

f.tight_layout()
if save_figs:
    plt.savefig("AsimovSimSignalFit.pdf")


########################################################################################################################
# Extract and fit our signal

# Signal = Data - Fitted Background
bin_values = pldict["Data"][2]
extracted_signal = bin_values - BackgroundFit(bin_centers, *popt_masked)

bin_centers = pldict["Signal"][0]
bin_edges = pldict["Signal"][1]
bin_values = pldict["Signal"][2]
bin_errors = pldict["Signal"][3]


def scale_signal(x, scale, popt_CB=popt_CB):
    return scale * CrystalBall(x, *popt_CB)


# Fit extracted signal with the crystal ball function
popt_final, pcov = curve_fit(scale_signal, bin_centers, extracted_signal, sigma=bin_errors, p0=[1.0])
NHiggs = int(popt_CB[0] * popt_final[0])


f, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(16, 12), gridspec_kw={"height_ratios": [3, 1]})
f.suptitle("Fitting extracted signal", fontsize=22)

ax1.plot(xs, scale_signal(xs, *popt_final), color="r", label="Fitted signal")
ax1.scatter(bin_centers, extracted_signal, color="k", label="Extracted signal")
ax1.axvline(popt_CB[-2], color="k", ls="--", alpha=0.3)
ax1.text(
    110,
    15000,
    r"$\alpha_{{scale}} = {:.3f}$".format(*popt_final) + "\n" + r"$N_{{Higgs}} = {:d}$".format(NHiggs),
    size=25,
    bbox=dict(facecolor="w", edgecolor="gray", boxstyle="round,pad=0.5"),
)
ax1.legend(loc="upper right", fontsize=20)
ax1.set_ylabel("Number of events", fontsize=20)
ax1.set_xticks(bin_edges[::2])
ax1.tick_params(axis="both", which="major", labelsize=20)
ax1.set_xlim((108, 140))  # Plot only relevant part of the signal
ax1.grid(True)

ax2.bar(bin_centers, extracted_signal / scale_signal(bin_centers, *popt_final) - 1, width=width, color="k")
ax2.axhline(0, color="k", ls="--", alpha=0.7)
ax2.set_xlabel(r"$m_{\mu \mu}$", fontsize=20)
ax2.set_ylabel("(Data-Pred.)/Pred.", fontsize=20)
ax2.set_xticks(bin_edges[::2])
ax2.tick_params(axis="both", which="major", labelsize=20)
ax2.set_xlim((108, 140))  # Plot only relevant part of the signal
ax2.set_ylim((-2, 2))
ax2.grid(True)

f.tight_layout()
if save_figs:
    plt.savefig("AsimovSignalFit.pdf")
