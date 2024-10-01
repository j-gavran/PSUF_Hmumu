# #################### #
# EXAMPLE: fit to data #
# #################### #

import matplotlib.pyplot as plt
import numpy as np
from atlas_fit_function import atlas_invMass_mumu
from scipy.optimize import curve_fit

########################################################################################################################
# User defined

labels = ["Background", "Signal", "Data"]

pldict = {}
for label in labels:
    with np.load("data/original_histograms/mass_mm_higgs_" + label + ".npz", "rb") as data:
        bin_centers = data["bin_centers"]
        bin_edges = data["bin_edges"]
        bin_values = data["bin_values"]
        bin_errors = data["bin_errors"]

    pldict[label] = [bin_centers, bin_edges, bin_values, bin_errors]

xs = np.linspace(bin_edges[0], bin_edges[-1], 301)
width = (np.max(xs) - np.min(xs)) / len(bin_centers)
chebyshev = False
save = True

#########################################################################################################################
# Fit simulated background

bin_centers = pldict["Background"][0]
bin_edges = pldict["Background"][1]
bin_values = pldict["Background"][2]
bin_errors = pldict["Background"][3]
xerrs = 0.5 * (bin_edges[1:] - bin_edges[:-1])


def Background(x, a, b, c, d, e, f, g, h, func=atlas_invMass_mumu):
    if chebyshev:
        # Choose Chebyshev polynomials as weights
        from numpy.polynomial.chebyshev import chebval

        return func(chebval(x, [a, b, c, d, e, f, g, h]), x)
    else:
        # Choose some weighting function
        return func(np.exp(a * x) + b * x**3 + c * x**2 + d * x + h, x)


def return_p0(chebyshev):
    if chebyshev:
        return (10**6, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    else:
        return (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 10**6)


popt, pcov = curve_fit(Background, bin_centers, bin_values, sigma=bin_errors, p0=return_p0(chebyshev))
perr = np.sqrt(np.diag(pcov))

f, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(16, 9), gridspec_kw={"height_ratios": [3, 1]})
f.suptitle("Fitting simulated background", fontsize=22)
ax1.errorbar(
    bin_centers,
    bin_values,
    bin_errors,
    xerrs,
    marker="o",
    markersize=5,
    color="k",
    ecolor="k",
    ls="",
    label="Original SimBkg histogram",
)
ax1.plot(xs, Background(xs, *popt), "r-", label="Simulated Background fit")
ax1.set_ylabel("Number of events", fontsize=20)
ax1.tick_params(axis="both", which="major", labelsize=20)
ax1.legend(fontsize=20)
# ax1.set_yscale('log')

ax2.bar(bin_centers, bin_values / Background(bin_centers, *popt) - 1, width=width, color="k")
ax2.axhline(0, color="k", ls="--", alpha=0.7)
ax2.set_xlabel(r"$m_{\mu \mu}$", fontsize=20)
ax2.set_ylabel("(Data-Pred.)/Pred.", fontsize=20)
ax2.set_xticks(bin_edges[::4])
ax2.tick_params(axis="both", which="major", labelsize=20)
ax2.grid(True)

f.tight_layout()
if save:
    plt.savefig("helpers/plots/SimBkg_fit.pdf")


########################################################################################################################
# Fit background from data

bin_centers = pldict["Data"][0]
bin_edges = pldict["Data"][1]
bin_values = pldict["Data"][2]
bin_errors = pldict["Data"][3]
xerrs = 0.5 * (bin_edges[1:] - bin_edges[:-1])

# Blind signal region
signal_region = (bin_centers < 120) | (bin_centers > 130)
bin_centers_masked = bin_centers[signal_region]
bin_values_masked = bin_values[signal_region]
bin_errors_masked = bin_errors[signal_region]
xerrs_masked = xerrs[signal_region]

popt_full, pcov_full = curve_fit(Background, bin_centers, bin_values, sigma=bin_errors, p0=return_p0(chebyshev))
popt_masked, pcov = curve_fit(
    Background, bin_centers_masked, bin_values_masked, sigma=bin_errors_masked, p0=return_p0(chebyshev)
)

f, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(16, 9), gridspec_kw={"height_ratios": [3, 1]})
f.suptitle("Fitting background from data", fontsize=22)

ax1.errorbar(
    bin_centers_masked,
    bin_values_masked,
    bin_errors_masked,
    xerrs_masked,
    marker="o",
    markersize=5,
    color="k",
    ecolor="k",
    ls="",
    label="Original Data histogram",
)
ax1.plot(xs, Background(xs, *popt_full), "g-", label="Data Background fit full", alpha=0.7)
ax1.plot(xs, Background(xs, *popt_masked), "r-", label="Data Background fit", alpha=0.7)
# ax1.text(140, 10**5, 'Full:\na: {:.3e}\nb: {:.3f}\nc: {:.3f}\nd: {:.3e}'.format(*popt_full), fontsize=20)
# ax1.text(150, 10**5, 'Blinded:\na: {:.3e}\nb: {:.3f}\nc: {:.3f}\nd: {:.3e}'.format(*popt_masked), fontsize=20)
ax1.set_ylabel("Number of events", fontsize=20)
ax1.tick_params(axis="both", which="major", labelsize=20)
ax1.legend(fontsize=20)
# ax1.set_yscale('log')

ax2.bar(
    bin_centers_masked,
    bin_values_masked / Background(bin_centers_masked, *popt_masked) - 1,
    width=width / 2.0,
    color="g",
)
ax2.bar(
    bin_centers_masked + width / 2.0,
    bin_values_masked / Background(bin_centers_masked, *popt_full) - 1,
    width=width / 2.0,
    color="r",
)
ax2.axhline(0, color="k", ls="--", alpha=0.7)
ax2.set_xlabel(r"$m_{\mu \mu}$", fontsize=20)
ax2.set_ylabel("(Data-Pred.)/Pred.", fontsize=20)
ax2.set_xticks(bin_edges[::4])
ax2.tick_params(axis="both", which="major", labelsize=20)
ax2.grid(True)

f.tight_layout()
if save:
    plt.savefig("helpers/plots/DataBkg_fit.pdf")


########################################################################################################################
# Extract signal (data - background fit)

extracted_signal = bin_values - Background(bin_centers, *popt_masked)

plt.figure(figsize=(12, 4.5))
plt.title("Extracted signal", fontsize=22)
plt.scatter(bin_centers, extracted_signal, color="k", label="Extracted signal")
plt.xlabel(r"$m_{\mu \mu}$", fontsize=20)
plt.ylabel("Number of events", fontsize=20)
plt.xticks(bin_edges[::4], bin_edges[::4].astype(int), size=20)
plt.yticks(size=20)
plt.legend(fontsize=20)
plt.tight_layout()
plt.grid()
if save:
    plt.savefig("helpers/plots/Extracted_signal.pdf")


########################################################################################################################
# Fit simulated signal

bin_centers = pldict["Signal"][0]
bin_edges = pldict["Signal"][1]
bin_values = pldict["Signal"][2]
bin_errors = pldict["Signal"][3]
xerrs = 0.5 * (bin_edges[1:] - bin_edges[:-1])


def CB(x, A, aL, aR, nL, nR, mCB, sCB):
    return np.piecewise(
        x,
        [(x - mCB) / sCB <= -aL, (x - mCB) / sCB >= aR],
        [
            lambda x: A
            * (nL / np.abs(aL)) ** nL
            * np.exp(-(aL**2) / 2)
            * (nL / np.abs(aL) - np.abs(aL) - (x - mCB) / sCB) ** (-nL),
            lambda x: A
            * (nR / np.abs(aR)) ** nR
            * np.exp(-(aR**2) / 2)
            * (nR / np.abs(aR) - np.abs(aR) + (x - mCB) / sCB) ** (-nR),
            lambda x: A * np.exp(-((x - mCB) ** 2) / (2 * sCB**2)),
        ],
    )


popt_CB, pcov = curve_fit(CB, bin_centers, bin_values, sigma=bin_errors, p0=[134.0, 1.5, 1.5, 5.0, 5.0, 124.6, 2.7])
perr = np.sqrt(np.diag(pcov))

plt.figure(figsize=(16, 9))
plt.title("Fitting simulated signal", fontsize=22)
plt.errorbar(
    bin_centers,
    bin_values,
    bin_errors,
    xerrs,
    marker="o",
    markersize=5,
    color="k",
    ecolor="k",
    ls="",
    label="Original histogram",
)
plt.plot(xs, CB(xs, *popt_CB), "r-", label="signal fit CB")
string = "CB parameters after fit:\n"
for par, pop in zip(
    [
        "A",
        r"$\alpha_L$",
        r"$\alpha_R$",
        r"$n_L$",
        r"$n_R$",
        r"$\overline{m}_{CB}$",
        r"$\sigma_{CB}$",
    ],
    popt_CB,
):
    string += par + f": {pop:.3f}\n"
plt.text(115, 75, string[:-1], size=20, bbox=dict(facecolor="none", edgecolor="gray", boxstyle="round,pad=0.5"))
plt.xlabel(r"$m_{\mu \mu}$", fontsize=20)
plt.ylabel("Number of events", fontsize=20)
plt.xticks(bin_edges[::4], bin_edges[::4].astype(int), size=20)
plt.yticks(size=20)
plt.legend(fontsize=20)
if save:
    plt.savefig("helpers/plots/CB_fit.pdf")


########################################################################################################################
# Scale our signal


def scale_signal(x, scale, popt_CB=popt_CB):
    return scale * CB(x, *popt_CB)


popt, pcov = curve_fit(scale_signal, bin_centers, extracted_signal, sigma=pldict["Data"][3], p0=[1.0])
NHiggs = int(popt_CB[0] * popt[0])

plt.figure(figsize=(12, 8))
plt.title("Fitted signal", fontsize=22)
plt.plot(xs, scale_signal(xs, popt), color="r", label="Fitted signal")
plt.errorbar(
    bin_centers,
    extracted_signal,
    pldict["Data"][3],
    xerrs,
    marker="o",
    markersize=5,
    color="k",
    ecolor="k",
    ls="",
    label="Extracted signal",
)
plt.text(
    130,
    -200,
    r"$\alpha_{{scale}} = {:.3f}$".format(*popt) + "\n" + r"$N_{{Higgs}} = {:d}$".format(NHiggs),
    size=20,
    bbox=dict(facecolor="w", edgecolor="gray", boxstyle="round,pad=0.5"),
)
plt.xlabel(r"$m_{\mu \mu}$", fontsize=20)
plt.ylabel("Number of events", fontsize=20)
plt.xticks(bin_edges[::4], bin_edges[::4].astype(int), size=20)
plt.yticks(size=20)
plt.legend(loc="upper right", fontsize=20)
plt.tight_layout()
plt.grid()
if save:
    plt.savefig("helpers/plots/final_fit.pdf")
