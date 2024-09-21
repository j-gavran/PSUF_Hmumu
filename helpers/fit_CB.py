# ##################### #
# EXAMPLE: Crystal Ball #
# ##################### #

import numpy as np
from scipy.optimize import curve_fit


def CrystalBall(x, A, aL, aR, nL, nR, mCB, sCB):
    condlist = [
        (x - mCB) / sCB <= -aL,
        (x - mCB) / sCB >= aR,
    ]
    funclist = [
        lambda x: A
        * (nL / np.abs(aL)) ** nL
        * np.exp(-(aL**2) / 2)
        * (nL / np.abs(aL) - np.abs(aL) - (x - mCB) / sCB) ** (-nL),
        lambda x: A
        * (nR / np.abs(aR)) ** nR
        * np.exp(-(aR**2) / 2)
        * (nR / np.abs(aR) - np.abs(aR) + (x - mCB) / sCB) ** (-nR),
        lambda x: A * np.exp(-((x - mCB) ** 2) / (2 * sCB**2)),
    ]
    return np.piecewise(x, condlist, funclist)


if __name__ == "__main__":
    fontsize = 16

    # Load the data
    fileName = "data/original_histograms/mass_mm_higgs_Signal.npz"

    with np.load(fileName) as data:
        bin_edges = data["bin_edges"]
        bin_centers = data["bin_centers"]
        bin_values = data["bin_values"]
        bin_errors = data["bin_errors"]

    # Fit
    popt, pcov = curve_fit(
        CrystalBall,
        bin_centers,
        bin_values,
        sigma=bin_errors,
        p0=[133.0, 1.5, 1.5, 3.7, 9.6, 124.5, 3.0],
    )
    perr = np.sqrt(np.diag(pcov))
    A, aL, aR, nL, nR, mCB, sCB = popt

    # Print out the parameters
    print(
        "A: {A:.3f}\naL: {aL:.3f}\naR: {aR:.3f}\nnL: {nL:.3f}\nnR: {nR:.3f}\nmCB: {mCB:.3f}\nsCB: {sCB:.3f}".format(
            A=A,
            aL=aL,
            aR=aR,
            nL=nL,
            nR=nR,
            mCB=mCB,
            sCB=sCB,
        )
    )

    # Fit values for plotting
    xs = np.linspace(110, 160, 501)
    fit_values = np.array(CrystalBall(xs, A, aL, aR, nL, nR, mCB, sCB))
