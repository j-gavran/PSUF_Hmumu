import matplotlib.pyplot as plt
import numpy as np


def partonLumi_uubar(x):
    p0 = 0.00703528
    p1 = -0.000174816
    p2 = 1.89616e-06
    p3 = -1.12038e-08
    p4 = 3.7537e-11
    p5 = -6.71198e-14
    p6 = 4.98316e-17

    return (
        p0
        + p1 * x
        + p2 * x * x
        + p3 * x * x * x
        + p4 * x * x * x * x
        + p5 * x * x * x * x * x
        + p6 * x * x * x * x * x * x
    )


def partonLumi_ddbar(x):
    p0 = 0.0052163
    p1 = -0.000130677
    p2 = 1.42682e-06
    p3 = -8.48218e-09
    p4 = 2.85847e-11
    p5 = -5.14001e-14
    p6 = 3.83665e-17

    return (
        p0
        + p1 * x
        + p2 * x * x
        + p3 * x * x * x
        + p4 * x * x * x * x
        + p5 * x * x * x * x * x
        + p6 * x * x * x * x * x * x
    )


def partonLumi_ssbar(x):
    p0 = 0.00227443
    p1 = -5.80923e-05
    p2 = 6.40022e-07
    p3 = -3.81913e-09
    p4 = 1.28814e-11
    p5 = -2.3144e-14
    p6 = 1.72444e-17

    return (
        p0
        + p1 * x
        + p2 * x * x
        + p3 * x * x * x
        + p4 * x * x * x * x
        + p5 * x * x * x * x * x
        + p6 * x * x * x * x * x * x
    )


def BW(s, ZMass, ZWidth):
    return 1.0 / ((s - ZMass * ZMass) * (s - ZMass * ZMass) + (ZMass * ZMass * ZWidth * ZWidth))


def atlas_invMass_mumu_core(x):
    ZMass = 91.200000
    ZWidth = 2.490000
    alphaEM = 7.297352e-3
    Nc = 3
    Qd = -1.0 / 3.0
    Qu = 2.0 / 3.0
    Qs = -1.0 / 3.0
    sin2W = 0.23126
    GF = 1.1663787e-5
    gAmu = -0.5
    gAu = 0.5
    gAd = -0.5
    gAs = -0.5
    gVmu = -0.5 + 2 * sin2W
    gVu = 0.5 - 4.0 / 3.0 * sin2W
    gVd = -0.5 + 2.0 / 3.0 * sin2W
    gVs = -0.5 + 2.0 / 3.0 * sin2W

    s = x * x
    alphaEM2 = alphaEM * alphaEM
    kappa = np.sqrt(2) * GF * ZMass * ZMass / (4 * np.pi * alphaEM)
    BWgam = BWz = BW(s, ZMass, ZWidth)
    S_Zgam = kappa * s * (s - ZMass * ZMass) * BWgam
    S_Z = kappa * kappa * s * s * BWz

    ME_ddbar = (
        4
        * np.pi
        * alphaEM2
        / (3 * Nc * s)
        * ((Qd * Qd) - 2 * Qd * gVmu * gVd * S_Zgam + (gAmu * gAmu + gVmu * gVmu) * (gAd * gAd + gVd * gVd) * S_Z)
    )
    ME_uubar = (
        4
        * np.pi
        * alphaEM2
        / (3 * Nc * s)
        * ((Qu * Qu) - 2 * Qu * gVmu * gVu * S_Zgam + (gAmu * gAmu + gVmu * gVmu) * (gAu * gAu + gVu * gVu) * S_Z)
    )
    ME_ssbar = (
        4
        * np.pi
        * alphaEM2
        / (3 * Nc * s)
        * ((Qs * Qs) - 2 * Qs * gVmu * gVs * S_Zgam + (gAmu * gAmu + gVmu * gVmu) * (gAs * gAs + gVs * gVs) * S_Z)
    )

    return (ME_ddbar * partonLumi_ddbar(x) + ME_uubar * partonLumi_uubar(x) + ME_ssbar * partonLumi_ssbar(x)) * x


def atlas_invMass_mumu(add, x):
    # add - extra correction to the function (polynomial, root, ... ? )
    return add * atlas_invMass_mumu_core(x)


if __name__ == "__main__":
    # the custom add function - add polynomials or other functions here
    def poly(x):
        return 1.0

    # use curve_fit to fit the function

    plt.figure(figsize=(16, 8))
    x = np.linspace(110.0, 200.0, 100)
    plt.plot(x, atlas_invMass_mumu(poly(x), x), lw=2)
    plt.yscale("log")
    plt.tight_layout()
    plt.savefig("helpers/plots/atlas_fit_function.pdf")
    plt.show()
