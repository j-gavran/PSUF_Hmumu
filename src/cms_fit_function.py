import numpy as np


def cms_fit(m, a_1, a_2, a_3):
    m_Z = 91.1876
    g_Z = 2.4952
    return np.exp(a_2 * m + a_3 * m ** 2) / ((m - m_Z) ** a_1 + (0.5 * g_Z) ** a_1)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    inFileName = "src/DATA/original_histograms/mass_mm_higgs_Background.npz"
    with np.load(inFileName) as data:
        bin_edges = data["bin_edges"]
        bin_centers = data["bin_centers"]
        bin_values = data["bin_values"]
        bin_errors = data["bin_errors"]

    popt, pcov = curve_fit(cms_fit, bin_centers, bin_values, sigma=bin_errors, p0=np.array([1.0, 1.0, 1.0]) * 0.001)

    print(popt)

    std = np.sqrt(np.diag(pcov))
    fit_values = cms_fit(bin_centers, *popt)

    print(fit_values)

    xerrs = 0.5 * (bin_edges[1:] - bin_edges[:-1])

    plt.figure()
    plt.errorbar(
        bin_centers, bin_values, bin_errors, xerrs, fmt="none", color="b", ecolor="b", label="Original histogram"
    )
    plt.plot(bin_centers, fit_values, "g-", label="fit")
    plt.legend()
    plt.tight_layout()
    plt.show()
