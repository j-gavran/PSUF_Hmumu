import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

inFileName = "src/DATA/original_histograms/mass_mm_higgs_Background.npz"
with np.load(inFileName) as data:
    bin_edges = data["bin_edges"]
    bin_centers = data["bin_centers"]
    bin_values = data["bin_values"]
    bin_errors = data["bin_errors"]


poly3 = lambda m, a, b, c, d: a + b * m + c * m ** 2 + d * m ** 3
popt, pcov = curve_fit(poly3, bin_centers, bin_values, sigma=bin_errors, p0=np.ones(4))

std = np.sqrt(np.diag(pcov))
fit_values = poly3(bin_centers, *popt)

xerrs = 0.5 * (bin_edges[1:] - bin_edges[:-1])

plt.figure()
plt.errorbar(bin_centers, bin_values, bin_errors, xerrs, fmt="none", color="b", ecolor="b", label="Original histogram")
plt.plot(bin_centers, fit_values, "g-", label="fit poly3")
plt.legend()
plt.tight_layout()
plt.show()
