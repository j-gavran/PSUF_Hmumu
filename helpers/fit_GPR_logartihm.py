# ######################### #
# EXAMPLE: GPR in log scale #
# ######################### #

import numpy as np
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor

# Different kernels available in sklearn:
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern

plt.rcParams.update({"axes.labelsize": 16, "xtick.labelsize": 14, "ytick.labelsize": 14})
# Set random seed for same results
np.random.seed(12345)

# Load data
inFileName = "data/original_histograms/mass_mm_higgs_Background.npz"
with np.load(inFileName) as data:
    bin_centers = data["bin_centers"]
    bin_values = np.log(data["bin_values"])
    bin_errors = data["bin_errors"] / data["bin_values"]  # How does the error scale when taking f(x) = ln(x)?

# Mask
signal_region = (119, 131)
mask = (bin_centers < signal_region[0]) | (bin_centers > signal_region[1])
bin_centers_masked = bin_centers[mask]
bin_values_masked = bin_values[mask]
bin_errors_masked = bin_errors[mask]

# Set hyper-parameter bounds for ConstantKernel
nEvts = np.max(bin_values)
const0 = 1.0
const_low = 1e-1
const_hi = 1e3

# Set hyper-parameter bounds for RBF kernel
RBF0 = 1.0
RBF_low = 1e-1
RBF_high = 1e2

# A) Define kernel: ConstantKernel * RBF:
kernel_RBF = ConstantKernel(const0, constant_value_bounds=(const_low, const_hi)) * RBF(
    RBF0, length_scale_bounds=(RBF_low, RBF_high)
)

# B) Define kernel: ConstantKernel * Matern:
kernel_Matern = ConstantKernel(const0, constant_value_bounds=(const_low, const_hi)) * Matern(
    RBF0, length_scale_bounds=(RBF_low, RBF_high), nu=1.5
)

# Transform x data into 2d vector!
X = np.atleast_2d(bin_centers_masked).T  # true datapoints
X_to_predict = np.atleast_2d(np.linspace(110, 160, 1000)).T  # what to predict
y = bin_values_masked

# Initialize Gaussian Process Regressor: !!! alpha = bin_errors, 2*bin_errors or bin_errors**2? Your task to figure out!!!
gp = GaussianProcessRegressor(kernel=kernel_RBF, n_restarts_optimizer=1, alpha=bin_errors_masked**2)

# Fit on X with values y
gp.fit(X, y)
print("Final kernel combination:\n", gp.kernel_)

# Predict
y_pred, sigma = gp.predict(X_to_predict, return_std=True)
y_pred_sparse, sigma_sparse = gp.predict(np.atleast_2d(bin_centers).T, return_std=True)

fig, axes = plt.subplot_mosaic([["main"], ["main"], ["main"], ["ratio"]], sharex=True, figsize=(8, 8))

# Main pad
axes["main"].set_title("Example GPR with RBF kernel", fontsize=20, fontweight="bold")
axes["main"].fill_between(X_to_predict.ravel(), y_pred - sigma, y_pred + sigma)
axes["main"].scatter(bin_centers_masked, bin_values_masked, color="r", linewidth=0.5, marker="o", s=25, label="Data")
axes["main"].scatter(
    bin_centers[~mask], bin_values[~mask], color="g", marker="+", s=100, label="Blinded data (not used in the fit)"
)
axes["main"].plot(X_to_predict, y_pred, color="k", label="GPR Prediction")
axes["main"].set_ylabel("ln(events/bin)", fontsize=16)
axes["main"].set_ylim((8, 12))
axes["main"].legend(fontsize=16)

# Ratio pad
axes["ratio"].errorbar(
    bin_centers, bin_values / y_pred_sparse, yerr=sigma_sparse, color="k", linewidth=0.0, elinewidth=0.5, marker="."
)
axes["ratio"].axhline(1, c="k", lw=1, alpha=0.7)
axes["ratio"].set_xlabel(r"$m_{\mu\mu}$ [GeV]", fontsize=16)
axes["ratio"].set_ylabel("Data/Pred.", fontsize=16)

# Make ratio plot labels symmetric around 1.
max = np.max(np.abs(axes["ratio"].get_yticks() - 1.0)) / 1.5
axes["ratio"].set_ylim((1.0 - max, 1.0 + max))
axes["ratio"].grid()

# Make an inner plot
axes["main"].plot([113, 119], [9.9, 10.9], "k--")
axes["main"].plot([138, 131], [9.9, 10.3], "k--")
ax_inner = fig.add_axes([0.2125, 0.3625, 0.4, 0.25])
ax_inner.set_title("zoom-in", fontsize=20)
ax_inner.scatter(bin_centers[~mask], bin_values[~mask], color="g", marker="+", s=100)
ax_inner.plot(X_to_predict, y_pred, color="k")
ax_inner.set_xlim(signal_region)
ax_inner.set_ylim((10.3, 11))
ax_inner.grid()

plt.savefig("helpers/plots/GPR_log.pdf")
