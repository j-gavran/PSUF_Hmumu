import glob
import logging
import re

import gpytorch
import matplotlib.pyplot as plt
import mplhep as hep
import numpy as np
import torch
from tqdm import tqdm
from uncertainties import unumpy

from helpers.visualize_data import simple_histogram_plot


def load_histogram(hist_file):
    with np.load(hist_file, "rb") as data:
        bin_edges = data["bin_edges"]
        bin_centers = data["bin_centers"]
        bin_values = data["bin_values"]
        bin_errors = data["bin_errors"]

    return [bin_centers, bin_edges, bin_values, bin_errors]


def prepare_histograms(saved_hist_path="src/DATA/original_histograms/", gamma=0.0):
    original_histograms = [f for f in glob.glob(saved_hist_path + "*.npz")]
    assert len(original_histograms) > 0, "No histograms found"

    histograms = dict()

    for f in original_histograms:
        match = re.search(r"mass_mm_(\w+)", f)  # needs to start with mass_mm_
        if match:
            region, label = match.group(1).split("_")
        else:
            logging.warning(f"no match for {f}")

        if region not in histograms:
            histograms[region] = dict()

        histograms[region][label] = load_histogram(f)

    # Asimov
    for region, labels in histograms.items():
        d_bin_centers, d_bin_edges, d_bin_values, d_bin_errors = labels["Data"]
        s_bin_centers, s_bin_edges, s_bin_values, s_bin_errors = labels["Signal"]

        histograms[region]["AsimovData"] = [
            d_bin_centers,
            d_bin_edges,
            d_bin_values + gamma * s_bin_values,
            np.sqrt(d_bin_errors**2 + (gamma * s_bin_errors) ** 2),
        ]
        histograms[region]["AsimovSignal"] = [
            s_bin_centers,
            s_bin_edges,
            s_bin_values + gamma * s_bin_values,
            s_bin_errors if gamma == 0.0 else gamma * s_bin_errors,
        ]

    # blinding
    for region, labels in histograms.items():
        bin_centers, bin_edges, bin_values, bin_errors = labels["Data"]
        blind_idx_c = (bin_centers <= 120) | (bin_centers >= 130)
        blind_idx_e = (bin_edges <= 120) | (bin_edges > 130)
        histograms[region]["BlindData"] = [
            bin_centers[blind_idx_c],
            bin_edges[blind_idx_e],
            bin_values[blind_idx_c],
            bin_errors[blind_idx_c],
        ]

    # MC
    for region, labels in histograms.items():
        b_bin_centers, b_bin_edges, b_bin_values, b_bin_errors = histograms[region]["Background"]
        s_bin_centers, s_bin_edges, s_bin_values, s_bin_errors = histograms[region]["Signal"]
        as_bin_centers, as_bin_edges, as_bin_values, as_bin_errors = histograms[region]["AsimovSignal"]

        histograms[region]["SignalPlusBackgroundMC"] = [
            b_bin_centers,
            b_bin_edges,
            b_bin_values + s_bin_values,
            np.sqrt(b_bin_errors**2 + s_bin_errors**2),
        ]
        histograms[region]["AsimovSignalPlusBackgroundMC"] = [
            b_bin_centers,
            b_bin_edges,
            b_bin_values + as_bin_values,
            np.sqrt(b_bin_errors**2 + as_bin_errors**2),
        ]

    return histograms


class RescaleData:
    def __init__(self, rescale_type):
        self.rescale_type = rescale_type
        self.unumpy_x = None
        self.params = None

    def set_data(self, x, x_std=None):
        if x_std is None:
            x_std = np.zeros_like(x)

        # Do error propagation automatically - https://en.wikipedia.org/wiki/Propagation_of_uncertainty
        self.unumpy_x = unumpy.uarray(x, x_std)
        return self

    def __call__(self, inverse=False):
        if self.rescale_type == "minmax":
            return self.minmax_rescale(inverse)
        elif self.rescale_type == "log":
            return self.log_rescale(inverse)
        elif self.rescale_type is None:
            return unumpy.nominal_values(self.unumpy_x), unumpy.std_devs(self.unumpy_x)
        else:
            raise ValueError(f"Unknown rescale type {self.rescale_type}")

    def minmax_rescale(self, inverse):
        if inverse:
            res = self.params[0] + (self.params[1] - self.params[0]) * self.unumpy_x
            return unumpy.nominal_values(res), unumpy.std_devs(res)
        else:
            self.params = [self.unumpy_x.min(), self.unumpy_x.max()]
            res = (self.unumpy_x - self.params[0]) / (self.params[1] - self.params[0])
            return unumpy.nominal_values(res), unumpy.std_devs(res)

    def log_rescale(self, inverse):
        if inverse:
            res = unumpy.exp(self.unumpy_x)
            return unumpy.nominal_values(res), unumpy.std_devs(res)
        else:
            res = unumpy.log(self.unumpy_x)
            return unumpy.nominal_values(res), unumpy.std_devs(res)


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, alpha=None):
        """See: https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Simple_GP_Regression.html#GPyTorch-Regression-Tutorial"""
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        if alpha is not None:
            self.alpha = torch.diag(alpha)
        else:
            self.alpha = None

        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        if self.alpha is not None and self.training:
            covar_x = self.covar_module(x) + self.alpha
        else:
            covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_gpr(train_x, train_y, var=None, lr=0.1, epochs=100):
    # Transform to torch tensors
    if not torch.is_tensor(train_x):
        train_x = torch.from_numpy(train_x.astype(np.float32))
    if not torch.is_tensor(train_y):
        train_y = torch.from_numpy(train_y.astype(np.float32))
    if var is not None and not torch.is_tensor(var):
        var = torch.from_numpy(var.astype(np.float32))

    # Initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    model = ExactGPModel(train_x, train_y, likelihood, alpha=var)

    # Find optimal model hyperparameters
    model.train()
    likelihood.train()

    # Use the adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # Includes GaussianLikelihood parameters

    # "Loss" for GPs - the marginal log likelihood
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    loss_lst, lengthscale_lst, noise_lst = [], [], []
    for _ in tqdm(range(epochs), desc="Training GPR"):
        optimizer.zero_grad()

        output = model(train_x)

        loss = -mll(output, train_y)
        loss.backward()

        loss_lst.append(loss.item())
        lengthscale_lst.append(model.covar_module.base_kernel.lengthscale.item())
        noise_lst.append(model.likelihood.noise.item())

        optimizer.step()

    return model, likelihood, [loss_lst, lengthscale_lst, noise_lst]


def test_gpr(model, likelihood, test_x):
    # Transform to torch tensors
    if not torch.is_tensor(test_x):
        test_x = torch.from_numpy(test_x.astype(np.float32))

    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        # Make predictions by feeding model through likelihood
        observed_pred = likelihood(model(test_x))
        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()

    return [lower.numpy(), upper.numpy()], observed_pred.mean.numpy()


def plot_all_histogram_regions(histograms, path="src/plots/"):
    simple_histogram_plot(histograms["all"]["BlindData"], color="k", ls="--", zorder=10)
    simple_histogram_plot(histograms["higgs"]["Signal"], color="r")
    simple_histogram_plot(histograms["higgs"]["AsimovData"], color="b")
    simple_histogram_plot(histograms["all"]["SignalPlusBackgroundMC"], color="g")

    plt.legend(["Blinded data", r"$\gamma \times$Signal", "Asimov dataset", "MC Signal + MC Background"], fontsize=12)
    plt.yscale("log")
    plt.ylabel("$N$")
    plt.xlabel(r"$m_{\mu\mu}$")
    plt.axvline(120, ls="--", c="k", lw=1)
    plt.axvline(130, ls="--", c="k", lw=1)
    plt.tight_layout()
    plt.savefig(path + "regions.pdf")
    plt.show()
    plt.close()


def plot_losses(loss_lst, lengthscale_lst, noise_lst, path="src/plots/"):
    epochs = len(loss_lst)

    plt.plot(range(epochs), loss_lst, lw=2)
    plt.ylabel("Loss")
    plt.xlabel("epochs")
    plt.savefig(path + "GPR_loss.pdf")
    plt.show()
    plt.close()

    plt.plot(range(epochs), lengthscale_lst, lw=2)
    plt.ylabel("Lengthscale")
    plt.xlabel("epochs")
    plt.savefig(path + "GPR_lengthscale.pdf")
    plt.show()
    plt.close()

    plt.plot(range(epochs), noise_lst, lw=2)
    plt.ylabel("Noise")
    plt.xlabel("epochs")
    plt.savefig(path + "GPR_noise.pdf")
    plt.show()
    plt.close()


def plot_predictions(lower_upper, train_x, train_y, test_x, mean, true_x=None, true_y=None, path="src/plots/"):
    lower, upper = lower_upper

    f, ax = plt.subplots(1, 1)
    ax.plot(train_x, train_y, "k*", label="Train data")
    ax.plot(test_x, mean, "b", label="Predicted mean")
    ax.fill_between(test_x, lower, upper, alpha=0.5, label="Confidence")

    if true_x is not None and true_y is not None:
        ax.scatter(true_x, true_y, color="r", zorder=10, alpha=0.9, s=10, label="True labels")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    plt.tight_layout()
    f.savefig(path + "GPR_predictions.pdf")
    plt.show()
    plt.close()


if __name__ == "__main__":
    # good tutorial for simple GP with gpytorch: https://d2l.ai/chapter_gaussian-processes/gp-inference.html

    hep.style.use(hep.style.ATLAS)

    histograms = prepare_histograms(gamma=100.0)

    # plot all regions
    plot_all_histogram_regions(histograms)

    # None, "minmax", "log"
    rescale_type = "minmax"

    # train data
    bin_centers, bin_edges, bin_values, bin_errors = histograms["higgs"]["BlindData"]
    train_y_data = RescaleData(rescale_type).set_data(bin_values, bin_errors)
    train_x_data = RescaleData(rescale_type).set_data(bin_centers, np.diff(bin_edges))

    # test data
    bin_centers, bin_edges, bin_values, bin_errors = histograms["higgs"]["Data"]
    test_y_data = RescaleData(rescale_type).set_data(bin_values, bin_errors)
    test_x_data = RescaleData(rescale_type).set_data(bin_centers, np.diff(bin_edges))

    # call rescale
    train_y, train_y_std = train_y_data()
    train_x, _ = train_x_data()

    test_y, test_y_std = test_y_data()
    test_x, test_x_std = test_x_data()

    # train GPR
    model, likelihood, losses = train_gpr(train_x, train_y, var=None, epochs=500)
    # model, likelihood, losses = train_gpr(train_x, train_y, var=train_y_std**2, epochs=500)
    plot_losses(*losses)

    # test GPR
    lower_upper, mean = test_gpr(model, likelihood, test_x)
    plot_predictions(lower_upper, train_x, train_y, test_x, mean, true_x=test_x, true_y=test_y)

    # scale back predictions
    std = lower_upper[1] - lower_upper[0]

    train_y_data.set_data(mean, std)
    test_x_data.set_data(test_x, test_x_std)

    # final prediction with error
    yp, yp_std = train_y_data(inverse=True)
    x, x_std = test_x_data(inverse=True)

    # save final prediction
    histograms["higgs"]["Predicted"] = [x, bin_edges, yp, yp_std]

    # plot final result
    plt.plot(x, yp, ls="--", c="k", lw=2, label="fit")
    plt.fill_between(x, yp - yp_std, yp + yp_std, alpha=0.5, label="confidence")
    plt.legend()
    plt.xlabel(r"$m_{\mu\mu}$")
    plt.ylabel(r"$N$")
    plt.tight_layout()
    plt.savefig("src/plots/GPR_torch_final_result.pdf")
    plt.show()
    plt.close()
