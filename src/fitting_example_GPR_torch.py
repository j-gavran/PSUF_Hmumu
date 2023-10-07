import glob
import logging
import re

import gpytorch
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


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


class SklearnLikeRBFKernel(gpytorch.kernels.RBFKernel):
    def __init__(self, *args, add_to_daig=None, **kwargs):
        """See: https://github.com/scikit-learn/scikit-learn/blob/d99b728b3a7952b2111cf5e0cb5d14f92c6f3a80/sklearn/gaussian_process/_gpr.py#L343"""
        super().__init__(*args, **kwargs)
        if add_to_daig is not None:
            self.add_to_daig = torch.diag(add_to_daig)
        else:
            self.add_to_daig = add_to_daig

    def forward(self, *args, **params):
        if self.training:
            return super().forward(*args, **params) + self.add_to_daig
        else:
            return super().forward(*args, **params)


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, alpha=None):
        """See: https://docs.gpytorch.ai/en/stable/examples/01_Exact_GPs/Simple_GP_Regression.html#GPyTorch-Regression-Tutorial"""
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()

        if alpha is not None:
            self.covar_module = gpytorch.kernels.ScaleKernel(SklearnLikeRBFKernel(add_to_daig=alpha))
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def train_gpr(train_x, train_y, var=None, lr=0.1, epochs=100, plot_losses=True):
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

    if plot_losses:
        plt.plot(range(epochs), loss_lst)
        plt.title("Loss")
        plt.savefig("src/plots/gpr_loss.pdf")
        plt.close()

        plt.plot(range(epochs), lengthscale_lst)
        plt.title("Lengthscale")
        plt.savefig("src/plots/gpr_lengthscale.pdf")
        plt.close()

        plt.plot(range(epochs), noise_lst)
        plt.title("Noise")
        plt.savefig("src/plots/gpr_noise.pdf")
        plt.close()

    return model, likelihood


def test_gpr(model, likelihood, test_x, train_x=None, train_y=None, plot=True):
    # Get into evaluation (predictive posterior) mode
    model.eval()
    likelihood.eval()

    with torch.no_grad():
        # Make predictions by feeding model through likelihood
        observed_pred = likelihood(model(test_x))
        # Get upper and lower confidence bounds
        lower, upper = observed_pred.confidence_region()

    if plot:
        f, ax = plt.subplots(1, 1)
        # Plot training data as black stars
        ax.plot(train_x.numpy(), train_y.numpy(), "k*")
        # Plot predictive means as blue line
        ax.plot(test_x.numpy(), observed_pred.mean.numpy(), "b")
        # Shade between the lower and upper confidence bounds
        ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)

        return f, ax

    return [lower.numpy(), upper.numpy()], observed_pred.mean.numpy()


if __name__ == "__main__":
    import mplhep as hep

    from src.visualize_data import simple_histogram_plot

    hep.style.use(hep.style.ATLAS)

    histograms = prepare_histograms(gamma=100.0)

    # plot all regions
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
    plt.savefig("src/plots/regions.pdf")

    # gpr
    train_bkg = histograms["higgs"]["BlindData"]
    test = histograms["higgs"]["Data"]

    # rescale to [0, 1]
    train_bin_centers, train_bin_edges, train_bin_values, train_bin_errors = train_bkg
    test_bin_centers, test_bin_edges, test_bin_values, test_bin_errors = test

    train_x = torch.from_numpy(train_bin_centers.astype(np.float32))
    train_x = (train_x - train_x.min()) / (train_x.max() - train_x.min())

    train_y = torch.from_numpy(train_bin_values.astype(np.float32))
    train_y = (train_y - train_y.min()) / (train_y.max() - train_y.min())

    test_x = torch.linspace(train_bin_centers[0], train_bin_centers[-1], 100)
    test_x = (test_x - test_x.min()) / (test_x.max() - test_x.min())

    noise_std = torch.from_numpy(train_bin_errors.astype(np.float32))
    noise_var = noise_std**2 * (train_y / torch.from_numpy(train_bin_values))

    # fit
    model, likelihood = train_gpr(train_x, train_y, var=noise_var, lr=0.1, epochs=100)

    # predict and plot
    f, ax = test_gpr(model, likelihood, test_x, train_x, train_y, plot=True)

    unblind_test = histograms["higgs"]["Data"]
    bin_centers, bin_edges, bin_values, bin_errors = unblind_test

    bin_centers = (bin_centers - bin_centers.min()) / (bin_centers.max() - bin_centers.min())
    bin_values = (bin_values - bin_values.min()) / (bin_values.max() - bin_values.min())

    ax.scatter(bin_centers, bin_values, color="r", zorder=10, alpha=0.9, s=10)
    ax.legend(["Observed Data", "Mean", "Confidence", "Unblinded Data"])

    f.savefig("src/plots/GPR_torch.pdf")
