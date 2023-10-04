import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
from matplotlib.ticker import AutoMinorLocator, LogLocator, NullFormatter


def visualize_histograms(
    labels,
    hist_name,
    path_to_original_histograms,
    output_filename,
    logy=True,
    ymax=5 * 10 ** 7,
    yloc=0.95,
    color_background="green",
    color_signal="red",
    color_data_bkg="blue",
    color_data="black",
    save=True,
):
    # Load data to dictionary pldict:
    pldict = {}

    for label in labels:
        with np.load(f"{path_to_original_histograms}{hist_name}_{label}.npz", "rb") as data:
            bin_edges = data["bin_edges"]
            bin_centers = data["bin_centers"]
            bin_values = data["bin_values"]
            bin_errors = data["bin_errors"]

        pldict[label] = [bin_centers, bin_edges, bin_values, bin_errors]

    f, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(6, 6), gridspec_kw={"height_ratios": [3, 1]})

    xs = pldict["Data"][0]
    xedges = pldict["Data"][1]
    xwidths = xedges[1:] - xedges[:-1]
    xerrs = 0.4 * xwidths
    ys = pldict["Data"][2]
    yerrs = pldict["Data"][3]
    bkgs = pldict["Background"][2]
    berrs = pldict["Background"][3]
    sigs = pldict["Signal"][2]
    serrs = pldict["Signal"][3]

    ax1.errorbar(xs, ys, xerr=xerrs, yerr=yerrs, fmt=".", color=color_data, markersize=2)
    ax1.hist(xs, xedges, weights=bkgs, histtype="step", color=color_background)
    ax1.hist(xs, xedges, weights=sigs, histtype="step", color=color_signal)
    ax1.fill_between(xs, sigs - serrs, sigs + serrs, color=color_signal, alpha=0.2)

    # Fine control over labels - histograms as lines, data with marker only...
    siglab = mlines.Line2D([], [], color=color_signal, markersize=10, label="Signal")
    bkglab = mlines.Line2D([], [], color=color_background, markersize=10, label="Background")
    dlab = mlines.Line2D([], [], color=color_data, marker=".", lw=0, markersize=10, label="Data")

    ax1.legend(handles=[siglab, bkglab, dlab], loc="upper right", frameon=0, bbox_to_anchor=(0.95, yloc))

    ax1.set_ylabel("Events/bin")
    ax1.set_xlabel(r"$m_{\mu\mu}$ [GeV]")

    ax1.set_xlim([xedges[0], xedges[-1]])
    ax1.set_ylim([0.01, ymax])

    # Prettyfy #1
    formatter = ticker.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-1, 1))
    ax1.yaxis.set_major_formatter(formatter)

    ax1.tick_params(labeltop=False, labelright=False)
    plt.xlabel(ax1.get_xlabel(), horizontalalignment="right", x=1.0)
    plt.ylabel(ax1.get_ylabel(), horizontalalignment="right", y=1.0)
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())
    ax1.tick_params(axis="y", direction="in", left=1, right=1, which="both")
    ax1.tick_params(axis="x", direction="in", bottom=1, top=1, which="both")
    ax1.tick_params(axis="both", which="major", length=12)
    ax1.tick_params(axis="both", which="minor", length=6)

    if logy:
        ax1.set_yscale("log")
        locmin = LogLocator(base=10.0, subs=(0.2, 0.4, 0.6, 0.8, 1, 2, 4, 6, 8, 9, 10))
        locmin = LogLocator(base=10.0, subs=(2, 4, 6, 8, 10))
        ax1.yaxis.set_minor_locator(locmin)
        ax1.yaxis.set_minor_formatter(NullFormatter())
        finelab = ax1.yaxis.get_ticklabels()
        finelab[1].set_visible(False)

    zvals = ys / bkgs
    ry = yerrs / ys
    rb = berrs / bkgs
    zerrs = zvals * np.sqrt(ry * ry + rb * rb)
    ax2.errorbar(xs, zvals, xerr=xerrs, yerr=zerrs, fmt="none", color=color_data_bkg, markersize=10)
    ax2.set_ylabel("Data/Bkg")

    # Fine y-label control for overlap
    finelab = ax2.yaxis.get_ticklabels()
    finelab[0].set_visible(False)
    if not logy:
        finelab[-1].set_visible(False)

    ax2.set_xlabel(r"$m_{\mu\mu}$ [GeV]")
    ax2.set_xlim([xedges[0], xedges[-1]])
    ax2.set_ylim([0.8, 1.2])
    ax2.axhline(1, color="k", lw=1)

    # Prettyfy #2
    ax2.tick_params(labeltop=False, labelright=False)
    plt.xlabel(ax2.get_xlabel(), horizontalalignment="right", x=1.0)
    plt.ylabel(ax2.get_ylabel(), horizontalalignment="center", y=0.5)

    ax2.xaxis.set_minor_locator(AutoMinorLocator())
    ax2.yaxis.set_minor_locator(AutoMinorLocator())
    ax2.tick_params(axis="y", direction="in", left=1, right=1, which="both")
    ax2.tick_params(axis="x", direction="in", bottom=1, top=1, which="both")
    ax2.tick_params(axis="both", which="major", length=12)
    ax2.tick_params(axis="both", which="minor", length=6)

    f.subplots_adjust(hspace=0)

    if save:
        plt.savefig(output_filename, format="pdf")
        plt.show()
        plt.close()
    else:
        return f, (ax1, ax2)


if __name__ == "__main__":
    labels = ["Background", "Signal", "Data"]
    hist_name = "mass_mm_higgs"

    path_to_original_histograms = "src/DATA/original_histograms/"
    output_filename = f"src/plots/{hist_name}.pdf"

    logy = True  # logscale
    ymax = 5 * 10 ** 7  # graph ymax value
    yloc = 0.95  # legend y location

    color_background = "green"
    color_signal = "red"
    color_data_bkg = "blue"
    color_data = "black"

    visualize_histograms(labels, hist_name, path_to_original_histograms, output_filename)
