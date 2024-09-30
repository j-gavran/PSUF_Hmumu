import os
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


def load_data(dataset, datadir):
    """Function for loading .h5 datasets"""
    infile = os.path.join(datadir, dataset + ".h5")

    print("Loading {}...".format(infile))

    store = pd.HDFStore(infile, "r")
    dataset = store["ntuple"]
    store.close()

    return dataset


def make_histograms(datadir_input, datadir_output, labels, datasets, n_bins, x_range=None, save_hist_name=None):
    if type(n_bins) is not int and x_range is not None:
        raise ValueError("If n_bins is a sequence, x_range must be None.")

    save_names = []

    for label, dataset in zip(labels, datasets):
        # Load dataset
        ds = load_data(dataset, datadir_input)

        print(f"Creating histogram for {label}...")

        # Get simulated (Background, Signal) or measured (Data) data
        all_events = ds["Muons_Minv_MuMu_Paper"]

        # Get correct weights
        wts = ds["CombWeight"]
        # for MC: sum of weights squared, for data: N
        wts2 = wts**2

        # Firstly, get correct number of bin_values
        bin_values, _ = np.histogram(all_events, bins=n_bins, range=x_range, weights=wts)  # wts!

        # Secondly, calculate bin_errors
        y, bin_edges = np.histogram(all_events, bins=n_bins, range=x_range, weights=wts2)  # wts2!
        bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
        bin_errors = np.sqrt(y)

        if save_hist_name is not None:
            save_name = f"{datadir_output}{save_hist_name}_{label}.npz"
        else:
            if x_range is None:
                save_name = f"{datadir_output}hist_range_custom_nbin-{len(n_bins)}_{label}.npz"
            else:
                save_name = f"{datadir_output}hist_range_{x_range[0]}-{x_range[1]}_nbin-{n_bins}_{label}.npz"

        save_names.append(save_name)

        # Save histogram to .npz file
        with open(save_name, "wb") as f:
            np.savez(f, bin_edges=bin_edges, bin_centers=bin_centers, bin_values=bin_values, bin_errors=bin_errors)

        f.close()

    return save_names


def load_histogram(path_to_hist, file_name, label, return_dct=False):
    """Load histogram from .npz file."""

    if label.lower() not in ["background", "signal", "data"]:
        raise ValueError("label must be one of ['Background', 'Signal', 'Data']")

    with np.load(f"{path_to_hist}/{file_name}_{label}.npz", "rb") as data:
        bin_edges = data["bin_edges"]
        bin_centers = data["bin_centers"]
        bin_values = data["bin_values"]
        bin_errors = data["bin_errors"]

    if return_dct:
        return {"centers": bin_centers, "edges": bin_edges, "values": bin_values, "errors": bin_errors}
    else:
        return bin_centers, bin_edges, bin_values, bin_errors


def url_download(url, data_dir, chunk_size=1024):
    """Downloads file from url to data_dir.

    Parameters
    ----------
    url : str
        URL of file to download.
    data_dir : str
        Downloaded in this directory (needs to exist).
    chunk_size : int, optional
        Chunk size for downloading, by default 1024

    Returns
    -------
    str
        File name of downloaded file.

    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    fname = data_dir + url.split("/")[-1]

    if Path(fname).is_file() is not True:
        print(f"started downloading from {url} ...")

        resp = requests.get(url, stream=True)
        total = int(resp.headers.get("content-length", 0))

        with open(fname, "wb") as file, tqdm(
            desc=fname, total=total, unit="iB", unit_scale=True, unit_divisor=1024
        ) as bar:
            for data in resp.iter_content(chunk_size=chunk_size):
                size = file.write(data)
                bar.update(size)
    else:
        print(f"already downloaded {fname}!")

    return fname


if __name__ == "__main__":
    # User defined: -> play around with n_bins and x_range
    datadir_input = "data/raw_data/"  # Directory to raw data, change this!
    datadir_output = "data/generated_histograms/"  # Directory to generated histograms, change/create this!

    # download files from cernbox
    url_download("https://cernbox.cern.ch/remote.php/dav/public-files/rd9b2rjzYxQ6jC2/mc_bkg_new.h5", datadir_input)
    url_download("https://cernbox.cern.ch/remote.php/dav/public-files/rd9b2rjzYxQ6jC2/mc_sig.h5", datadir_input)
    url_download("https://cernbox.cern.ch/remote.php/dav/public-files/rd9b2rjzYxQ6jC2/data.h5", datadir_input)

    x_range = (110, 160)  # m_mumu energy interval (110.,160.) GeV for Higgs

    # If n_bins is an int, it defines the number of equal-width bins in the given range.
    # If n_bins is a sequence, it defines a monotonically increasing array of bin edges,
    # including the rightmost edge, allowing for non-uniform bin widths.
    n_bins = 60

    ds_bkg = "mc_bkg_new"  # filename for Background simulations (.h5)
    ds_sig = "mc_sig"  # filename for Signal simulations (.h5)
    ds_data = "data"  # filename for Measured data (.h5)

    datasets = [ds_bkg, ds_sig, ds_data]
    labels = ["Background", "Signal", "Data"]

    dirs = make_histograms(
        datadir_input,
        datadir_output,
        labels,
        datasets,
        n_bins,
        x_range=x_range,
        save_hist_name="my_hist",
    )

    print(f"Saved histograms to {dirs} with {n_bins}.")
