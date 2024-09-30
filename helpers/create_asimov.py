import os

import numpy as np


def make_asimov_data(multiply, data_path, histo_name, labels, save_asimov):
    """The Asimov dataset is a hypothetical dataset that is used to estimate the expected sensitivity of a particle
    physics experiment. It is constructed by adding a multiple of the signal to the dataset, where the multiple is chosen
    such that the signal has the same number of events as would be expected from a given theoretical model. The resulting
    dataset is then used to calculate the expected significance of the signal.

    """

    # Load data
    pldict = {}
    for label in labels:
        fileName = os.path.join(data_path, histo_name + label + ".npz")
        with np.load(fileName, "rb") as data:
            bin_centers = data["bin_centers"]
            bin_edges = data["bin_edges"]
            bin_values = data["bin_values"]
            bin_errors = data["bin_errors"]

        pldict[label] = [bin_centers, bin_edges, bin_values, bin_errors]

    # Add multiple of signal to Data and Signal
    pldict["Data"][2] += multiply * pldict["Signal"][2]
    pldict["Data"][3] = np.sqrt(pldict["Data"][3] ** 2 + (multiply * pldict["Signal"][3]) ** 2)

    pldict["Signal"][2] += multiply * pldict["Signal"][2]
    pldict["Signal"][3] = multiply * pldict["Signal"][3]

    # Save Asimov data
    if save_asimov:
        for label in labels:
            fileName = histo_name + "Asimov" + label + ".npz"
            np.savez(
                os.path.join(data_path, fileName),
                bin_centers=pldict[label][0],
                bin_edges=pldict[label][1],
                bin_values=pldict[label][2],
                bin_errors=pldict[label][3],
            )

    return pldict


if __name__ == "__main__":
    multiply = 100
    data_path = "data/generated_histograms"
    histo_name = "my_hist_"
    labels = ["Background", "Signal", "Data"]
    save_asimov = False

    make_asimov_data(multiply, data_path, histo_name, labels, save_asimov)
