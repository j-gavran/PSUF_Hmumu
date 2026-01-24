from __future__ import annotations

import glob
import os
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from uncertainties import unumpy

from helpers.create_histograms import url_download


class HistMaker:
    def __init__(
        self, data_dir_input: str = "data/raw_data", data_dir_output: str = "data/generated_histograms"
    ) -> None:
        """Download and create histograms for H->mumu example.

        Parameters
        ----------
        data_dir_input : str, optional
            Directory containing raw .h5 data files, by default "data/raw_data".
        data_dir_output : str, optional
            Directory for generated histogram .npz files, by default "data/generated_histograms".

        Properties
        ----------
        hist_paths : dict[str, str]
            Dictionary with paths to generated histogram .npz files.

        Methods
        -------
        download()
            Download datasets from CERNBox links.
        make(x_range, n_bins, save_name="hist", remake=False)
            Generate histograms with uniform binning.
        make_from_array(n_bins, save_name="hist", remake=False)
            Generate histograms with custom bin edges.

        Examples
        --------
        >>> hm = HistMaker()
        >>> hm.download()
        >>> hm.make(x_range=(0, 150), n_bins=50 , save_name="hist_uniform")
        >>> hm.make_from_array(n_bins=np.array([0, 50, 100, 120, 130, 140, 150]), save_name="hist_custom")

        """
        self.data_dir_input = data_dir_input
        self.data_dir_output = data_dir_output

        os.makedirs(self.data_dir_input, exist_ok=True)
        os.makedirs(self.data_dir_output, exist_ok=True)

        self._datasets = ["mc_bkg_new", "mc_sig", "data"]
        self._labels = ["Background", "Signal", "Data"]

        self._hist_paths: dict[str, str] | None = None

        self._cernbox_links = [
            "https://cernbox.cern.ch/remote.php/dav/public-files/c9QNNQiPU92BBGG/mc_bkg_new.h5",
            "https://cernbox.cern.ch/remote.php/dav/public-files/fbvcC6YD67C1eZq/mc_sig.h5",
            "https://cernbox.cern.ch/remote.php/dav/public-files/uO4rp4vRckMROy3/data.h5",
        ]

    @property
    def hist_paths(self) -> dict[str, str]:
        if self._hist_paths is None:
            raise ValueError("Histogram paths have not been set yet.")
        return self._hist_paths

    def download(self) -> None:
        for cernbox_link in self._cernbox_links:
            url_download(cernbox_link, self.data_dir_input)

    def _get_hist_paths(self, save_name: str) -> dict[str, str]:
        return {label: f"{self.data_dir_output}/{save_name}_{label}.npz" for label in self._labels}

    def _load_data(self, dataset: str) -> pd.DataFrame:
        infile = os.path.join(self.data_dir_input, f"{dataset}.h5")

        print(f"Loading dataset from {infile}...")

        store = pd.HDFStore(infile, "r")
        dataset = store["ntuple"]
        store.close()

        return dataset

    def _make_histograms(
        self,
        n_bins: int | np.ndarray,
        x_range: tuple[float, float] | None,
        weights_modifier: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        if type(n_bins) is np.ndarray and x_range is not None:
            raise ValueError("If n_bins is an array, x_range must be None.")

        for label, dataset in zip(self._labels, self._datasets):
            # Load dataset
            ds = self._load_data(dataset)

            print(f"Creating histogram for {label}...")

            # Get simulated (Background, Signal) or measured (Data) data
            all_events = ds["Muons_Minv_MuMu_Paper"]

            # Get correct weights
            wts = ds["CombWeight"]

            # For MC: sum of weights squared, for data: N
            if label == "Data":
                wts2 = wts
            elif weights_modifier is None:
                wts2 = wts**2
            else:
                wts2 = weights_modifier(wts)

            # Firstly, get correct number of bin_values
            bin_values, _ = np.histogram(all_events, bins=n_bins, range=x_range, weights=wts)  # wts!

            # Secondly, calculate bin_errors
            y, bin_edges = np.histogram(all_events, bins=n_bins, range=x_range, weights=wts2)  # wts2!
            bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])
            bin_errors = np.sqrt(y)

            # Save histogram to .npz file
            with open(self.hist_paths[label], "wb") as f:
                np.savez(f, bin_edges=bin_edges, bin_centers=bin_centers, bin_values=bin_values, bin_errors=bin_errors)

            f.close()

    def _check_missing(self) -> None:
        h5_files = glob.glob(os.path.join(self.data_dir_input, "*.h5"))
        present_datasets = {os.path.splitext(os.path.basename(f))[0] for f in h5_files}

        missing = set(self._datasets) - present_datasets
        if missing:
            raise FileNotFoundError(f"Missing datasets: {missing}. Please run the download method first.")

    def _check_if_exists(self) -> bool:
        return all([os.path.isfile(f) for f in self.hist_paths.values()])

    def make(
        self,
        x_range: tuple[float, float],
        n_bins: int,
        save_name: str = "hist",
        weights_modifier: Callable[[np.ndarray], np.ndarray] | None = None,
        remake: bool = False,
    ) -> dict[str, str]:
        self._check_missing()

        self._hist_paths = self._get_hist_paths(save_name=save_name)

        if not remake and self._check_if_exists():
            print("Histograms already exist. Skipping creation.")
            return self.hist_paths

        self._make_histograms(n_bins=n_bins, x_range=x_range, weights_modifier=weights_modifier)

        return self.hist_paths

    def make_from_array(
        self,
        bins: np.ndarray,
        save_name: str = "hist",
        weights_modifier: Callable[[np.ndarray], np.ndarray] | None = None,
        remake: bool = False,
    ) -> dict[str, str]:
        self._check_missing()

        self._hist_paths = self._get_hist_paths(save_name=save_name)

        if not remake and self._check_if_exists():
            print("Histograms already exist. Skipping creation.")
            return self.hist_paths

        self._make_histograms(n_bins=bins, x_range=None, weights_modifier=weights_modifier)

        return self.hist_paths


@dataclass(frozen=True)
class Hist:
    edges: np.ndarray
    centers: np.ndarray
    values: np.ndarray
    errors: np.ndarray

    @property
    def uarray(self) -> unumpy.uarray:
        return unumpy.uarray(self.values, self.errors)

    @classmethod
    def from_uarray(cls, edges: np.ndarray, centers: np.ndarray, uarr: unumpy.uarray) -> Hist:
        return cls(edges=edges, centers=centers, values=unumpy.nominal_values(uarr), errors=unumpy.std_devs(uarr))

    def __add__(self, other: Hist) -> Hist:
        if not np.array_equal(self.edges, other.edges):
            raise ValueError("Cannot add Hist objects with different bin edges.")

        result = self.uarray + other.uarray
        return self.from_uarray(self.edges, self.centers, result)

    def __sub__(self, other: Hist) -> Hist:
        if not np.array_equal(self.edges, other.edges):
            raise ValueError("Cannot subtract Hist objects with different bin edges.")

        result = self.uarray - other.uarray
        return self.from_uarray(self.edges, self.centers, result)

    def __mul__(self, other: int | float) -> Hist:
        if isinstance(other, (int, float)):
            result = self.uarray * other
            return self.from_uarray(self.edges, self.centers, result)

        raise ValueError("Can only multiply Hist by a scalar (int or float).")

    def __truediv__(self, other: Hist) -> Hist:
        if not np.array_equal(self.edges, other.edges):
            raise ValueError("Cannot divide Hist objects with different bin edges.")

        result = self.uarray / other.uarray
        return self.from_uarray(self.edges, self.centers, result)

    def __repr__(self) -> str:
        return (
            f"Hist(n_bins={len(self.centers)}, "
            f"range=[{self.edges[0]:.3g}, {self.edges[-1]:.3g}], "
            f"total={np.sum(self.values):.3g})"
        )

    def __str__(self) -> str:
        lines = [
            f"Histogram with {len(self.centers)} bins",
            f"  Range: [{self.edges[0]:.4g}, {self.edges[-1]:.4g}]",
            f"  Total entries: {np.sum(self.values):.4g}",
            f"  Mean: {np.average(self.centers, weights=self.values):.4g}",
            f"  Max bin value: {np.max(self.values):.4g}",
        ]
        return "\n".join(lines)


@dataclass(frozen=True)
class RegionHist:
    background: Hist
    signal: Hist
    data: Hist

    def __repr__(self) -> str:
        return f"RegionHist(background={repr(self.background)}, signal={repr(self.signal)}, data={repr(self.data)})"

    def __str__(self) -> str:
        lines = ["Region Histograms:", ""]

        for name, hist in [("Background", self.background), ("Signal", self.signal), ("Data", self.data)]:
            lines.append(f"{name}:")
            hist_str = str(hist).replace("\n", "\n  ")
            lines.append(f"  {hist_str}")
            lines.append("")

        return "\n".join(lines)


class HistLoader:
    def __init__(self, data_dir_output: str = "data/generated_histograms") -> None:
        """Load histograms from .npz files for H->mumu example. Loads Data, Signal and Background histograms.

        Parameters
        ----------
        hist_paths : dict[str, str]
            Dictionary with paths to histogram .npz files.
        data_dir_output : str, optional
            Directory for generated histogram .npz files, by default "data/generated_histograms".

        """
        self.data_dir_output = data_dir_output

    def _load_histogram(self, hist_file: str) -> Hist:
        if not os.path.isfile(hist_file):
            raise FileNotFoundError(f"Histogram file {hist_file} does not exist.")

        with np.load(hist_file) as data:
            return Hist(
                edges=data["bin_edges"],
                centers=data["bin_centers"],
                values=data["bin_values"],
                errors=data["bin_errors"],
            )

    def load(self, hist_paths: dict[str, str]) -> RegionHist:
        if set(hist_paths.keys()) != {"Background", "Signal", "Data"}:
            raise ValueError("hist_paths must contain keys: 'Background', 'Signal', 'Data'.")

        return RegionHist(
            background=self._load_histogram(hist_paths["Background"]),
            signal=self._load_histogram(hist_paths["Signal"]),
            data=self._load_histogram(hist_paths["Data"]),
        )
