import os
from pathlib import Path

import requests
from tqdm import tqdm


def url_download(url: str, data_dir: str, chunk_size: int = 1024) -> None:
    """Downloads file from url to data_dir.

    Parameters
    ----------
    url : str
        URL of file to download.
    data_dir : str
        Downloaded in this directory (needs to exist).
    chunk_size : int, optional
        Chunk size for downloading, by default 1024

    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    fname = os.path.join(data_dir, url.split("/")[-1])

    if Path(fname).is_file() is not True:
        print(f"started downloading {fname} from {url} ...")

        resp = requests.get(url, stream=True)
        total = int(resp.headers.get("content-length", 0))

        with (
            open(fname, "wb") as file,
            tqdm(desc=fname, total=total, unit="iB", unit_scale=True, unit_divisor=1024) as bar,
        ):
            for data in resp.iter_content(chunk_size=chunk_size):
                size = file.write(data)
                bar.update(size)
    else:
        print(f"already downloaded {fname}!")
