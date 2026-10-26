"""
Put the tracking ROInet weights where the tracking pipeline looks for them
(``tempfile.gettempdir()``) before the tests run.

The CI workflow caches ``~/roicat_ci_cache/ROInet.zip`` between runs, so an
outage at the hosting sites does not fail the integration tests. On a cache
miss, ``ROInet_embedder`` downloads the file, trying each default URL in order.
"""
import shutil
import tempfile
from pathlib import Path

from roicat import ROInet, util

## Download into the cache directory unless a file with the right hash is there
dir_cache = Path("~/roicat_ci_cache").expanduser()
params_network = util.get_default_parameters(pipeline='tracking')['ROInet']['network']
ROInet.ROInet_embedder(
    dir_networkFiles=str(dir_cache),
    download_method='check_local_first',
    download_url=params_network['download_url'],
    download_hash=params_network['download_hash'],
    verbose=True,
)

## Copy to the directory the tracking pipeline reads from
path_dest = Path(tempfile.gettempdir()) / "ROInet.zip"
shutil.copyfile(dir_cache / "ROInet.zip", path_dest)
print(f"Copied ROInet.zip to {path_dest}")
