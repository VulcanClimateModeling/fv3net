import os

import numpy as np
import glob
import xarray

from synth import generate_nudging


def test_generate_nudging(tmpdir):
    generate_nudging(
        tmpdir,
        [np.datetime64("2016-08-01T00:15:00"), np.datetime64("2016-08-01T00:30:00")],
    )

    # open all the files
    for path in tmpdir.listdir('*.zarr'):
        xarray.open_zarr(os.path.join(tmpdir, path))
