import os

import xarray as xr

import synth


def test_generate_fine_res_correct_files(tmpdir):
    synth.generate_fine_res(tmpdir)
