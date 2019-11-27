import pytest
import xarray as xr

from vcm.misc import rundir

FV_CORE_IN_RESTART = "./RESTART/fv_core.res.tile6.nc"
FV_CORE_IN_RESTART_WITH_TIMESTEP = "./RESTART/20180605.000000.fv_core.res.tile6.nc"

FV_CORE_FINAL = rundir.RestartFile(
    path="./RESTART/fv_core.res.tile6.nc.0000",
    category="fv_core.res",
    tile=6,
    subtile=0,
)

FV_CORE_IN_RESTART_FULL_COMPLEXITY = rundir.RestartFile(
    path="./RESTART/20180605.000000.fv_core.res.tile6.nc.0000",
    _time="20180605.000000",
    category="fv_core.res",
    tile=6,
    subtile=0,
)

FV_CORE_INPUT = rundir.RestartFile(
    path="./INPUT/fv_core.res.tile6.nc", category="fv_core.res", tile=6, subtile=0
)


@pytest.mark.parametrize(
    "path",
    [
        FV_CORE_IN_RESTART,
        FV_CORE_IN_RESTART_FULL_COMPLEXITY.path,
        FV_CORE_IN_RESTART_WITH_TIMESTEP,
    ],
)
def test__restart_regexp_matches(path):
    reg = rundir.RestartFile._restart_regexp()
    matches = reg.search(FV_CORE_IN_RESTART_WITH_TIMESTEP)
    assert matches is not None


@pytest.mark.parametrize("restartfile", [FV_CORE_IN_RESTART_FULL_COMPLEXITY])
def test_RestartFile_from_path(restartfile: rundir.RestartFile):
    spec = rundir.RestartFile.from_path(restartfile.path)
    assert spec == restartfile


@pytest.mark.parametrize(
    "file,expected",
    [
        (FV_CORE_FINAL, False),
        (FV_CORE_IN_RESTART_FULL_COMPLEXITY, False),
        (FV_CORE_INPUT, True),
    ],
)
def test_RestartFile_is_initial_time(file, expected):
    assert file.is_initial_time == expected


@pytest.mark.parametrize(
    "file,expected",
    [
        (FV_CORE_FINAL, True),
        (FV_CORE_IN_RESTART_FULL_COMPLEXITY, False),
        (FV_CORE_INPUT, False),
    ],
)
def test_RestartFile_is_final_time(file, expected):
    assert file.is_final_time == expected


@pytest.mark.skip()
def test_restart_files_at_url():
    url = "gs://vcm-ml-data/2019-10-28-X-SHiELD-2019-10-05-multiresolution-extracted/one-step-run/C48/20160801.003000/rundir"  # noqa
    ds = rundir.open_restarts(
        url, initial_time="20160801.003000", final_time="20160801.004500"
    )
    grid = xr.open_mfdataset(
        "20160801.003000/rundir/grid_spec.tile?.nc", concat_dim="tile", combine="nested"
    )
    ds = ds.merge(grid)
    print(ds)
