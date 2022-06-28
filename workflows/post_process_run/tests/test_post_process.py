import os
import numpy as np
import xarray as xr
from fv3post.post_process import (
    parse_rundir,
    process_item,
    open_tiles,
    cast_time,
    clear_encoding,
    drop_time_average_info_vars,
    FORTRAN_TIME_AVERAGING_VARS,
)
import tempfile
from datetime import datetime

TEST_CHUNKS = {"a.zarr": {"time": 5}}


def test_parse_rundir_mocked_walker():
    walker = [
        (
            ".",
            ["diags.zarr", "INPUT", "OUTPUT"],
            ["a.tile1.nc", "a.tile2.nc", "a.tile3.nc", "randomfile"],
        ),
        ("./diags.zarr", ["a"], [".zattrs"],),
        ("./diags.zarr/a", [], ["0", "1", ".zattrs"],),
        ("./INPUT", [], ["restart.nc"],),
    ]
    tiles, zarrs, other = parse_rundir(walker)

    assert tiles == ["./a.tile1.nc", "./a.tile2.nc", "./a.tile3.nc"]
    assert zarrs == ["./diags.zarr"]
    assert set(other) == {"./INPUT/restart.nc", "./randomfile"}


def test_parse_rundir_os_walk_integration(tmpdir):
    # Setup directory structure
    zarr = tmpdir.mkdir("diags.zarr")
    input_ = tmpdir.mkdir("INPUT")

    zarr.join("0").write("")
    zarr.join("1").write("")
    tmpdir.join("a.tile1.nc").write("")
    tmpdir.join("a.tile2.nc").write("")
    tmpdir.join("a.tile3.nc").write("")
    tmpdir.join("randomfile").write("")

    input_.join("restart.nc").write("")

    tiles, zarrs, other = parse_rundir(os.walk(str(tmpdir)))

    assert set(tiles) == {
        f"{tmpdir}/a.tile1.nc",
        f"{tmpdir}/a.tile2.nc",
        f"{tmpdir}/a.tile3.nc",
    }
    assert set(zarrs) == {f"{tmpdir}/diags.zarr"}
    assert set(other) == {f"{tmpdir}/INPUT/restart.nc", f"{tmpdir}/randomfile"}


def test_process_item_dataset(tmpdir):
    d_in = str(tmpdir)
    localpath = str(tmpdir.join("diags.zarr"))
    ds = xr.Dataset(
        {"a": (["time", "x"], np.ones((200, 10)))}, attrs={"path": localpath}
    )
    with tempfile.TemporaryDirectory() as d_out:
        process_item(ds, d_in, d_out, TEST_CHUNKS)
        xr.open_zarr(d_out + "/diags.zarr")


def test_process_item_empty_dataset(tmpdir):
    d_in = str(tmpdir)
    ds = xr.Dataset()
    with tempfile.TemporaryDirectory() as d_out:
        process_item(ds, d_in, d_out, TEST_CHUNKS)


def test_process_item_str(tmpdir):
    txt = "hello"
    d_in = str(tmpdir)
    path = tmpdir.join("afile.txt")
    path.write(txt)

    with tempfile.TemporaryDirectory() as d_out:
        process_item(str(path), d_in, d_out, TEST_CHUNKS)
        with open(d_out + "/afile.txt") as f:
            assert f.read() == txt


def test_process_item_str_nested(tmpdir):
    txt = "hello"
    d_in = str(tmpdir)
    path = tmpdir.mkdir("nest").join("afile.txt")
    path.write(txt)

    with tempfile.TemporaryDirectory() as d_out:
        process_item(str(path), d_in, d_out, TEST_CHUNKS)
        with open(d_out + "/nest/afile.txt") as f:
            assert f.read() == txt


def test_process_item_broken_symlink(tmpdir):
    fake_path = str(tmpdir.join("idontexist"))
    broken_link = str(tmpdir.join("broken_link"))
    os.symlink(fake_path, broken_link)
    with tempfile.TemporaryDirectory() as d_out:
        process_item(broken_link, str(tmpdir), d_out, TEST_CHUNKS)


def test_open_tiles_netcdf_data(tmpdir):
    ds = xr.Dataset({"a": (["time", "x"], np.ones((200, 10)))})
    tiles = []
    for i in range(1, 7):
        path = f"{tmpdir}/a.tile{i}.nc"
        ds.to_netcdf(path)
        tiles.append(path)

    out = open_tiles(tiles, str(tmpdir), chunks=TEST_CHUNKS)
    saved_ds = list(out)[0]

    assert isinstance(saved_ds, xr.Dataset)
    # check for variable "a"
    saved_ds["a"]


def test_cast_time_no_coord():
    ds_no_coord = xr.Dataset({"a": (["time", "x"], np.ones((1, 10)))})
    output_no_coord = cast_time(ds_no_coord)
    xr.testing.assert_allclose(ds_no_coord, output_no_coord)


def test_cast_time_with_coord():
    ds_with_coord = xr.Dataset(
        {"a": (["time", "x"], np.ones((1, 10)))},
        coords={"time": [datetime(2016, 8, 1)]},
    )
    output_with_coord = cast_time(ds_with_coord)
    assert isinstance(output_with_coord.time.values[0], np.datetime64)


def test_clear_encoding_with_coords():
    # incoming chunks may not align with data
    x_coord = xr.Variable(["x"], np.arange(10), encoding={"chunks": 6})
    time_coord = xr.Variable(
        ["time"], [datetime(2016, 8, 1)], encoding={"chunks": 1024}
    )
    ds_with_coord = xr.Dataset(
        {"a": (["time", "x"], np.ones((1, 10)))},
        coords={"time": time_coord, "x": x_coord},
    )
    clear_encoding(ds_with_coord)
    ds_no_encoding = xr.Dataset(
        {"a": (["time", "x"], np.ones((1, 10)))},
        coords={"time": [datetime(2016, 8, 1)], "x": np.arange(10)},
    )
    xr.testing.assert_identical(ds_with_coord, ds_no_encoding)


def test_drop_time_average_info_vars():
    dt = np.timedelta64(1, "s")
    ds_with_interval_vars = xr.Dataset(
        {
            "a": (["time", "x"], np.ones((2, 10))),
            "average_T1": (["time", "x"], np.ones((2, 10))),
            "average_T2": (["time", "x"], np.ones((2, 10))),
            "average_DT": (["time", "x"], np.full((2, 10), dt)),
        },
    )

    ds_dropped = drop_time_average_info_vars(ds_with_interval_vars)
    for time_avg_var in FORTRAN_TIME_AVERAGING_VARS:
        assert time_avg_var not in ds_dropped
    assert ds_dropped.attrs["time_averaging_interval"] == dt


def test_drop_time_average_info_vars_not_present():
    ds_no_interval_vars = xr.Dataset({"a": (["time", "x"], np.ones((2, 10)))})

    ds_dropped = drop_time_average_info_vars(ds_no_interval_vars)
    assert "a" in ds_dropped
