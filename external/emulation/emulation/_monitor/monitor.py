import dataclasses
import logging
import os
import json
from typing import Mapping, Tuple, Any
import cftime
import numpy as np
import xarray as xr
from datetime import datetime, timedelta
from mpi4py import MPI
import tensorflow as tf
import emulation.serialize

from pace.util import ZarrMonitor, CubedSpherePartitioner, Quantity, TilePartitioner
from .._typing import FortranState
from emulation._time import translate_time


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TIME_FMT = "%Y%m%d.%H%M%S"
DIMS_MAP = {
    1: ["sample"],
    2: ["sample", "z"],
}


@dataclasses.dataclass
class StorageConfig:
    """Storage configuration

    Attributes:
        output_freq_sec: output frequency in seconds to save
            nc and/or zarr files at
        output_start_sec: do not save any outputs before this time
        var_meta_path: path to variable metadata added to saved field
            attributes. if not set, then the environmental variable VAR_META_PATH
            will be used. If THAT isn't set then no metadata other than 'unknown'
            units will be set.
        save_nc: save all state fields to netcdf, default
            is true
        save_zarr: save all statefields to zarr, default
            is true
        save_tfrecord: save all statefields to tfrecord
    """

    var_meta_path: str = ""
    output_freq_sec: int = 10_800
    output_start_sec: int = 0
    save_nc: bool = True
    save_zarr: bool = True
    save_tfrecord: bool = False


def _load_monitor(layout):

    tile_partitioner = TilePartitioner(layout)
    partitioner = CubedSpherePartitioner(tile_partitioner)

    output_zarr = os.path.join(os.getcwd(), "state_output.zarr")
    output_monitor = ZarrMonitor(output_zarr, partitioner, mpi_comm=MPI.COMM_WORLD)
    logger.info(f"Initialized zarr monitor at: {output_zarr}")
    return output_monitor


def _remove_io_suffix(key: str):
    if key.endswith("_input"):
        var_key = key[:-6]
        logger.debug(f"Removed _input with result {var_key} for metadata mapping")
    elif key.endswith("_output"):
        var_key = key[:-7]
        logger.debug(f"Removed _output with result {var_key} for metadata mapping")
    else:
        var_key = key

    return var_key


def _get_attrs(key: str, metadata: Mapping):
    key = _remove_io_suffix(key)
    if key in metadata:
        meta = dict(**metadata[key])
        meta = {k: json.dumps(v) for k, v in meta.items()}
    else:
        logger.debug(f"No metadata found for {key}... skipping")
        meta = {}

    return meta


def _convert_to_quantities(state, metadata):

    quantities = {}
    for key, data in state.items():
        data = np.squeeze(data.astype(np.float32))
        data_t = data.T
        dims = DIMS_MAP[data.ndim]
        attrs = _get_attrs(key, metadata)
        units = attrs.pop("units", "unknown")
        quantities[key] = Quantity(data_t, dims, units)
        # Access to private member could break TODO: Quantity kwarg for attrs?
        quantities[key]._attrs.update(attrs)

    return quantities


def _convert_to_xr_dataset(state, metadata):

    dataset = {}
    for key, data in state.items():
        data = np.squeeze(data.astype(np.float32))
        data_t = data.T
        dims = DIMS_MAP[data.ndim]
        attrs = _get_attrs(key, metadata)
        attrs["units"] = attrs.pop("units", "unknown")
        dataset[key] = xr.DataArray(data_t, dims=dims, attrs=attrs)

    return xr.Dataset(dataset)


def _create_nc_path():

    nc_dump_path = os.path.join(os.getcwd(), "netcdf_output")
    if not os.path.exists(nc_dump_path):
        os.makedirs(nc_dump_path, exist_ok=True)

    return nc_dump_path


def _store_netcdf(state, time, nc_dump_path, metadata):

    logger.debug(f"Model fields: {list(state.keys())}")
    logger.info(f"Storing state to netcdf on rank {MPI.COMM_WORLD.Get_rank()}")
    ds = _convert_to_xr_dataset(state, metadata)
    rank = MPI.COMM_WORLD.Get_rank()
    coords = {"time": time, "tile": rank}
    ds = ds.assign_coords(coords)
    filename = f"state_{time.strftime(TIME_FMT)}_{rank}.nc"
    out_path = os.path.join(nc_dump_path, filename)
    ds.to_netcdf(out_path)


def _store_zarr(state, time, monitor, metadata):

    logger.info(f"Storing zarr model state on rank {MPI.COMM_WORLD.Get_rank()}")
    logger.debug(f"Model fields: {list(state.keys())}")
    state = _convert_to_quantities(state, metadata)
    state["time"] = time
    monitor.store(state)


class _TFRecordStore:

    PARSER_FILE: str = "parser.tf"
    TIME: str = "time"

    def __init__(self, root: str, rank: int):
        self.rank = rank
        self.root = root
        tf.io.gfile.makedirs(self.root)
        self._tf_writer = tf.io.TFRecordWriter(
            path=os.path.join(self.root, f"rank{self.rank}.tfrecord")
        )
        self._called = False

    def _save_parser_if_needed(self, state_tf: Mapping[str, tf.Tensor]):
        # needs the state to get the parser so cannot be run in __init__
        if not self._called and self.rank == 0:
            parser = emulation.serialize.get_parser(state_tf)
            tf.saved_model.save(parser, os.path.join(self.root, self.PARSER_FILE))
            self._called = True

    def _convert_to_tensor(
        self, time: cftime.DatetimeJulian, state: Mapping[str, np.ndarray],
    ) -> Mapping[str, tf.Tensor]:
        state_tf = {key: tf.convert_to_tensor(state[key].T) for key in state}
        time = datetime(
            time.year, time.month, time.day, time.hour, time.minute, time.second
        )
        n = max([state[key].shape[0] for key in state])
        state_tf[self.TIME] = tf.convert_to_tensor([time.isoformat()] * n)
        return state_tf

    def __call__(self, state: Mapping[str, np.ndarray], time: cftime.DatetimeJulian):
        state_tf = self._convert_to_tensor(time, state)
        self._save_parser_if_needed(state_tf)
        self._tf_writer.write(emulation.serialize.serialize_tensor_dict(state_tf))
        # need to flush after every call since there are no finalization hooks
        # in the model
        self._tf_writer.flush()


class StorageHook:
    """
    Singleton class for configuring from the environment for
    the store function used during fv3gfs-runtime by call-py-fort.

    Instanced at the top-level of `_monitor`
    """

    def __init__(
        self,
        output_freq_sec: int,
        output_start_sec: int = 0,
        dt_sec: int = 900,
        layout: Tuple[int, int] = (1, 1),
        metadata: Any = {},
        save_nc: bool = True,
        save_zarr: bool = True,
        save_tfrecord: bool = False,
    ):
        self.name = "emulation storage monitor"

        self.output_freq_sec = output_freq_sec
        self.output_start_sec = output_start_sec
        self.metadata = metadata
        self.save_nc = save_nc
        self.save_zarr = save_zarr
        self.save_tfrecord = save_tfrecord
        self.initial_time = None
        self.dt_sec = dt_sec

        if self.save_zarr:
            self.monitor = _load_monitor(layout)
        else:
            self.monitor = None

        if self.save_nc:
            self.nc_dump_path = _create_nc_path()

        if self.save_tfrecord:
            rank = MPI.COMM_WORLD.Get_rank()
            self._store_tfrecord = _TFRecordStore("tfrecords", rank)

    def _store_data_at_time(self, time: cftime.DatetimeJulian):

        elapsed: timedelta = time - self.initial_time

        logger.debug(f"Time elapsed after increment: {elapsed}")
        logger.debug(
            f"Output frequency modulus: {elapsed.seconds % self.output_freq_sec}"
        )
        return (elapsed.seconds % self.output_freq_sec == 0) and (
            elapsed.seconds >= self.output_start_sec
        )

    def store(self, state: FortranState) -> None:
        """
        Hook function for storing the fortran state used by call_py_fort.
        Stores everything that resides in the state at the time.

        'model_time' is expected to be in the state and is removed
        for each storage call.  All other variables are expected to
        correspond to DIMS_MAP after a transpose.

        Args:
            state: Fortran state fields
        """

        state = dict(**state)
        time = translate_time(state.pop("model_time"))

        if self.initial_time is None:
            self.initial_time = time

        # add increment since we are in the middle of timestep
        increment = timedelta(seconds=self.dt_sec)
        if self._store_data_at_time(time + increment):

            logger.debug(
                f"Store flags: save_zarr={self.save_zarr}, save_nc={self.save_nc}"
            )

            if self.save_zarr:
                _store_zarr(state, time, self.monitor, self.metadata)

            if self.save_nc:
                _store_netcdf(state, time, self.nc_dump_path, self.metadata)

            if self.save_tfrecord:
                self._store_tfrecord(state, time)
