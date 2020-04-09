import os
import shutil
import tempfile
import secrets
import string
import logging
from datetime import timedelta
from typing import Any, Callable, List
from typing.io import BinaryIO

import apache_beam as beam
import xarray as xr
from apache_beam.io import filesystems

from vcm.cloud.fsspec import get_fs
from vcm import parse_timestep_str_from_path, parse_datetime_from_str
from vcm.cubedsphere.constants import TIME_FMT

logger = logging.getLogger(__name__)


class CombineSubtilesByKey(beam.PTransform):
    """Transform for combining subtiles of cubed-sphere data in a beam PCollection.

    This transform operates on a PCollection of `(key, xarray dataarray)`
    tuples. For most instances, the tile number should be in the `key`.

    See the tests for an example.
    """

    def expand(self, pcoll):
        return pcoll | beam.GroupByKey() | beam.MapTuple(self._combine)

    @staticmethod
    def _combine(key, datasets):
        return key, xr.combine_by_coords(datasets)


class WriteToNetCDFs(beam.PTransform):
    """Transform for writing xarray Datasets to netCDF either remote or local
netCDF files.

    Saves a collection of `(key, dataset)` based on a naming function

    Attributes:

        name_fn: the function to used to translate the `key` to a local
            or remote url. Let an element of the input PCollection be given by `(key,
            ds)`, where ds is an xr.Dataset, then this transform will save `ds` as a
            netCDF file at the URL given by `name_fn(key)`. If this functions returns
            a string beginning with `gs://`, this transform will save the netCDF
            using Google Cloud Storage, otherwise it will be local file.

    Example:

        >>> from fv3net.pipelines import common
        >>> import os
        >>> import xarray as xr
        >>> input_data = [('a', xr.DataArray([1.0], name='name').to_dataset())]
        >>> input_data
        [('a', <xarray.Dataset>
        Dimensions:  (dim_0: 1)
        Dimensions without coordinates: dim_0
        Data variables:
            name     (dim_0) float64 1.0)]
        >>> import apache_beam as beam
        >>> with beam.Pipeline() as p:
        ...     (p | beam.Create(input_data)
        ...        | common.WriteToNetCDFs(lambda letter: f'{letter}.nc'))
        ...
        >>> os.system('ncdump -h a.nc')
        netcdf a {
        dimensions:
            dim_0 = 1 ;
        variables:
            double name(dim_0) ;
                name:_FillValue = NaN ;
        }
        0

    """

    def __init__(self, name_fn: Callable[[Any], str]):
        self.name_fn = name_fn

    def _process(self, key, elm: xr.Dataset):
        """Save a netCDF to a path which is determined from `key`

        This works for any url support by apache-beam's built-in FileSystems_ class.

        .. _FileSystems:
            https://beam.apache.org/releases/pydoc/2.6.0/apache_beam.io.filesystems.html#apache_beam.io.filesystems.FileSystems

        """
        path = self.name_fn(key)
        dest: BinaryIO = filesystems.FileSystems.create(path)

        # use a file-system backed buffer in case the data is too large to fit in memory
        tmp = tempfile.mktemp()
        try:
            elm.to_netcdf(tmp)
            with open(tmp, "rb") as src:
                shutil.copyfileobj(src, dest)
        finally:
            dest.close()
            os.unlink(tmp)

    def expand(self, pcoll):
        return pcoll | beam.MapTuple(self._process)


def list_timesteps(path: str) -> List[str]:
    """
    Returns the unique timesteps at a path. Note that any path with a
    timestep matching the parsing check will be returned from this
    function.

    Args:
        path: local or remote path to directory containing timesteps

    Returns:
        sorted list of all timesteps within path
    """
    try:
        file_list = get_fs(path).ls(path)
    except FileNotFoundError:
        file_list = []
    timesteps = []
    for current_file in file_list:
        try:
            timestep = parse_timestep_str_from_path(current_file)
            timesteps.append(timestep)
        except ValueError:
            # not a timestep directory
            continue
    return sorted(timesteps)


def subsample_timesteps_at_interval(
    timesteps: List[str], sampling_interval: int,
) -> List[str]:
    """
    Subsample a list of timesteps at the specified interval (in minutes). Raises
    a ValueError if requested interval of output does not align with available
    timesteps.

    Args:
        timesteps: A list of all available timestep strings.  Assumed to
            be in the format described by vcm.cubedsphere.constants.TIME_FMT
        sampling_interval: The interval to subsample the list in minutes
    
    Returns:
        A subsampled list of the input timesteps at the desired interval.
    """
    logger.info(
        f"Subsampling available timesteps to every {sampling_interval} minutes."
    )
    current_time = parse_datetime_from_str(timesteps[0])
    last_time = parse_datetime_from_str(timesteps[-1])
    available_times = set(timesteps)
    subsampled_timesteps = [timesteps[0]]
    try:
        delta = timedelta(minutes=sampling_interval)
    except TypeError:
        logger.warning(
            f"No sampling interval specifed, returning first value only."
        )
    else:
        while current_time < last_time:
            next_time = current_time + delta
            next_time_str = next_time.strftime(TIME_FMT)
            if next_time_str in available_times:
                subsampled_timesteps.append(next_time_str)

            current_time = next_time
        num_subsampled = len(subsampled_timesteps)
        if num_subsampled < 2:
            logger.warning(
                f"Desired subsampling interval of {sampling_interval} minutes "
                f"longer than sequence duration, or is misaligned with the "
                f"dataset interval. Using only first timestep."
            )

    return subsampled_timesteps


def get_alphanumeric_unique_tag(tag_length: int) -> str:
    """Generates a random alphanumeric string (a-z0-9) of a specified length"""

    if tag_length < 1:
        raise ValueError("Unique tag length should be 1 or greater.")

    use_chars = string.ascii_lowercase + string.digits
    short_id = "".join([secrets.choice(use_chars) for i in range(tag_length)])
    return short_id


def dump_nc(ds: xr.Dataset, f):
    # to_netcdf closes file, which will delete the buffer
    # need to use a buffer since seek doesn't work with GCSFS file objects
    with tempfile.TemporaryDirectory() as dirname:
        url = os.path.join(dirname, "tmp.nc")
        ds.to_netcdf(url, engine="h5netcdf")
        with open(url, "rb") as tmp1:
            shutil.copyfileobj(tmp1, f)
