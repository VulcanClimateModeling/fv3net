import os
import shutil
import tempfile
from typing import Any, Callable
from typing.io import BinaryIO

import apache_beam as beam
import xarray as xr
from apache_beam.io import filesystems


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


class ArraysToZarr(beam.PTransform):
    """Write a PCollection of Dataset objects to zarr.
    
    The dims of each Dataset must be identical, but the data are combined accross multiple coords
    
    The data are stored in the same chunks as the input dataset sequence
    """

    def __init__(self, store):
        self.store = store

    def expand(self, pcoll):
        global_metadata = pcoll | "CombineCoordinates" >> beam.Map(get_metadata) | beam.CombineGlobally(coords_union)
        zarr_group = global_metadata | "Initialize Zarr" >> beam.Map(_initialize_zarr, store=self.store)
        return pcoll | "PutDatasetInZarr" >> beam.Map(_put_in_zarr, global_metadata=beam.pvalue.AsSingleton(global_metadata),
                                                      zarr_group=beam.pvalue.AsSingleton(global_zarr))


def _initialize_zarr(metadata, store):
    pass


def _put_in_zarr(dataset, global_metadata, zarr_group):
    local_metadata = get_metadata(dataset)
    for name in dataset:
        idx = get_index(name, local_metadata, global_metadata)
        global_zarr[key][idx] = np.asarray(dataset[name])
    return


def get_metadata(ds: xr.Dataset):
    return {
        "dims": {key: ds[key].dims for key in ds},
        "coords": ds.coords,
        "names": list(ds),
        "attrs": {key: ds[key].attrs for key in ds}
    }


def coords_union(coords):
    pass


def get_index(name, local_metadata, global_metadata):
    pass
