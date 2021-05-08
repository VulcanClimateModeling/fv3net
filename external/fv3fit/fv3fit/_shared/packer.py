from typing import (
    Iterable,
    TextIO,
    List,
    Dict,
    Tuple,
    cast,
    Mapping,
    Sequence,
)
import numpy as np
import xarray as xr
import pandas as pd
import yaml


def _feature_dims(data: xr.Dataset, sample_dim: str) -> Sequence[str]:
    return [str(dim) for dim in data.dims.keys() if dim != sample_dim]


def _unique_dim_name(
    data: xr.Dataset, sample_dim: str, feature_dim_name_2d_var: str = "feature"
) -> str:
    feature_dims = _feature_dims(data, sample_dim)
    if len(feature_dims) > 0:
        feature_dim_name = "_".join(["feature"] + list(feature_dims))
    else:
        feature_dim_name = feature_dim_name_2d_var
    if sample_dim == feature_dim_name:
        raise ValueError(
            f"The sample dim name ({sample_dim}) cannot be the same "
            f"as the feature dim name ({feature_dim_name})"
        )
    return feature_dim_name


def pack(data: xr.Dataset, sample_dim: str) -> Tuple[np.ndarray, pd.MultiIndex]:
    feature_dim_name = _unique_dim_name(data, sample_dim)
    stacked = data.to_stacked_array(feature_dim_name, sample_dims=[sample_dim])
    return (
        stacked.transpose(sample_dim, feature_dim_name).data,
        stacked.indexes[feature_dim_name],
    )


def unpack(
    data: np.ndarray, sample_dim: str, feature_index: pd.MultiIndex
) -> xr.Dataset:
    if len(data.shape) == 1:
        data = data[:, None]
    da = xr.DataArray(
        data, dims=[sample_dim, "feature"], coords={"feature": feature_index}
    )
    return da.to_unstacked_dataset("feature")


class ArrayPacker:
    """
    A class to handle converting xarray datasets to and from numpy arrays.

    Used for ML training/prediction.
    """

    def __init__(self, sample_dim_name, pack_names: Iterable[str]):
        """Initialize the ArrayPacker.

        Args:
            sample_dim_name: dimension name to treat as the sample dimension   dfdfdff
            pack_names: variable pack_names to pack
        """
        self._pack_names = list(pack_names)
        self._n_features: Dict[str, int] = {}
        self._sample_dim_name = sample_dim_name
        self._dims: Dict[str, Sequence[str]] = {}

    @property
    def pack_names(self) -> List[str]:
        """variable pack_names being packed"""
        return self._pack_names

    @property
    def sample_dim_name(self) -> str:
        """name of sample dimension"""
        return self._sample_dim_name

    @property
    def feature_counts(self) -> dict:
        return self._n_features.copy()

    @property
    def _total_features(self):
        return sum(self._n_features[name] for name in self._pack_names)

    def to_array(self, dataset: xr.Dataset) -> np.ndarray:
        """Convert dataset into a 2D array with [sample, feature] dimensions.

        Variable names inserted into the array are passed on initialization of this
        object. Each of those variables in the dataset must have the sample
        dimension name indicated when this object was initialized, and at most one
        more dimension, considered the feature dimension.

        The first time this is called, the length of the feature dimension for each
        variable is stored, and can be retrieved on `packer.feature_counts`.

        On subsequent calls, the feature dimensions are broadcast
        to have this length. This ensures the returned array has the same shape on
        subsequent calls, and allows packing a dataset of scalars against
        [sample, feature] arrays.
        
        Args:
            dataset: dataset containing variables in self.pack_names to pack

        Returns:
            array: 2D [sample, feature] array with data from the dataset
        """
        if len(self._n_features) == 0:
            self._n_features.update(
                _count_features_2d(self.pack_names, dataset, self._sample_dim_name)
            )
            for name in self.pack_names:
                self._dims[name] = cast(Tuple[str], dataset[name].dims)
        for var in self.pack_names:
            if dataset[var].dims[0] != self.sample_dim_name:
                dataset[var] = dataset[var].transpose()
        array = _to_array_2d(dataset, self.pack_names, self.feature_counts)
        return array

    def to_dataset(self, array: np.ndarray) -> xr.Dataset:
        """Restore a dataset from a 2D [sample, feature] array.

        Restores dimension names, but does not restore coordinates or attributes.

        Can only be called after `to_array` is called.

        Args:
            array: 2D [sample, feature] array

        Returns:
            dataset: xarray dataset with data from the given array
        """
        if len(array.shape) > 2:
            raise NotImplementedError("can only restore 2D arrays to datasets")
        if len(self._n_features) == 0:
            raise RuntimeError(
                "must pack at least once before unpacking, "
                "so dimension lengths are known"
            )
        return to_dataset(array, self.pack_names, self._dims, self.feature_counts)

    def dump(self, f: TextIO):
        return yaml.safe_dump(
            {
                "n_features": self._n_features,
                "pack_names": self._pack_names,
                "sample_dim_name": self._sample_dim_name,
                "dims": self._dims,
            },
            f,
        )

    @classmethod
    def load(cls, f: TextIO):
        data = yaml.safe_load(f.read())
        packer = cls(data["sample_dim_name"], data["pack_names"])
        packer._n_features = data["n_features"]
        packer._dims = data["dims"]
        return packer


def _to_array_2d(
    dataset: xr.Dataset, pack_names: Sequence[str], feature_counts: Mapping[str, int],
):
    """
    Convert dataset into a 2D array with [sample, feature] dimensions.

    The first dimension of each variable to pack is assumed to be the sample dimension,
    and the second (if it exists) is assumed to be the feature dimension.
    Each variable must be 1D or 2D.
    
    Args:
        dataset: dataset containing variables in self.pack_names to pack
        pack_names: names of variables to pack
        feature_counts: number of features for each variable

    Returns:
        array: 2D [sample, feature] array with data from the dataset
    """
    # we can assume here that the first dimension is the sample dimension
    n_samples = dataset[pack_names[0]].shape[0]
    total_features = sum(feature_counts[name] for name in pack_names)

    array = np.empty([n_samples, total_features])

    i_start = 0
    for name in pack_names:
        n_features = feature_counts[name]
        if n_features > 1:
            array[:, i_start : i_start + n_features] = dataset[name]
        else:
            array[:, i_start] = dataset[name]
        i_start += n_features
    return array


def to_dataset(
    array: np.ndarray,
    pack_names: Iterable[str],
    dimensions: Mapping[str, Sequence[str]],
    feature_counts: Mapping[str, int],
):
    """Restore a dataset from a 2D [sample, feature] array.

    Restores dimension names, but does not restore coordinates or attributes.

    Can only be called after `to_array` is called.

    Args:
        array: 2D [sample, feature] array
        pack_names: names of variables to unpack
        dimensions: mapping which provides a list of dimensions for each variable
        feature_counts: mapping which provides a number of features for each variable

    Returns:
        dataset: xarray dataset with data from the given array
    """
    data_vars = {}
    i_start = 0
    for name in pack_names:
        n_features = feature_counts[name]
        if n_features > 1:
            data_vars[name] = (
                dimensions[name],
                array[:, i_start : i_start + n_features],
            )
        else:
            data_vars[name] = (dimensions[name], array[:, i_start])
        i_start += n_features
    return xr.Dataset(data_vars)  # type: ignore


def _count_features_2d(
    quantity_names: Iterable[str], dataset: xr.Dataset, sample_dim_name: str
) -> Mapping[str, int]:
    """
    count features for (sample[, z]) arrays
    """
    for name in quantity_names:
        if len(dataset[name].dims) > 2:
            value = dataset[name]
            raise ValueError(
                "can only pack 1D/2D (sample[, z]) "
                f"variables, recieved value for {name} with dimensions {value.dims}"
            )
    return_dict = {}
    for name in quantity_names:
        value = dataset[name]
        if len(value.dims) == 1 and value.dims[0] == sample_dim_name:
            return_dict[name] = 1
        elif value.dims[0] != sample_dim_name:
            raise ValueError(
                f"cannot pack value for {name} whose first dimension is not the "
                f"sample dimension ({sample_dim_name}), has dims {value.dims}"
            )
        else:
            return_dict[name] = value.shape[1]
    return return_dict


def unpack_matrix(
    x_packer: ArrayPacker, y_packer: ArrayPacker, matrix: np.ndarray
) -> xr.Dataset:
    """Unpack a matrix

    Args:
        x_packer: packer for the rows of the matrix
        y_packer: packer for the columns of the matrix
        matrix: the matrix to be unpacked
    Returns:
        a Dataset

    """
    jacobian_dict = {}
    j = 0
    for in_name in x_packer.pack_names:
        i = 0
        for out_name in y_packer.pack_names:
            size_in = x_packer.feature_counts[in_name]
            size_out = y_packer.feature_counts[out_name]

            jacobian_dict[(in_name, out_name)] = xr.DataArray(
                matrix[i : i + size_out, j : j + size_in], dims=[out_name, in_name],
            )
            i += size_out
        j += size_in

    return xr.Dataset(jacobian_dict)  # type: ignore
