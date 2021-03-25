from typing import (
    Iterable,
    TextIO,
    List,
    Dict,
    Tuple,
    cast,
    Mapping,
    Sequence,
    Optional,
)
import numpy as np
import xarray as xr
import pandas as pd
import yaml
import tensorflow as tf


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


class Unpack(tf.keras.layers.Layer):
    def __init__(
        self,
        *,
        pack_names: Sequence[str],
        n_features: Mapping[str, int],
        feature_dim: int,
    ):
        super().__init__()
        self.pack_names = pack_names
        self.n_features = n_features
        self.feature_dim = feature_dim
        if feature_dim not in (1, 2):
            raise NotImplementedError(self.feature_dim)

    def call(self, inputs):
        i = 0
        return_tensors = []
        for name in self.pack_names:
            features = self.n_features[name]
            if self.feature_dim == 1:
                return_tensors.append(inputs[:, i : i + features])
            elif self.feature_dim == 2:
                return_tensors.append(inputs[:, :, i : i + features])
            else:
                raise NotImplementedError(self.feature_dim)
            i += features
        return return_tensors

    def get_config(self):
        return {
            "pack_names": self.pack_names,
            "n_features": self.n_features,
            "feature_dim": self.feature_dim,
        }


class ArrayPacker:
    """
    A class to handle converting xarray datasets to and from numpy arrays.

    Used for ML training/prediction.
    """

    def __init__(self, sample_dim_name, pack_names: Iterable[str]):
        """Initialize the ArrayPacker.

        Args:
            sample_dim_name: dimension name to treat as the sample dimension
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

    def pack_layer(self):
        if len(self.pack_names) > 1:
            return tf.keras.layers.Concatenate()
        else:
            raise NotImplementedError(
                "pack layer only implemented for multiple pack variables, "
                "avoid adding a pack layer when len(obj.pack_names) is 1"
            )

    def unpack_layer(self, feature_dim: int):
        # have to store this as a local scope variable
        # so that serialization does not cause self to be serialized
        return Unpack(
            pack_names=self.pack_names,
            n_features=self._n_features,
            feature_dim=feature_dim,
        )

    def to_array(self, dataset: xr.Dataset, is_3d: bool = False) -> np.ndarray:
        """Convert dataset into a 2D array with [sample, feature] dimensions or
        3D array with [sample, time, feature] dimensions.

        Dimensions are inferred from non-sample dimensions, and assumes all
        arrays in the dataset have a shape of (sample) and (sample, feature)
        or all arrays in the dataset have a shape of (sample, time) or
        (sample, time, feature).

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
            dataset: dataset containing variables in self.pack_names to pack,
                dimensionality must match value of is_3d
            is_3d: if True, pack to a 3D array. This can't be detected automatically
                because sometimes all packed variables are scalars

        Returns:
            array: 2D [sample, feature] array with data from the dataset
        """
        if len(self._n_features) == 0:
            self._n_features.update(
                count_features(
                    self.pack_names, dataset, self._sample_dim_name, is_3d=is_3d
                )
            )
            for name in self.pack_names:
                self._dims[name] = cast(Tuple[str], dataset[name].dims)
            self._coords = cast(Mapping[str, xr.IndexVariable], dataset.coords)
        for var in self.pack_names:
            if dataset[var].dims[0] != self.sample_dim_name:
                dataset[var] = dataset[var].transpose()
        array = to_array(dataset, self.pack_names, self.feature_counts, is_3d=is_3d)
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
        all_dims = {}
        for name, dims in self._dims.items():
            if len(dims) <= 2:
                all_dims[name] = dims
            elif len(dims) == 3:
                # relevant when we to_array on a 3D dataset and want to restore a slice
                # of it (time snapshot) to a 2D dataset
                all_dims[name] = [dims[0], dims[2]]  # no time dimension
            else:
                raise RuntimeError(dims)
        return to_dataset(array, self.pack_names, all_dims, self.feature_counts)

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


def to_array(
    dataset: xr.Dataset,
    pack_names: Sequence[str],
    feature_counts: Mapping[str, int],
    is_3d: bool = False,
):
    """
    Convert dataset into a 2D array with [sample, feature] dimensions
    or 3D array with [sample, time, feature] dimensions.

    2D or 3D is selected based on whether any packed variables have multiple non-sample
    dimensions.

    The first dimension of each variable to pack is assumed to be the sample dimension,
    and the second (if it exists) is assumed to be the feature dimension.
    Each variable must be 1D or 2D.
    
    Args:
        dataset: dataset containing variables in self.pack_names to pack,
            dimensionality must match value of is_3d
        pack_names: names of variables to pack
        feature_counts: number of features for each variable
        is_3d: if True, output a 3D array. Useful when packing
            only scalars to avoid the time dimension being treated as feature

    Returns:
        array: 2D [sample, feature] array with data from the dataset
    """
    # we can assume here that the first dimension is the sample dimension
    n_samples = dataset[pack_names[0]].shape[0]
    total_features = sum(feature_counts[name] for name in pack_names)

    if is_3d:
        max_dims = 3
        # can assume all variables have [sample, time] dimensions
        n_times = dataset[pack_names[0]].shape[1]
        array = np.empty([n_samples, n_times, total_features])
    else:
        array = np.empty([n_samples, total_features])

    i_start = 0
    for name in pack_names:
        n_features = feature_counts[name]
        if is_3d:  # assume sample, time, feature arrays
            if n_features > 1:
                array[:, :, i_start : i_start + n_features] = dataset[name]
            else:
                array[:, :, i_start] = dataset[name]
        else:  # assume sample, feature arrays
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
            assert len(dimensions[name]) == 2, (name, dimensions[name])
            data_vars[name] = (
                dimensions[name],
                array[:, i_start : i_start + n_features],
            )
        else:
            data_vars[name] = (dimensions[name], array[:, i_start])
        i_start += n_features
    return xr.Dataset(data_vars)  # type: ignore


def count_features(
    quantity_names: Iterable[str],
    dataset: xr.Dataset,
    sample_dim_name: str,
    is_3d: bool = False,
) -> Mapping[str, int]:
    """Count the number of ML outputs corresponding to a set of quantities in a dataset.

    The first dimension of all variables indicated must be the sample dimension,
    and they must have at most one other dimension (treated as the "feature" dimension).

    Args:
        quantity_names: names of variables to include in the count
        dataset: a dataset containing the indicated variables,
            dimensionality must match value of is_3d
        sample_dim_name: dimension to treat as the "sample" dimension, any other
            dimensions are treated as a "feature" dimension.
        is_3d: pass as True if the dataset contains a time dimension
            [sample, time, feature]
    """
    for name in quantity_names:
        if len(dataset[name].dims) > 3:
            value = dataset[name]
            raise ValueError(
                "can only pack 1D/2D (sample[, z]) or 2D/3D (sample, time[, z]) "
                f"variables, recieved value for {name} with dimensions {value.dims}"
            )
    if is_3d:
        return _count_features_3d(quantity_names, dataset, sample_dim_name)
    elif any(len(dataset[name].dims) == 3 for name in quantity_names):
        # for safety, we want users to explicitly state they want to work on 3D
        raise ValueError("passed dataset has 3D variables, but is_3d is False")
    else:
        return _count_features_2d(quantity_names, dataset, sample_dim_name)


def _count_features_2d(
    quantity_names: Iterable[str], dataset: xr.Dataset, sample_dim_name: str
) -> Mapping[str, int]:
    """
    count features for (sample[, z]) arrays
    """
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


def _count_features_3d(
    quantity_names: Iterable[str], dataset: xr.Dataset, sample_dim_name: str
) -> Mapping[str, int]:
    """
    count features for (sample, time[, z]) arrays
    """
    return_dict = {}
    for name in quantity_names:
        value = dataset[name]
        if len(value.dims) == 2 and value.dims[0] == sample_dim_name:
            return_dict[name] = 1
        elif value.dims[0] != sample_dim_name:
            raise ValueError(
                f"cannot pack value for {name} whose first dimension is not the "
                f"sample dimension ({sample_dim_name}), has dims {value.dims}"
            )
        else:
            return_dict[name] = value.shape[2]
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
