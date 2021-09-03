from typing import cast, Sequence, Hashable, Iterable, TypeVar
import warnings
import xarray as xr


T = TypeVar("T", xr.DataArray, xr.Dataset)


def get_variables(ds: xr.Dataset, variables: Iterable[Hashable]) -> xr.Dataset:
    """ds[...] is very confusing function from a typing perspective and should be
    avoided in long-running pipeline codes. This function introduces a type-stable
    alternative that works better with mypy.

    In particular, ds[('a' , 'b' ,'c')] looks for a variable named ('a', 'b', 'c') which
    usually doesn't exist, so it causes a key error. but ds[['a', 'b', 'c']] makes a
    dataset only consisting of the variables 'a', 'b', and 'c'. This causes tons of
    hard to find errors.
    """
    variables = list(variables)
    return cast(xr.Dataset, ds[variables])


def _validate_stack_dims(
    ds, dims, allowed_broadcast_dims=(), allowed_broadcast_vars=()
):
    """Don't broadcast arrays"""
    ds_ = ds.to_dataset() if isinstance(ds, xr.DataArray) else ds
    for variable in ds_:
        if variable in allowed_broadcast_vars:
            continue
        var_dims = ds_[variable].dims
        broadcast_dims = set(dims) - (set(var_dims) | set(allowed_broadcast_dims))
        if len(broadcast_dims) > 0:
            raise ValueError(
                f"{variable} will be broadcast to include unallowed dimensions "
                f"{broadcast_dims}. This could greatly increase the size of dataset."
            )


def stack_once(
    ds: T,
    dim,
    dims: Sequence[Hashable],
    allowed_broadcast_dims: Sequence[Hashable] = (),
    allowed_broadcast_vars: Sequence[Hashable] = (),
):
    """Stack once raising ValueError if any unexpected broadcasting occurs"""
    _validate_stack_dims(ds, dims, allowed_broadcast_dims, allowed_broadcast_vars)
    return ds.stack({dim: dims})


def warn_if_intersecting(old: Iterable[Hashable], new: Iterable[Hashable]):
    """
    Warn if renaming to new data keys will overwrite names (e.g., in a xr.Dataset)
    via an overlap with old keys or from duplication in new keys.

    Args:
        old: Original keys to check against
        new: Incoming keys to check for duplicates or existence in old
    """
    duplicates = {item for item in new if list(new).count(item) > 1}
    overlap = set(old) & set(new)
    overwrites = duplicates | overlap
    if len(overwrites) > 0:
        warnings.warn(
            UserWarning(
                f"Unsafe renaming of keys detected. Overlap: {overlap}"
                f"  Duplicates: {duplicates}"
            )
        )
