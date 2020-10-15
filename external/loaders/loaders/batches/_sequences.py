import os
import xarray as xr
import glob
import joblib
import collections
from copy import deepcopy
from functools import partial
from numpy.random import RandomState
from typing import (
    Callable,
    Sequence,
    MutableMapping,
    TypeVar,
    Hashable,
    Any,
    Optional,
    Union,
)

T = TypeVar("T")


class BaseSequence(Sequence[T]):
    def local(self, path, n_jobs=4):
        os.makedirs(path, exist_ok=True)
        joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(self._save_item)(path, i) for i in range(len(self))
        )
        return Local(os.path.abspath(path))

    def _save_item(self, path, i):
        item = self[i]
        cleaned = item.drop("sample")
        cleaned.to_netcdf(os.path.join(path, "%05d.nc" % i))

    def take(self, n):
        return Take(self, n)

    def map(self, func):
        return FunctionOutputSequence(func, self)


class Take(BaseSequence[T]):
    def __init__(self, parent_seq, n):
        self._seq = parent_seq
        self.n = n

    def __getitem__(self, i):
        if i < len(self):
            return self._seq[i]
        else:
            raise IndexError()

    def __len__(self):
        return self.n


class Local(BaseSequence[T]):
    def __init__(self, path):
        self.path = path

    @property
    def files(self):
        return sorted(glob.glob(os.path.join(self.path, "*.nc")))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return xr.open_dataset(self.files[i])


class FunctionOutputSequence(BaseSequence[T]):
    """A wrapper over a sequence of function arguments passed into a function.

    Attributes:
        attrs: a dictionary of metadata.
    """

    attrs: MutableMapping[Hashable, Any]

    def __init__(self, func: Callable[..., T], args_sequence: Sequence[Any]):
        """
        Args:
            func: the function to call, which takes in one argument
            args_sequence: a sequence of arguments
        Returns:
            result_sequence: a sequence of function results
        """
        if not isinstance(args_sequence, collections.Sequence):
            raise TypeError(f"args_sequence must be a sequence, got {args_sequence}")
        self._func = func
        self._args = args_sequence
        self.attrs = {}

    def __getitem__(self, item: Union[int, slice]) -> T:

        if isinstance(item, int):
            return self._func(self._args[item])
        elif isinstance(item, slice):
            return self._slice_selection(item)
        else:
            TypeError(f"Invalid argument type of {type(item)} passed into __getitem__.")

    def _slice_selection(self, selection: slice):
        seq = self.__class__(self._func, self._args[selection])
        seq.attrs.update(deepcopy(self.attrs))
        return seq

    def __len__(self) -> int:
        return len(self._args)


def shuffle(
    sequence: Sequence[Any], seed: Optional[int] = None
) -> FunctionOutputSequence:
    """
    Shuffle a sequence by creating a new FunctionOutputSequence
    with shuffled indices as arguments.  Preserves potentially lazy
    operations on input sequence __getitem__ calls by shuffling
    index arguments.

    Args:
        sequence:  Input sequence to have access indices shuffled
        seed: Seed for random number generator used for shuffling
    Returns:
        A new shuffled sequence
    """
    random = RandomState(seed)
    seq_len = len(sequence)
    shuffled = random.choice(seq_len, size=seq_len, replace=False).tolist()
    func = partial(_simple_getitem, sequence)
    return FunctionOutputSequence(func, shuffled)


def _simple_getitem(sequence: Sequence[Any], item: Union[int, slice]):
    return sequence[item]
