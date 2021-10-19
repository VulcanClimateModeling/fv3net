import logging
import threading
import queue
import xarray as xr
import numpy as np
import tensorflow as tf
from typing import Sequence, Tuple, List, Any, Iterable, Hashable

from ..._shared.packer import pack, Unpacker
from ..._shared import SAMPLE_DIM_NAME

logger = logging.getLogger(__name__)


class _XyArraySequence(tf.keras.utils.Sequence):
    """
    Wrapper object converting a sequence of batch datasets
    to a sequence of input/output numpy arrays.
    """

    def __init__(
        self,
        input_variables: Iterable[Hashable],
        output_variables: Iterable[Hashable],
        dataset_sequence: Sequence[xr.Dataset],
    ):
        self.input_variables = list(input_variables)
        self.output_variables = list(output_variables)
        self.dataset_sequence = dataset_sequence

    def __len__(self) -> int:
        return len(self.dataset_sequence)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        ds = self.dataset_sequence[idx]
        X, _ = pack(ds[self.input_variables], SAMPLE_DIM_NAME)
        y, _ = pack(ds[self.output_variables], SAMPLE_DIM_NAME)
        return X, y

    def get_unpackers(self) -> Tuple[Unpacker, Unpacker]:
        ds = self.dataset_sequence[0]
        _, X_unpacker = pack(ds[self.input_variables], SAMPLE_DIM_NAME)
        _, y_unpacker = pack(ds[self.output_variables], SAMPLE_DIM_NAME)
        return X_unpacker, y_unpacker


class _XyMultiArraySequence(tf.keras.utils.Sequence):
    """
    Wrapper object converting a sequence of batch datasets
    to a sequence of tuples of input/output numpy arrays.

    These tuples contain one unpacked numpy array for each input/output,
    in contrast to _XyArraySequence which is specialized to the case
    of a single input/output of packed arrays.
    """

    def __init__(
        self,
        X_names: Sequence[str],
        y_names: Sequence[str],
        dataset_sequence: Sequence[xr.Dataset],
    ):
        self.X_names = X_names
        self.y_names = y_names
        self.dataset_sequence = dataset_sequence

    def __len__(self) -> int:
        return len(self.dataset_sequence)

    def __getitem__(self, idx) -> Tuple[np.ndarray, np.ndarray]:
        ds = self.dataset_sequence[idx]
        X = tuple(ds[name].values for name in self.X_names)
        y = tuple(ds[name].values for name in self.y_names)
        return X, y


class _ThreadedSequencePreLoader(tf.keras.utils.Sequence):
    """
    Wrapper object for using a threaded pre-load to provide
    items for a generator.

    Note: This might not preserve strict sequence ordering
        ... but its faster.  Beware that it can load up to
        max_queue_size + num_workers into memory at the
        same time.
    """

    def __init__(
        self,
        seq: tf.keras.utils.Sequence,
        num_workers: int = 4,
        max_queue_size: int = 6,
    ):
        logger.debug(
            f"Initializing threaded batch loader with {num_workers} workers"
            f" and max queue size of {max_queue_size}"
        )
        self._seq = seq
        self.num_workers = num_workers
        self.max_queue_size = max_queue_size

    def __len__(self):
        return len(self._seq)

    def __getitem__(self, index) -> Any:
        return self._seq[index]

    def __iter__(self):

        init_q = queue.Queue()
        for idx in list(range(len(self))):
            init_q.put(idx)

        event = threading.Event()
        preloaded = queue.Queue(maxsize=self.max_queue_size)

        producers: List[threading.Thread] = [
            threading.Thread(
                target=self._produce_loaded_batches, args=(init_q, preloaded, event)
            )
            for i in range(self.num_workers)
        ]

        # Start workers
        for thread in producers:
            thread.start()
            logger.debug(f"Started worker thread {thread.ident}")

        # Generator on preloaded batches
        for i in range(len(self)):
            yield preloaded.get()

        # stop threads
        event.set()
        for thread in producers:
            logger.debug(f"Joining worker thread {thread.ident}")
            thread.join()

    def _produce_loaded_batches(self, src_q, dst_q, event):
        while not event.is_set():

            try:
                item = src_q.get(timeout=5)
            except queue.Empty:
                continue

            dst_q.put(self[item])
            src_q.task_done()
            logger.debug(f"Loadded batch #{item}")
