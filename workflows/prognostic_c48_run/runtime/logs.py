import datetime
import json
import logging
import contextlib
import tempfile
import sys
import os
from typing import Any, Mapping, Union
import cftime


@contextlib.contextmanager
def capture_stream(stream):
    """fork is not compatible with mpi so this won't work
    
    The wait line below hangs indefinitely.
    """

    # create a pipe to communicate with the fork
    r, w = os.pipe()

    # fork the process
    pid = os.fork()

    if pid > 0:
        # parent process:
        # close the reading end, we won't need this
        os.close(r)
        try:
            orig_file_handle = os.dup(stream.fileno())
            # overwrite the streams fileno with a the pipe to be read by the forked
            # process below
            os.dup2(w, stream.fileno())
            yield
        finally:
            # restore the original file handle
            os.dup2(orig_file_handle, stream.fileno())

            os.close(w)
            os.wait()
    else:
        # child process:
        # close the writing end, we won't need this
        os.close(w)
        with os.fdopen(r) as f:
            for line in f:
                logging.debug(line.strip())
        sys.exit(0)


@contextlib.contextmanager
def capture_stream_mpi(stream, logger_name="fv3gfs"):

    # parent process:
    # close the reading end, we won't need this
    with tempfile.TemporaryFile() as out:
        try:
            orig_file_handle = os.dup(stream.fileno())
            # overwrite the streams fileno with a the pipe to be read by the forked
            # process below
            os.dup2(out.fileno(), stream.fileno())
            yield
        finally:
            # restore the original file handle
            os.dup2(orig_file_handle, stream.fileno())

            # print logging info
            logger = logging.getLogger(logger_name)
            out.seek(0)
            for line in out:
                logger.debug(line.strip().decode("UTF-8"))


def captured_stream(func):
    def myfunc(*args, **kwargs):
        with capture_stream_mpi(sys.stdout):
            return func(*args, **kwargs)

    return myfunc


def capture_fv3gfs_funcs():
    """Surpress stderr and stdout from all fv3gfs functions"""
    import fv3gfs.wrapper as wrapper  # noqa

    for func in ["step_dynamics", "step_physics", "initialize", "cleanup"]:
        setattr(wrapper, func, captured_stream(getattr(wrapper, func)))


def setup_file_logger(name: str):
    """Configure a logger which streams to name.txt as well as stderr."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(f"{name}.txt")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(fmt="%(levelname)s:%(name)s:%(message)s"))

    logger.addHandler(fh)
    logger.addHandler(ch)


def log_mapping(
    time: Union[datetime.datetime, cftime.DatetimeJulian],
    content: Mapping[str, Any],
    logger_name: str,
):
    """Serialize a mapping 'content' to logger_name using JSON.
    
    Warning:
        content must be a mapping whose keys and values are serializable by JSON.
    """
    dt = datetime.datetime(
        time.year, time.month, time.day, time.hour, time.minute, time.second
    )
    msg = json.dumps({"time": dt.isoformat(), **content})
    logging.getLogger(logger_name).info(msg)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    with capture_stream(sys.stdout):
        print("should appear")
        print("should appear")
        print("should appear")
