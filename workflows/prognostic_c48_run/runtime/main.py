import gc
import logging
from mpi4py import MPI

import tensorflow as tf
from runtime.loop import TimeLoop
import pace.util as util
import runtime

STATISTICS_LOG_NAME = "statistics"
PROFILES_LOG_NAME = "profiles"

logging.basicConfig(level=logging.INFO)

logging.getLogger("pace.util").setLevel(logging.WARN)
logging.getLogger("fsspec").setLevel(logging.WARN)
logging.getLogger("urllib3").setLevel(logging.WARN)

logger = logging.getLogger(__name__)


def main():
    comm = MPI.COMM_WORLD

    config = runtime.get_config()
    wrapper = runtime.get_wrapper(config)
    runtime.capture_fv3gfs_funcs(wrapper)
    wrapper.initialize()

    partitioner = util.CubedSpherePartitioner.from_namelist(runtime.get_namelist())
    for name in [STATISTICS_LOG_NAME, PROFILES_LOG_NAME]:
        runtime.setup_file_logger(name)

    loop = TimeLoop(config, wrapper, comm=comm)

    diag_files = runtime.get_diagnostic_files(
        config.diagnostics, partitioner, comm, initial_time=loop.time
    )
    if comm.rank == 0:
        runtime.write_chunks(config)

    writer = tf.summary.create_file_writer(f"tensorboard/rank_{comm.rank}")

    with writer.as_default():
        for time, diagnostics in loop:

            if comm.rank == 0:
                logger.debug(f"diags: {list(diagnostics.keys())}")

            averages = runtime.globally_average_2d_diagnostics(
                comm, diagnostics, exclude=loop._states_to_output
            )
            profiles = runtime.globally_sum_3d_diagnostics(
                comm, diagnostics, ["specific_humidity_limiter_active"]
            )
            if comm.rank == 0:
                runtime.log_mapping(time, averages, STATISTICS_LOG_NAME)
                runtime.log_mapping(time, profiles, PROFILES_LOG_NAME)

            for diag_file in diag_files:
                diag_file.observe(time, diagnostics)

    # Diag files *should* flush themselves on deletion but
    # fv3gfs.wrapper.cleanup short-circuits the usual python deletion
    # mechanisms
    for diag_file in diag_files:
        diag_file.flush()

    loop.log_global_timings()
    return wrapper


if __name__ == "__main__":

    wrapper = main()
    # need to cleanup any python objects that may have MPI operations before
    # calling wrapper.cleanup
    # this avoids the following error message:
    #
    #    Attempting to use an MPI routine after finalizing MPICH
    gc.collect()
    wrapper.cleanup()
