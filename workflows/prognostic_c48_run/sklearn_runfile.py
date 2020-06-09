import logging
from typing import Mapping, Set

import fsspec
import zarr
from sklearn.externals import joblib
from sklearn.utils import parallel_backend
import xarray as xr

import fv3gfs
from fv3gfs._wrapper import get_time
import fv3util
import runtime
from fv3net.regression.sklearn.wrapper import SklearnWrapper

from mpi4py import MPI

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

State = Mapping[str, xr.DataArray]

# following variables are required no matter what feature set is being used
TEMP = "air_temperature"
SPHUM = "specific_humidity"
DELP = "pressure_thickness_of_atmospheric_layer"
PRECIP_RATE = "surface_precipitation_rate"
REQUIRED_VARIABLES = {TEMP, SPHUM, DELP, PRECIP_RATE}

cp = 1004
gravity = 9.81


class RenamingAdapter:
    """Adapter object for renaming"""

    def __init__(self, model: SklearnWrapper, rename_in, rename_out=None):
        self.model = model
        self.rename_in = rename_in
        if rename_out is None:
            self.rename_out = dict(zip(rename_in.values(), rename_in.keys()))
        else:
            self.rename_out = rename_out

    @staticmethod
    def _rename(ds: xr.Dataset, rename: Mapping[str, str]) -> xr.Dataset:
        all_names = set(ds.data_vars) | set(ds.dims) | set(rename)
        rename_restricted = {key: rename[key] for key in all_names}
        return ds.rename(rename_restricted)

    def _rename_inputs(self, ds):
        # TODO typehints
        return self._rename(ds, self.rename_in)

    def _rename_outputs(self, ds):
        return self._rename(ds, self.rename_out)

    @property
    def variables(self) -> Set[str]:
        return {self.rename_out.get(var, var) for var in self.model.input_vars_}

    def predict(self, arg: xr.Dataset, sample_dim: str) -> xr.Dataset:
        input_ = self._rename_inputs(arg)
        prediction = self.model.predict(input_, sample_dim)
        return self._rename_outputs(prediction)


def compute_diagnostics(state, diags):

    net_moistening = (diags["dQ2"] * state[DELP] / gravity).sum("z")
    physics_precip = state[PRECIP_RATE]

    return dict(
        net_moistening=(net_moistening)
        .assign_attrs(units="kg/m^2/s")
        .assign_attrs(description="column integrated ML model moisture tendency"),
        net_heating=(diags["dQ1"] * state[DELP] / gravity * cp)
        .sum("z")
        .assign_attrs(units="W/m^2")
        .assign_attrs(description="column integrated ML model heating"),
        water_vapor_path=(state[SPHUM] * state[DELP] / gravity)
        .sum("z")
        .assign_attrs(units="mm")
        .assign_attrs(description="column integrated water vapor"),
        physics_precip=(physics_precip)
        .assign_attrs(units="kg/m^2/s")
        .assign_attrs(
            description="surface precipitation rate due to parameterized physics"
        ),
    )


def rename_diagnostics(diags):
    """Postfix ML output names with _diagnostic and create zero-valued outputs in
    their stead. Function operates in place."""
    for variable in ["net_moistening", "net_heating"]:
        attrs = diags[variable].attrs
        diags[f"{variable}_diagnostic"] = diags[variable].assign_attrs(
            description=attrs["description"] + " (diagnostic only)"
        )
        diags[variable] = xr.zeros_like(diags[variable]).assign_attrs(attrs)


def open_model(config):
    # Load the model
    rename = config.get("input_variable_standard_names", {})
    with fsspec.open(config["model"], "rb") as f:
        model = joblib.load(f)
    return RenamingAdapter(model, rename)


def predict(model: RenamingAdapter, state: State) -> State:
    """Given ML model and state, return tendency prediction."""
    stacked = xr.Dataset(state).stack(sample=["x", "y"])
    with parallel_backend("threading", n_jobs=1):
        output = model.predict(stacked, "sample").unstack("sample")
    return {key: output[key] for key in output}


def apply(state: State, tendency: State, dt: float) -> State:
    """Given state and tendency prediction, return updated state.
    Returned state only includes variables updated by ML model."""
    with xr.set_options(keep_attrs=True):
        updated = {
            SPHUM: state[SPHUM] + tendency["dQ2"] * dt,
            TEMP: state[TEMP] + tendency["dQ1"] * dt,
        }
    return updated


if __name__ == "__main__":
    args = runtime.get_config()
    NML = runtime.get_namelist()
    TIMESTEP = NML["coupler_nml"]["dt_atmos"]

    times = []
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # change into run directoryy
    MPI.COMM_WORLD.barrier()  # wait for master rank to write run directory

    do_only_diagnostic_ml = args["scikit_learn"].get("diagnostic_ml", False)

    # open zarr tape for output
    if rank == 0:
        GROUP = zarr.open_group(args["scikit_learn"]["zarr_output"], mode="w")
    else:
        GROUP = None

    GROUP = comm.bcast(GROUP, root=0)

    if rank == 0:
        logger.info("Downloading Sklearn Model")
        MODEL = open_model(args["scikit_learn"])
        logger.info("Model downloaded")
    else:
        MODEL = None

    MODEL = comm.bcast(MODEL, root=0)
    variables = list(MODEL.variables | REQUIRED_VARIABLES)
    if rank == 0:
        logger.debug(f"Prognostic run requires variables: {variables}")

    if rank == 0:
        logger.info(f"Timestep: {TIMESTEP}")

    fv3gfs.initialize()
    for i in range(fv3gfs.get_step_count()):
        if rank == 0:
            logger.debug(f"Dynamics Step")
        fv3gfs.step_dynamics()
        if rank == 0:
            logger.debug(f"Physics Step")
        fv3gfs.step_physics()

        if rank == 0:
            logger.debug(f"Getting state variables: {variables}")
        state = {
            key: value.data_array
            for key, value in fv3gfs.get_state(names=variables).items()
        }

        if rank == 0:
            logger.debug("Computing RF updated variables")
        tendency = predict(MODEL, state)

        if do_only_diagnostic_ml:
            updated_state = {}
        else:
            updated_state = apply(state, tendency, dt=TIMESTEP)

        if rank == 0:
            logger.debug("Setting Fortran State")
        print(updated_state)
        fv3gfs.set_state(
            {
                key: fv3util.Quantity.from_data_array(value)
                for key, value in updated_state.items()
            }
        )

        diagnostics = compute_diagnostics(state, tendency)
        if do_only_diagnostic_ml:
            rename_diagnostics(diagnostics)

        if i == 0:
            writers = runtime.init_writers(GROUP, comm, diagnostics)
        runtime.append_to_writers(writers, diagnostics)

        times.append(get_time())

    fv3gfs.cleanup()
