from typing import Sequence, MutableMapping, Hashable, Tuple, Optional
import dataclasses
from datetime import timedelta
import logging
import intake
import cftime
import xarray as xr
from runtime.types import State, Diagnostics, Tendencies
import fv3gfs.util
from vcm.catalog import catalog as CATALOG
from vcm.safe import get_variables


logger = logging.getLogger(__name__)

QuantityState = MutableMapping[Hashable, fv3gfs.util.Quantity]


def get_timesteps(
    init_time: cftime.DatetimeJulian, timestep_seconds: float, n_timesteps: int
) -> Sequence[cftime.DatetimeJulian]:
    """Get sequence of model timesteps
    
    Args
        init_time (cftime.DatetimeJulian): model run start time
        timestep_seconds (float): model timestep
        n_timesteps: number of timesteps in model run
    
    Returns: Sequence of cftime.DatetimeJulian objects corresponding to ends
        of model run timesteps
    
    """
    return [
        init_time + timedelta(seconds=timestep_seconds * i)
        for i in range(1, n_timesteps + 1)
    ]


@dataclasses.dataclass
class PrescriberConfig:
    """Configuration for a prescriber object to set states in the model from an external source

    Attributes:
        dataset_key (str): location of the dataset that provides prescribe values;
            the routine first will try `dataset_key` as a `vcm.catalog` key, and if
            not present it will attempt to use it as a (local or remote) path to
            a zarr dataset
        variables (Sequence[str]): sequence of variable names in the dataset to prescribe
        consolidated (bool): optional, whether desired dataset has consolidated metadata;
            defaults to True

    Example::

        PrescriberConfig(
            dataset_key="gs://vcm-ml-intermediate/2021-03-fine-res-surface-radiative-fluxes/fine-res-surface-radiative-fluxes.zarr",
            variables=[
                "override_for_time_adjusted_total_sky_downward_shortwave_flux_at_surface", 
                "override_for_time_adjusted_total_sky_downward_longwave_flux_at_surface", 
                "override_for_time_adjusted_total_sky_net_shortwave_flux_at_surface",
            ]
        )

    """  # noqa

    dataset_key: str
    variables: Sequence[str]
    consolidated: bool = True


class Prescriber:

    net_moistening = "net_moistening"

    def __init__(
        self,
        config: PrescriberConfig,
        communicator: fv3gfs.util.CubedSphereCommunicator,
        timesteps: Optional[Sequence[cftime.DatetimeJulian]] = None,
    ):
        """Create a Prescriber object
        
        Args:
            config (PrescriberConfig),
            communicator (fv3gfs.util.CubedSphereCommunicator),
            timesteps (Sequence[cftime.DatetimeJulian]): optional sequence specifying
                all wrapper timesteps for which data is required; if not supplied,
                defaults to downloading entire time dimension of `dataset_key`
        """
        self._config = config
        self._communicator = communicator
        self._timesteps = timesteps
        prescribed_ds, time_coord = self._load_prescribed_ds()
        self._prescribed_ds: xr.Dataset = self._scatter_prescribed_ds(
            prescribed_ds, time_coord
        )

    def _load_prescribed_ds(
        self,
    ) -> Tuple[Optional[xr.Dataset], Optional[xr.DataArray]]:
        prescribed_ds: Optional[xr.Dataset]
        time_coord: Optional[xr.DataArray]
        if self._communicator.rank == 0:
            prescribed_ds, time_coord = _get_prescribed_ds(
                self._config.dataset_key,
                list(self._config.variables),
                self._timesteps,
                self._config.consolidated,
            )
        else:
            prescribed_ds, time_coord = None, None
        return prescribed_ds, time_coord

    def _scatter_prescribed_ds(
        self, prescribed_ds: Optional[xr.Dataset], time_coord: Optional[xr.DataArray]
    ) -> xr.Dataset:
        if isinstance(prescribed_ds, xr.Dataset):
            scattered_ds = _quantity_state_to_ds(
                self._communicator.scatter_state(_ds_to_quantity_state(prescribed_ds))
            )
        else:
            scattered_ds = _quantity_state_to_ds(self._communicator.scatter_state())
        time_coord = self._communicator.comm.bcast(time_coord, root=0)
        scattered_ds = scattered_ds.assign_coords({"time": time_coord})
        return scattered_ds

    def __call__(self, time, state):
        diagnostics: Diagnostics = {}
        prescribed_timestep: xr.Dataset = self._prescribed_ds.sel(time=time)
        state_updates: State = {
            name: prescribed_timestep[name] for name in prescribed_timestep.data_vars
        }
        for name in state_updates.keys():
            diagnostics[name] = state_updates[name]
        tendency: Tendencies = {}
        return tendency, diagnostics, state_updates

    def get_diagnostics(self, state, tendency):
        return {}

    def get_momentum_diagnostics(self, state, tendency):
        return {}


def _get_prescribed_ds(
    dataset_key: str,
    variables: Sequence[str],
    timesteps: Optional[Sequence[cftime.DatetimeJulian]],
    consolidated: bool = True,
) -> Tuple[xr.Dataset, xr.DataArray]:
    logger.info(f"Setting up dataset for state setting: {dataset_key}")
    ds = _open_ds(dataset_key, consolidated)
    ds = get_variables(ds, variables)
    if timesteps is not None:
        ds = ds.reindex(time=timesteps, method="bfill")
    time_coord = ds.coords["time"]
    return ds.drop_vars(names="time").load(), time_coord


def _open_ds(dataset_key: str, consolidated: bool) -> xr.Dataset:
    try:
        ds = CATALOG[dataset_key].to_dask()
    except KeyError:
        ds = intake.open_zarr(dataset_key, consolidated=consolidated).to_dask()
    return ds


def _ds_to_quantity_state(ds: xr.Dataset) -> QuantityState:
    quantity_state: QuantityState = {
        variable: fv3gfs.util.Quantity.from_data_array(ds[variable])
        for variable in ds.data_vars
    }
    return quantity_state


def _quantity_state_to_ds(quantity_state: QuantityState) -> xr.Dataset:
    ds = xr.Dataset(
        {
            variable: quantity_state[variable].data_array
            for variable in quantity_state.keys()
        }
    )
    return ds
