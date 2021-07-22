import os
import re
import vcm
import numpy as np
import xarray as xr

from functools import partial
from toolz import groupby
from typing import (
    Dict,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
    Hashable,
    Iterable,
    MutableMapping,
    Callable,
    cast,
)

from ._transformations import KeyMap
from .._utils import assign_net_physics_terms
from ..constants import (
    DERIVATION_FV3GFS_COORD,
    DERIVATION_SHIELD_COORD,
    RENAMED_SHIELD_DIAG_VARS,
)
from ._base import GeoMapper
from ._high_res_diags import open_high_res_diags
from ._merged import MergeOverlappingData
from loaders._config import mapper_functions
from loaders.typing import Mapper

Time = str
Tile = int
K = Tuple[Time, Tile]


DESCRIPTIVE_NAMES: Mapping[Hashable, Hashable] = {
    "T": "air_temperature",
    "t_dt_fv_sat_adj_coarse": "air_temperature_saturation_adjustment",
    "t_dt_nudge_coarse": "air_temperature_nudging",
    "t_dt_phys_coarse": "air_temperature_physics",
    "eddy_flux_vulcan_omega_temp": "air_temperature_unresolved_flux",
    "T_vulcan_omega_coarse": "air_temperature_total_resolved_flux",
    "T_storage": "air_temperature_storage",
    "sphum": "specific_humidity",
    "qv_dt_fv_sat_adj_coarse": "specific_humidity_saturation_adjustment",
    "qv_dt_phys_coarse": "specific_humidity_physics",
    "eddy_flux_vulcan_omega_sphum": "specific_humidity_unresolved_flux",
    "sphum_vulcan_omega_coarse": "specific_humidity_total_resolved_flux",
    "sphum_storage": "specific_humidity_storage",
    "vulcan_omega_coarse": "omega",
}


def _fv_sat_adj_metadata(field: str, field_units: str) -> Dict[str, str]:
    """Return the metadata attrs dict for" the saturation adjustment tendency of
    a field."""
    return {
        "units": f"{field_units}/s",
        "long_name": "tendency of {field} due "
        "to dynamical core saturation adjustment",
    }


def _nudging_metadata(field: str, field_units: str) -> Dict[str, str]:
    """Return the metadata attrs dict for" the nudging tendency of a field."""
    return {
        "units": f"{field_units}/s",
        "long_name": "tendency of {field} due " "to SHiELD nudging",
    }


def _physics_metadata(field: str, field_units: str) -> Dict[str, str]:
    """Return the metadata attrs dict for" the physics tendency of a field."""
    return {
        "units": f"{field_units}/s",
        "long_name": "tendency of {field} due " "to physics",
        "description": "sum of microphysics and any other parameterized process",
    }


def _storage_metadata(field: str, field_units: str) -> Dict[str, str]:
    """Return the metadata attrs dict for" the storage tendency of a field."""
    return {
        "units": f"{field_units}/s",
        "long_name": "storage tendency of {field}",
        "description": f"partial time derivative of {field} for fixed x, "
        "y, and output model level.  Sum of all the budget tendencies.",
    }


def _convergence_metadata(field: str, field_units: str) -> Dict[str, str]:
    """Return the metadata attrs dict for" the vertical eddy flux
    convergence tendency of a field."""
    return {
        "units": f"{field_units}/s",
        "long_name": "vertical eddy flux convergence tendency of {field}",
    }


TENDENCY_METADATA = {
    "air_temperature_saturation_adjustment": _fv_sat_adj_metadata(
        "air_temperature", "K"
    ),
    "air_temperature_nudging": _nudging_metadata("air_temperature", "K"),
    "air_temperature_physics": _physics_metadata("air_temperature", "K"),
    "air_temperature_storage": _storage_metadata("air_temperature", "K"),
    "air_temperature_convergence": _convergence_metadata("air_temperature", "K"),
    "specific_humidity_saturation_adjustment": _fv_sat_adj_metadata(
        "specific_humidity", "kg/kg"
    ),
    "specific_humidity_nudging": _nudging_metadata("specific_humidity", "kg/kg"),
    "specific_humidity_physics": _physics_metadata("specific_humidity", "kg/kg"),
    "specific_humidity_storage": _storage_metadata("specific_humidity", "kg/kg"),
    "specific_humidity_convergence": _convergence_metadata(
        "specific_humidity", "kg/kg"
    ),
}


def eddy_flux_coarse(unresolved_flux, total_resolved_flux, omega, field):
    """Compute re-coarsened eddy flux divergence from re-coarsed data
    """
    return unresolved_flux + (total_resolved_flux - omega * field)


def _center_to_interface(f: np.ndarray) -> np.ndarray:
    """Interpolate vertically cell centered data to the interface
    with linearly extrapolated inputs"""
    f_low = 2 * f[..., 0] - f[..., 1]
    f_high = 2 * f[..., -1] - f[..., -2]
    pad = np.concatenate([f_low[..., np.newaxis], f, f_high[..., np.newaxis]], axis=-1)
    return (pad[..., :-1] + pad[..., 1:]) / 2


def _convergence(eddy: np.ndarray, delp: np.ndarray) -> np.ndarray:
    """Compute vertical convergence of a cell-centered flux.

    This flux is assumed to vanish at the vertical boundaries
    """
    padded = _center_to_interface(eddy)
    # pad interfaces assuming eddy = 0 at edges
    return -np.diff(padded, axis=-1) / delp


def convergence(eddy: xr.DataArray, delp: xr.DataArray, dim: str = "p") -> xr.DataArray:
    return xr.apply_ufunc(
        _convergence,
        eddy,
        delp,
        input_core_dims=[[dim], [dim]],
        output_core_dims=[[dim]],
        dask="parallelized",
        output_dtypes=[eddy.dtype],
    )


class FineResolutionBudgetTiles(Mapping[Tuple[str, int], xr.Dataset]):
    """A Mapping to a fine-res-q1-q2 dataset"""

    def __init__(self, url):
        self._fs = vcm.cloud.get_fs(url)
        self._url = url
        self.files = self._fs.glob(os.path.join(url, "*.nc"))
        if len(self.files) == 0:
            raise ValueError("No file detected")

    def _parse_file(self, url) -> Tuple[str, int]:
        pattern = r"tile(.)\.nc"
        match = re.search(pattern, url)
        if match is None:
            raise ValueError(
                f"invalid url, must contain pattern {pattern} but received {url}"
            )
        date = vcm.parse_timestep_str_from_path(url)
        tile = match.group(1)
        return date, int(tile)

    def __getitem__(self, key):
        return vcm.open_remote_nc(self._fs, self._find_file(key))

    def _find_file(self, key):
        return [file for file in self.files if self._parse_file(file) == key][-1]

    def keys(self):
        return [self._parse_file(file) for file in self.files]

    def __len__(self):
        return len(self.keys())

    def __iter__(self):
        return iter(self.keys())


class GroupByTime(GeoMapper):
    def __init__(self, tiles: Mapping[K, xr.Dataset]):
        def fn(key):
            time, _ = key
            return time

        self._tiles = tiles
        self._time_lookup = groupby(fn, self._tiles.keys())

    def keys(self):
        return self._time_lookup.keys()

    def __getitem__(self, time: Time) -> xr.Dataset:
        datasets = [self._tiles[key] for key in self._time_lookup[time]]
        tiles = range(len(datasets))
        return xr.concat(datasets, dim="tile").assign_coords(tile=tiles)


class FineResolutionSources(GeoMapper):
    def __init__(
        self,
        fine_resolution_time_mapping: Mapping[Time, xr.Dataset],
        drop_vars: Sequence[str] = ("step", "time"),
        dim_order: Sequence[str] = ("tile", "z", "y", "x"),
        rename_vars: Optional[Mapping[Hashable, Hashable]] = None,
    ):
        self._time_mapping = fine_resolution_time_mapping
        self._drop_vars = drop_vars
        self._dim_order = dim_order
        self._rename_vars = rename_vars or {}

    def keys(self):
        return self._time_mapping.keys()

    def __getitem__(self, time: Time) -> xr.Dataset:
        time_slice = self._time_mapping[time].rename(DESCRIPTIVE_NAMES)
        return (
            self._derived_budget_ds(time_slice)
            .drop_vars(names=self._drop_vars, errors="ignore")
            .rename(self._rename_vars)
            .transpose(*self._dim_order)
        )

    def _compute_coarse_eddy_flux_convergence_ds(
        self, budget_time_ds: xr.Dataset, field: str, vertical_dimension: str
    ) -> xr.Dataset:
        eddy_flux = eddy_flux_coarse(
            budget_time_ds[f"{field}_unresolved_flux"],
            budget_time_ds[f"{field}_total_resolved_flux"],
            budget_time_ds["omega"],
            budget_time_ds[field],
        )
        budget_time_ds[f"{field}_convergence"] = convergence(
            eddy_flux, budget_time_ds["delp"], dim=vertical_dimension
        )
        return budget_time_ds

    def _add_tendency_term_metadata(self, budget_time_ds: xr.Dataset) -> xr.Dataset:
        for variable, attrs in TENDENCY_METADATA.items():
            if variable in budget_time_ds:
                budget_time_ds[variable] = budget_time_ds[variable].assign_attrs(
                    **attrs
                )
        return budget_time_ds

    def _derived_budget_ds(
        self,
        budget_time_ds: xr.Dataset,
        variable_prefixes: Mapping[str, str] = None,
        apparent_source_terms: Sequence[str] = (
            "physics",
            "saturation_adjustment",
            "convergence",
        ),
        vertical_dimension: str = "pfull",
    ) -> xr.Dataset:

        if variable_prefixes is None:
            variable_prefixes = {
                "air_temperature": "Q1",
                "specific_humidity": "Q2",
            }

        for variable_name, apparent_source_name in variable_prefixes.items():
            budget_time_ds = (
                budget_time_ds.pipe(
                    self._compute_coarse_eddy_flux_convergence_ds,
                    variable_name,
                    vertical_dimension,
                )
                .pipe(
                    self._insert_budget_dQ,
                    variable_name,
                    f"d{apparent_source_name}",
                    apparent_source_terms,
                )
                .pipe(
                    self._insert_budget_pQ, variable_name, f"p{apparent_source_name}",
                )
            )
        budget_time_ds = (
            budget_time_ds.pipe(self._insert_physics)
            .pipe(assign_net_physics_terms)
            .pipe(self._add_tendency_term_metadata)
        )
        return budget_time_ds

    @staticmethod
    def _insert_budget_dQ(
        budget_time_ds: xr.Dataset,
        variable_name: str,
        apparent_source_name: str,
        apparent_source_terms: Sequence[str],
    ) -> xr.Dataset:
        """Insert dQ (really Q) from other budget terms"""

        source_vars = [f"{variable_name}_{term}" for term in apparent_source_terms]
        apparent_source = (
            vcm.safe.get_variables(budget_time_ds, source_vars)
            .to_array(dim="variable")
            .sum(dim="variable")
        )
        budget_time_ds = budget_time_ds.assign({apparent_source_name: apparent_source})

        units = budget_time_ds[f"{variable_name}_{apparent_source_terms[0]}"].attrs.get(
            "units", None
        )
        budget_time_ds[apparent_source_name].attrs.update(
            {"name": f"apparent source of {variable_name}"}
        )
        if units is not None:
            budget_time_ds[apparent_source_name].attrs.update({"units": units})

        return budget_time_ds

    @staticmethod
    def _insert_budget_pQ(
        budget_time_ds: xr.Dataset, variable_name: str, apparent_source_name: str,
    ) -> xr.Dataset:
        """Insert pQ = 0 in the fine-res budget case"""

        budget_time_ds = budget_time_ds.assign(
            {apparent_source_name: xr.zeros_like(budget_time_ds[f"{variable_name}"])}
        )

        budget_time_ds[apparent_source_name].attrs[
            "name"
        ] = f"coarse-res physics tendency of {variable_name}"

        units = budget_time_ds[f"{variable_name}"].attrs.get("units", None)
        if units is not None:
            budget_time_ds[apparent_source_name].attrs["units"] = f"{units}/s"

        return budget_time_ds

    @staticmethod
    def _insert_physics(
        budget_time_ds: xr.Dataset,
        physics_varnames: Iterable[str] = RENAMED_SHIELD_DIAG_VARS.values(),
    ) -> xr.Dataset:

        template_2d_var = budget_time_ds["air_temperature"].isel({"pfull": 0})

        physics_vars: MutableMapping[Hashable, Hashable] = {}
        for var in physics_varnames:
            physics_var = xr.full_like(template_2d_var, fill_value=0.0)
            physics_vars[var] = physics_var

        return budget_time_ds.assign(physics_vars)


def open_fine_resolution_budget(url: str) -> Mapping[str, xr.Dataset]:
    """Open a mapping interface to the fine resolution budget data

    Example:

        >>> from loaders import open_fine_resolution_budget
        >>> loader = open_fine_resolution_budget('gs://vcm-ml-scratch/noah/2020-05-19/')
        >>> len(loader)
        479
        >>> loader['20160805.202230']
        <xarray.Dataset>
        Dimensions:                         (grid_xt: 48, grid_yt: 48, pfull: 79, tile: 6)
        Coordinates:
            time                            object 2016-08-05 20:22:30
            step                            <U6 'middle'
        * tile                            (tile) int64 1 2 3 4 5 6
        Dimensions without coordinates: grid_xt, grid_yt, pfull
        Data variables:
            air_temperature                          (tile, pfull, grid_yt, grid_xt) float32 235.28934 ... 290.56107
            air_temperature_convergence              (tile, grid_yt, grid_xt, pfull) float32 4.3996937e-07 ... 1.7985441e-06
            air_temperature_microphysics             (tile, pfull, grid_yt, grid_xt) float32 0.0 ... -5.5472506e-06
            air_temperature_nudging                  (tile, pfull, grid_yt, grid_xt) float32 0.0 ... 2.0156076e-06
            air_temperature_physics                  (tile, pfull, grid_yt, grid_xt) float32 2.3518855e-06 ... -3.3252392e-05
            air_temperature_unresolved_flux          (tile, pfull, grid_yt, grid_xt) float32 0.26079428 ... 0.6763954
            air_temperature_total_resolved_flux      (tile, pfull, grid_yt, grid_xt) float32 0.26079428 ... 0.6763954
            air_temperature_storage                  (tile, pfull, grid_yt, grid_xt) float32 0.000119928314 ... 5.2825694e-06
            specific_humidity                        (tile, pfull, grid_yt, grid_xt) float32 5.7787e-06 ... 0.008809893
            specific_humidity_convergence            (tile, grid_yt, grid_xt, pfull) float32 -6.838638e-14 ... -1.7079346e-08
            specific_humidity_microphysics           (tile, pfull, grid_yt, grid_xt) float32 0.0 ... 1.6763515e-09
            specific_humidity_physics                (tile, pfull, grid_yt, grid_xt) float32 -1.961625e-14 ... 5.385441e-09
            specific_humidity_unresolved_flux        (tile, pfull, grid_yt, grid_xt) float32 6.4418755e-09 ... 2.0072384e-05
            specific_humidity_total_resolved_flux    (tile, pfull, grid_yt, grid_xt) float32 6.4418755e-09 ... 2.0072384e-05
            specific_humidity_storage                (tile, pfull, grid_yt, grid_xt) float32 -6.422655e-11 ... -5.3609618e-08
    """  # noqa
    tiles = FineResolutionBudgetTiles(url)
    return GroupByTime(tiles)


@mapper_functions.register
def open_fine_res_apparent_sources(
    data_path: str,
    shield_diags_path: str = None,
    offset_seconds: Union[int, float] = 0,
) -> Mapper:
    """Open a derived mapping interface to the fine resolution budget, grouped
        by time and with derived apparent sources

    Args:
        data_path (str): path to fine res dataset
        shield_diags_path: path to directory containing a zarr store of SHiELD
            diagnostics coarsened to the nudged model resolution (optional)
        offset_seconds: amount to shift the keys forward by in seconds. For
            example, if the underlying data contains a value at the key
            "20160801.000730", a value off 450 will shift this forward 7:30
            minutes, so that this same value can be accessed with the key
            "20160801.001500"
    """

    rename_vars: Mapping[Hashable, Hashable] = {
        "grid_xt": "x",
        "grid_yt": "y",
        "pfull": "z",
        "delp": "pressure_thickness_of_atmospheric_layer",
    }
    dim_order = ("tile", "z", "y", "x")
    drop_vars = ("step",)

    fine_resolution_sources_mapper = FineResolutionSources(
        open_fine_resolution_budget(data_path),
        drop_vars=drop_vars,
        dim_order=dim_order,
        rename_vars=rename_vars,
    )

    shift_timestamp = cast(
        Callable[[xr.Dataset], xr.Dataset],
        partial(vcm.shift_timestamp, seconds=offset_seconds),
    )

    shifted_mapper = KeyMap(shift_timestamp, fine_resolution_sources_mapper)

    if shield_diags_path is not None:
        shield_diags_mapper = open_high_res_diags(shield_diags_path)
        final_mapper: Mapper = MergeOverlappingData(
            shield_diags_mapper,
            shifted_mapper,
            source_name_left=DERIVATION_SHIELD_COORD,
            source_name_right=DERIVATION_FV3GFS_COORD,
        )
    else:
        final_mapper = shifted_mapper

    return final_mapper
