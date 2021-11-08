import logging
import warnings
import xarray as xr

import vcm

SECONDS_PER_DAY = 86400

logger = logging.getLogger(__name__)


def physics_variables(ds: xr.Dataset) -> xr.Dataset:
    """
    Compute selected derived variables from a physics dataset and merge them back in.
    
    Args:
        ds: Dataset to calculated derived values from and merge to

    Note:
        Derived variables are linear combinations of other variables with no reductions.
    """
    arrays = []
    for func in [
        _column_pq1,
        _column_pq2,
        _column_dq1,
        _column_dq2,
        _column_q1,
        _column_q2,
        _total_precip_to_surface,
        _column_dqu,
        _column_dqv,
        _column_nq1,
        _column_nq2,
        _column_dq1_or_nq1,
        _column_dq2_or_nq2,
        _water_vapor_path,
        _minus_column_q2,
    ]:
        try:
            arrays.append(func(ds))
        except (KeyError, AttributeError):  # account for ds[var] and ds.var notations
            logger.warning(f"Missing variable for calculation in {func.__name__}")
    return ds.merge(xr.merge(arrays))


def _column_pq1(ds: xr.Dataset) -> xr.DataArray:
    DSWRFsfc_name = "DSWRFsfc_from_RRTMG" if "DSWRFsfc_from_RRTMG" in ds else "DSWRFsfc"
    USWRFsfc_name = "USWRFsfc_from_RRTMG" if "USWRFsfc_from_RRTMG" in ds else "USWRFsfc"
    DLWRFsfc_name = "DLWRFsfc_from_RRTMG" if "DLWRFsfc_from_RRTMG" in ds else "DLWRFsfc"
    net_heating_arg_labels = [
        DLWRFsfc_name,
        DSWRFsfc_name,
        "ULWRFsfc",
        "ULWRFtoa",
        USWRFsfc_name,
        "USWRFtoa",
        "DSWRFtoa",
        "SHTFLsfc",
        "PRATEsfc",
    ]
    net_heating_args = [ds[var] for var in net_heating_arg_labels]
    column_pq1 = vcm.net_heating(*net_heating_args)
    column_pq1.attrs = {
        "long_name": "<pQ1> column integrated heating from physics",
        "units": "W/m^2",
    }
    return column_pq1.rename("column_integrated_pQ1")


def _column_pq2(ds: xr.Dataset) -> xr.DataArray:
    evap = vcm.latent_heat_flux_to_evaporation(ds.LHTFLsfc)
    column_pq2 = SECONDS_PER_DAY * (evap - ds.PRATEsfc)
    column_pq2.attrs = {
        "long_name": "<pQ2> column integrated moistening from physics",
        "units": "mm/day",
    }
    return column_pq2.rename("column_integrated_pQ2")


def _column_dq1(ds: xr.Dataset) -> xr.DataArray:
    if "net_heating_due_to_machine_learning" in ds:
        warnings.warn(
            "'net_heating_due_to_machine_learning' is a deprecated variable name. "
            "It will not be supported in future versions of fv3net. Use "
            "'column_heating_due_to_machine_learning' instead.",
            DeprecationWarning,
        )
        # fix isochoric vs isobaric transition issue
        column_dq1 = 716.95 / 1004 * ds.net_heating_due_to_machine_learning
    elif "net_heating" in ds:
        warnings.warn(
            "'net_heating' is a deprecated variable name. "
            "It will not be supported in future versions of fv3net. Use "
            "'column_heating_due_to_machine_learning' instead.",
            DeprecationWarning,
        )
        # fix isochoric vs isobaric transition issue
        column_dq1 = 716.95 / 1004 * ds.net_heating
    elif "column_heating_due_to_machine_learning" in ds:
        column_dq1 = ds.column_heating_due_to_machine_learning
    elif "storage_of_internal_energy_path_due_to_machine_learning" in ds:
        column_dq1 = ds.storage_of_internal_energy_path_due_to_machine_learning
    else:
        # assume given dataset is for a baseline or verification run
        column_dq1 = xr.zeros_like(ds.PRATEsfc)
    column_dq1.attrs = {
        "long_name": "<dQ1> column integrated heating from ML",
        "units": "W/m^2",
    }
    return column_dq1.rename("column_integrated_dQ1")


def _column_dq2(ds: xr.Dataset) -> xr.DataArray:
    if "net_moistening_due_to_machine_learning" in ds:
        column_dq2 = SECONDS_PER_DAY * ds.net_moistening_due_to_machine_learning
    elif "storage_of_specific_humidity_path_due_to_machine_learning" in ds:
        column_dq2 = (
            SECONDS_PER_DAY
            * ds.storage_of_specific_humidity_path_due_to_machine_learning
        )
    elif "net_moistening" in ds:
        # for backwards compatibility
        warnings.warn(
            "'net_moistening' is a deprecated variable name. "
            "It will not be supported in future versions of fv3net. Use "
            "'net_moistening_due_to_machine_learning' instead.",
            DeprecationWarning,
        )
        column_dq2 = SECONDS_PER_DAY * ds.net_moistening
    else:
        # assume given dataset is for a baseline or verification run
        column_dq2 = xr.zeros_like(ds.PRATEsfc)
    column_dq2.attrs = {
        "long_name": "<dQ2> column integrated moistening from ML",
        "units": "mm/day",
    }
    return column_dq2.rename("column_integrated_dQ2")


def _column_dqu(ds: xr.Dataset) -> xr.DataArray:
    if "column_integrated_dQu" in ds:
        warnings.warn(
            "'column_integrated_dQu' is a deprecated variable name. "
            "It will not be supported in future versions of fv3net. Use "
            "'column_integrated_dQu_stress' (units of Pa) instead.",
            DeprecationWarning,
        )
        # convert from m/s/s to Pa by multiplying by surface pressure divided by gravity
        column_dqu = 100000 / 9.8065 * ds.column_integrated_dQu
    elif "column_integrated_dQu_stress" in ds:
        column_dqu = ds.column_integrated_dQu_stress
    else:
        # assume given dataset has no ML prediction of momentum tendencies
        column_dqu = xr.zeros_like(ds.PRATEsfc)
    column_dqu.attrs = {
        "long_name": "<dQu> column integrated eastward wind tendency from ML",
        "units": "Pa",
    }
    return column_dqu.rename("column_int_dQu")


def _column_dqv(ds: xr.Dataset) -> xr.DataArray:
    if "column_integrated_dQv" in ds:
        warnings.warn(
            "'column_integrated_dQv' is a deprecated variable name. "
            "It will not be supported in future versions of fv3net. Use "
            "'column_integrated_dQv_stress' (units of Pa) instead.",
            DeprecationWarning,
        )
        # convert from m/s/s to Pa by multiplying by surface pressure divided by gravity
        column_dqv = 100000 / 9.8065 * ds.column_integrated_dQv
    else:
        # assume given dataset has no ML prediction of momentum tendencies
        column_dqv = xr.zeros_like(ds.PRATEsfc)
    column_dqv.attrs = {
        "long_name": "<dQv> column integrated northward wind tendency from ML",
        "units": "Pa",
    }
    return column_dqv.rename("column_int_dQv")


def _column_q1(ds: xr.Dataset) -> xr.DataArray:
    applied_physics_name = "storage_of_internal_energy_path_due_to_applied_physics"
    if applied_physics_name in ds:
        column_q1 = ds[applied_physics_name] + _column_nq1(ds)
    else:
        column_q1 = _column_pq1(ds) + _column_dq1(ds) + _column_nq1(ds)
    column_q1.attrs = {
        "long_name": "<Q1> column integrated heating from physics + ML + nudging",
        "units": "W/m^2",
    }
    return column_q1.rename("column_integrated_Q1")


def _column_q2(ds: xr.Dataset) -> xr.DataArray:
    applied_physics_name = "storage_of_specific_humidity_path_due_to_applied_physics"
    if applied_physics_name in ds:
        column_q2 = SECONDS_PER_DAY * ds[applied_physics_name] + _column_nq2(ds)
    else:
        column_q2 = _column_pq2(ds) + _column_dq2(ds) + _column_nq2(ds)
    column_q2.attrs = {
        "long_name": "<Q2> column integrated moistening from physics + ML + nudging",
        "units": "mm/day",
    }
    return column_q2.rename("column_integrated_Q2")


def _column_nq1(ds: xr.Dataset) -> xr.DataArray:
    if "column_heating_nudge" in ds:
        # name for column integrated temperature nudging in nudge-to-obs
        column_nq1 = ds.column_heating_nudge
    elif "int_t_dt_nudge" in ds:
        # name for column-integrated temperature nudging in X-SHiELD runs
        column_nq1 = ds.int_t_dt_nudge
    elif "net_heating_due_to_nudging" in ds:
        # old name for column integrated temperature nudging in nudge-to-fine
        warnings.warn(
            "'net_heating_due_to_nudging' is a deprecated variable name. "
            "It will not be supported in future versions of fv3net. Use "
            "'column_heating_due_to_nudging' instead.",
            DeprecationWarning,
        )
        # fix isochoric vs isobaric transition issue
        column_nq1 = 716.95 / 1004 * ds.net_heating_due_to_nudging
    elif "column_heating_due_to_nudging" in ds:
        column_nq1 = ds.column_heating_due_to_nudging
    else:
        # assume given dataset is for a run without temperature nudging
        column_nq1 = xr.zeros_like(ds.PRATEsfc)
    column_nq1.attrs = {
        "long_name": "<nQ1> column integrated heating from nudging",
        "units": "W/m^2",
    }
    return column_nq1.rename("column_integrated_nQ1")


def _column_nq2(ds: xr.Dataset) -> xr.DataArray:
    if "column_moistening_nudge" in ds:
        # name for column integrated humidity nudging in nudge-to-obs
        column_nq2 = SECONDS_PER_DAY * ds.column_moistening_nudge
    elif "net_moistening_due_to_nudging" in ds:
        # name for column integrated humidity nudging in nudge-to-fine
        column_nq2 = SECONDS_PER_DAY * ds.net_moistening_due_to_nudging
    else:
        # assume given dataset is for a run without humidity nudging
        column_nq2 = xr.zeros_like(ds.PRATEsfc)
    column_nq2.attrs = {
        "long_name": "<nQ2> column integrated moistening from nudging",
        "units": "mm/day",
    }
    return column_nq2.rename("column_integrated_nQ2")


def _total_precip_to_surface(ds: xr.Dataset) -> xr.DataArray:
    if "total_precipitation_rate" in ds:
        # this is calculated in the prognostic and nudge-to-fine runs
        total_precip_to_surface = ds.total_precipitation_rate * SECONDS_PER_DAY
    else:
        # in the baseline case total_precip_to_surface and physics precip are the
        # same because _column_nq2 and _column_dq2 are zero; in the N2O case
        # total_precip_to_surface needs to be computed here, limited to positive
        # values; finally, this is also a backward compatible fix to overwrite
        # timestep-dependent `total_precip` accumulations calculated previously in
        # the prognostic and nudge-to-fine runs
        total_precip_to_surface = (
            ds.PRATEsfc * SECONDS_PER_DAY - _column_dq2(ds) - _column_nq2(ds)
        )
        total_precip_to_surface = total_precip_to_surface.where(
            total_precip_to_surface >= 0, 0
        )
    total_precip_to_surface.attrs = {
        "long_name": "total precip to surface (max(PRATE-<dQ2>-<nQ2>, 0))",
        "units": "mm/day",
    }
    return total_precip_to_surface.rename("total_precip_to_surface")


def _column_dq1_or_nq1(ds: xr.Dataset, tol=1.0e-12) -> xr.DataArray:
    """<dQ1>+<nQ1> with appropriate long name if either is zero. Useful for movies."""
    column_dq1 = _column_dq1(ds)
    column_nq1 = _column_nq1(ds)
    column_dq1_or_nq1 = column_dq1 + column_nq1
    if abs(column_nq1).max() < tol:
        long_name = "<dQ1> column integrated moistening from ML"
    elif abs(column_dq1).max() < tol:
        long_name = "<nQ1> column integrated moistening from nudging"
    else:
        long_name = "<dQ1> + <nQ1> column integrated moistening from ML + nudging"
    column_dq1_or_nq1.attrs = {"long_name": long_name, "units": "W/m^2"}
    return column_dq1_or_nq1.rename("column_integrated_dQ1_or_nQ1")


def _column_dq2_or_nq2(ds: xr.Dataset, tol=1.0e-12) -> xr.DataArray:
    """<dQ2>+<nQ2> with appropriate long name if either is zero. Useful for movies."""
    column_dq2 = _column_dq2(ds)
    column_nq2 = _column_nq2(ds)
    column_dq2_or_nq2 = column_dq2 + column_nq2
    if abs(column_nq2).max() < tol:
        long_name = "<dQ2> column integrated moistening from ML"
    elif abs(column_dq2).max() < tol:
        long_name = "<nQ2> column integrated moistening from nudging"
    else:
        long_name = "<dQ2> + <nQ2> column integrated moistening from ML + nudging"
    column_dq2_or_nq2.attrs = {"long_name": long_name, "units": "mm/day"}
    return column_dq2_or_nq2.rename("column_integrated_dQ2_or_nQ2")


def _water_vapor_path(ds: xr.Dataset) -> xr.DataArray:
    if "water_vapor_path" in ds:
        result = ds.water_vapor_path
    else:
        # if necessary, back out from total water path and condensate paths
        result = ds.PWAT - ds.iw - ds.VIL
    result.attrs = {"long_name": "water vapor path", "units": "mm"}
    return result.rename("water_vapor_path")


def _minus_column_q2(ds: xr.Dataset) -> xr.DataArray:
    result = -_column_q2(ds)
    result.attrs = {"long_name": "-<Q2> column integrated drying", "units": "mm/day"}
    return result.rename("minus_column_integrated_q2")
