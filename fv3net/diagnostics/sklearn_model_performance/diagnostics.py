import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.stats import binned_statistic_2d
import warnings
import xarray as xr
import visualize as viz
import vcm
from vcm.cubedsphere.regridz import regrid_to_common_pressure
from vcm.select import mask_to_surface_type
from fv3net.diagnostics import get_latlon_grid_coords_set, EXAMPLE_CLIMATE_LATLON_COORDS

from .data import integrate_for_Q, lower_tropospheric_stability

kg_m2s_to_mm_day = (1e3 * 86400) / 997.0
SEC_PER_DAY = 86400

SAMPLE_DIM = "sample"
matplotlib.use("Agg")


def plot_diagnostics(ds_pred, ds_fv3, ds_shield, output_dir, dpi_figures, names):
    """ Makes figures for predictions on test data

    Args:
        ds_pred, ds_fv3, ds_shield [xarray dataset]: contains variables from
            ML prediction, FV3 target, and SHiELD.
        slmsk [xarray dataarray]: surface type variable information
        data_vars [List[str]]: data variables to keep in concatenated dataset
        output_dir: location to write figures to
        dpi_figures: dict of dpi for figures

    Returns:
        dict of header keys and image path list values for passing to the html
        report template
    """
    report_sections = {}
    ds = xr.concat([ds_pred, ds_fv3, ds_shield], dim="dataset")

    # for convenience, separate the land/sea data
    figs = _map_plot_dQ_versus_total(
        ds, names["init_time_dim"], names["mappable_var_kwargs"]
    )
    fig_pe_ml, fig_pe_ml_frac, fig_heating_ml, fig_heating_ml_frac = figs
    fig_pe_ml.savefig(os.path.join(output_dir, "dQ2_vertical_integral_map.png"))
    fig_pe_ml_frac.savefig(os.path.join(output_dir, "dQ2_frac_of_PE.png"))
    fig_heating_ml.savefig(os.path.join(output_dir, "dQ1_vertical_integral_map.png"))
    fig_heating_ml_frac.savefig(os.path.join(output_dir, "dQ1_frac_of_heating.png"))

    report_sections["ML model contributions to Q1 and Q2"] = [
        "dQ2_vertical_integral_map.png",
        "dQ2_frac_of_PE.png",
        "dQ1_vertical_integral_map.png",
        "dQ1_frac_of_heating.png",
    ]

    # LTS
    _plot_lower_troposphere_stability(
        ds_pred, ds_fv3, ds_shield, names, lat_max=20
    ).savefig(os.path.join(output_dir, "LTS_vs_Q.png"), dpi=dpi_figures["LTS"])
    report_sections["Lower tropospheric stability vs humidity"] = ["LTS_vs_Q.png"]

    # Vertical dQ2 profiles over land and ocean
    for sfc_type in ["sea", "land"]:
        _make_vertical_profile_plots(
            vcm.mask_to_surface_type(ds_pred, sfc_type)["dQ2"],
            vcm.mask_to_surface_type(ds_fv3, sfc_type)["dQ2"],
            vcm.mask_to_surface_type(ds_shield, sfc_type)["net_precipitation"],
            delp=vcm.mask_to_surface_type(ds_pred, sfc_type)[
                names["var_pressure_thickness"]
            ],
            stack_dims=names["stack_dims"],
            coord_z_center=names["coord_z_center"],
            units="[kg/kg/s]",
            title=f"{sfc_type}: dQ2 vertical profile",
        ).savefig(
            os.path.join(output_dir, f"vertical_profile_dQ2_{sfc_type}.png"),
            dpi=dpi_figures["dQ2_pressure_profiles"],
        )
    report_sections["dQ2 pressure level profiles"] = [
        "vertical_profile_dQ2_land.png",
        "vertical_profile_dQ2_sea.png",
    ]
    plt.close("all")

    # plot P-E across the diurnal cycle
    grid = ds[[names["var_lat"], names["var_lon"]]]
    local_coords = get_latlon_grid_coords_set(
        grid,
        EXAMPLE_CLIMATE_LATLON_COORDS,
        var_lat=names["var_lat"],
        var_lon=names["var_lon"],
        coord_x_center=names["coord_x_center"],
        coord_y_center=names["coord_y_center"],
    )
    ds["local_time"] = vcm.local_time(
        ds, time=names["init_time_dim"], lon_var=names["var_lon"]
    )

    viz.plot_diurnal_cycle(
        mask_to_surface_type(
            ds[["net_precipitation", names["var_land_sea_mask"], "local_time"]], "sea"
        ),
        "net_precipitation",
        stack_dims=names["stack_dims"],
        title="ocean",
    ).savefig(
        os.path.join(output_dir, "diurnal_cycle_P-E_sea.png"),
        dpi=dpi_figures["diurnal_cycle"],
    )
    viz.plot_diurnal_cycle(
        mask_to_surface_type(
            ds[["net_precipitation", names["var_land_sea_mask"], "local_time"]], "land"
        ),
        "net_precipitation",
        stack_dims=names["stack_dims"],
        title="land",
    ).savefig(
        os.path.join(output_dir, "diurnal_cycle_P-E_land.png"),
        dpi=dpi_figures["diurnal_cycle"],
    )
    for location_name, coords in local_coords.items():
        viz.plot_diurnal_cycle(
            ds[["net_precipitation", "local_time"]].sel(coords),
            "net_precipitation",
            stack_dims=names["stack_dims"],
            title=location_name,
            ylabel="P-E [mm/day]",
        ).savefig(
            os.path.join(output_dir, f"diurnal_cycle_P-E_{location_name}.png"),
            dpi=dpi_figures["diurnal_cycle"],
        )
    report_sections["Diurnal cycle, P-E"] = [
        "diurnal_cycle_P-E_sea.png",
        "diurnal_cycle_P-E_land.png",
    ] + [f"diurnal_cycle_P-E_{location_name}.png" for location_name in local_coords]
    plt.close("all")

    # plot column heating across the diurnal cycle
    viz.plot_diurnal_cycle(
        mask_to_surface_type(
            ds[["net_heating", names["var_land_sea_mask"], "local_time"]], "sea"
        ),
        "net_heating",
        stack_dims=names["stack_dims"],
        title="sea",
    ).savefig(
        os.path.join(output_dir, "diurnal_cycle_heating_sea.png"),
        dpi=dpi_figures["diurnal_cycle"],
    )
    viz.plot_diurnal_cycle(
        mask_to_surface_type(
            ds[["net_heating", names["var_land_sea_mask"], "local_time"]], "land"
        ),
        "net_heating",
        stack_dims=names["stack_dims"],
        title="land",
    ).savefig(
        os.path.join(output_dir, "diurnal_cycle_heating_land.png"),
        dpi=dpi_figures["diurnal_cycle"],
    )

    for location_name, coords in local_coords.items():
        viz.plot_diurnal_cycle(
            ds[["net_heating", "local_time"]].sel(coords),
            "net_heating",
            stack_dims=names["stack_dims"],
            title=location_name,
            ylabel="heating [W/m$^2$]",
        ).savefig(
            os.path.join(output_dir, f"diurnal_cycle_heating_{location_name}.png"),
            dpi=dpi_figures["diurnal_cycle"],
        )
    report_sections["Diurnal cycle, heating"] = [
        "diurnal_cycle_heating_sea.png",
        "diurnal_cycle_heating_land.png",
    ] + [f"diurnal_cycle_heating_{location_name}.png" for location_name in local_coords]
    plt.close("all")

    # map plot variables and compare across prediction/ C48 /coarsened high res data
    _plot_comparison_maps(
        ds,
        "net_precipitation",
        init_time_dim=names["init_time_dim"],
        time_index_selection=None,
        map_var_kwargs=names["mappable_var_kwargs"],
        plot_cube_kwargs={"cbar_label": "time avg, P-E [mm/day]"},
    ).savefig(
        os.path.join(output_dir, "P-E_time_avg.png"), dpi=dpi_figures["map_plot_3col"]
    )
    _plot_comparison_maps(
        ds,
        "net_precipitation",
        init_time_dim=names["init_time_dim"],
        time_index_selection=[0, -1],
        map_var_kwargs=names["mappable_var_kwargs"],
        plot_cube_kwargs={"cbar_label": "timestep snapshot, P-E [mm/day]"},
    ).savefig(
        os.path.join(output_dir, "P-E_time_snapshots.png"),
        dpi=dpi_figures["map_plot_3col"],
    )
    report_sections["P-E"] = ["P-E_time_avg.png", "P-E_time_snapshots.png"]

    _plot_comparison_maps(
        ds,
        "net_heating",
        init_time_dim=names["init_time_dim"],
        time_index_selection=None,
        map_var_kwargs=names["mappable_var_kwargs"],
        plot_cube_kwargs={"cbar_label": "time avg, column heating [W/m$^2$]"},
    ).savefig(
        os.path.join(output_dir, "column_heating_time_avg.png"),
        dpi=dpi_figures["map_plot_3col"],
    )
    _plot_comparison_maps(
        ds,
        "net_heating",
        init_time_dim=names["init_time_dim"],
        time_index_selection=[0, -1],
        map_var_kwargs=names["mappable_var_kwargs"],
        plot_cube_kwargs={"cbar_label": "timestep snapshot, column heating [W/m$^2$]"},
    ).savefig(
        os.path.join(output_dir, "column_heating_snapshots.png"),
        dpi=dpi_figures["map_plot_3col"],
    )
    report_sections["Column heating"] = [
        "column_heating_time_avg.png",
        "column_heating_snapshots.png",
    ]
    plt.close("all")
    return report_sections


# Below are plotting functions specific to this diagnostic workflow


def _plot_comparison_maps(
    ds,
    var,
    init_time_dim="initial_time",
    time_index_selection=None,
    map_var_kwargs=None,
    plot_cube_kwargs=None,
):
    # map plot a variable and compare across prediction/ C48 /coarsened high res data
    matplotlib.rcParams["figure.dpi"] = 200
    plt.clf()
    plot_cube_kwargs = plot_cube_kwargs or {}

    if not time_index_selection:
        map_var = viz.mappable_var(ds.mean(init_time_dim), var, **map_var_kwargs)
    else:
        map_var = viz.mappable_var(
            ds.isel({init_time_dim: time_index_selection}), var, **map_var_kwargs
        )
        plot_cube_kwargs["row"] = init_time_dim
    fig = viz.plot_cube(map_var, col="dataset", **plot_cube_kwargs)[0]
    if isinstance(time_index_selection, int):
        time_label = (
            ds[init_time_dim]
            .values[time_index_selection]
            .strftime("%Y-%m-%d, %H:%M:%S")
        )
        plt.suptitle(time_label)
    return fig


def _make_vertical_profile_plots(
    da_pred,
    da_fv3,
    da_high_res_split_var,
    delp,
    units,
    stack_dims,
    coord_z_center,
    title=None,
):
    """Creates vertical profile plots of dQ2 for drying/moistening columns

    Args:
        da_pred (xr data array): data array for ML prediction of 3d variable
        da_fv3 (xr data array): data array for FV3 target of 3d variable
        da_high_res_split_var (xr data array): data array for coarsened high res,
            e.g. net precip
        variable to divide pos/neg columns
        units (str): [description]
        output_dir (str): [description]
        plot_filename (str, optional): [description].
             Defaults to f"vertical_profile.png".
        title (str, optional): [description]. Defaults to None.
    """

    plt.clf()
    fig = plt.figure()
    pos_mask, neg_mask = (da_high_res_split_var > 0, da_high_res_split_var < 0)
    da_pred = regrid_to_common_pressure(da_pred, delp, coord_z_center)
    da_fv3 = regrid_to_common_pressure(da_fv3, delp, coord_z_center)

    da_pred_pos_PE = da_pred.where(pos_mask)
    da_pred_neg_PE = da_pred.where(neg_mask)
    da_target_pos_PE = da_fv3.where(pos_mask)
    da_target_neg_PE = da_fv3.where(neg_mask)

    pressure = da_pred.pressure.values / 100.0
    profiles_kwargs = zip(
        [da_pred_pos_PE, da_target_pos_PE, da_pred_neg_PE, da_target_neg_PE],
        [
            {"label": "P-E > 0, prediction", "color": "blue", "linestyle": "-"},
            {"label": "P-E > 0, target", "color": "blue", "linestyle": "--"},
            {"label": "P-E < 0, prediction", "color": "orange", "linestyle": "-"},
            {"label": "P-E < 0, target", "color": "orange", "linestyle": "--"},
        ],
    )

    for data, kwargs in profiles_kwargs:
        stack_dims = [dim for dim in stack_dims if dim in data.dims]
        data_mean = np.mean(np.nan_to_num(data.stack(sample=stack_dims).values), axis=1)
        plt.plot(pressure, data_mean, **kwargs)

    plt.xlabel("Pressure [hPa]")
    plt.ylabel(units)
    if title:
        plt.title(title)
    plt.legend()
    return fig


def _plot_lower_troposphere_stability(ds_pred, ds_test, ds_hires, names, lat_max=20):
    warnings.filterwarnings("ignore", message="invalid value encountered in less")
    lat_mask = abs(ds_test[names["var_lat"]]) < lat_max

    ds_test["net_precip_pred"] = ds_pred["net_precipitation"]
    ds_test["net_precip_hires"] = ds_hires["net_precipitation"]
    ds_test = (
        vcm.mask_to_surface_type(ds_test, "sea")
        .where(lat_mask)
        .stack(sample=names["stack_dims"])
        .dropna("sample")
    )
    ds_test["pressure"] = vcm.pressure_at_midpoint_log(
        ds_test[names["var_pressure_thickness"]], dim=names["coord_z_center"]
    )

    Q = [
        integrate_for_Q(p, qt)
        for p, qt in zip(
            ds_test["pressure"].values.T, ds_test[names["var_sphum"]].values.T
        )
    ]
    LTS = lower_tropospheric_stability(
        ds_test[names["var_temp"]],
        ds_test[names["var_pressure_thickness"]],
        ds_test[names["var_sfc_temp"]],
        names["coord_z_center"],
    )
    fig = plt.figure(figsize=(16, 4))

    ax1 = fig.add_subplot(131)
    hist = ax1.hist2d(LTS.values, Q, bins=20)
    cbar1 = fig.colorbar(hist[3], ax=ax1)
    cbar1.set_label("count")
    ax1.set_xlabel("LTS [K]")
    ax1.set_ylabel("Q [mm]")

    ax2 = fig.add_subplot(132)
    bin_values_pred, x_edge, y_edge, _ = binned_statistic_2d(
        LTS.values, Q, ds_test["net_precip_pred"].values, statistic="mean", bins=20
    )
    X, Y = np.meshgrid(x_edge, y_edge)
    PE = ax2.pcolormesh(X, Y, bin_values_pred.T, vmin=-10, vmax=100)
    cbar2 = fig.colorbar(PE, ax=ax2)
    cbar2.set_label("P-E [mm/d]")
    ax2.set_xlabel("LTS [K]")
    ax2.set_ylabel("Q [mm]")
    ax2.set_title("Avg predicted P-E")

    ax3 = fig.add_subplot(133)
    bin_values_hires, x_edge, y_edge, _ = binned_statistic_2d(
        LTS.values, Q, ds_test["net_precip_hires"].values, statistic="mean", bins=20
    )
    bin_error = bin_values_pred - bin_values_hires
    PE_err = ax3.pcolormesh(X, Y, bin_error.T)
    cbar3 = fig.colorbar(PE_err, ax=ax3)
    cbar3.set_label("P-E [mm/d]")
    ax3.set_xlabel("LTS [K]")
    ax3.set_ylabel("Q [mm]")
    ax3.set_title("Avg P-E error (predicted - high res)")
    return fig


def _map_plot_dQ_versus_total(ds, init_time_dim, map_var_kwargs):
    """ Produces plots of the residual components dQ of column heating
    and moistening for comparison to total quantities

    Args:
        ds (xarray dataset): dataset with "dataset" dimension denoting whether
        the dQ quantity was from the high-low res tendency or the ML model prediction

    Returns:
        Figure objects for plots of ML predictions of {column heating, P-E}
        for both the absolute ML prediction value as well as the ML
        prediction as a fraction of the total quantity (ML + physics)
    """
    ds = ds.assign(
        {
            "net_precipitation_ml_frac_of_total": ds["net_precipitation_ml"]
            / ds["net_precipitation"],
            "net_heating_ml_frac_of_total": ds["net_heating_ml"] / ds["net_heating"],
        }
    )
    fig_pe_ml = viz.plot_cube(
        viz.mappable_var(ds, "net_precipitation_ml", **map_var_kwargs).mean(
            init_time_dim
        ),
        col="dataset",
    )[0]
    fig_pe_ml.suptitle("P-E [mm/d]: residual dQ2")
    fig_pe_ml_frac = viz.plot_cube(
        viz.mappable_var(
            ds, "net_precipitation_ml_frac_of_total", **map_var_kwargs
        ).mean(init_time_dim),
        col="dataset",
        vmin=-1,
        vmax=1,
    )[0]
    fig_pe_ml_frac.suptitle("P-E: dQ residual as fraction of total")

    fig_heating_ml = viz.plot_cube(
        viz.mappable_var(ds, "net_heating_ml", **map_var_kwargs).mean(init_time_dim),
        col="dataset",
    )[0]
    fig_heating_ml.suptitle("heating [W/m$^2$], ML contribution")
    fig_heating_ml_frac = viz.plot_cube(
        viz.mappable_var(ds, "net_heating_ml_frac_of_total", **map_var_kwargs).mean(
            init_time_dim
        ),
        col="dataset",
        vmin=-1,
        vmax=1,
    )[0]
    fig_heating_ml_frac.suptitle("heating: ML prediction as fraction of total")

    return fig_pe_ml, fig_pe_ml_frac, fig_heating_ml, fig_heating_ml_frac
