import matplotlib.pyplot as plt
import numpy as np
from typing import Sequence, Union, Mapping
import xarray as xr

import fv3viz as visualize
from .utils import units_from_name

# grid info for the plot_cube function
MAPPABLE_VAR_KWARGS = {
    "coord_x_center": "x",
    "coord_y_center": "y",
    "coord_x_outer": "x_interface",
    "coord_y_outer": "y_interface",
    "coord_vars": {
        "lonb": ["y_interface", "x_interface", "tile"],
        "latb": ["y_interface", "x_interface", "tile"],
        "lon": ["y", "x", "tile"],
        "lat": ["y", "x", "tile"],
    },
}


def plot_profile_var(
    ds: xr.Dataset,
    var: str,
    dpi: int = 100,
    derivation_dim: str = "derivation",
    domain_dim: str = "domain",
    dataset_dim: str = "dataset",
    derivation_plot_coords: Sequence[str] = ("target", "predict"),
    xlim: Sequence[float] = None,
    xticks: Union[Sequence[float], np.ndarray] = None,
):
    if dataset_dim in ds[var].dims:
        ds[var] = ds[var].mean("dataset")
    if derivation_dim in ds[var].dims:
        facet_grid = (
            ds[var]
            .sel({derivation_dim: list(derivation_plot_coords)})
            .plot(y="z", hue=derivation_dim, col=domain_dim)
        )
    facet_grid.set_titles(template="{value}", maxchar=40)
    f = facet_grid.fig
    for ax in facet_grid.axes.flatten():
        ax.invert_yaxis()
        ax.plot([0, 0], [1, 79], "k-")
        ax.set_xlabel(f"{var} {units_from_name(var)}")
        if xlim:
            ax.set_xlim(xlim)
    f.set_size_inches([17, 3.5])
    f.set_dpi(dpi)
    f.suptitle(var.replace("_", " "))
    return f


def plot_column_integrated_var(
    ds: xr.Dataset,
    var: str,
    derivation_plot_coords: Sequence[str],
    derivation_dim: str = "derivation",
    dataset_dim: str = "dataset",
    dpi: int = 100,
    vmax: Union[int, float] = None,
):
    f, _, _, _, facet_grid = visualize.plot_cube(
        visualize.mappable_var(
            ds.sel(derivation=derivation_plot_coords), var, **MAPPABLE_VAR_KWARGS
        ),
        col=derivation_dim,
        row=dataset_dim if dataset_dim in ds.dims else None,
        vmax=vmax,
    )
    facet_grid.set_titles(template="{value} ", maxchar=40)
    num_datasets = len(ds[dataset_dim]) if dataset_dim in ds.dims else 1
    f.set_size_inches([14, 3.5 * num_datasets])
    f.set_dpi(dpi)
    f.suptitle(f'{var.replace("_", " ")} {units_from_name(var)}')
    return f


def plot_diurnal_cycles(
    ds_diurnal: xr.Dataset,
    var: str,
    derivation_plot_coords: Sequence[str],
    dpi: int = 100,
):
    ds_diurnal = ds_diurnal.sel(derivation=derivation_plot_coords)

    facetgrid = ds_diurnal[var].squeeze().plot(hue="derivation", col="surface_type")

    facetgrid.set_titles(template="{value}", maxchar=40)
    f = facetgrid.fig
    axes = facetgrid.axes
    for ax in axes.flatten():
        ax.grid(axis="y")
        ax.set_xlabel("local_time [hrs]")
        ax.set_ylabel(units_from_name(var))
        ax.set_xlim([0, 23])
        ax.set_xticks(np.linspace(0, 24, 13))
    f.set_size_inches([12, 4])
    f.set_dpi(dpi)
    f.suptitle(var)
    f.tight_layout()
    return f


def _plot_generic_data_array(
    da: xr.DataArray,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    xlim: Sequence[float] = None,
    ylim: Sequence[float] = None,
):
    fig = plt.figure()
    da.plot()
    if xlabel:
        plt.xlabel(xlabel)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    units = units_from_name(da.name) or ""
    ylabel = ylabel or units
    title = title or " ".join([da.name.replace("_", " ").replace("-", ",")])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    return fig


def plot_zonal_average(
    data: xr.DataArray,
    rename_axes: Mapping = None,
    title: str = None,
    plot_kwargs: Mapping = None,
):
    fig = plt.figure()
    units = units_from_name(data.name) or ""
    title = f"{title or data.name} {units}"
    plot_kwargs = plot_kwargs or {}
    rename_axes = rename_axes or {
        "lat_interp": "Latitude [deg]",
        "pressure": "Pressure [Pa]",
    }
    data = data.rename(rename_axes).rename(title)
    data.plot(yincrease=False, x="Latitude [deg]", robust=True, **plot_kwargs)
    return fig
