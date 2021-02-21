from ._constants import (
    COORD_X_CENTER,
    COORD_Y_CENTER,
    COORD_X_OUTER,
    COORD_Y_OUTER,
    VAR_LON_CENTER,
    VAR_LAT_CENTER,
    VAR_LON_OUTER,
    VAR_LAT_OUTER,
)
from ._plot_helpers import (
    _infer_color_limits,
    _get_var_label,
    _remove_redundant_dims,
    _min_max_from_percentiles,
)
from ._masking import _mask_antimeridian_quads
import xarray as xr
import numpy as np
from matplotlib import pyplot as plt
import warnings
from functools import partial

try:
    from cartopy import crs as ccrs
except ImportError:
    pass

# global

_COORD_VARS = {
    VAR_LON_OUTER: [COORD_Y_OUTER, COORD_X_OUTER, "tile"],
    VAR_LAT_OUTER: [COORD_Y_OUTER, COORD_X_OUTER, "tile"],
    VAR_LON_CENTER: [COORD_Y_CENTER, COORD_X_CENTER, "tile"],
    VAR_LAT_CENTER: [COORD_Y_CENTER, COORD_X_CENTER, "tile"],
}


def plot_cube(
    plottable_variable: xr.Dataset,
    plotting_function: str = "pcolormesh",
    ax: plt.axes = None,
    row: str = None,
    col: str = None,
    col_wrap: int = None,
    projection: "ccrs.Projection" = None,
    colorbar: bool = True,
    cmap_percentiles_lim: bool = True,
    cbar_label: str = None,
    coastlines: bool = True,
    coastlines_kwargs: dict = None,
    **kwargs,
):
    """ Plots tiled cubed sphere grids onto a global map projection

    Args:
        plottable_variable (xr.Dataset):
            Dataset containing variable to plotted via pcolormesh, along with
            coordinate variables (lat, latb, lon, lonb). This dataset object
            can be created from the helper function :py:func:`mappable_var`, which takes
            in an fv3gfs restart or diagnostic dataset along with the name of
            the variable to be plotted.
        plotting_function (str, optional):
            Name of matplotlib 2-d plotting function. Available options are
            "pcolormesh", "contour", and "contourf". Defaults to "pcolormesh".
        ax (plt.axes, optional):
            Axes onto which the map should be plotted; must be created with
            a cartopy projection argument. If not supplied, axes are generated
            with a projection. If ax is suppled, faceting is disabled.
        row (str, optional):
            Name of diemnsion to be faceted along subplot rows. Must not be a
            tile, lat, or lon dimension.  Defaults to no row facets.
        col (str, optional):
            Name of diemnsion to be faceted along subplot columns. Must not be
            a tile, lat, or lon dimension. Defaults to no column facets.
        col_wrap (int, optional):
            If only one of `col`, `row` is specified, number of columns to plot
            before wrapping onto next row. Defaults to None, i.e. no limit.
        projection (ccrs.Projection, optional):
            Cartopy projection object to be used in creating axes. Ignored if
            cartopy geo-axes are supplied.  Defaults to Robinson projection.
        colorbar (bool, optional):
            Flag for whether to plot a colorbar. Defaults to True.
        cmap_percentiles_lim(bool, optional):
            If False, use the absolute min/max to set color limits. If True, use 2/98
            percentile values.
        cbar_label (str, optional):
            If provided, use this as the color bar label.
        coastlines (bool, optinal):
            Whether to plot coastlines on map. Default True.
        coastlines_kwargs (dict, optional):
            Dict of arguments to be passed to cartopy axes's `coastline`
            function if `coastlines` flag is set to True.
        **kwargs:
            Additional keyword arguments to be passed to the plotting function.

    Returns:
        figure (plt.Figure):
            matplotlib figure object onto which axes grid is created
        axes (np.ndarray):
            Array of `plt.axes` objects assocated with map subplots if faceting;
            otherwise array containing single axes object.
        handles (list):
            List or nested list of matplotlib object handles associated with
            map subplots if faceting; otherwise list of single object handle.
        cbar (plt.colorbar):
            object handle associated with figure, if `colorbar`
            arg is True, else None.
        facet_grid (xarray.plot.facetgrid):
            xarray plotting facetgrid for multi-axes case. In single-axes case,
            retunrs None.

    Example:
        # plot diag winds at two times
        fig, axes, hs, cbar, facet_grid = plot_cube(
            mappable_var(diag_ds, 'VGRD850').isel(time = slice(2, 4)),
            plotting_function = "contourf",
            col = "time",
            coastlines = True,
            colorbar = True,
            vmin = -20,
            vmax = 20
        )
    """
    var_name = list(plottable_variable.data_vars)[0]
    array = plottable_variable[var_name].values
    if cmap_percentiles_lim:
        xmin, xmax = _min_max_from_percentiles(array)
    else:
        xmin, xmax = np.nanmin(array), np.nanmax(array)
    vmin = kwargs["vmin"] if "vmin" in kwargs else None
    vmax = kwargs["vmax"] if "vmax" in kwargs else None
    cmap = kwargs["cmap"] if "cmap" in kwargs else None
    kwargs["vmin"], kwargs["vmax"], kwargs["cmap"] = _infer_color_limits(
        xmin, xmax, vmin, vmax, cmap
    )

    _plot_func_short = partial(
        plot_cube_axes,
        lat=plottable_variable.lat.values,
        lon=plottable_variable.lon.values,
        latb=plottable_variable.latb.values,
        lonb=plottable_variable.lonb.values,
        plotting_function=plotting_function,
        **kwargs,
    )

    projection = ccrs.Robinson() if not projection else projection

    if ax is None and (row or col):
        # facets
        facet_grid = xr.plot.FacetGrid(
            data=plottable_variable,
            row=row,
            col=col,
            col_wrap=col_wrap,
            subplot_kws={"projection": projection},
        )
        facet_grid = facet_grid.map(_plot_func_short, var_name)
        fig = facet_grid.fig
        axes = facet_grid.axes
        handles = facet_grid._mappables
    else:
        # single axes
        if ax is None:
            fig, ax = plt.subplots(1, 1, subplot_kw={"projection": projection})
        else:
            fig = ax.figure
        handle = _plot_func_short(array, ax=ax)
        axes = np.array(ax)
        handles = [handle]
        facet_grid = None

    if coastlines:
        coastlines_kwargs = dict() if not coastlines_kwargs else coastlines_kwargs
        [ax.coastlines(**coastlines_kwargs) for ax in axes.flatten()]

    if colorbar:
        if row or col:
            fig.subplots_adjust(
                bottom=0.1, top=0.9, left=0.1, right=0.8, wspace=0.02, hspace=0.02
            )
            cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
        else:
            fig.subplots_adjust(wspace=0.25)
            cb_ax = ax.inset_axes([1.05, 0, 0.02, 1])
        cbar = plt.colorbar(handles[0], cax=cb_ax, extend="both")
        cbar.set_label(
            _get_var_label(plottable_variable[var_name].attrs, cbar_label or var_name)
        )
    else:
        cbar = None

    return fig, axes, handles, cbar, facet_grid


def mappable_var(
    ds: xr.Dataset,
    var_name: str,
    coord_x_center: str = COORD_X_CENTER,
    coord_y_center: str = COORD_Y_CENTER,
    coord_x_outer: str = COORD_X_OUTER,
    coord_y_outer: str = COORD_Y_OUTER,
    coord_vars: dict = _COORD_VARS,
):
    """ Converts a restart or diagnostic dataset into a format for plotting
    across cubed-sphere tiles
    
    Note that the default coordinate names and grid variable coordinates are for FV3
    restart and diagnostic file formats. If plotting prognostic-run python diagnostic
    zarrs, use the following kwargs:
    .. ::
    
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
        
    while if plotting prognostic run report diagnostics variables use the following:
    .. ::
    
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

    Args:
        ds (xr.Dataset):
            Dataset containing the variable to be plotted, along with grid variables.
        var_name (str):
            Name of variable to be plotted.
        coord_x_center (str):
            name of the x-coordinate describing cell centers
        coord_y_center (str):
            name of the y-coordinate describing cell centers
        coord_x_outer (str):
            name of the x-coordinate describing cell interfaces
        coord_y_outer (str):
            name of the y-coordinate describing cell interfaces
        coord_vars (Mapping[str, Sequence[str]]):
            mapping of names of grid variables, which must include latitudes and
            longitudes of both cell centers and bounds, to their sequence of
            coordinate names

    Returns:
        ds (xr.Dataset):
            Dataset containing variable to be plotted as well as grid
            coordinates variables, which are renamed and ordered for
            plotting. Intended as first argument to :py:func:`plot_cube`.

    Example:
    .. ::
        
        # plot diag winds at two times
        axes, hs, cbar = plot_cube(
            mappable_var(diag_ds, 'VGRD850').isel(time = slice(2, 4)),
            plotting_function = "contourf",
            col = "time",
            coastlines = True,
            colorbar = True,
            vmin = -20,
            vmax = 20

        )
    """
    for var, dims in coord_vars.items():
        ds[var] = _remove_redundant_dims(ds[var], required_dims=dims)
        ds[var] = ds[var].transpose(*dims)

    first_dims = [coord_y_center, coord_x_center, "tile"]
    rest = [dim for dim in ds[[var_name]].dims if dim not in first_dims]
    xpose_dims = first_dims + rest
    new_ds = ds[[var_name]].copy().transpose(*xpose_dims)

    for grid_var in coord_vars:
        new_ds = new_ds.assign_coords(coords={grid_var: ds[grid_var]})

    for coord in [coord_y_center, coord_x_center, coord_y_outer, coord_x_outer]:
        if coord in new_ds.coords:
            new_ds = new_ds.drop(coord)

    return new_ds


def pcolormesh_cube(lat, lon, array, ax=None, **kwargs):
    """Plots tiled cubed sphere.

    This function applies nan to gridcells which cross the antimeridian, and then
    iteratively plots rectangles of array which avoid nan gridcells. This is done to
    avoid artifacts when plotting gridlines with the `edgecolor` argument.

    Args:
        lat (np.ndarray):
            Array of latitudes with dimensions (tile, ny + 1, nx + 1).
            Should be given at cell corners.
        lon (np.ndarray):
            Array of longitudes with dimensions (tile, ny + 1, nx + 1).
            Should be given at cell corners.
        array (np.ndarray):
            Array of variables values at cell centers, of dimensions (tile, ny, nx)
        ax (plt.axes, optional)
            Matplotlib geoaxes object onto which plotting function will be
            called. Default None uses current axes.
        **kwargs:
            Keyword arguments to be passed to plotting function.

    Returns:
        p_handle (obj):
            matplotlib object handle associated with a segment of the map subplot
    """
    if lat.shape != lon.shape:
        raise ValueError("lat and lon should have the same shape")
    if ax is None:
        ax = plt.gca()
    central_longitude = ax.projection.proj4_params["lon_0"]
    array = np.where(
        _mask_antimeridian_quads(lon.T, central_longitude), array.T, np.nan
    ).T
    # oddly a PlateCarree transform seems to be needed here even for non-PlateCarree
    # projections?? very puzzling, but it seems to be the case.
    kwargs["transform"] = kwargs.get("transform", ccrs.PlateCarree())
    kwargs["vmin"] = kwargs.get("vmin", np.nanmin(array))
    kwargs["vmax"] = kwargs.get("vmax", np.nanmax(array))

    for tile in range(array.shape[0]):
        x = center_longitudes(lon[tile, :, :], central_longitude)
        y = lat[tile, :, :]
        for x_plot, y_plot, array_plot in _segment_plot_inputs(x, y, array[tile, :, :]):
            p_handle = ax.pcolormesh(x_plot, y_plot, array_plot, **kwargs)
    return p_handle


def _segment_plot_inputs(x, y, masked_array):
    """Takes in two arrays at corners of grid cells and an array at grid cell centers
    which may contain NaNs. Yields 3-tuples of rectangular segments of
    these arrays which cover all non-nan points without duplicates, and don't contain
    NaNs.
    """
    is_nan = np.isnan(masked_array)
    if np.sum(is_nan) == 0:  # contiguous section, just plot it
        if np.product(masked_array.shape) > 0:
            yield (x, y, masked_array)
    else:
        x_nans = np.sum(is_nan, axis=1) / is_nan.shape[1]
        y_nans = np.sum(is_nan, axis=0) / is_nan.shape[0]
        if x_nans.max() >= y_nans.max():  # most nan-y line is in first dimension
            i_split = x_nans.argmax()
            if x_nans[i_split] == 1.0:  # split cleanly along line
                yield from _segment_plot_inputs(
                    x[: i_split + 1, :], y[: i_split + 1, :], masked_array[:i_split, :],
                )
                yield from _segment_plot_inputs(
                    x[i_split + 1 :, :],
                    y[i_split + 1 :, :],
                    masked_array[i_split + 1 :, :],
                )
            else:
                # split to create segments of complete nans
                # which subsequent recursive calls will split on and remove
                i_start = 0
                i_end = 1
                while i_end < is_nan.shape[1]:
                    while (
                        i_end < is_nan.shape[1]
                        and is_nan[i_split, i_start] == is_nan[i_split, i_end]
                    ):
                        i_end += 1
                    # we have a largest-possible contiguous segment of nans/not nans
                    yield from _segment_plot_inputs(
                        x[:, i_start : i_end + 1],
                        y[:, i_start : i_end + 1],
                        masked_array[:, i_start:i_end],
                    )
                    i_start = i_end  # start the next segment
        else:
            # put most nan-y line in first dimension
            # so the first part of this if block catches it
            yield from _segment_plot_inputs(
                x.T, y.T, masked_array.T,
            )


def center_longitudes(lon_array, central_longitude):
    return np.where(
        lon_array < (central_longitude + 180.0) % 360.0, lon_array, lon_array - 360.0,
    )


def _validate_cube_shape(lat_shape, lon_shape, latb_shape, lonb_shape, array_shape):
    if (lon_shape[-1] != 6) or (lat_shape[-1] != 6) or (array_shape[-1] != 6):
        raise ValueError(
            """Last axis of each array must have six elements for
            cubed-sphere tiles."""
        )

    if (
        (lon_shape[0] != lat_shape[0])
        or (lat_shape[0] != array_shape[0])
        or (lon_shape[1] != lat_shape[1])
        or (lat_shape[1] != array_shape[1])
    ):
        raise ValueError(
            """Horizontal axis lengths of lat and lon must be equal to
            those of array."""
        )

    if (len(lonb_shape) != 3) or (len(latb_shape) != 3) or (len(array_shape) != 3):
        raise ValueError("Lonb, latb, and data_var each must be 3-dimensional.")

    if (lonb_shape[-1] != 6) or (latb_shape[-1] != 6) or (array_shape[-1] != 6):
        raise ValueError(
            "Tile axis of each array must have six elements for cubed-sphere tiles."
        )

    if (
        (lonb_shape[0] != latb_shape[0])
        or (latb_shape[0] != (array_shape[0] + 1))
        or (lonb_shape[1] != latb_shape[1])
        or (latb_shape[1] != (array_shape[1] + 1))
    ):
        raise ValueError(
            """Horizontal axis lengths of latb and lonb
            must be one greater than those of array."""
        )

    if (len(lon_shape) != 3) or (len(lat_shape) != 3) or (len(array_shape) != 3):
        raise ValueError("Lon, lat, and data_var each must be 3-dimensional.")


def plot_cube_axes(
    array: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    latb: np.ndarray,
    lonb: np.ndarray,
    plotting_function: str,
    ax: plt.axes = None,
    **kwargs,
):
    """ Plots tiled cubed sphere for a given subplot axis,
        using np.ndarrays for all data

    The `edgecolor` argument produces artifacts with this approach for pcolormesh.
    To plot pcolormesh, you should consider using :py:func:`pcolormesh_cube`.

    Args:
        array (np.ndarray):
            Array of variables values at cell centers, of dimensions (npy, npx,
            tile)
        lat (np.ndarray):
            Array of latitudes of cell centers, of dimensions (npy, npx, tile)
        lon (np.ndarray):
            Array of longitudes of cell centers, of dimensions (npy, npx, tile)
        latb (np.ndarray):
            Array of latitudes of cell edges, of dimensions (npy + 1, npx + 1,
            tile)
        lonb (np.ndarray):
            Array of longitudes of cell edges, of dimensions (npy + 1, npx + 1,
            tile)
        plotting_function (str):
            Name of matplotlib 2-d plotting function. Available options are
            "pcolormesh", "contour", and "contourf".
        ax (plt.axes, optional)
            Matplotlib geoaxes object onto which plotting function will be
            called. Default None uses current axes.
        **kwargs:
            Keyword arguments to be passed to plotting function.

    Returns:
        p_handle (obj):
            matplotlib object handle associated with map subplot

    Example:
        _, ax = plt.subplots(1, 1, subplot_kw = {'projection': ccrs.Robinson()})
        h = plot_cube_axes(
            ds['T'].isel(time = 0, pfull = 40).values.transpose([1, 2, 0]),
            ds['lat'].values,
            ds['lon'].values,
            ds['latb'].values,
            ds['lonb'].values,
            "contour",
            ax
        )
    """
    _validate_cube_shape(lon.shape, lat.shape, lonb.shape, latb.shape, array.shape)

    if ax is None:
        ax = plt.gca()

    if plotting_function in ["pcolormesh", "contour", "contourf"]:
        _plotting_function = getattr(ax, plotting_function)
    else:
        raise ValueError(
            """Plotting functions only include pcolormesh, contour,
            and contourf."""
        )

    if "vmin" not in kwargs:
        kwargs["vmin"] = np.nanmin(array)

    if "vmax" not in kwargs:
        kwargs["vmax"] = np.nanmax(array)

    if plotting_function != "pcolormesh":
        if "levels" not in kwargs:
            kwargs["n_levels"] = 11 if "n_levels" not in kwargs else kwargs["n_levels"]
            kwargs["levels"] = np.linspace(
                kwargs["vmin"], kwargs["vmax"], kwargs["n_levels"]
            )

    central_longitude = ax.projection.proj4_params["lon_0"]

    masked_array = np.where(
        _mask_antimeridian_quads(lonb, central_longitude), array, np.nan
    )

    for tile in range(6):
        if plotting_function == "pcolormesh":
            x = lonb[:, :, tile]
            y = latb[:, :, tile]
        else:
            # contouring
            x = center_longitudes(lon[:, :, tile], central_longitude)
            y = lat[:, :, tile]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p_handle = _plotting_function(
                x, y, masked_array[:, :, tile], transform=ccrs.PlateCarree(), **kwargs
            )

    ax.set_global()

    return p_handle
