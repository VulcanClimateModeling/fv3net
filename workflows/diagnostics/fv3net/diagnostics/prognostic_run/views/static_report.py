#!/usr/bin/env python

from typing import Dict, Iterable, List, Mapping, Sequence, Tuple
import os
import xarray as xr
import fsspec
import numpy as np
import pandas as pd
import holoviews as hv

from fv3net.diagnostics.prognostic_run.computed_diagnostics import (
    ComputedDiagnosticsList,
    RunDiagnostics,
    RunMetrics,
)

import fv3viz
import vcm
from report import create_html, Link, OrderedList, RawHTML
from report.holoviews import HVPlot, get_html_header
from .matplotlib import (
    plot_2d_matplotlib,
    plot_cubed_sphere_map,
    plot_histogram,
    plot_histogram2d,
)
from fv3net.diagnostics.prognostic_run.constants import (
    PERCENTILES,
    PRECIP_RATE,
    TOP_LEVEL_METRICS,
    DEFAULT_FIGURE_WIDTH,
    DEFAULT_FIGURE_HEIGHT,
    REFERENCE_HOVMOLLER_DURATION_SECONDS,
    MovieUrls,
)
from fv3net.diagnostics._shared.constants import WVP, COL_DRYING

import logging

import warnings

warnings.filterwarnings(
    "ignore", message="Creating an ndarray from ragged nested sequences"
)
warnings.filterwarnings("ignore", message="All-NaN slice encountered")

logging.basicConfig(level=logging.INFO)

hv.extension("bokeh")
COLOR_CYCLE = hv.Cycle(fv3viz.wong_palette)
fv3viz.use_colorblind_friendly_style()
PUBLIC_GCS_DOMAIN = "https://storage.googleapis.com"
JQUERY_CDN = "https://ajax.googleapis.com/ajax/libs/jquery/3.6.0/jquery.min.js"
DATATABLES_CSS_CDN = "https://cdn.datatables.net/1.11.5/css/jquery.dataTables.css"
DATATABLES_JS_CDN = "https://cdn.datatables.net/1.11.5/js/jquery.dataTables.js"
MovieManifest = Sequence[Tuple[str, str]]
PublicLinks = Dict[str, List[Tuple[str, str]]]


def get_datatables_header() -> str:
    header = "<style>th{font-size: 1.15em}\n.firstcolumn{font-size: 1.15em}</style>"
    header += f'\n<script src="{JQUERY_CDN}"></script>'
    header += f'\n<link rel="stylesheet" type="text/css" href="{DATATABLES_CSS_CDN}">'
    header += f'\n<script src="{DATATABLES_JS_CDN}"></script>'
    header += """
        <script>
            $(document).ready(function() {
                $('table.display').DataTable( {
                    "scrollX":        "100%",
                    "scrollY":        "700px",
                    "scrollCollapse": true,
                    "paging":         false,
                    "columnDefs": [{"targets": [0], "className": "firstcolumn"}]
                } );
                $('table.dataframe').DataTable();
            } );
        </script>"""
    return header


def get_header() -> str:
    return get_html_header() + "\n" + get_datatables_header()


def upload(html: str, url: str, content_type: str = "text/html"):
    """Upload to a local or remote path, setting the content type if remote

    Setting the content type is necessary for viewing the uploaded object in a
    the web browser (e.g. it is a webpage or image).

    """
    with fsspec.open(url, "w") as f:
        f.write(html)

    if url.startswith("gs"):
        fs = fsspec.filesystem("gs")
        fs.setxattrs(url, content_type=content_type)


class PlotManager:
    """An object for managing lists of plots in an extensible way

    New plotting functions can be registered using the ``register`` method.

    All plotting functions registered by the object will be called in sequence on
    the data passed to `make_plots``.

    We could extend this class in the future to have even more features
    (e.g. parallel plot generation, exception handling, etc)

    """

    def __init__(self):
        self._diags = []

    def register(self, func):
        """Register a given function as a diagnostic

        This can be used to generate a new set of plots to appear the html reports
        """
        self._diags.append(func)
        return func

    def make_plots(self, data) -> Iterable:
        for func in self._diags:
            yield func(data)


def plot_1d(run_diags: RunDiagnostics, varfilter: str) -> HVPlot:
    """Plot all diagnostics whose name includes varfilter. Plot is overlaid across runs.
    All matching diagnostics must be 1D."""

    hmap = hv.HoloMap(kdims=["variable", "run"])
    vars_to_plot = run_diags.matching_variables(varfilter)
    for run in run_diags.runs:
        for varname in vars_to_plot:
            v = run_diags.get_variable(run, varname).rename("value")
            style = "solid" if run_diags.is_baseline(run) else "dashed"
            color = "black" if run_diags.is_verification(run) else COLOR_CYCLE
            long_name = v.long_name
            hmap[(long_name, run)] = hv.Curve(v, label=varfilter).options(
                line_dash=style, color=color
            )
    return HVPlot(_set_opts_and_overlay(hmap))


def plot_1d_min_max_with_region_bar(
    run_diags: RunDiagnostics, varfilter_min: str, varfilter_max: str,
) -> HVPlot:
    """Plot all diagnostics whose name includes varfilter. Plot is overlaid across runs.
    All matching diagnostics must be 1D."""
    hmap = hv.HoloMap(kdims=["variable", "region", "run"])

    variables_to_plot = run_diags.matching_variables(varfilter_min)

    for run in run_diags.runs:
        for min_var in variables_to_plot:
            max_var = min_var.replace(varfilter_min, varfilter_max)
            vmin = run_diags.get_variable(run, min_var).rename("min")
            vmax = run_diags.get_variable(run, max_var).rename("max")
            style = "solid" if run_diags.is_baseline(run) else "dashed"
            color = "black" if run_diags.is_verification(run) else COLOR_CYCLE
            long_name = vmin.long_name
            region = min_var.split("_")[-1]
            # Area plot doesn't automatically add correct y label
            ylabel = f'{vmin.attrs["long_name"]} {vmin.attrs["units"]}'
            hmap[(long_name, region, run)] = hv.Area(
                (vmin.time, vmin, vmax), label="Min/max", vdims=["y", "y2"]
            ).options(line_dash=style, color=color, alpha=0.6, ylabel=ylabel)
    return HVPlot(_set_opts_and_overlay(hmap))


def plot_1d_with_region_bar(run_diags: RunDiagnostics, varfilter: str) -> HVPlot:
    """Plot all diagnostics whose name includes varfilter. Plot is overlaid across runs.
    Region will be selectable through a drop-down bar. Region is assumed to be part of
    variable name after last underscore. All matching diagnostics must be 1D."""
    hmap = hv.HoloMap(kdims=["variable", "region", "run"])
    vars_to_plot = run_diags.matching_variables(varfilter)
    for run in run_diags.runs:
        for varname in vars_to_plot:
            v = run_diags.get_variable(run, varname).rename("value")
            style = "solid" if run_diags.is_baseline(run) else "dashed"
            color = "black" if run_diags.is_verification(run) else COLOR_CYCLE
            long_name = v.long_name
            region = varname.split("_")[-1]
            hmap[(long_name, region, run)] = hv.Curve(v, label=varfilter,).options(
                line_dash=style, color=color
            )
    return HVPlot(_set_opts_and_overlay(hmap))


def _set_opts_and_overlay(hmap, overlay="run"):
    return (
        hmap.opts(norm={"framewise": True}, plot=dict(width=850, height=500))
        .overlay(overlay)
        .opts(
            legend_position="right",
            bgcolor="gainsboro",
            show_grid=True,
            gridstyle=dict(grid_line_color="white", grid_line_width=0.5),
        )
    )


def _parse_diurnal_component_fields(varname: str):

    # diags key format: diurn_component_<varname>_diurnal_<sfc_type>
    tokens = varname.split("_")
    short_varname = tokens[2]
    surface_type = tokens[-1]

    return short_varname, surface_type


def diurnal_component_plot(
    run_diags: RunDiagnostics,
    run_attr_name="run",
    diurnal_component_name="diurn_component",
) -> HVPlot:

    hmap = hv.HoloMap(kdims=["run", "surface_type", "short_varname"])
    variables_to_plot = run_diags.matching_variables(diurnal_component_name)

    for run in run_diags.runs:
        for varname in variables_to_plot:
            v = run_diags.get_variable(run, varname).rename("value")
            short_vname, surface_type = _parse_diurnal_component_fields(varname)
            hmap[(run, surface_type, short_vname)] = hv.Curve(
                v, label=diurnal_component_name
            ).options(color=COLOR_CYCLE)
    return HVPlot(_set_opts_and_overlay(hmap, overlay="short_varname"))


# Initialize diagnostic managers
# following plot managers will be passed the data from the diags.nc files
timeseries_plot_manager = PlotManager()
zonal_mean_plot_manager = PlotManager()
hovmoller_plot_manager = PlotManager()
zonal_pressure_plot_manager = PlotManager()
diurnal_plot_manager = PlotManager()
histogram_plot_manager = PlotManager()
tropical_hovmoller_plot_manager = PlotManager()

# this will be passed the data from the metrics.json files
metrics_plot_manager = PlotManager()


# Routines for plotting the "diagnostics"
@timeseries_plot_manager.register
def rms_plots(diagnostics: Iterable[xr.Dataset]) -> HVPlot:
    return plot_1d(diagnostics, varfilter="rms_global")


@timeseries_plot_manager.register
def spatial_mean_plots(diagnostics: Iterable[xr.Dataset]) -> HVPlot:
    return plot_1d_with_region_bar(diagnostics, varfilter="spatial_mean")


@timeseries_plot_manager.register
def spatial_minmax_plots(diagnostics: Iterable[xr.Dataset]) -> HVPlot:
    return plot_1d_min_max_with_region_bar(
        diagnostics, varfilter_min="spatial_min", varfilter_max="spatial_max"
    )


@timeseries_plot_manager.register
def custom_timeseries_plots(diagnostics: Iterable[xr.Dataset]) -> HVPlot:
    return plot_1d(diagnostics, varfilter="timeseries")


@zonal_mean_plot_manager.register
def zonal_mean_plots(diagnostics: Iterable[xr.Dataset]) -> HVPlot:
    return plot_1d(diagnostics, varfilter="zonal_and_time_mean")


@hovmoller_plot_manager.register
def zonal_mean_hovmoller_plots(diagnostics: Iterable[xr.Dataset]) -> RawHTML:
    return plot_2d_matplotlib(
        diagnostics, "zonal_mean_value", ["time", "latitude"], n_jobs=-1
    )


@hovmoller_plot_manager.register
def zonal_mean_hovmoller_bias_plots(diagnostics: Iterable[xr.Dataset]) -> RawHTML:
    return plot_2d_matplotlib(
        diagnostics, "zonal_mean_bias", ["time", "latitude"], n_jobs=-1
    )


def infer_duration_seconds(diagnostics: RunDiagnostics) -> float:
    run, *_ = diagnostics.runs
    time = diagnostics.get_variable(run, "high_frequency_time")
    delta = time.isel(high_frequency_time=-1) - time.isel(high_frequency_time=0)
    return (delta / np.timedelta64(1, "s")).item()


def infer_tropical_hovmoller_figsize(diagnostics: RunDiagnostics,) -> Tuple[int, int]:
    """A heuristic to infer a reasonable size and aspect ratio for Hovmoller plots"""
    duration_seconds = infer_duration_seconds(diagnostics)
    height_ratio = duration_seconds / REFERENCE_HOVMOLLER_DURATION_SECONDS
    reference_width = DEFAULT_FIGURE_WIDTH
    reference_height = DEFAULT_FIGURE_HEIGHT
    height = min(reference_height * height_ratio, 10 * reference_height)
    return (reference_width, height)


@tropical_hovmoller_plot_manager.register
def deep_tropical_meridional_mean_hovmoller_plots(
    diagnostics: RunDiagnostics,
) -> RawHTML:
    figsize = infer_tropical_hovmoller_figsize(diagnostics)
    return plot_2d_matplotlib(
        diagnostics,
        "deep_tropical_meridional_mean_value",
        ["longitude", "high_frequency_time"],
        figsize=figsize,
        n_jobs=-1,
    )


def time_mean_cubed_sphere_maps(
    diagnostics: Iterable[xr.Dataset], metrics: pd.DataFrame
) -> HVPlot:
    return plot_cubed_sphere_map(
        diagnostics,
        metrics,
        "time_mean_value",
        metrics_for_title={"Mean": "time_and_global_mean_value"},
        n_jobs=-1,
    )


def time_mean_bias_cubed_sphere_maps(
    diagnostics: Iterable[xr.Dataset], metrics: pd.DataFrame
) -> HVPlot:
    return plot_cubed_sphere_map(
        diagnostics,
        metrics,
        "time_mean_bias",
        metrics_for_title={
            "Mean": "time_and_global_mean_bias",
            "RMSE": "rmse_of_time_mean",
        },
        n_jobs=-1,
    )


@zonal_pressure_plot_manager.register
def zonal_pressure_plots(diagnostics: Iterable[xr.Dataset]) -> RawHTML:
    return plot_2d_matplotlib(
        diagnostics,
        "pressure_level_zonal_time_mean",
        ["latitude", "pressure"],
        yincrease=False,
        ylabel="Pressure [Pa]",
        n_jobs=-1,
    )


@zonal_pressure_plot_manager.register
def zonal_pressure_bias_plots(diagnostics: Iterable[xr.Dataset]) -> RawHTML:
    return plot_2d_matplotlib(
        diagnostics,
        "pressure_level_zonal_bias",
        ["latitude", "pressure"],
        contour=True,
        cmap="RdBu_r",
        yincrease=False,
        ylabel="Pressure [Pa]",
        n_jobs=-1,
    )


@diurnal_plot_manager.register
def diurnal_cycle_plots(diagnostics: Iterable[xr.Dataset]) -> HVPlot:
    return plot_1d_with_region_bar(diagnostics, varfilter="diurnal")


@diurnal_plot_manager.register
def diurnal_cycle_component_plots(diagnostics: Iterable[xr.Dataset]) -> HVPlot:
    return diurnal_component_plot(diagnostics)


@histogram_plot_manager.register
def precip_histogram_plots(diagnostics: Iterable[xr.Dataset]) -> HVPlot:
    return plot_histogram(
        diagnostics, f"{PRECIP_RATE}_histogram", xscale="log", yscale="log"
    )


@histogram_plot_manager.register
def water_vapor_path_histogram_plots(diagnostics: Iterable[xr.Dataset]) -> HVPlot:
    return plot_histogram(diagnostics, f"{WVP}_histogram")


@histogram_plot_manager.register
def histogram2d_plots(diagnostics: Iterable[xr.Dataset]) -> HVPlot:
    return plot_histogram2d(diagnostics, WVP, COL_DRYING)


@histogram_plot_manager.register
def conditional_average_plots(diagnostics: Iterable[xr.Dataset]) -> HVPlot:
    return plot_1d(diagnostics, varfilter="conditional_average")


# Routines for plotting the "metrics"
# New plotting routines can be registered here.
@metrics_plot_manager.register
def time_mean_bias_metrics(metrics: RunMetrics) -> RawHTML:
    return metric_type_table(metrics, "time_and_global_mean_bias")


@metrics_plot_manager.register
def rmse_time_mean_metrics(metrics: RunMetrics) -> RawHTML:
    return metric_type_table(metrics, "rmse_of_time_mean")


@metrics_plot_manager.register
def rmse_time_mean_land_metrics(metrics: RunMetrics) -> RawHTML:
    return metric_type_table(metrics, "rmse_of_time_mean_land")


@metrics_plot_manager.register
def rmse_time_mean_sea_metrics(metrics: RunMetrics) -> RawHTML:
    return metric_type_table(metrics, "rmse_of_time_mean_sea")


@metrics_plot_manager.register
def rmse_3day_metrics(metrics: RunMetrics) -> RawHTML:
    return metric_type_table(metrics, "rmse_3day")


@metrics_plot_manager.register
def drift_metrics(metrics: RunMetrics) -> RawHTML:
    return metric_type_table(metrics, "drift_3day")


def _get_metric_type_df(metrics: RunMetrics, metric_type: str) -> pd.DataFrame:
    variables = sorted(metrics.get_metric_variables(metric_type))
    table = {}
    for varname in variables:
        units = metrics.get_metric_units(metric_type, varname, metrics.runs[0])
        column_name = f"{varname} [{units}]"
        table[column_name] = [
            metrics.get_metric_value(metric_type, varname, run) for run in metrics.runs
        ]
    return pd.DataFrame(table, index=metrics.runs)


def metric_type_table(metrics: RunMetrics, metric_type: str) -> RawHTML:
    """Return HTML table of all metrics of type metric_type.

    Args:
        metrics: Computed metrics.
        metric_type: Label for type of metric, e.g. "rmse_5day".

    Returns:
        HTML representation of table of metric values with rows of all variables
        available for given metric_type and columns of runs.
    """
    df = _get_metric_type_df(metrics, metric_type).transpose()
    header = f"<h3>{metric_type}</h3>"
    return RawHTML(header + df.to_html(justify="center", float_format="{:.2f}".format))


def _get_metric_df(metrics, metric_names):
    table = {}
    for metric_type, variables in metric_names.items():
        for variable in variables:
            units = metrics.get_metric_units(metric_type, variable, metrics.runs[0])
            column_name = f"{variable} {metric_type} [{units}]"
            table[column_name] = [
                metrics.get_metric_value(metric_type, variable, run)
                for run in metrics.runs
            ]
    return pd.DataFrame(table, index=metrics.runs)


def metric_table(
    metrics: RunMetrics, metric_names: Mapping[str, Sequence[str]]
) -> RawHTML:
    """Return HTML table for all metrics specified in metric_names.

    Args:
        metrics: Computed metrics.
        metric_names: A mapping from metric_types to sequences of variable names.
            For example, {"rmse_5day": ["h500", "tmp850"]}.

    Returns:
        HTML representation of table of metric values with rows of all metrics
        specified in metric_names and columns of runs.
    """
    df = _get_metric_df(metrics, metric_names).transpose()
    return RawHTML(df.to_html(justify="center", float_format="{:.2f}".format))


navigation = OrderedList(
    Link("Home", "index.html"),
    Link("Process diagnostics", "process_diagnostics.html"),
    Link("Latitude versus time hovmoller", "hovmoller.html"),
    Link("Time-mean maps", "maps.html"),
    Link("Time-mean zonal-pressure profiles", "zonal_pressure.html"),
)
navigation = [navigation]  # must be iterable for create_html template


def render_index(metadata, diagnostics, metrics, movie_links):
    sections_index = {
        "Links": navigation,
        "Top-level metrics": [metric_table(metrics, TOP_LEVEL_METRICS)],
        "Timeseries": list(timeseries_plot_manager.make_plots(diagnostics)),
        "Zonal mean": list(zonal_mean_plot_manager.make_plots(diagnostics)),
        "Complete metrics": list(metrics_plot_manager.make_plots(metrics)),
    }
    return create_html(
        title="Prognostic run report",
        metadata={**metadata, **render_links(movie_links)},
        sections=sections_index,
        html_header=get_header(),
    )


def render_hovmollers(metadata, diagnostics):
    sections_hovmoller = {
        "Links": navigation,
        "Zonal mean value and bias": list(
            hovmoller_plot_manager.make_plots(diagnostics)
        ),
    }
    return create_html(
        title="Latitude versus time hovmoller plots",
        metadata=metadata,
        sections=sections_hovmoller,
        html_header=get_header(),
    )


def render_maps(metadata, diagnostics, metrics):
    # the plotting functions here require two inputs so can't use a PlotManager
    sections = {
        "Links": navigation,
        "Time-mean maps": [
            time_mean_cubed_sphere_maps(diagnostics, metrics),
            time_mean_bias_cubed_sphere_maps(diagnostics, metrics),
        ],
    }
    return create_html(
        title="Time-mean maps",
        metadata=metadata,
        sections=sections,
        html_header=get_header(),
    )


def render_zonal_pressures(metadata, diagnostics):
    sections_zonal_pressure = {
        "Links": navigation,
        "Zonal mean values at pressure levels": list(
            zonal_pressure_plot_manager.make_plots(diagnostics)
        ),
    }
    return create_html(
        title="Pressure versus latitude plots",
        metadata=metadata,
        sections=sections_zonal_pressure,
        html_header=get_header(),
    )


def render_process_diagnostics(metadata, diagnostics, metrics):
    percentile_names = {f"percentile_{p}": [PRECIP_RATE] for p in PERCENTILES}
    sections = {
        "Links": navigation,
        "Precipitation percentiles": [metric_table(metrics, percentile_names)],
        "Diurnal cycle": list(diurnal_plot_manager.make_plots(diagnostics)),
        "Precipitation and water vapor path": list(
            histogram_plot_manager.make_plots(diagnostics)
        ),
        "Tropical Hovmoller plots": list(
            tropical_hovmoller_plot_manager.make_plots(diagnostics)
        ),
    }
    return create_html(
        title="Process diagnostics",
        metadata=metadata,
        sections=sections,
        html_header=get_header(),
    )


def _html_link(url, tag):
    return f"<a href='{url}'>{tag}</a>"


def render_links(link_dict):
    """Render links to html

    Args:
        link_dict: dict where keys are names, and values are lists (url,
            text_to_display). For example::

                {"column_moistening.mp4": [(url_to_qv, "specific humidity"), ...]}
    """
    return {
        key: " ".join([_html_link(url, tag) for (url, tag) in links])
        for key, links in link_dict.items()
    }


def _movie_name(url: str) -> str:
    return os.path.basename(url)


def _movie_output_path(root: str, run_name: str, movie_name: str) -> str:
    return os.path.join(root, "_movies", run_name, movie_name)


def _get_movie_manifest(movie_urls: MovieUrls, output: str) -> MovieManifest:
    """Return manifest of report output location for each movie in movie_urls.

    Args:
        movie_urls: the URLs for the movies to be included in report.
        output: the location where the report will be saved.

    Returns:
        Tuples of source URL and output path for each movie."""
    manifest = []
    for run_name, urls in movie_urls.items():
        for url in urls:
            output_path = _movie_output_path(output, run_name, _movie_name(url))
            manifest.append((url, output_path))
    return manifest


def _get_public_links(movie_urls: MovieUrls, output: str) -> PublicLinks:
    """Get the public links at which each movie can be opened in a browser."""
    public_links: PublicLinks = {}
    for run_name, urls in movie_urls.items():
        for url in urls:
            movie_name = _movie_name(url)
            output_path = _movie_output_path(output, run_name, movie_name)
            if output_path.startswith("gs://"):
                public_link = output_path.replace("gs:/", PUBLIC_GCS_DOMAIN)
            else:
                public_link = output_path
            public_links.setdefault(movie_name, []).append((public_link, run_name))
    return public_links


def make_report(computed_diagnostics: ComputedDiagnosticsList, output):
    metrics = computed_diagnostics.load_metrics_from_diagnostics()
    movie_urls = computed_diagnostics.find_movie_urls()
    metadata, diagnostics = computed_diagnostics.load_diagnostics()
    manifest = _get_movie_manifest(movie_urls, output)
    public_links = _get_public_links(movie_urls, output)
    for source, target in manifest:
        vcm.cloud.copy(source, target, content_type="video/mp4")

    pages = {
        "index.html": render_index(metadata, diagnostics, metrics, public_links),
        "hovmoller.html": render_hovmollers(metadata, diagnostics),
        "maps.html": render_maps(metadata, diagnostics, metrics),
        "zonal_pressure.html": render_zonal_pressures(metadata, diagnostics),
        "process_diagnostics.html": render_process_diagnostics(
            metadata, diagnostics, metrics
        ),
    }

    for filename, html in pages.items():
        upload(html, os.path.join(output, filename))


def _register_report(subparsers):
    parser = subparsers.add_parser("report", help="Generate a static html report.")
    parser.add_argument("input", help="Directory containing multiple run diagnostics.")
    parser.add_argument("output", help="Location to save report html files.")
    parser.set_defaults(func=main)


def _register_report_from_urls(subparsers):
    parser = subparsers.add_parser(
        "report-from-urls",
        help="Generate a static html report from list of diagnostics.",
    )
    parser.add_argument(
        "inputs",
        help="Folders containing diags.nc. Will be labeled with "
        "increasing numbers in report.",
        nargs="+",
    )
    parser.add_argument(
        "-o", "--output", help="Location to save report html files.", required=True
    )
    parser.set_defaults(func=main_new)


def _register_report_from_json(subparsers):
    parser = subparsers.add_parser(
        "report-from-json",
        help="Generate report from diagnostics listed in given JSON file.",
    )
    parser.add_argument(
        "input",
        help="Path to JSON file with list of mappings, each with name and url keys. "
        "Given URLs must be for run diagnostics folders.",
    )
    parser.add_argument("output", help="Location to save report html files.")
    parser.add_argument(
        "-r",
        "--urls-are-rundirs",
        action="store_true",
        help="Use if the URLs in given JSON file are for fv3gfs run directories "
        "instead of run diagnostics folders.",
    )
    parser.set_defaults(func=main_json)


def register_parser(subparsers):
    _register_report(subparsers)
    _register_report_from_urls(subparsers)
    _register_report_from_json(subparsers)


def main(args):
    computed_diagnostics = ComputedDiagnosticsList.from_directory(args.input)
    make_report(computed_diagnostics, args.output)


def main_new(args):
    computed_diagnostics = ComputedDiagnosticsList.from_urls(args.inputs)
    make_report(computed_diagnostics, args.output)


def main_json(args):
    computed_diagnostics = ComputedDiagnosticsList.from_json(
        args.input, args.urls_are_rundirs
    )
    make_report(computed_diagnostics, args.output)
