#!/usr/bin/env python

import json
from typing import Iterable
import os
import xarray as xr
import fsspec
import pandas as pd
from pathlib import Path
import argparse
import holoviews as hv
from report import create_html
from report.holoviews import HVPlot, get_html_header

hv.extension("bokeh")
PUBLIC_GCS_DOMAIN = "https://storage.googleapis.com"


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


def get_variables_with_dims(ds, dims):
    return ds.drop([key for key in ds if set(ds[key].dims) != set(dims)])


def convert_time_index_to_datetime(ds, dim):
    return ds.assign_coords({dim: ds.indexes[dim].to_datetimeindex()})


def detect_rundirs(bucket: str, fs: fsspec.AbstractFileSystem):
    diag_ncs = fs.glob(os.path.join(bucket, "*", "diags.nc"))
    if len(diag_ncs) < 2:
        raise ValueError(
            "Plots require more than 1 diagnostic directory in"
            f" {bucket} for holoviews plots to display correctly."
        )
    return [Path(url).parent.name for url in diag_ncs]


def load_diags(bucket, rundirs):
    metrics = {}
    for rundir in rundirs:
        path = os.path.join(bucket, rundir, "diags.nc")
        with fsspec.open(path, "rb") as f:
            metrics[rundir] = xr.open_dataset(f, engine="h5netcdf").compute()
    return metrics


def _yield_metric_rows(metrics):
    """yield rows to be combined into a dataframe
    """
    for run in metrics:
        for name in metrics[run]:
            yield {
                "run": run,
                "metric": name,
                "value": metrics[run][name]["value"],
                "units": metrics[run][name]["units"],
            }


def _parse_metadata(run: str):
    baseline_s = "-baseline"

    if run.endswith(baseline_s):
        baseline = True
    else:
        baseline = False

    return {"run": run, "baseline": baseline}


def load_metrics(bucket, rundirs):
    metrics = {}
    for rundir in rundirs:
        path = os.path.join(bucket, rundir, "metrics.json")
        with fsspec.open(path, "rb") as f:
            metrics[rundir] = json.load(f)

    return metrics


def get_movie_links(bucket, rundirs, fs, domain=PUBLIC_GCS_DOMAIN):
    movie_links = {}
    for rundir in rundirs:
        movie_paths = fs.glob(os.path.join(bucket, rundir, "*.mp4"))
        for gcs_path in movie_paths:
            movie_name = os.path.basename(gcs_path)
            if movie_name not in movie_links:
                movie_links[movie_name] = ""
            public_url = os.path.join(domain, gcs_path)
            movie_links[movie_name] += " " + _html_link(public_url, rundir)
    return movie_links


def _html_link(url, tag):
    return f"<a href='{url}'>{tag}</a>"


def _longest_run(diagnostics: Iterable[xr.Dataset]) -> xr.Dataset:
    max_length = 0
    for ds in diagnostics:
        if ds.sizes["time"] > max_length:
            longest_ds = ds
            max_length = ds.sizes["time"]
    return longest_ds


def holomap_filter(time_series, varfilter, run_attr_name="run"):
    p = hv.Cycle("Colorblind")
    hmap = hv.HoloMap(kdims=["variable", "run"])
    for ds in time_series:
        for varname in ds:
            if varfilter in varname:
                v = ds[varname]
                style = "solid" if ds.attrs["baseline"] else "dashed"
                run = ds.attrs[run_attr_name]
                long_name = ds[varname].long_name
                hmap[(long_name, run)] = hv.Curve(v, label=varfilter).options(
                    line_dash=style, color=p
                )
    return hmap


def holomap_filter_with_region_bar(time_series, varfilter, run_attr_name="run"):
    p = hv.Cycle("Colorblind")
    hmap = hv.HoloMap(kdims=["variable", "region", "run"])
    for ds in time_series:
        for varname in ds:
            if varfilter in varname:
                try:
                    v = ds[varname]
                except KeyError:
                    pass
                else:
                    style = "solid" if ds.attrs["baseline"] else "dashed"
                    run = ds.attrs[run_attr_name]
                    long_name = ds[varname].long_name
                    region = varname.split("_")[-1]
                    hmap[(long_name, region, run)] = hv.Curve(
                        v.rename("value"), label=varfilter,
                    ).options(line_dash=style, color=p)
    return hmap


def _add_hmap_opts(hmap, overlay="run"):
    return (
        hmap.opts(norm={"framewise": True}, plot=dict(width=850, height=500))
        .overlay(overlay)
        .opts(legend_position="right")
    )


def time_series_plot(time_series: Iterable[xr.Dataset], varfilter: str) -> HVPlot:
    return HVPlot(_add_hmap_opts(holomap_filter(time_series, varfilter)))


def time_series_plot_with_region_bar(
    time_series: Iterable[xr.Dataset], varfilter: str
) -> HVPlot:
    return HVPlot(
        _add_hmap_opts(holomap_filter_with_region_bar(time_series, varfilter))
    )


def _parse_diurnal_component_fields(varname: str):

    # diags key format: diurn_component_<varname>_diurnal_<sfc_type>
    tokens = varname.split("_")
    short_varname = tokens[2]
    surface_type = tokens[-1]

    return short_varname, surface_type


def _get_verification_diagnostics(ds: xr.Dataset) -> xr.Dataset:
    """Back out verification timeseries from prognostic run value and bias"""
    verif_diagnostics = {}
    verif_attrs = {"run": "verification", "baseline": True}
    mean_bias_pairs = {
        "spatial_mean": "mean_bias",
        "diurn_component": "diurn_bias",
        "zonal_and_time_mean": "zonal_bias",
    }
    for mean_filter, bias_filter in mean_bias_pairs.items():
        mean_vars = [var for var in ds if mean_filter in var]
        for var in mean_vars:
            matching_bias_var = var.replace(mean_filter, bias_filter)
            if matching_bias_var in ds:
                # verification = prognostic - bias
                verif_diagnostics[var] = ds[var] - ds[matching_bias_var]
                verif_diagnostics[var].attrs = ds[var].attrs
    return xr.Dataset(verif_diagnostics, attrs=verif_attrs)


def diurnal_component_plot(
    time_series: Iterable[xr.Dataset],
    run_attr_name="run",
    diurnal_component_name="diurn_component",
) -> HVPlot:

    p = hv.Cycle("Colorblind")
    hmap = hv.HoloMap(kdims=["run", "surface_type", "short_varname"])

    for ds in time_series:
        for varname in ds:
            if diurnal_component_name in varname:
                v = ds[varname]
                short_vname, surface_type = _parse_diurnal_component_fields(varname)
                run = ds.attrs[run_attr_name]
                hmap[(run, surface_type, short_vname)] = hv.Curve(
                    v, label=diurnal_component_name
                ).options(color=p)

    return HVPlot(_add_hmap_opts(hmap, overlay="short_varname"))


# Initialize diagnostic managers
# diag_plot_manager will be passed the data from the diags.nc files
diag_plot_manager = PlotManager()
# this will be passed the data from the metrics.json files
metrics_plot_manager = PlotManager()


# Routines for plotting the "diagnostics"
@diag_plot_manager.register
def rms_plots(time_series: Iterable[xr.Dataset]) -> HVPlot:
    return time_series_plot(time_series, varfilter="rms_global")


@diag_plot_manager.register
def spatial_mean_plots(time_series: Iterable[xr.Dataset]) -> HVPlot:
    return time_series_plot_with_region_bar(time_series, varfilter="spatial_mean")


@diag_plot_manager.register
def zonal_mean_plots(time_series: Iterable[xr.Dataset]) -> HVPlot:
    return time_series_plot(time_series, varfilter="zonal_and_time_mean")


@diag_plot_manager.register
def diurnal_cycle_plots(time_series: Iterable[xr.Dataset]) -> HVPlot:
    return time_series_plot_with_region_bar(time_series, varfilter="diurnal")


@diag_plot_manager.register
def diurnal_cycle_component_plots(time_series: Iterable[xr.Dataset]) -> HVPlot:
    return diurnal_component_plot(time_series)


# Routines for plotting the "metrics"
# New plotting routines can be registered here.
@metrics_plot_manager.register
def time_mean_bias_metrics(metrics: pd.DataFrame) -> hv.HoloMap:
    return generic_metric_plot(metrics, "time_and_global_mean_bias")


@metrics_plot_manager.register
def rmse_metrics(metrics: pd.DataFrame) -> hv.HoloMap:
    return generic_metric_plot(metrics, "rmse")


@metrics_plot_manager.register
def drift_metrics(metrics: pd.DataFrame) -> hv.HoloMap:
    return generic_metric_plot(metrics, "drift")


def generic_metric_plot(metrics: pd.DataFrame, name: str) -> hv.HoloMap:
    hmap = hv.HoloMap(kdims=["metric"])
    bar_opts = dict(norm=dict(framewise=True), plot=dict(width=600))
    metrics_contains_name = False
    for metric in metrics.metric.unique():
        if metric.startswith(name):
            metrics_contains_name = True
            s = metrics[metrics.metric == metric]
            bars = hv.Bars((s.run, s.value), hv.Dimension("Run"), s.units.iloc[0])
            hmap[metric] = bars
    if metrics_contains_name:
        return HVPlot(hmap.opts(**bar_opts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")

    args = parser.parse_args()
    bucket = args.input

    # get run information
    fs, _, _ = fsspec.get_fs_token_paths(bucket)
    rundirs = detect_rundirs(bucket, fs)
    run_table = pd.DataFrame.from_records(_parse_metadata(run) for run in rundirs)
    run_table_lookup = run_table.set_index("run")

    # load diagnostics
    diags = load_diags(bucket, rundirs)
    # keep all vars that have only these dimensions
    dims = ["time", "local_time", "latitude"]
    diagnostics = [
        xr.merge([get_variables_with_dims(ds, [dim]) for dim in dims]).assign_attrs(
            run=key, **run_table_lookup.loc[key]
        )
        for key, ds in diags.items()
    ]
    diagnostics = [convert_time_index_to_datetime(ds, "time") for ds in diagnostics]

    # hack to add verification data from longest set of diagnostics as new run
    # TODO: generate separate diags.nc file for verification data and load that in here
    longest_run_ds = _longest_run(diagnostics)
    diagnostics.append(_get_verification_diagnostics(longest_run_ds))

    # load metrics
    nested_metrics = load_metrics(bucket, rundirs)
    metric_table = pd.DataFrame.from_records(_yield_metric_rows(nested_metrics))

    # generate all plots
    sections = {"Diagnostics": list(diag_plot_manager.make_plots(diagnostics))}
    if not metric_table.empty:
        metrics = pd.merge(run_table, metric_table, on="run")
        sections["Metrics"] = list(metrics_plot_manager.make_plots(metrics))

    # get metadata
    run_urls = {key: ds.attrs["url"] for key, ds in diags.items()}
    verification_datasets = [ds.attrs["verification"] for ds in diags.values()]
    if any([verification_datasets[0] != item for item in verification_datasets]):
        raise ValueError(
            "Report cannot be generated with diagnostics computed against "
            "different verification datasets."
        )
    verification_label = {"verification dataset": verification_datasets[0]}
    movie_links = get_movie_links(bucket, rundirs, fs)

    html = create_html(
        title="Prognostic run report",
        metadata={**verification_label, **run_urls, **movie_links},
        sections=sections,
        html_header=get_html_header(),
    )
    upload(html, args.output, content_type="text/html")


if __name__ == "__main__":
    main()
