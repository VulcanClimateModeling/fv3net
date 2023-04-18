# flake8: noqa

from datetime import timedelta
import os
from typing import Optional, Tuple
import fv3fit
from fv3fit.pytorch import DEVICE
from matplotlib import pyplot as plt
import xarray as xr
from vcm.catalog import catalog
import fv3viz
import cartopy.crs as ccrs
import numpy as np
import vcm
from time import time
from fv3fit.pytorch.cyclegan.cyclegan_trainer import get_percentile_error

GRID = catalog["grid/c48"].read()
AREA = GRID.area
land_sea_mask = catalog["landseamask/c48"].read().land_sea_mask == 1

TO_MM_DAY = 86400 / 0.997

# for older data
# C48_I_TRAIN_END = 11688
# C384_I_TRAIN_END = 2920

# for "march" data and for multi-climate "march" data
C48_I_TRAIN_END = 14600
C384_I_TRAIN_END = 14600

# for testing
# C48_I_TRAIN_END = 50
# C384_I_TRAIN_END = 50


def predict_dataset(
    c48_real: xr.Dataset, c384_real: xr.Dataset, cyclegan: fv3fit.pytorch.CycleGAN
) -> Tuple[xr.Dataset, xr.Dataset]:
    c48_real = c48_real[cyclegan.state_variables]
    c384_real = c384_real[cyclegan.state_variables]
    c384_list = []
    c48_list = []
    for i in range(len(c384_real.perturbation)):
        nt = len(c384_real.time)
        start = time()
        c384_list.append(cyclegan.predict(c48_real.isel(perturbation=i)))
        print(f"Predicted {nt} samples of c384 in {time() - start} seconds")
        c48_list.append(cyclegan.predict(c384_real.isel(perturbation=i), reverse=True))
    c384_gen = xr.concat(c384_list, dim="perturbation")
    c48_gen = xr.concat(c48_list, dim="perturbation")
    return c48_gen, c384_gen
    # return c48_real[cyclegan.state_variables], c384_real[cyclegan.state_variables]


class MeanStdAggregator:
    def __init__(self):
        self.n_samples = 0
        self.sum = None
        self.sum_squared = None

    def add(self, x: xr.DataArray):
        # must be float64 as the std calc is very sensitive to rounding errors
        if self.sum is None:
            self.sum = x.astype(np.float64).mean("time").values
            self.sum_squared = (x.astype(np.float64) ** 2).mean("time").values
        else:
            self.sum += x.astype(np.float64).mean("time").values
            self.sum_squared += (x.astype(np.float64) ** 2).mean("time").values
        self.n_samples += 1

    def get_mean(self) -> np.ndarray:
        return self.sum / self.n_samples

    def get_std(self) -> np.ndarray:
        return np.sqrt(
            self.sum_squared / self.n_samples - (self.sum / self.n_samples) ** 2
        )


class HistogramAggregator:
    def __init__(self, bins: np.ndarray):
        self.bins = bins
        self.sum = None

    def add(self, x: xr.DataArray):
        array = x.values
        array = array.reshape(-1, *array.shape[-4:])
        out = np.empty((array.shape[0], len(self.bins) - 1), dtype=int)
        for i in range(array.shape[0]):
            out[i] = np.histogram(array[i].flatten(), bins=self.bins)[0]
        out = out.reshape(*x.shape[:-4], len(self.bins) - 1)
        if self.sum is None:
            self.sum = out
        else:
            self.sum += out

    def get(self) -> np.ndarray:
        return self.sum


def get_mean(
    c48_real: xr.Dataset,
    c384_real: xr.Dataset,
    c48_gen: xr.Dataset,
    c384_gen: xr.Dataset,
    varname: str,
):
    def reduction(ds: xr.Dataset) -> xr.Dataset:
        return ds.mean("time")

    return get_reduction(c48_real, c384_real, c48_gen, c384_gen, varname, reduction)


def get_std(
    c48_real: xr.Dataset,
    c384_real: xr.Dataset,
    c48_gen: xr.Dataset,
    c384_gen: xr.Dataset,
    varname: str,
):
    def reduction(ds: xr.Dataset) -> xr.Dataset:
        return ds.std("time")

    return get_reduction(c48_real, c384_real, c48_gen, c384_gen, varname, reduction)


def get_reduction(
    c48_real: xr.Dataset,
    c384_real: xr.Dataset,
    c48_gen: xr.Dataset,
    c384_gen: xr.Dataset,
    varname: str,
    reduction,
) -> xr.DataArray:
    c48_real = time_to_diurnal(c48_real)
    c384_real = time_to_diurnal(c384_real)
    c48_gen = time_to_diurnal(c48_gen)
    c384_gen = time_to_diurnal(c384_gen)
    # c48_quantile = time_to_diurnal(c48_quantile)
    # c384_quantile = time_to_diurnal(c384_quantile)
    c48_real_reduced = reduction(c48_real[varname])
    c384_real_reduced = reduction(c384_real[varname])
    c48_gen_reduced = reduction(c48_gen[varname])
    c384_gen_reduced = reduction(c384_gen[varname])
    # c48_quantile_reduced = reduction(c48_quantile[varname])
    # c384_quantile_reduced = reduction(c384_quantile[varname])

    c48 = xr.concat([c48_real_reduced, c48_gen_reduced], dim="source").assign_coords(
        source=["real", "gen"]
    )
    c384 = xr.concat([c384_real_reduced, c384_gen_reduced], dim="source").assign_coords(
        source=["real", "gen"]
    )
    return xr.concat([c48, c384], dim="grid").assign_coords(grid=["C48", "C384"])


def time_to_diurnal(ds: xr.Dataset) -> xr.Dataset:
    """
    Converts a dataset with a time dimension in 3h increments
    to a dataset with a 'diurnal' dimension of length 8 and a new 'time' dimension.

    Does this by reshaping the time dimension into two dimensions. If the initial
    time dimension is of length N, the new 'time' dimension will be of length N/8,
    and the new 'diurnal' dimension will be of length 8.

    If not all days have 8 time steps, the last partial day will be dropped.
    """
    initial_time = ds["time"][0]
    diurnal_list = []
    new_time = ds["time"].isel(time=slice(0, None, 8))
    for i in range(8):
        ds_hour = ds.isel(time=slice(i, None, 8))
        if "time" in ds.coords:
            ds_hour = ds_hour.drop("time").assign_coords({"time": new_time})
        diurnal_list.append(ds_hour)
    n_days = min([len(ds_hour.time) for ds_hour in diurnal_list])
    ds = xr.concat(
        [ds.isel(time=slice(0, n_days)) for ds in diurnal_list], dim="diurnal"
    )
    ds["initial_time"] = initial_time
    return ds


def process_dataset(
    c48_real: xr.Dataset,
    c384_real: xr.Dataset,
    c48_gen: xr.Dataset,
    c384_gen: xr.Dataset,
    use_train_reference: bool = True,
) -> xr.Dataset:
    # first, denormalize data
    for varname in c48_gen.data_vars.keys():
        try:
            c48_real = c48_real.assign(
                **{
                    varname: c48_real[varname] * c48_real[varname + "_std"]
                    + c48_real[varname + "_mean"]
                }
            )
            c48_gen = c48_gen.assign(
                **{
                    varname: c48_gen[varname] * c48_real[varname + "_std"]
                    + c48_real[varname + "_mean"]
                }
            )
            c384_real = c384_real.assign(
                **{
                    varname: c384_real[varname] * c384_real[varname + "_std"]
                    + c384_real[varname + "_mean"]
                }
            )
            c384_gen = c384_gen.assign(
                **{
                    varname: c384_gen[varname] * c384_real[varname + "_std"]
                    + c384_real[varname + "_mean"]
                }
            )
        except KeyError:
            print(
                "Assuming de-normalized data as no mean or std was found for variable {}".format(
                    varname
                )
            )
    c48_real = c48_real[c48_gen.data_vars.keys()]
    c384_real = c384_real[c384_gen.data_vars.keys()]
    # c48_quantile, c384_quantile = quantile_map(c48_real, c384_real)
    # c48_quantile, c384_quantile = c48_real.copy(deep=True), c384_real.copy(deep=True)
    out_vars = {}
    for varname in c48_real.data_vars.keys():
        if use_train_reference:
            ref_c48_real = c48_real.isel(time=slice(0, C48_I_TRAIN_END))
            ref_c384_real = c384_real.isel(time=slice(0, C384_I_TRAIN_END))
        else:
            ref_c48_real = c48_real.isel(time=slice(C48_I_TRAIN_END, None))
            ref_c384_real = c384_real.isel(time=slice(C384_I_TRAIN_END, None))
        print("getting histogram")
        out_vars[f"{varname}_hist_bins"], out_vars[f"{varname}_hist"] = get_histogram(
            ref_c48_real, ref_c384_real, c48_gen, c384_gen, varname
        )
        print("getting mean")
        out_vars[f"{varname}_mean"] = get_mean(
            ref_c48_real, ref_c384_real, c48_gen, c384_gen, varname
        )
        print("getting std")
        out_vars[f"{varname}_std"] = get_std(
            ref_c48_real, ref_c384_real, c48_gen, c384_gen, varname
        )
    return xr.Dataset(out_vars)


def get_histogram(
    c48_real: xr.Dataset,
    c384_real: xr.Dataset,
    c48_gen: xr.Dataset,
    c384_gen: xr.Dataset,
    varname: str,
):

    # must ensure bins fill whole range, but we can't load all the data
    # at once for histogram computation, so we get min/max from each dataset
    # and then use the max/min of those to get the bin min/max values
    # vmin = min(
    #     [
    #         c48_real[varname].min().values,
    #         c384_real[varname].min().values,
    #         c48_gen[varname].min().values,
    #         c384_gen[varname].min().values,
    #         # c48_quantile[varname].min().values,
    #         # c384_quantile[varname].min().values,
    #     ]
    # )
    # vmax = max(
    #     [
    #         c48_real[varname].max().values,
    #         c384_real[varname].max().values,
    #         c48_gen[varname].max().values,
    #         c384_gen[varname].max().values,
    #         # c48_quantile[varname].max().values,
    #         # c384_quantile[varname].max().values,
    #     ]
    # )
    vmin = -0.001
    vmax = 0.0175

    bin_edges = np.linspace(vmin, vmax, 150)

    def reduction(da: xr.DataArray):
        def _hist(x: np.ndarray):
            in_shape = x.shape
            x = x.reshape(-1, *x.shape[-4:])
            out = np.empty((x.shape[0], len(bin_edges) - 1), dtype=int)
            for i in range(x.shape[0]):
                out[i] = np.histogram(x[i].flatten(), bins=bin_edges)[0]
            return out.reshape(*in_shape[:-4], len(bin_edges) - 1)

        return xr.apply_ufunc(
            _hist,
            da,
            input_core_dims=(["time", "tile", "x", "y"],),
            output_core_dims=(["hist"],),
            output_sizes={"hist": len(bin_edges) - 1},
            vectorize=True,
            dask="parallelized",
            output_dtypes=[int],
            dask_gufunc_kwargs={"allow_rechunk": True},
        )

    return (
        bin_edges,
        get_reduction(c48_real, c384_real, c48_gen, c384_gen, varname, reduction),
    )


# def quantile_map(c48: xr.Dataset, c384: xr.Dataset) -> Tuple[xr.Dataset, xr.Dataset]:
#     mappings_c48_to_c384 = {}
#     mappings_c384_to_c48 = {}
#     for varname in c48.data_vars.keys():
#         print(f"training quantile mapping for {varname}")
#         transform_c48 = sklearn.preprocessing.QuantileTransformer(
#             subsample=int(1_000_000), output_distribution="uniform"
#         )
#         transform_c48.fit(c48[varname].isel(time=slice(C48_I_TRAIN_END, None)).values.flatten().reshape(-1, 1))
#         transform_c384 = sklearn.preprocessing.QuantileTransformer(
#             subsample=int(1_000_000), output_distribution="uniform"
#         )
#         transform_c384.fit(c384[varname].isel(time=slice(C384_I_TRAIN_END, None)).values.flatten().reshape(-1, 1))
#         mappings_c48_to_c384[varname] = QuantileMapping(transform_c48, transform_c384)
#         mappings_c384_to_c48[varname] = QuantileMapping(transform_c384, transform_c48)

#     transformed = {}
#     for varname in c384.data_vars.keys():
#         print(f"performing c384 to c48 quantile mapping for {varname}")
#         var_test = c384[varname].isel(time=slice(C384_I_TRAIN_END, None))
#         array = mappings_c384_to_c48[varname](
#             var_test.values.flatten().reshape(-1, 1)
#         ).reshape(var_test.shape)
#         transformed[varname] = xr.DataArray(
#             array, dims=var_test.dims, coords=var_test.coords
#         )
#     c48_quantile = xr.Dataset(transformed)
#     transformed = {}
#     for varname in c48.data_vars.keys():
#         print(f"performing c48 to c384 quantile mapping for {varname}")
#         var_test = c48[varname].isel(time=slice(C48_I_TRAIN_END, None))
#         array = mappings_c48_to_c384[varname](
#             var_test.values.flatten().reshape(-1, 1)
#         ).reshape(var_test.shape)
#         transformed[varname] = xr.DataArray(
#             array, dims=var_test.dims, coords=var_test.coords
#         )
#     c384_quantile = xr.Dataset(transformed)
#     return c48_quantile, c384_quantile


# class QuantileMapping:
#     def __init__(self, transform_in, transform_out):
#         self.transform_in = transform_in
#         self.transform_out = transform_out

#     def __call__(self, x):
#         return self.transform_out.inverse_transform(self.transform_in.transform(x))

#     def dump(self, path):
#         with open(path, "wb") as f:
#             pickle.dump(self, f)


# def train_quantile_mapping(c48_real: xr.Dataset, c384_real: xr.Dataset, varname: str):
#     transform_c48 = sklearn.preprocessing.QuantileTransformer(
#         subsample=int(1_000_000), output_distribution="uniform"
#     )
#     transform_c384 = sklearn.preprocessing.QuantileTransformer(
#         subsample=int(1_000_000), output_distribution="uniform"
#     )
#     transform_c48.fit(c48_real[varname].values.flatten().reshape(-1, 1))
#     transform_c384.fit(c384_real[varname].values.flatten().reshape(-1, 1))
#     return QuantileMapping(transform_c48, transform_c384)


def plot_mean_all(ds, varname, label: str):
    fig, ax = plt.subplots(
        len(ds.perturbation),
        4,
        figsize=(18, 3 * len(ds.perturbation) + 0.5),
        subplot_kw={"projection": ccrs.Robinson()},
    )
    if len(ax.shape) == 1:
        ax = ax[None, :]
    print(f"Recording pattern biases for {label}")
    for i, climate in enumerate(ds.perturbation.values):
        mean = (
            ds[f"{varname}_mean"].sel(grid="C384").isel(perturbation=i).mean("diurnal")
        ) * TO_MM_DAY
        mean_c48_real = (
            ds[f"{varname}_mean"]
            .sel(grid="C48", source="real")
            .isel(perturbation=i)
            .mean("diurnal")
        ) * TO_MM_DAY
        vmin = mean.min().values
        vmax = mean.max().values

        bias = mean - mean.sel(source="real")
        bias_c48_real = mean_c48_real - mean.sel(source="real")

        ax[i, 0].set_title(f"{climate} C384")
        fv3viz.plot_cube(
            ds=GRID.merge(
                xr.Dataset({varname: mean.sel(source="real")}), compat="override"
            ),
            var_name=varname,
            ax=ax[i, 0],
            vmin=vmin,
            vmax=vmax,
        )
        ax[i, 1].set_title(f"{climate} C384 ML")
        fv3viz.plot_cube(
            ds=GRID.merge(
                xr.Dataset({varname: mean.sel(source="gen")}), compat="override"
            ),
            var_name=varname,
            ax=ax[i, 1],
            vmin=vmin,
            vmax=vmax,
        )

        def weighted_mean(da, weights):
            return (da * weights).mean() / weights.mean()

        def weighted_std(da, weights):
            return (((da ** 2) * weights).mean() / weights.mean()) ** 0.5

        # gen_bias_mean = bias.sel(source="gen").mean().values
        # gen_bias_std = bias.sel(source="gen").std().values
        # gen_bias_land_mean = bias.sel(source="gen").where(land_sea_mask).mean().values
        # gen_bias_land_std = bias.sel(source="gen").where(land_sea_mask).std().values
        # c48_bias_mean = bias_c48_real.mean().values
        # c48_bias_std = bias_c48_real.std().values
        # c48_bias_land_mean = bias_c48_real.where(land_sea_mask).mean().values
        # c48_bias_land_std = bias_c48_real.where(land_sea_mask).std().values

        gen_bias_mean = weighted_mean(bias.sel(source="gen"), AREA).values
        gen_bias_std = weighted_std(bias.sel(source="gen"), AREA).values
        gen_bias_land_mean = weighted_mean(
            bias.sel(source="gen"), AREA * land_sea_mask
        ).values
        gen_bias_land_std = weighted_std(
            bias.sel(source="gen"), AREA * land_sea_mask
        ).values
        c48_bias_mean = weighted_mean(bias_c48_real, AREA).values
        c48_bias_std = weighted_std(bias_c48_real, AREA).values
        c48_bias_land_mean = weighted_mean(bias_c48_real, AREA * land_sea_mask).values
        c48_bias_land_std = weighted_std(bias_c48_real, AREA * land_sea_mask).values

        bias_min = min(bias.sel(source="gen").min().values, bias_c48_real.min().values)
        bias_max = max(bias.sel(source="gen").max().values, bias_c48_real.max().values)
        bias_max = max(abs(bias_min), abs(bias_max))
        bias_min = -bias_max

        fv3viz.plot_cube(
            ds=GRID.merge(
                xr.Dataset({f"{varname}_gen_bias": bias.sel(source="gen")}),
                compat="override",
            ),
            var_name=f"{varname}_gen_bias",
            ax=ax[i, 2],
            vmin=bias_min,
            vmax=bias_max,
        )
        ax[i, 2].set_title("{} C384 ML Bias".format(climate))
        fv3viz.plot_cube(
            ds=GRID.merge(
                xr.Dataset({f"{varname}_c48_bias": bias_c48_real}), compat="override"
            ),
            var_name=f"{varname}_c48_bias",
            ax=ax[i, 3],
            vmin=bias_min,
            vmax=bias_max,
        )
        ax[i, 3].set_title("{} C48 Bias".format(climate))
        print(
            f"{climate} & {c48_bias_mean:.2e} & {gen_bias_mean:.2e} & {c48_bias_std:.2e} & {gen_bias_std:.2e} & {c48_bias_land_mean:.2e} & {gen_bias_land_mean:.2e} & {c48_bias_land_std:.2e} & {gen_bias_land_std:.2e} \\\\"
        )

    plt.tight_layout()
    fig.savefig(f"./plots/{label}-mean.png", dpi=100)


def plot_hist_all(ds, varname, label: str):
    fig, ax = plt.subplots(
        len(ds.perturbation), 2, figsize=(10, 1 + 2.5 * len(ds.perturbation)),
    )
    if len(ax.shape) == 1:
        ax = ax[None, :]
    for i, climate in enumerate(ds.perturbation.values):
        plot_cdf(ds.isel(perturbation=i), varname, ax=ax[i, 0])
        ax[i, 0].set_yscale("log")
        ax[i, 0].set_title(f"{climate} CDF\n" + ax[i, 0].get_title())
        ax[i, 0].set_ylim(1e-10, 1.0)
        plot_hist(ds.isel(perturbation=i), varname, ax=ax[i, 1])
        ax[i, 1].set_yscale("log")
        ax[i, 1].set_title(f"{climate} PDF\n" + ax[i, 1].get_title())
    plt.tight_layout()
    fig.savefig(f"./plots/{label}-histogram.png", dpi=100)


def plot_cdf_all(ds, varname, label: str):
    fig, ax = plt.subplots(2, 2, figsize=(10, 8),)
    ax = ax.flatten()
    for i, climate in enumerate(ds.perturbation.values):
        print(f"Getting CDF statistics for {climate}")
        plot_cdf(ds.isel(perturbation=i), varname, ax=ax[i])
        ax[i].set_yscale("log")
        ax[i].set_title(climate)
        n_samples = np.sum(ds[f"{varname}_hist"].sel(source="real", grid="C384").values)
        ax[i].set_ylim(1.0 / n_samples, 1.0)

    # find first bin with non-zero value
    first_bin = np.where(
        ds[f"{varname}_hist"].sum(["perturbation", "source", "grid"]).values.flatten()
        > 0
    )[0][0]
    # find last bin with non-zero value
    last_bin = (
        np.where(
            ds[f"{varname}_hist"]
            .sum(["perturbation", "source", "grid"])
            .values.flatten()
            > 0
        )[0][-1]
        + 1
    )
    edges = ds[f"{varname}_hist_bins"].values * TO_MM_DAY
    for i in range(4):
        ax[i].set_xlim(edges[first_bin], edges[last_bin])
    plt.tight_layout()
    fig.savefig(f"./plots/{label}-cdf.png", dpi=100)


def plot_hist_c384(ds, varname, label: str):
    fig, ax = plt.subplots(1, 2, figsize=(10, 1 + 2.5 * len(ds.perturbation)),)

    def norm_hist(ds, source, grid):
        base_hist = ds.sel(source=source, grid=grid)[f"{varname}_hist"].sum("diurnal")
        return base_hist.values.flatten() / base_hist.sum().values

    edges = ds[f"{varname}_hist_bins"].values
    edges = ds[f"{varname}_hist_bins"].values
    for i, climate in enumerate(ds.perturbation.values):
        ax[0].step(
            edges[:-1],
            norm_hist(ds.isel(perturbation=i), source="real", grid="C384"),
            where="post",
            alpha=0.5,
            label=climate,
        )
        ax[1].step(
            edges[:-1],
            norm_hist(ds.isel(perturbation=i), source="real", grid="C384"),
            where="post",
            alpha=0.5,
            label=climate,
        )
    ax[1].set_yscale("log")
    ax[0].set_title(f"C384 PRATEsfc")
    ax[1].set_title(f"C384 PRATEsfc (log)")
    plt.legend()
    plt.tight_layout()
    fig.savefig(f"./plots/{label}-histogram_c384.png", dpi=100)


def plot_hist(ds, varname, ax):
    def norm_hist(source, grid):
        base_hist = ds.sel(source=source, grid=grid)[f"{varname}_hist"].sum("diurnal")
        return base_hist.values.flatten() / base_hist.sum().values

    edges = ds[f"{varname}_hist_bins"].values * TO_MM_DAY

    ax.step(
        edges[:-1],
        norm_hist(source="real", grid="C48"),
        where="post",
        alpha=0.5,
        label="c48_real",
    )
    ax.step(
        edges[:-1],
        norm_hist(source="real", grid="C384"),
        where="post",
        alpha=0.5,
        label="c384_real",
    )
    ax.step(
        edges[:-1],
        norm_hist(source="gen", grid="C48"),
        where="post",
        alpha=0.5,
        label="c48_gen",
    )
    ax.step(
        edges[:-1],
        norm_hist(source="gen", grid="C384"),
        where="post",
        alpha=0.5,
        label=f"c384_gen",
    )
    # find first bin with non-zero value
    first_bin = np.where(
        ds[f"{varname}_hist"].sum(["diurnal", "source", "grid"]).values.flatten() > 0
    )[0][0]
    # find last bin with non-zero value
    last_bin = (
        np.where(
            ds[f"{varname}_hist"].sum(["diurnal", "source", "grid"]).values.flatten()
            > 0
        )[0][-1]
        + 1
    )
    ax.set_xlim(edges[first_bin], edges[last_bin])
    ax.legend(loc="upper left")
    ax.set_xlabel(varname)
    ax.set_ylabel("probability density")
    ax.set_title(f"{varname}")


def plot_cdf(ds, varname, ax):
    def norm_hist(source, grid):
        base_hist = ds.sel(source=source, grid=grid)[f"{varname}_hist"]
        return base_hist.values.flatten() / base_hist.sum().values

    def pct_error(source, grid, percentile):
        hist_pred = ds.sel(source=source, grid=grid)[f"{varname}_hist"].values
        hist_real = ds.sel(source="real", grid="C384")[f"{varname}_hist"].values
        return get_percentile_error(edges, hist_pred, hist_real, percentile)

    def cdf(source, grid):
        p_norm = norm_hist(source, grid)
        return 1.0 - np.concatenate([[0], np.cumsum(p_norm)])

    edges = ds[f"{varname}_hist_bins"].values * TO_MM_DAY

    ax.step(
        edges, cdf(source="real", grid="C48"), where="post", alpha=0.5, label="C48",
    )
    ax.step(
        edges, cdf(source="real", grid="C384"), where="post", alpha=0.5, label="C384",
    )
    ax.step(
        edges, cdf(source="gen", grid="C48"), where="post", alpha=0.5, label="c48_gen",
    )
    ax.step(
        edges,
        cdf(source="gen", grid="C384"),
        where="post",
        alpha=0.5,
        label=f"C384 (ML)",
    )
    err_99_c48 = pct_error(source="real", grid="C48", percentile=0.99)
    err_999_c48 = pct_error(source="real", grid="C48", percentile=0.999)
    err_9999_c48 = pct_error(source="real", grid="C48", percentile=0.9999)
    err_99_c384 = pct_error(source="gen", grid="C384", percentile=0.99)
    err_999_c384 = pct_error(source="gen", grid="C384", percentile=0.999)
    err_9999_c384 = pct_error(source="gen", grid="C384", percentile=0.9999)

    print(
        f" & {err_99_c48} & {err_99_c384} & {err_999_c48} & {err_999_c384} & {err_9999_c48} & {err_9999_c384} \\\\"
    )

    ax.legend(loc="upper right")
    ax.set_xlabel(varname)
    ax.set_ylabel("1 - CDF")
    ax.set_title(f"{varname}")


def plot_diurnal_means(ds, varname):
    for i, _ in enumerate(ds.perturbation.values):
        plot_diurnal_mean_for_climate(ds, i, varname)


def plot_diurnal_mean_for_climate(ds, i_perturbation, varname):
    climate = ds.perturbation.values[i_perturbation]
    ds = ds.isel(perturbation=i_perturbation)
    fig, ax = plt.subplots(
        len(ds.diurnal),
        3,
        figsize=(15, 1 + 2.5 * len(ds.diurnal)),
        subplot_kw={"projection": ccrs.Robinson()},
    )
    if len(ax.shape) == 1:
        ax = ax[None, :]
    for i_diurnal in range(len(ds.diurnal)):
        data = [
            ds.isel(diurnal=i_diurnal).sel(source="real", grid="C384")[
                f"{varname}_mean"
            ]
            * TO_MM_DAY,
            ds.isel(diurnal=i_diurnal).sel(source="gen", grid="C384")[f"{varname}_mean"]
            * TO_MM_DAY,
            ds.isel(diurnal=i_diurnal).sel(source="real", grid="C48")[f"{varname}_mean"]
            * TO_MM_DAY,
        ]
        vmin = min(d.min().values for d in data)
        vmax = max(d.max().values for d in data)
        for i_data, d in enumerate(data):
            fv3viz.plot_cube(
                ds=GRID.merge(xr.Dataset({varname: d}), compat="override"),
                var_name=varname,
                ax=ax[i_diurnal, i_data],
                vmin=vmin,
                vmax=vmax,
            )
    plt.tight_layout()
    fig.savefig(f"diurnal_{climate}.png", dpi=100)


def plot_diurnal_cycle(ds, initial_time, varname, label: str):
    # Sahel region of Africa
    # lon -8 to 35
    # lat -8 to 8
    # in_region = xr.ufuncs.logical_and(
    #     xr.ufuncs.logical_and(
    #         GRID["lon"] < 35,
    #         GRID["lon"] > -8,
    #     ),
    #     xr.ufuncs.logical_and(
    #         GRID["lat"] < 8,
    #         GRID["lat"] > -8,
    #     ),
    # ).astype(int)
    ds["PRATEsfc_var"] = ds["PRATEsfc_std"] ** 2
    diurnal_times = xr.DataArray(
        [initial_time + timedelta(hours=i) for i in range(0, 24, 3)],
        dims=["diurnal"],
        name="time",
    )
    local_time = vcm.local_time(
        GRID.merge(xr.Dataset({"time": diurnal_times}), compat="override"),
        time="time",
        lon_var="lon",
    )
    ds = (
        (ds * AREA)
        .where(land_sea_mask)
        .groupby_bins(local_time, bins=np.arange(0, 25, 3))
        .mean()
    ) / AREA.mean()
    data = ds[f"PRATEsfc_mean"] * TO_MM_DAY
    total_mean = data.mean("group_bins")
    total_std = ds["PRATEsfc_var"].mean("group_bins") ** 0.5 * TO_MM_DAY
    fig, ax = plt.subplots(
        len(ds.perturbation), 2, figsize=(10, 1 + 2.5 * len(ds.perturbation)),
    )
    if len(ax.shape) == 1:
        ax = ax[None, :]
    x = np.arange(0, 24, 24 / len(ds.group_bins))
    for i, climate in enumerate(ds.perturbation.values):
        real_c384 = dict(perturbation=climate, source="real", grid="C384")
        gen_c384 = dict(perturbation=climate, source="gen", grid="C384")
        real_c48 = dict(perturbation=climate, source="real", grid="C48")
        ax[i, 0].plot(
            x,
            data.sel(perturbation=climate, source="real", grid="C384").values,
            label="real",
        )
        ax[i, 0].plot(
            x,
            data.sel(perturbation=climate, source="gen", grid="C384").values,
            label="gen",
        )
        ax[i, 0].plot(
            x,
            data.sel(perturbation=climate, source="real", grid="C48").values,
            label="c48",
        )
        ax[i, 1].plot(
            x,
            (data.sel(**real_c384).values - total_mean.sel(**real_c384).values)
            / total_std.sel(**real_c384).values,
            label="real",
        )
        ax[i, 1].plot(
            x,
            (data.sel(**gen_c384).values - total_mean.sel(**gen_c384).values)
            / total_std.sel(**gen_c384).values,
            label="gen",
        )
        ax[i, 1].plot(
            x,
            (data.sel(**real_c48).values - total_mean.sel(**real_c48).values)
            / total_std.sel(**real_c48).values,
            label="c48",
        )
        # ax[i].plot(
        #     x,
        #     data.sel(perturbation=climate, source="quantile", grid="C384").values,
        #     label="quantile",
        # )
        for j in (0, 1):
            # ax[i].set_xticks(x)
            # ax[i].set_xticklabels([f"{i}:00" for i in range(0, 24, 3)])
            # ax[i].set_xlabel("local time")
            # ax[i].set_ylabel(varname)
            # ax[i].set_title(climate)
            # ax[i].legend()
            ax[i, j].set_xticks(x)
            ax[i, j].set_xticklabels([f"{i}:00" for i in range(0, 24, 3)])
            ax[i, j].set_xlabel("local time")
            ax[i, j].set_title(climate)
            ax[i, j].legend()
        ax[i, 0].set_ylabel(varname)
        ax[i, 1].set_ylabel(f"{varname} (normalized)")
    plt.tight_layout()
    fig.savefig(f"./plots/{label}-diurnal_cycle.png", dpi=100)


def save_to_netcdf(ds, filename):
    tmp_filename = "tmp_{perturbation}_{source}_{grid}.nc"
    for perturbation in ds.perturbation.values:
        for source in ds.source.values:
            for grid in ds.grid.values:
                this_filename = tmp_filename.format(
                    perturbation=perturbation, source=source, grid=grid
                )
                ds.sel(perturbation=perturbation, source=source, grid=grid).to_netcdf(
                    this_filename
                )
    datasets = {}
    # must call open_mfdataset only along one dimension, then merge the other two
    for perturbation in ds.perturbation.values:
        datasets[perturbation] = []
        for source in ds.source.values:
            ds1 = xr.open_mfdataset(
                tmp_filename.format(perturbation=perturbation, source=source, grid="*"),
                combine="nested",
                concat_dim="grid",
            )
            datasets[perturbation].append(ds1)
    perturbation_datasets = []
    for perturbation in ds.perturbation.values:
        perturbation_datasets.append(xr.concat(datasets[perturbation], dim="source"))
    ds = xr.concat(perturbation_datasets, dim="perturbation")
    ds.to_netcdf(filename)


class DatasetAggregator:
    def __init__(self):
        vmin = -0.001
        vmax = 0.0175
        bin_edges = np.linspace(vmin, vmax, 150)
        # TODO: time_to_diurnal doesn't remove time dimension, only converts it to days. handle this.
        self._dims = ("diurnal", "perturbation", "tile", "x", "y")
        self._mean_std_c48_real = MeanStdAggregator()
        self._mean_std_c48_gen = MeanStdAggregator()
        self._mean_std_c384_real = MeanStdAggregator()
        self._mean_std_c384_gen = MeanStdAggregator()
        self._histogram_c48_real = HistogramAggregator(bins=bin_edges)
        self._histogram_c48_gen = HistogramAggregator(bins=bin_edges)
        self._histogram_c384_real = HistogramAggregator(bins=bin_edges)
        self._histogram_c384_gen = HistogramAggregator(bins=bin_edges)

    def add(self, c48_real, c384_real, c48_gen, c384_gen):
        c48_real_diurnal = time_to_diurnal(c48_real)
        c384_real_diurnal = time_to_diurnal(c384_real)
        c48_gen_diurnal = time_to_diurnal(c48_gen)
        c384_gen_diurnal = time_to_diurnal(c384_gen)
        # time can be out-of-order because we reduce along it inside the aggregators
        assert tuple(c48_real_diurnal["PRATEsfc"].dims[1:]) == (
            "perturbation",
            "tile",
            "time",
            "x",
            "y",
        )
        assert tuple(c48_gen_diurnal["PRATEsfc"].dims[1:]) == (
            "perturbation",
            "time",
            "tile",
            "x",
            "y",
        )
        assert tuple(c384_real_diurnal["PRATEsfc"].dims[1:]) == (
            "perturbation",
            "tile",
            "time",
            "x",
            "y",
        )
        assert tuple(c384_gen_diurnal["PRATEsfc"].dims[1:]) == (
            "perturbation",
            "time",
            "tile",
            "x",
            "y",
        )
        self._mean_std_c48_real.add(c48_real_diurnal["PRATEsfc"])
        self._mean_std_c384_real.add(c384_real_diurnal["PRATEsfc"])
        self._mean_std_c48_gen.add(c48_gen_diurnal["PRATEsfc"])
        self._mean_std_c384_gen.add(c384_gen_diurnal["PRATEsfc"])
        c48_real_daily = c48_real.coarsen(time=8).mean()
        c384_real_daily = c384_real.coarsen(time=8).mean()
        c48_gen_daily = c48_gen.coarsen(time=8).mean()
        c384_gen_daily = c384_gen.coarsen(time=8).mean()
        self._histogram_c48_real.add(c48_real_daily["PRATEsfc"])
        self._histogram_c384_real.add(c384_real_daily["PRATEsfc"])
        self._histogram_c48_gen.add(c48_gen_daily["PRATEsfc"])
        self._histogram_c384_gen.add(c384_gen_daily["PRATEsfc"])

    def get_dataset(self) -> xr.Dataset:
        mean_c48_real = xr.DataArray(
            self._mean_std_c48_real.get_mean(), dims=self._dims
        )
        std_c48_real = xr.DataArray(self._mean_std_c48_real.get_std(), dims=self._dims)
        mean_c384_real = xr.DataArray(
            self._mean_std_c384_real.get_mean(), dims=self._dims
        )
        std_c384_real = xr.DataArray(
            self._mean_std_c384_real.get_std(), dims=self._dims
        )
        mean_c48_gen = xr.DataArray(self._mean_std_c48_gen.get_mean(), dims=self._dims)
        std_c48_gen = xr.DataArray(self._mean_std_c48_gen.get_std(), dims=self._dims)
        mean_c384_gen = xr.DataArray(
            self._mean_std_c384_gen.get_mean(), dims=self._dims
        )
        std_c384_gen = xr.DataArray(self._mean_std_c384_gen.get_std(), dims=self._dims)
        histogram_c48_real = xr.DataArray(
            self._histogram_c48_real.get(), dims=["perturbation", "hist"]
        )
        histogram_c384_real = xr.DataArray(
            self._histogram_c384_real.get(), dims=["perturbation", "hist"]
        )
        histogram_c48_gen = xr.DataArray(
            self._histogram_c48_gen.get(), dims=["perturbation", "hist"]
        )
        histogram_c384_gen = xr.DataArray(
            self._histogram_c384_gen.get(), dims=["perturbation", "hist"]
        )
        c48_real = xr.Dataset(
            {
                "PRATEsfc_mean": mean_c48_real,
                "PRATEsfc_std": std_c48_real,
                "PRATEsfc_hist": histogram_c48_real,
            }
        )
        c384_real = xr.Dataset(
            {
                "PRATEsfc_mean": mean_c384_real,
                "PRATEsfc_std": std_c384_real,
                "PRATEsfc_hist": histogram_c384_real,
            }
        )
        c48_gen = xr.Dataset(
            {
                "PRATEsfc_mean": mean_c48_gen,
                "PRATEsfc_std": std_c48_gen,
                "PRATEsfc_hist": histogram_c48_gen,
            }
        )
        c384_gen = xr.Dataset(
            {
                "PRATEsfc_mean": mean_c384_gen,
                "PRATEsfc_std": std_c384_gen,
                "PRATEsfc_hist": histogram_c384_gen,
            }
        )
        c48 = xr.concat([c48_real, c48_gen], dim="source").assign_coords(
            source=["real", "gen"]
        )
        c384 = xr.concat([c384_real, c384_gen], dim="source").assign_coords(
            source=["real", "gen"]
        )
        ds = xr.concat([c48, c384], dim="grid").assign_coords(grid=["C48", "C384"])
        ds["PRATEsfc_hist_bins"] = (("hist_bins",), self._histogram_c48_real.bins)
        return ds


if __name__ == "__main__":
    fv3fit.set_random_seed(0)
    CHECKPOINT_PATH = "gs://vcm-ml-experiments/cyclegan/checkpoints/c48_to_c384/"
    EVALUATE_ON_TRAIN = False
    # BASE_NAME, EPOCH = (
    #     "20230208-183103-cdda934c", 17  # precip-only, properly normalized, +45 epochs
    #     # "20230217-220447-a693c405", 62  # lr=2e-6, xyz features
    # )
    for BASE_NAME, label, EPOCH in [
        # climate-normalized models
        # ("20230208-183103-cdda934c", "no_xyz-e45-e17", 17),  # precip-only, properly normalized, +45 epochs
        # https://wandb.ai/ai2cm/cyclegan_c48_to_c384/runs/u8wdoeam for 45+17 run
        # ("20230217-220447-a693c405", "xyz-lr-2e-6-e78", 78),  # lr=2e-6, xyz features
        # ("20230217-220447-a693c405", "xyz-lr-2e-6-e25", 25),  # lr=2e-6, xyz features
        # ("20230217-220543-6e87853c", "xyz-lr-2e-5-e07", 7),  # lr=2e-5, xyz features
        # ("20230217-220543-6e87853c", "xyz-lr-2e-5-e79", 79),  # lr=2e-5, xyz features
        # ("20230217-220450-264af48e", "xyz-lr-2e-4-e04", 4),  # lr=2e-4, xyz features
        # ("20230217-220450-264af48e", "xyz-lr-2e-4-e77", 77),  # lr=2e-4, xyz features
        # ("20230227-184934-4f398f77", "xyz-2e-5-e80-2e-6-e32", 32),
        # ("20230227-184948-b84d7a60", "xyz-2e-5-e07-2e-6-e32", 32),
        # ("20230227-185639-f6bd7c4c", "xyz-2e-6-e17-2e-7-e32", 32),
        # no-demean models
        # ("20230302-000418-3c10c358", "no-demean-2e-5-e75", 75),
        # ("20230302-000015-699b0906", "no-demean-2e-6-e75", 75),
        # no climate-normalization models
        # ("20230303-203000-7ddd1a30", "denorm-2e-6-e65", 65),
        # ("20230303-203211-7bc0b30b", "denorm-2e-5-e12", 12),
        # ("20230303-203211-7bc0b30b", "denorm-2e-5-e63", 63),
        # ("20230303-203310-73494d8b", "denorm-2e-5-3x3-e67", 67),
        # ("20230303-203306-f753d490", "denorm-2e-6-3x3-e69", 69),
        # ("20230308-231603-db43b4ab", "denorm-diurnal-2e-5-3x3-e06", 6),
        # ("20230308-232851-15c84e3f", "denorm-diurnal-2e-6-3x3-e05", 5),
        # new-data models
        # ("20230308-221136-b82ccf7f", "march-diurnal-2e-6-3x3-e14", 14),
        # ("20230308-221136-b82ccf7f", "march-diurnal-2e-6-3x3-e33", 33),
        # ("20230313-211437-4a624ff9", "march-diurnal-2e-6-3x3-train-as-val-e06", 6),
        # ("20230313-211452-034a9ec9", "march-diurnal-2e-5-3x3-train-as-val-e06", 6),
        # ("20230313-211352-c94b7408", "march-2e-6-geo-bias-e07", 7),
        # ("20230314-214027-54366191", "lr-1e-4-decay-0.63096", 24),
        # ("20230314-213709-fc95b736", "lr-1e-4-decay-0.79433", 24),
        # ("20230314-214051-25b2a902", "lr-1e-3-decay-0.63096", 24),
        # ("20230316-151658-9017348e", "lr-1e-4-decay-0.79433-no-geo-bias", 35),
        # ("20230316-144944-b3143932", "lr-1e-4-decay-0.89125", 35),
        # ("20230316-182507-2cfb6254", "lr-1e-6-decay-0.89125", 35),
        # ("20230316-182557-17cdb0a6", "lr-1e-5-decay-0.89125", 34),
        # ("20230314-213709-fc95b736", "lr-1e-4-decay-0.79433", 50),
        # ("20230314-214027-54366191", "lr-1e-4-decay-0.63096", 50),
        # ("20230314-214051-25b2a902", "lr-1e-3-decay-0.63096", 50),
        # multi-climate new-data models
        ("20230329-221949-9d8e8abc", "prec-lr-1e-4-decay-0.63096-full", 16),
        ("20230330-174749-899f5c19", "prec-lr-1e-5-decay-0.63096-full", 23),
    ]:
        label = label + f"-e{EPOCH:02d}"
        fv3fit.set_random_seed(0)
        print(f"Loading {BASE_NAME} epoch {EPOCH}")
        cyclegan: fv3fit.pytorch.CycleGAN = fv3fit.load(
            # "gs://vcm-ml-experiments/cyclegan/checkpoints/c48_to_c384/20230130-231729-82b939d9-epoch_075/"  # precip-only
            # "gs://vcm-ml-experiments/cyclegan/checkpoints/c48_to_c384/20230202-233100-c5d574a4-epoch_045/"  # precip-only, properly normalized
            # "gs://vcm-ml-experiments/cyclegan/checkpoints/c48_to_c384/20230208-183103-cdda934c-epoch_017/"  # precip-only, properly normalized, +45 epochs
            # "gs://vcm-ml-experiments/cyclegan/checkpoints/c48_to_c384/20230217-220447-a693c405-epoch_040/"  # lr=2e-6, xyz features
            CHECKPOINT_PATH
            + BASE_NAME
            + f"-epoch_{EPOCH:03d}/"
        ).to(DEVICE)
        VARNAME = "PRATEsfc"
        initial_time = xr.open_zarr(
            "gs://vcm-ml-experiments/mcgibbon/2023-03-29/fine-combined.zarr"
        )["time"][0].values

        if EVALUATE_ON_TRAIN:
            BASE_NAME = "train-" + BASE_NAME
            label = "train-" + label

        PROCESSED_FILENAME = f"./processed/processed-agg-{BASE_NAME}-e{EPOCH}.nc"
        # label = "subset_5-" + label

        def plot(ds):
            # plot_mean_all(ds, VARNAME)
            print("plotting diurnal cycle")
            plot_diurnal_cycle(ds, initial_time, VARNAME, label)
            # print("plotting c384 histograms")
            # plot_hist_c384(ds, VARNAME, label)
            print("plotting cdfs")
            plot_cdf_all(ds, VARNAME, label)
            print("plotting means")
            plot_mean_all(ds, VARNAME, label)
            plt.close("all")
            # print("plotting diurnal means")
            # plot_diurnal_means(ds, VARNAME)

        if not os.path.exists(PROCESSED_FILENAME):
            print(f"Calculating processed data for {PROCESSED_FILENAME}")
            # c384_real_all: xr.Dataset = (
            #     xr.open_zarr(f"./fine-combined{data_suffix}.zarr/").rename(
            #         {"grid_xt": "x", "grid_yt": "y"}
            #     )
            # )  # .isel(time=slice(0, None, 5))
            # c48_real_all: xr.Dataset = (
            #     xr.open_zarr(f"./coarse-combined{data_suffix}.zarr/").rename(
            #         {"grid_xt": "x", "grid_yt": "y"}
            #     )
            # )  # .isel(time=slice(0, None, 5))
            c384_real_all: xr.Dataset = (
                xr.open_zarr(
                    "gs://vcm-ml-experiments/mcgibbon/2023-03-29/fine-combined.zarr"
                ).rename({"grid_xt": "x", "grid_yt": "y"})
            )  # .isel(time=slice(0, 60))  # .isel(time=slice(0, None, 5))
            c48_real_all: xr.Dataset = (
                xr.open_zarr(
                    "gs://vcm-ml-experiments/mcgibbon/2023-03-29/coarse-combined.zarr"
                ).rename({"grid_xt": "x", "grid_yt": "y"})
            )  # .isel(time=slice(0, 60))  # .isel(time=slice(0, None, 5))
            # C48_I_TRAIN_END = 30
            # C384_I_TRAIN_END = 30
            if not EVALUATE_ON_TRAIN:
                c48_real_in = c48_real_all.isel(
                    time=slice(C48_I_TRAIN_END, None)
                ).transpose(..., "x", "y")
                c384_real_in = c384_real_all.isel(
                    time=slice(C384_I_TRAIN_END, None)
                ).transpose(..., "x", "y")
            else:
                c48_real_in = c48_real_all.isel(
                    time=slice(None, C48_I_TRAIN_END)
                ).transpose(..., "x", "y")
                c384_real_in = c384_real_all.isel(
                    time=slice(None, C384_I_TRAIN_END)
                ).transpose(..., "x", "y")
            aggregator = DatasetAggregator()
            nt_bin = 73 * 8
            nt_final = len(c48_real_in.time) // nt_bin * nt_bin
            assert nt_bin % 8 == 0  # bins must be whole days, or silent bugs occur
            for i_time in range(0, nt_final, nt_bin):
                print(f"Predicting {i_time} / {len(c48_real_in.time)}")
                c48_real = c48_real_in.isel(time=slice(i_time, i_time + nt_bin))
                c384_real = c384_real_in.isel(time=slice(i_time, i_time + nt_bin))
                c48_gen, c384_gen = predict_dataset(c48_real, c384_real, cyclegan,)
                aggregator.add(c48_real, c384_real, c48_gen, c384_gen)
                ds = aggregator.get_dataset()
                ds = ds.assign_coords(perturbation=c48_real_in.perturbation)
                plot(ds)
            # c48_gen, c384_gen = predict_dataset(
            #     c48_real_in,
            #     c384_real_in,
            #     cyclegan,
            # )
            # ds = process_dataset(c48_real_all, c384_real_all, c48_gen, c384_gen)
            ds = aggregator.get_dataset()
            ds = ds.assign_coords(perturbation=c48_real_in.perturbation)
            ds.to_netcdf(PROCESSED_FILENAME)
        else:
            print(f"Loading processed data from {PROCESSED_FILENAME}")

        ds = xr.open_dataset(PROCESSED_FILENAME)
        plot(ds)
