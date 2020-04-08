import matplotlib
import matplotlib.pyplot as plt
import os
import vcm

from vcm.cubedsphere.constants import GRID_VARS


matplotlib.use("Agg")


def plot_metrics(ds_metrics, output_dir, dpi_figures):
    report_sections = {}
    # R^2 vs pressure
    _plot_r2_pressure_profile(ds_metrics).savefig(
        os.path.join(output_dir, f"r2_pressure_levels.png"),
        dpi=dpi_figures["R2_pressure_profiles"],
    )
    report_sections["R^2 vs pressure levels"] = ["r2_vs_pressure_levels.png"]

    # RMSE maps
    report_sections["Root mean squared error maps"] = []
    for var in ["net_precipitation", "net_heating"]:
        for target_dataset_name in ds_metrics.target_dataset_names.values:
            filename = f"rmse_map_{var}_{target_dataset_name}.png"
            _plot_rmse_map(ds_metrics, var, target_dataset_name).savefig(
                os.path.join(output_dir, filename), dpi=dpi_figures["map_plot_single"]
            )
            report_sections["Root mean squared error maps"].append(filename)

    return report_sections


def _plot_rmse_map(ds, var, target_dataset_name):
    plt.close("all")
    data_var = f"rmse_{var}_vs_{target_dataset_name}"
    fig = vcm.plot_cube(
        vcm.mappable_var(ds[GRID_VARS + [data_var]], data_var), vmin=0, vmax=2
    )[0]
    return fig


def _plot_r2_pressure_profile(ds):
    plt.close("all")
    fig = plt.figure()
    for surface, surface_line in zip(["global", "land", "sea"], ["-", ":", "--"]):
        for var, var_color in zip(["dQ1", "dQ2"], ["orange", "blue"]):
            plt.plot(
                ds["pressure"],
                ds[f"r2_{var}_pressure_levels_{surface}"],
                color=var_color,
                linestyle=surface_line,
                label=f"{var}, {surface}",
            )
    plt.xlabel("pressure [HPa]")
    plt.ylabel("$R^2$")
    plt.legend()
    return fig
