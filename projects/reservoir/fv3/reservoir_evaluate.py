# takes prediction and calculates evaluation scores
# todo apply land sea mask
import numpy as np
import xarray as xr
from sklearn.metrics import r2_score
import argparse
import glob
import os
import matplotlib.pyplot as plt
import csv
import vcm

"""example call:
python reservoir_evaluate.py
--data_path /home/paulah/data/era5/fv3-halo-0-masked/val
--input_path /home/paulah/fv3net-offline-reservoirs/hybrid-full-sub-24-halo-4-masked
--n_synchronize 200"""

DELTA_T = 604800  # 7 days in seconds


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path", help="path to data directory, should contain subfolders /tile-x"
    )
    parser.add_argument("--input_path", help="path to save predictions")
    parser.add_argument("--n_synchronize", type=int, default=100)
    parser.add_argument("--plots_or_not", type=str, default="no_plots")
    return parser.parse_args()


def main(args):
    # load data
    dataset = xr.concat(
        [
            xr.open_dataset(
                glob.glob(os.path.join(args.data_path, f"tile-{r}/*.nc"))[0]
            )
            for r in range(6)
        ],
        dim="tile",
    )
    dataset = dataset.transpose("time", "tile", "y", "x")
    # set mask TRue if mask_field is 0 and False if mask_field is 1
    dataset["mask_field"] = dataset.mask_field == 0

    # load single step and rollout predictions
    single_step_prediction = xr.open_dataset(
        args.input_path + "/single_step_prediction.nc"
    )
    rollout_prediction = xr.open_dataset(args.input_path + "/rollout.nc")

    # calculate tendencies need exactly the length of target tendency
    single_step_prediction_tendency = (
        dataset.sst.shift(time=-1) - single_step_prediction.sst
    ) / DELTA_T
    # for rollout tendency use rollout prediction to calculate
    rollout_prediction_tendency = (
        rollout_prediction.sst.shift(time=-1) - rollout_prediction.sst
    ) / DELTA_T

    # cases
    cases = ["single_step", "rollout", "single_step_tend", "rollout_tend"]
    trues = [
        dataset.sst.isel(time=slice(1, None)),
        dataset.sst.isel(time=slice(args.n_synchronize + 1, None)),
        dataset.sst_tendency.isel(time=slice(1, -1)),
        dataset.sst_tendency.isel(time=slice(args.n_synchronize, -2)),
    ]
    predictions = [
        single_step_prediction.sst.isel(time=slice(None, -1)),
        rollout_prediction.sst.isel(time=slice(None, -1)),
        single_step_prediction_tendency.isel(time=slice(1, -1)),
        rollout_prediction_tendency.isel(time=slice(None, -2)),
    ]

    # calculate scores function on four cases
    scores = []
    for case, true, pred in zip(cases, trues, predictions):
        if "tend" in case:
            scores.append(
                calculate_scores(
                    DELTA_T * true,
                    DELTA_T * pred,
                    dataset.mask_field[: len(true), ...],
                    case,
                )
            )
        else:
            scores.append(
                calculate_scores(true, pred, dataset.mask_field[: len(true), ...], case)
            )

    # combine all scores into one dictionary
    single_step_scores = scores[0]
    rollout_scores_full = scores[1]
    single_step_scores_tend = scores[2]
    rollout_scores_tend = scores[3]

    # combine single_step_scores, rollout_scores_full,
    # single_step_scores_tend, rollout_scores_tend into one dictionary
    combined_scores = {
        **single_step_scores,
        **rollout_scores_full,
        **single_step_scores_tend,
        **rollout_scores_tend,
    }
    # save scores as csv
    w = csv.writer(open(args.input_path + "/scores.csv", "w"))
    # loop over dictionary keys and values
    for key, val in combined_scores.items():
        # write every key and value to file
        w.writerow([key, val])

    # create plots
    # check arggument if we want to create plots
    if args.plots_or_not == "plots":
        plot_path = args.input_path + "/plots"
        if not os.path.exists(plot_path):
            os.makedirs(plot_path)
        # go through four cases single_step, rollout, single_step_tend, rollout_tend
        for case, true, pred in zip(cases, trues, predictions):
            create_timeseries_plots(
                true, pred, dataset.mask_field[: len(true), ...], case, plot_path
            )
            create_spatial_mean_plots(
                true, pred, dataset.mask_field[: len(true), ...], case, plot_path
            )
            create_spatial_r2_plot(
                true, pred, dataset.mask_field[: len(true), ...], case, plot_path
            )
            create_spatial_error_mean_plots(
                true, pred, dataset.mask_field[: len(true), ...], case, plot_path
            )
            create_spatial_abs_error_mean_plots(
                true, pred, dataset.mask_field[: len(true), ...], case, plot_path
            )
            if "tend" not in case:
                create_oni_plot(true, pred, case, plot_path)


def swap_axes(data):
    """ swap axes from (time, tile, y, x) to (tile, time, x y)"""
    return np.swapaxes(data, 0, 1)


def calculate_scores(target, pred, mask, case):
    """ calculate scores for target and pred"""
    # dimensions of target and pred are (time, tile, x, y)
    # make sure to mask out land
    target = target.data
    pred = pred.data
    scores = {}
    squared_difference_masked = np.ma.array((target - pred) ** 2, mask=mask)
    difference_masked = np.ma.array(pred - target, mask=mask)
    absolute_difference_masked = np.ma.array(np.abs(target - pred), mask=mask)
    scores[case + "_mse"] = np.mean(squared_difference_masked)
    scores[case + "_rmse"] = np.sqrt(scores[case + "_mse"])
    scores[case + "_mae"] = np.mean(absolute_difference_masked)
    scores[case + "_mean_bias"] = np.mean(difference_masked)
    # calculate r2 score over timeseries, ignoring nans
    r2_scores = []
    for tile in range(6):
        # initialze array to store r2 scores for each tile
        area_r2_scores = np.zeros((target.shape[-1], target.shape[-1]))
        for i in range(target.shape[-1]):
            for j in range(target.shape[-1]):
                if not mask[0, tile, i, j]:
                    # to make it more interpretable we set r2 scores below -1 to -1
                    area_r2_scores[i, j] = max(
                        r2_score(target[:, tile, i, j], pred[:, tile, i, j],), -1
                    )
        r2_scores.append(np.mean(np.ma.array(area_r2_scores, mask=mask[0, tile, ...])))
    # mean over tiles
    scores[case + "_r2"] = np.mean(r2_scores)

    # add scores per tile
    scores[case + "_mse_tile"] = np.mean(squared_difference_masked, axis=(0, 2, 3))
    scores[case + "_rmse_tile"] = np.sqrt(scores[case + "_mse_tile"])
    scores[case + "_mae_tile"] = np.mean(absolute_difference_masked, axis=(0, 2, 3))
    scores[case + "_mean_bias_tile"] = np.mean(difference_masked, axis=(0, 2, 3))
    scores[case + "_r2_tile"] = np.array(r2_scores)
    return scores


def calculate_spatial_r2(target, pred, mask):
    spatial_r2 = np.zeros((6, target.shape[-1], target.shape[-1]))
    for tile in range(6):
        # initialze array to store r2 scores for each tile

        for i in range(target.shape[-1]):
            for j in range(target.shape[-1]):
                if not mask[0, tile, i, j]:
                    # to make it more interpretable we set r2 scores below -1 to -1
                    spatial_r2[tile, i, j] = max(
                        r2_score(target[:, tile, i, j], pred[:, tile, i, j],), -1
                    )

    return spatial_r2


def create_timeseries_plots(true, pred, mask, case, path):
    # this function is going to be called four times for every case
    # create prediction plots
    true = true.data
    pred = pred.data
    create_timeseries_mean_plots(true, pred, mask, case, path)
    create_timeseries_random_plots(true, pred, mask, case, path)
    # create error plots
    error = true - pred
    create_error_timeseries_mean_plots(error, mask, case, path)
    pass


def create_timeseries_mean_plots(true, pred, mask, case, path):
    # make plot with 7 subplots, one global plot and 6 tiles
    fig, axs = plt.subplots(7, 1, figsize=(10, 15))
    # plot global
    axs[0].plot(true.mean(axis=(1, 2, 3)), label="true")
    axs[0].plot(pred.mean(axis=(1, 2, 3)), label="pred")
    # add legend
    axs[0].legend()
    # add title
    axs[0].set_title("Global mean")
    # plot tiles
    for tile in range(6):
        axs[tile + 1].plot(true[:, tile, :, :].mean(axis=(1, 2)), label="true")
        axs[tile + 1].plot(pred[:, tile, :, :].mean(axis=(1, 2)), label="pred")
        # add title
        axs[tile + 1].set_title("Tile " + str(tile))
        # add legend
        axs[tile + 1].legend()
    # save plot
    plt_path = path + "/mean_timeseries_" + case + ".png"
    plt.savefig(plt_path)


def create_error_timeseries_mean_plots(error, mask, case, path):
    # smae plot as create_timeseries_mean_plots but with error
    # make plot with 7 subplots, one global plot and 6 tiles
    fig, axs = plt.subplots(7, 1, figsize=(10, 15))
    # plot global error
    axs[0].plot(error.mean(axis=(1, 2, 3)), label="error")
    # add title
    axs[0].set_title("Global mean error")
    # add legend
    axs[0].legend()
    # plot tiles erros
    for tile in range(6):
        axs[tile + 1].plot(error[:, tile, :, :].mean(axis=(1, 2)), label="error")
        # add title
        axs[tile + 1].set_title("Tile " + str(tile))
        # add legend
        axs[tile + 1].legend()
    # save plot
    plt_path = path + "/mean_error_timeseries_" + case + ".png"
    plt.savefig(plt_path)


def create_timeseries_random_plots(true, pred, mask, case, path):
    # make plot with 7 subplots, one global plot and 6 tiles
    fig, axs = plt.subplots(6, 2, figsize=(13, 20))

    # plot tiles
    for tile in range(6):
        # choose a random location todo: make sure it is not land
        y = np.random.randint(0, true.shape[-2])
        x = np.random.randint(0, true.shape[-1])
        while mask[0, tile, x, y]:
            y = np.random.randint(0, true.shape[-2])
            x = np.random.randint(0, true.shape[-1])
        # plot timeseires at random location
        axs[tile, 0].plot(true[:, tile, x, y], label="true")
        axs[tile, 0].plot(pred[:, tile, x, y], label="pred")

        # add title that shows tile number and location
        axs[tile, 0].set_title("Tile " + str(tile) + " at location " + str((x, y)))

        # add legend
        axs[tile, 0].legend()

        y = np.random.randint(0, true.shape[-2])
        x = np.random.randint(0, true.shape[-1])
        while mask[0, tile, x, y]:
            y = np.random.randint(0, true.shape[-2])
            x = np.random.randint(0, true.shape[-1])
        # plot random timeseries
        axs[tile, 1].plot(true[:, tile, x, y], label="true")
        axs[tile, 1].plot(pred[:, tile, x, y], label="pred")
        # add title
        axs[tile, 1].set_title("Tile " + str(tile) + " at location " + str((x, y)))
        # add legend
        axs[tile, 1].legend()
    # save plot
    plt_path = path + "/random_timeseries_" + case + ".png"
    plt.savefig(plt_path)


def create_spatial_mean_plots(true_dataset, pred_dataset, mask, case, path):
    # make plot with 2 subplots
    fig, axs = plt.subplots(2, 1, figsize=(10, 15))
    # plot true
    plt.subplot(2, 1, 1)
    vcm.cubedsphere.to_cross(true_dataset.mean("time"), x="x", y="y").plot(
        cmap="coolwarm"
    )
    plt.title("true")
    # plot pred
    plt.subplot(2, 1, 2)
    vcm.cubedsphere.to_cross(pred_dataset.mean("time"), x="x", y="y").plot(
        cmap="coolwarm"
    )
    plt.title("pred")
    # save plot
    plt_path = path + "/spatial_mean_" + case + ".png"
    plt.savefig(plt_path)
    plt.close()


def create_spatial_r2_plot(true_dataset, pred_dataset, mask, case, path):
    # what colormap should i use for r2 score?
    # cmap = plt.cm.get_cmap('RdBu')
    # make one plot
    r2_scores = pred_dataset.copy()
    r2_scores.data = calculate_spatial_r2(true_dataset.data, pred_dataset.data, mask)[
        np.newaxis, ...
    ].repeat(true_dataset.shape[0], axis=0)
    plt.figure(figsize=(16, 10))
    vcm.cubedsphere.to_cross(r2_scores.mean("time"), x="x", y="y").plot(
        cmap="RdBu", vmin=-1, vmax=1
    )

    plt_path = path + "/spatial_r2_" + case + ".png"
    plt.savefig(plt_path)
    plt.close()


def create_spatial_error_mean_plots(true_dataset, pred_dataset, mask, case, path):
    plt.figure(figsize=(16, 10))
    vcm.cubedsphere.to_cross(
        true_dataset.mean(dim="time") - pred_dataset.mean(dim="time"), x="x", y="y"
    ).plot()

    plt.show()
    plt.savefig(path + "/spatial_error_mean_" + case + ".png")
    plt.close()


def create_spatial_abs_error_mean_plots(true_dataset, pred_dataset, mask, case, path):
    abs_mean = pred_dataset.copy()
    abs_mean.data = np.abs(true_dataset.data - pred_dataset.data)
    plt.figure(figsize=(16, 10))
    vcm.cubedsphere.to_cross(abs_mean.mean(dim="time"), x="x", y="y").plot(cmap="Reds")

    plt.show()
    plt.savefig(path + "/spatial_abs_error_mean_" + case + ".png")
    plt.close()


def create_spatial_error_random_plots(true_dataset, pred_dataset, mask, case, path):
    pass


def create_oni_plot(true, pred, case, path):
    # calculate 3 month roling average
    weeks_window = 13  # 13 weeks in 3 months
    # set 30 year sst average over nino 3.4 region
    # (from the last 30 years of training data)
    thirty_year_average = 299.92
    true = true.data
    pred = pred.data

    three_month_rolling_avergage_prediction = [
        (
            np.nanmean(pred[i : i + weeks_window, 3, 33:, 20:28])
            + np.nanmean(pred[i : i + weeks_window, 4, :14, 20:28])
        )
        * 0.5
        for i in range(true.shape[0] - weeks_window)
    ]
    three_month_rolling_avergage_true = [
        (
            np.nanmean(true[i : i + weeks_window, 3, 33:, 20:28])
            + np.nanmean(true[i : i + weeks_window, 4, :14, 20:28])
        )
        * 0.5
        for i in range(true.shape[0] - weeks_window)
    ]
    fig, axs = plt.subplots(1, 1, figsize=(15, 10))
    plt.plot(
        -(
            np.ones((true.shape[0] - weeks_window,)) * thirty_year_average
            - np.array(three_month_rolling_avergage_true)
        ),
        label="ERA5",
        linewidth=2,
    )
    plt.plot(
        -(
            np.ones((true.shape[0] - weeks_window,)) * thirty_year_average
            - np.array(three_month_rolling_avergage_prediction)
        ),
        label="Reservoir rollout",
        linewidth=2,
    )

    plt.plot(np.zeros((true.shape[0] - weeks_window,)), color="black", linestyle="--")
    plt.plot(
        0.5 * np.ones((true.shape[0] - weeks_window,)), color="gray", linestyle="--"
    )
    plt.plot(
        -0.5 * np.ones((true.shape[0] - weeks_window,)), color="gray", linestyle="--"
    )
    plt.ylim(-2.5, 2.5)
    plt.ylabel("Oceanic Nino Index", fontsize=20)
    plt.xlabel("Weeks into validation period (2010-2014)", fontsize=20)
    plt.legend(fontsize=20)
    # save plot
    plt_path = path + "/oni_plot_" + case + ".png"
    plt.savefig(plt_path)
    plt.close()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)