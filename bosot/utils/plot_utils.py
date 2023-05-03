# Copyright (c) 2023 Robert Bosch GmbH
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

# Author: Matthias Bitzer, matthias.bitzer3@de.bosch.com
from gpflow.utilities.utilities import print_summary
import numpy as np
from numpy.lib.shape_base import expand_dims
import pandas as pd
from pandas.plotting import scatter_matrix
import os
import matplotlib.pyplot as plt
import matplotlib
from bosot.models.gp_model import GPModel
from bosot.utils.plotter import Plotter
from bosot.utils.plotter2D import Plotter2D


def active_learning_nd_plot(x_data, y_data, save_plot=False, file_name=None, file_path=None):
    column_names = ["x" + str(i) for i in range(1, x_data.shape[1] + 1)] + ["y"]
    data = np.concatenate((x_data, y_data), axis=1)
    df = pd.DataFrame(data=data, columns=column_names)
    scatter_matrix(df, alpha=1.0, figsize=(6, 6), diagonal="kde")
    if save_plot:
        plt.savefig(os.path.join(file_path, file_name))
        plt.close()
    else:
        plt.show()


def active_learning_1d_plot(
    x_grid,
    pred_mu_grid,
    pred_sigma_grid,
    x_data,
    y_data,
    x_query,
    y_query,
    gt_available=False,
    gt_x=None,
    gt_f=None,
    save_plot=False,
    file_name=None,
    file_path=None,
):
    plotter_object = Plotter(1)
    if gt_available:
        plotter_object.add_gt_function(np.squeeze(gt_x), np.squeeze(gt_f), "black", 0)
    plotter_object.add_datapoints(np.squeeze(x_data), np.squeeze(y_data), "r", 0)
    plotter_object.add_datapoints(x_query, y_query, "green", 0)
    plotter_object.add_predictive_dist(np.squeeze(x_grid), np.squeeze(pred_mu_grid), np.squeeze(pred_sigma_grid), 0)
    if save_plot:
        plotter_object.save_fig(file_path, file_name)
    else:
        plotter_object.show()


def active_learning_1d_plot_multioutput(
    x_grid, pred_mu_grid, pred_sigma_grid, x_data, y_data, x_query, y_query, save_plot=False, file_name=None, file_path=None
):
    output_dim = y_data.shape[1]
    assert output_dim == pred_mu_grid.shape[1]
    plotter_object = Plotter(output_dim)
    for m in range(0, output_dim):
        plotter_object.add_datapoints(np.squeeze(x_data), np.squeeze(y_data[:, m]), "r", m)
        plotter_object.add_datapoints(x_query, y_query[m], "green", m)
        plotter_object.add_predictive_dist(np.squeeze(x_grid), np.squeeze(pred_mu_grid[:, m]), np.squeeze(pred_sigma_grid[:, m]), m)
    if save_plot:
        plotter_object.save_fig(file_path, file_name)
    else:
        plotter_object.show()


def active_learning_2d_plot(
    x_grid, acquisition_values_grid, pred_mu_grid, y_over_grid, x_data, x_query, save_plot=False, file_name=None, file_path=None
):
    plotter_object = Plotter2D(3)
    plotter_object.add_gt_function(x_grid, np.squeeze(acquisition_values_grid), "RdBu_r", 14, 0)
    plotter_object.add_datapoints(x_data, "black", 0)
    if len(x_query.shape) == 1:
        x_query = np.expand_dims(x_query, axis=0)

    plotter_object.add_datapoints(x_query, "green", 0)
    min_y = np.min(y_over_grid)
    max_y = np.max(y_over_grid)
    levels = np.linspace(min_y, max_y, 100)
    plotter_object.add_gt_function(x_grid, np.squeeze(y_over_grid), "seismic", levels, 1)
    plotter_object.add_datapoints(x_data, "black", 1)
    plotter_object.add_datapoints(x_query, "green", 1)
    plotter_object.add_gt_function(x_grid, np.squeeze(pred_mu_grid), "seismic", levels, 2)
    plotter_object.add_datapoints(x_data, "black", 2)
    plotter_object.add_datapoints(x_query, "green", 2)
    if save_plot:
        plotter_object.save_fig(file_path, file_name)
    else:
        plotter_object.show()


def active_learning_2d_plot_without_gt(
    x_grid, acquisition_values_grid, pred_mu_grid, x_data, x_query, save_plot=False, file_name=None, file_path=None
):
    plotter_object = Plotter2D(2)
    plotter_object.add_gt_function(x_grid, np.squeeze(acquisition_values_grid), "RdBu_r", 14, 0)
    plotter_object.add_datapoints(x_data, "black", 0)
    if len(x_query.shape) == 1:
        x_query = np.expand_dims(x_query, axis=0)

    plotter_object.add_datapoints(x_query, "green", 0)
    min_y = np.min(pred_mu_grid)
    max_y = np.max(pred_mu_grid)
    levels = np.linspace(min_y, max_y, 100)
    plotter_object.add_gt_function(x_grid, np.squeeze(pred_mu_grid), "seismic", levels, 1)
    plotter_object.add_datapoints(x_data, "black", 1)
    plotter_object.add_datapoints(x_query, "green", 1)
    if save_plot:
        plotter_object.save_fig(file_path, file_name)
    else:
        plotter_object.show()


def safe_active_learning_1d_plot(
    x_grid,
    pred_mu,
    pred_sigma,
    safety_mu,
    safety_sigma,
    safe_grid,
    x_data,
    y_data,
    safety_data,
    x_query,
    y_query,
    save_plot=False,
    file_name=None,
    file_path=None,
):
    plotter_object = Plotter(2)
    plotter_object.add_datapoints(np.squeeze(x_data), np.squeeze(y_data), "r", 0)
    plotter_object.add_datapoints(x_query, y_query, "green", 0)
    plotter_object.add_predictive_dist(np.squeeze(x_grid), np.squeeze(pred_mu), np.squeeze(pred_sigma), 0)
    plotter_object.add_datapoints(np.squeeze(x_data), np.squeeze(safety_data), "r", 1)
    plotter_object.add_predictive_dist(np.squeeze(x_grid), np.squeeze(safety_mu), np.squeeze(safety_sigma), 1)
    plotter_object.add_safety_region(safe_grid, 1)
    if save_plot:
        plotter_object.save_fig(file_path, file_name)
    else:
        plotter_object.show()


def safe_active_learning_2d_plot(
    x_grid,
    pred_mu,
    pred_sigma,
    safety_mu,
    safety_sigma,
    safety_estimate_over_grid,
    true_safety_over_grid,
    y_over_grid,
    x_data,
    x_query,
    save_plot=False,
    file_name=None,
    file_path=None,
):
    plotter_object = Plotter2D(5)
    plotter_object.add_gt_function(x_grid, np.squeeze(pred_sigma), "RdBu_r", 14, 0)
    plotter_object.add_datapoints(x_data, "red", 0)
    plotter_object.add_datapoints(np.expand_dims(x_query, axis=0), "green", 0)
    levels_safety = np.linspace(-1.0, 1.02, 500)
    plotter_object.add_gt_function(x_grid, safety_estimate_over_grid, "RdBu_r", levels_safety, 1)
    plotter_object.add_datapoints(x_data, "red", 1)
    plotter_object.add_datapoints(np.expand_dims(x_query, axis=0), "green", 1)

    plotter_object.add_gt_function(x_grid, np.squeeze(true_safety_over_grid), "RdBu_r", levels_safety, 2)
    plotter_object.add_datapoints(x_data, "red", 2)
    plotter_object.add_datapoints(np.expand_dims(x_query, axis=0), "green", 2)

    min_y = np.min(y_over_grid)
    max_y = np.max(y_over_grid)
    max_abs = max(np.abs(max_y), np.abs(min_y))
    max_y = max_y + 0.5 * max_abs
    min_y = min_y - 0.5 * max_abs

    levels = np.linspace(min_y, max_y, 100)
    plotter_object.add_gt_function(x_grid, np.squeeze(y_over_grid), "seismic", levels, 3)
    plotter_object.add_datapoints(x_data, "red", 3)
    plotter_object.add_datapoints(np.expand_dims(x_query, axis=0), "green", 3)
    plotter_object.add_gt_function(x_grid, np.squeeze(pred_mu), "seismic", levels, 4)
    plotter_object.add_datapoints(x_data, "red", 4)
    plotter_object.add_datapoints(np.expand_dims(x_query, axis=0), "green", 4)
    if save_plot:
        plotter_object.save_fig(file_path, file_name)
    else:
        plotter_object.show()


def set_font_sizes(font_size, only_axis=True):
    if only_axis:
        matplotlib.rc("xtick", labelsize=font_size)
        matplotlib.rc("ytick", labelsize=font_size)
    else:
        font = {"family": "normal", "size": font_size}

        matplotlib.rc("font", **font)


if __name__ == "__main__":
    x = np.random.uniform(0, 1, size=(100, 5))

    y = np.random.uniform(0, 1, size=(100, 1))
    active_learning_nd_plot(x, y, False)
