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
import matplotlib.pyplot as plt
import numpy as np
import os


class Plotter:
    def __init__(self, num_axes):
        self.num_axes = num_axes
        self.fig, self.axes = plt.subplots(num_axes, 1)

    def add_gt_function(self, x, ground_truth, color, ax_num, sort_x=True):
        if sort_x:
            sorted_indexes = np.argsort(x)
            self.give_axes(ax_num).plot(x[sorted_indexes], ground_truth[sorted_indexes], color=color)
        else:
            self.give_axes(ax_num).plot(x, ground_truth, color=color)

    def add_datapoints(self, x_data, y_data, color, ax_num):
        self.give_axes(ax_num).plot(x_data, y_data, "o", color=color)

    def give_axes(self, ax_num):
        if self.num_axes > 1:
            return self.axes[ax_num]
        else:
            return self.axes

    def add_posterior_functions(self, x, predictions, ax_num):
        num_predictions = predictions.shape[0]
        for i in range(0, num_predictions):
            self.give_axes(ax_num).plot(x, predictions[i], color="r", linewidth="0.5")

    def add_predictive_dist(self, x, pred_mu, pred_sigma, ax_num, sort_x=True):
        if sort_x:
            sorted_index = np.argsort(x)
            x = x[sorted_index]
            pred_mu = pred_mu[sorted_index]
            pred_sigma = pred_sigma[sorted_index]
        axes = self.give_axes(ax_num)
        axes.plot(x, pred_mu, color="g")
        axes.fill_between(x, pred_mu - pred_sigma, pred_mu + pred_sigma, alpha=0.8, color="b")
        axes.fill_between(x, pred_mu - 2 * pred_sigma, pred_mu + 2 * pred_sigma, alpha=0.3, color="b")

    def add_hline(self, y_value, color, ax_num):
        self.give_axes(ax_num).axhline(y_value, color=color, linestyle="--")

    def add_vline(self, x_value, color, ax_num):
        self.give_axes(ax_num).axvline(x_value, color=color, linestyle="--")

    def add_safety_region(self, safe_x, ax_num):
        min_y = self.give_axes(ax_num).get_ylim()[0]
        self.give_axes(ax_num).plot(safe_x, np.repeat(min_y, safe_x.shape[0]), "_", color="green")

    def save_fig(self, file_path, file_name):
        plt.savefig(os.path.join(file_path, file_name))
        plt.close()

    def show(self):
        plt.show()
