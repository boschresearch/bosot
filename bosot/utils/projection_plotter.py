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
from matplotlib.pyplot import plot
import numpy as np
from bosot.kernels.kernel_factory import KernelFactory
from bosot.configs.kernels.rbf_configs import BasicRBFConfig
from bosot.models.base_model import BaseModel
from bosot.models.model_factory import ModelFactory
from bosot.configs.models.gp_model_config import GPModelFastConfig
from bosot.utils.plotter import Plotter


class ProjectionPlotter:
    def __init__(self):
        self.dicretization_steps = 100
        self.plotter_object = Plotter(1)

    def reset_plotter(self):
        self.plotter_object = Plotter(1)

    def initialize(self, x_data: np.array, y_data: np.array):
        self.input_dimension = x_data.shape[1]
        self.a = np.min(x_data, axis=0)
        self.b = np.max(x_data, axis=0)
        kernel_config = BasicRBFConfig(input_dimension=self.input_dimension)
        model_config = GPModelFastConfig(kernel_config=kernel_config)
        self.gt_model = ModelFactory().build(model_config)
        self.gt_model.infer(x_data, y_data)

    def set_line(self, dimension_index: int, anchor_point: np.array):
        self.dimension_index = dimension_index
        self.anchor_point = anchor_point

    def plot_prediction_at_dimension(self, prediction_model: BaseModel):
        line, final_line = self.construct_line_data(self.dimension_index, self.anchor_point)
        final_line_mu, final_line_sigma = prediction_model.predictive_dist(final_line)
        gt_mu, _ = self.gt_model.predictive_dist(final_line)
        self.plotter_object.add_gt_function(line, np.squeeze(gt_mu), "black", 0)
        self.plotter_object.add_predictive_dist(line, final_line_mu, final_line_sigma, 0)

    def construct_line_data(self, dimension_index, anchor_point):
        line_data = []
        line = np.linspace(self.a[dimension_index], self.b[dimension_index], self.dicretization_steps)
        for p in line:
            point = anchor_point.copy()
            point[dimension_index] = p
            line_data.append(point)
        return line, np.array(line_data)

    def show(self):
        self.plotter_object.show()

    def save_fig(self, file_path, file_name):
        self.plotter_object.save_fig(file_path, file_name)

    def plot_closest_data_points(self, x_data: np.array, y_data: np.array, n_closest: int):
        _, line_data = self.construct_line_data(self.dimension_index, self.anchor_point)
        distances = []
        for x in x_data:
            assert len(line_data - x) == len(line_data)
            distance = np.min(np.linalg.norm(line_data - x, axis=0))
            distances.append(distance)
        distances = np.array(distances)
        shortest = np.argsort(distances)
        n_closest_indexes = shortest[:n_closest]
        x_to_plot = x_data[n_closest_indexes][:, self.dimension_index]
        y_to_plot = y_data[n_closest_indexes]
        self.plotter_object.add_datapoints(x_to_plot, y_to_plot, "green", 0)


if __name__ == "__main__":
    pass
