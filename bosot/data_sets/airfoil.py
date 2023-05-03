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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from enum import Enum
from bosot.utils.plot_utils import active_learning_1d_plot, active_learning_2d_plot, active_learning_nd_plot
from bosot.data_sets.enums import InputPreprocessingType, OutputPreprocessingType
from bosot.utils.utils import normalize_data, min_max_normalize_data
from bosot.data_sets.standard_data_set import StandardDataSet
import os


class Airfoil(StandardDataSet):
    def __init__(self, base_path, file_name="airfoil_self_noise.dat"):
        super().__init__()
        self.file_path = os.path.join(base_path, file_name)
        self.input_preprocessing_type = InputPreprocessingType.MIN_MAX_NORMALIZATION
        self.output_preprocessing_type = OutputPreprocessingType.NORMALIZATION
        self.name = "Airfoil"

    def load_data_set(self):
        data = np.loadtxt(self.file_path)
        x_frequency = np.expand_dims(data[:, 0], axis=1)
        x_angle = np.expand_dims(data[:, 1], axis=1)
        x_chord_length = np.expand_dims(data[:, 2], axis=1)
        x_stream_vel = np.expand_dims(data[:, 3], axis=1)
        x_thickness = np.expand_dims(data[:, 4], axis=1)
        airfoil_noise = np.expand_dims(data[:, 5], axis=1)
        self.x = np.concatenate((x_frequency, x_angle, x_chord_length, x_stream_vel, x_thickness), axis=1)
        self.y = airfoil_noise
        self.length = len(self.y)
        print(self.length)
        if self.input_preprocessing_type == InputPreprocessingType.NORMALIZATION:
            self.x = normalize_data(self.x)
        elif self.input_preprocessing_type == InputPreprocessingType.MIN_MAX_NORMALIZATION:
            self.x = min_max_normalize_data(self.x)

        if self.output_preprocessing_type == OutputPreprocessingType.NORMALIZATION:
            self.y = normalize_data(self.y)

    def plot_scatter(self):
        xs, ys = self.sample(300)
        active_learning_nd_plot(xs, ys)


if __name__ == "__main__":
    data_loader = Airfoil("YOUR-OWN-DATA-PATH")
    data_loader.load_data_set()
    print(data_loader.length)
    # data_loader.get_close_samples(10,1.5)
    data_loader.plot_scatter()
