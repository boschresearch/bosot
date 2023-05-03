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
from enum import Enum

from pandas.core import base
from bosot.data_sets.base_data_set import BaseDataset
from bosot.utils.plot_utils import active_learning_1d_plot, active_learning_2d_plot, active_learning_nd_plot
from bosot.data_sets.enums import InputPreprocessingType, OutputPreprocessingType
from bosot.utils.utils import normalize_data, min_max_normalize_data
from bosot.data_sets.standard_data_set import StandardDataSet
import os


class PowerPlant(StandardDataSet):
    def __init__(self, base_path, file_name="power_plant_data.csv"):
        super().__init__()
        self.file_path = os.path.join(base_path, file_name)
        self.input_preprocessing_type = InputPreprocessingType.MIN_MAX_NORMALIZATION
        self.output_preprocessing_type = OutputPreprocessingType.NORMALIZATION
        self.name = "PowerPlant"

    def load_data_set(self):
        df = pd.read_csv(self.file_path, sep=",")
        x_list = []
        label_name = "PE"
        print(df.columns)
        assert label_name in df.columns
        for col_name in df.columns:
            if not col_name == label_name:
                x_list.append(np.expand_dims(df[col_name].to_numpy(), axis=1))
        y = np.expand_dims(df[label_name].to_numpy(), axis=1)
        self.length = y.shape[0]
        self.x = np.concatenate(x_list, axis=1)
        self.y = y
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
    data_loader = PowerPlant("YOUR-OWN-DATA-PATH")
    data_loader.load_data_set()
    x_train, y_train, x_test, y_test = data_loader.sample_train_test(False, 100, 200, 0.8)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    print(data_loader.length)
    data_loader.plot_scatter()
