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
from bosot.utils.plotter import Plotter
import os


class AirlinePassenger(StandardDataSet):
    def __init__(self, base_path, file_name="airline-passengers.csv"):
        super().__init__()
        self.file_path = os.path.join(base_path, file_name)
        self.input_preprocessing_type = InputPreprocessingType.MIN_MAX_NORMALIZATION
        self.output_preprocessing_type = OutputPreprocessingType.NORMALIZATION
        self.name = "AirlinePassenger"

    def load_data_set(self):
        df = pd.read_csv(self.file_path, sep=",")
        string_to_decimal = lambda x: float(x.split("-")[0]) + (float(x.split("-")[1]) - 1.0) / 12.0
        self.x = np.expand_dims(df["Month"].apply(string_to_decimal).to_numpy(), axis=1)
        self.y = np.expand_dims(df["Passengers"].to_numpy(), axis=1)

        self.length = len(self.y)
        print(self.length)
        if self.input_preprocessing_type == InputPreprocessingType.NORMALIZATION:
            self.x = normalize_data(self.x)
        elif self.input_preprocessing_type == InputPreprocessingType.MIN_MAX_NORMALIZATION:
            self.x = min_max_normalize_data(self.x)

        if self.output_preprocessing_type == OutputPreprocessingType.NORMALIZATION:
            self.y = normalize_data(self.y)

    def plot(self):
        fig, ax = plt.subplots(1, 1)
        ax.plot(self.x, self.y, ".", color="black")
        plt.show()


if __name__ == "__main__":
    passenger_data = AirlinePassenger("YOUR-DATA-PATH")
    passenger_data.load_data_set()
    passenger_data.plot()
