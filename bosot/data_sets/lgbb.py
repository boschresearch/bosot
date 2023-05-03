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
from bosot.data_sets.base_data_set import BaseDataset
import os


class OutputType(Enum):
    LIFT = 0
    ROLL = 1
    YAW = 2
    PITCH = 3
    DRAG = 4


class LGBB(BaseDataset):
    def __init__(self, base_path, file_name="lgbb_original.txt", normalize_output=True):
        self.file_path = os.path.join(base_path, file_name)
        # add small aritficial noise - as response surface is deterministic
        self.observation_noise = 0.01
        self.add_noise = True
        self.exclude_outlier = True
        self.output_type = OutputType.LIFT
        self.beta = 0.0
        self.normalize_output = normalize_output
        self.name = "LGBB"

    def get_name(self):
        return self.name

    def load_data_set(self):
        df = pd.read_csv(self.file_path, sep=" ", skiprows=21)
        df_beta_0 = df[df["beta"] == self.beta]

        if self.output_type == OutputType.LIFT:
            key = "lift"
            high = df_beta_0[key].quantile(0.99)
            df_filtered = df_beta_0[(df_beta_0[key] < high)]
            x1 = df_filtered["mach"].to_numpy() / 6
            x2 = (df_filtered["alpha"].to_numpy() + 5) / 35
            self.y = np.expand_dims(df_filtered["lift"].to_numpy(), axis=1)
        elif self.output_type == OutputType.ROLL:
            x1 = df_beta_0["mach"].to_numpy() / 6
            x2 = (df_beta_0["alpha"].to_numpy() + 5) / 35
            self.y = np.expand_dims(df_beta_0["roll"].to_numpy(), axis=1)
        elif self.output_type == OutputType.YAW:
            key = "yaw"
            low = df_beta_0[key].quantile(0.01)
            high = df_beta_0[key].quantile(0.99)
            df_filtered = df_beta_0[(df_beta_0[key] < high) & (df_beta_0[key] > low)]
            x1 = df_filtered["mach"].to_numpy() / 6
            x2 = (df_filtered["alpha"].to_numpy() + 5) / 35
            self.y = np.expand_dims(df_filtered[key].to_numpy(), axis=1)
        elif self.output_type == OutputType.PITCH:
            key = "pitch"
            low = df_beta_0[key].quantile(0.01)
            high = df_beta_0[key].quantile(0.99)
            df_filtered = df_beta_0[(df_beta_0[key] < high) & (df_beta_0[key] > low)]
            x1 = df_filtered["mach"].to_numpy() / 6
            x2 = (df_filtered["alpha"].to_numpy() + 5) / 35
            self.y = np.expand_dims(df_filtered[key].to_numpy(), axis=1)
        elif self.output_type == OutputType.DRAG:
            key = "drag"
            low = df_beta_0[key].quantile(0.01)
            high = df_beta_0[key].quantile(0.99)
            df_filtered = df_beta_0[(df_beta_0[key] < high) & (df_beta_0[key] > low)]
            x1 = df_filtered["mach"].to_numpy() / 6
            x2 = (df_filtered["alpha"].to_numpy() + 5) / 35
            self.y = np.expand_dims(df_filtered[key].to_numpy(), axis=1)

        if self.normalize_output:
            mean_y = np.mean(self.y)
            std_y = np.std(self.y)
            self.y = (self.y - mean_y) / std_y

        self.x = np.stack((x1, x2), axis=1)
        self.length = x1.shape[0]

    def get_complete_dataset(self, add_noise=True):
        n = self.y.shape[0]
        if add_noise:
            noise = np.random.randn(n, 1) * self.observation_noise
            y = self.y + noise
        else:
            y = self.y
        return self.x, y

    def sample(self, n, random_x=None, expand_dims=None):
        indexes = np.random.choice(self.length, n, replace=False)
        x_sample = self.x[indexes]
        f_sample = self.y[indexes]
        noise = np.random.randn(n, 1) * self.observation_noise
        if self.add_noise:
            y_sample = f_sample + noise
        else:
            y_sample = f_sample
        return x_sample, y_sample

    def sample_train_test(self, use_absolute: bool, n_train: int, n_test: int, fraction_train: float):
        if use_absolute:
            assert n_train < self.length
            n = n_train + n_test
            if n > self.length:
                n = self.length
                print("Test + Train set exceeds number of datapoints - use n-n_train test points")
        else:
            n = self.length
            n_train = int(fraction_train * n)
            n_test = n - n_train
        indexes = np.random.choice(self.length, n, replace=False)
        train_indexes = indexes[:n_train]
        assert len(train_indexes) == n_train
        test_indexes = indexes[n_train:]
        if use_absolute and n_train + n_test <= self.length:
            assert len(test_indexes) == n_test
        noise = np.random.randn(self.length, 1) * self.observation_noise
        if self.add_noise:
            y = self.y + noise
        else:
            y = self.y
        x_train = self.x[train_indexes]
        y_train = self.y[train_indexes]
        x_test = self.x[test_indexes]
        y_test = self.y[test_indexes]
        return x_train, y_train, x_test, y_test

    def plot(self):
        xs, ys = self.sample(700)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.scatter(xs[:, 0], xs[:, 1], ys, marker=".")
        plt.show()


if __name__ == "__main__":
    data_loader = LGBB("YOUR-DATA-PATH")
    data_loader.load_data_set()
    # print(data_loader.sample_only_in_small_box_and_safe(10,0.3,0.2))
    # data_loader.plot_regime(left=True)
    # data_loader.sample_only_one_regime(10)
    # data_loader.plot_safe(0.90)
    x, y = data_loader.get_complete_dataset()
    assert y.shape[1] == 1
    x_train, y_train, x_test, y_test = data_loader.sample_train_test(True, 700, 200, 0.8)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    data_loader.plot()
    # data_loader.plot_regime(0.15,True)
