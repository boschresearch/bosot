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
from bosot.data_sets.base_data_set import BaseDataset
import numpy as np


class StandardDataSet(BaseDataset):
    def __init__(self):
        self.x: np.array
        self.y: np.array
        self.length: int
        self.name: str

    def get_complete_dataset(self):
        return self.x, self.y

    def sample(self, n):
        indexes = np.random.choice(self.length, n, replace=False)
        x_sample = self.x[indexes]
        y_sample = self.y[indexes]
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
        x_train = self.x[train_indexes]
        y_train = self.y[train_indexes]
        x_test = self.x[test_indexes]
        y_test = self.y[test_indexes]
        return x_train, y_train, x_test, y_test

    def get_name(self):
        return self.name
