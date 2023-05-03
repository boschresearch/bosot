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
import numpy as np
from typing import Tuple, Optional
from abc import ABC, abstractmethod


class BaseDataset(ABC):
    @abstractmethod
    def load_data_set(self):
        """
        loads dataset (probably from some file - implementation dependent)
        """
        raise NotImplementedError

    @abstractmethod
    def get_complete_dataset(self, **kwargs) -> Tuple[np.array, np.array]:
        """
        Retrieves the complete dataset (of size n with input dimensions d and output dimensions m) as numpy arrays

        Returns
        np.array - x (input values) with shape (n,d)
        np.array - y (output values) with shape (n,m)
        """
        raise NotImplementedError

    @abstractmethod
    def sample(self, n: int, **kwargs) -> Tuple[np.array, np.array]:
        """
        retrieves sample of size n from dataset (without replacement)

        Returns
        np.array - x (input values) with shape (n,d)
        np.array - y (output values) with shape (n,m)
        """
        raise NotImplementedError

    @abstractmethod
    def sample_train_test(self, use_absolute: bool, n_train: int, n_test: int, fraction_train: float):
        """
        retrieves train and test data (mutually exclusive samples) either in absoulte numbers or as fraction from the complete dataset

        Arguments:
            use_absolute - bool specifying if absolute numbers of training and test data should be used or fraction of complete dataset
            n_train - int specifying how many training datapoints should be sampled
            n_test - int scpecifying how many test datapoints should be sampled
            fraction_train - fraction of complete dataset that is used as training data

        Returns
        np.array - x train data with shape (n_train,d)
        np.array - y train data with shape (n_train,m)
        np.array - x test data with shape (n_test,d)
        np.array - y test data with shape (n_test,m)
        """
        raise NotImplementedError

    @abstractmethod
    def get_name(self):
        """
        method to get name of dataset
        """
        raise NotImplementedError
