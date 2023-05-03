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
from abc import ABC, abstractmethod
import tensorflow as tf
import numpy as np


class RegularizedKernelInterface(ABC):
    """
    Interface that a kernel needs to admit such that regularizer of the kernel is added to loss in GPModel
    """

    @abstractmethod
    def regularization_loss(self, x_data: np.array) -> tf.Tensor:
        """
        Method for retrieving the loss that should be added to the marginal likelihood loss of a gp (loss is regularizer for kernel HP's)

        Arguments:
        x_data - np.array with anchor input data shape (n,d)

        Returns:
        tf.Tensor - single float tf.Tensor with the loss value
        """
        raise NotImplementedError
