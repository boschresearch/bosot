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
from typing import List, Optional, Tuple
import gpflow
from bosot.kernels.base_object_kernel import BaseObjectKernel
from gpflow.models.gpr import GPR_with_posterior
import numpy as np
import tensorflow as tf
from gpflow.config import default_float
from gpflow.mean_functions import Constant, MeanFunction
from bosot.models.object_mean_functions import Zero


class ObjectGPR(GPR_with_posterior):
    """
    This is a class that extends the standard Gpflow gpr model from array to objects as input elements
    the only difference here is that X_data is now a list of objects rather than an np.array
    """

    def __init__(
        self,
        data: Tuple[List[object], np.array],
        kernel: BaseObjectKernel,
        noise_variance: float = 1.0,
        mean_function: MeanFunction = Zero(),
    ):
        assert isinstance(kernel, BaseObjectKernel)
        X_data, Y_data = data
        X_placeholder = np.zeros((len(Y_data), 1))
        # Initialize standard GPR model - without actual X data
        super().__init__(data=(X_placeholder, Y_data), kernel=kernel, mean_function=mean_function, noise_variance=noise_variance)
        # override actual data tuple
        Y_data_tf = self.array_to_tensor(Y_data)
        self.data = (X_data, Y_data_tf)

    def array_to_tensor(self, array):
        if tf.is_tensor(array):
            return array
        elif isinstance(array, np.ndarray):
            return tf.convert_to_tensor(array)
        return tf.convert_to_tensor(array, dtype=default_float())
