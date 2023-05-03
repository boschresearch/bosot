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
import gpflow
from typing import Tuple
import numpy as np

gpflow.config.set_default_float(np.float64)
f64 = gpflow.utilities.to_default_float
from tensorflow_probability import distributions as tfd
from bosot.kernels.base_elementary_kernel import BaseElementaryKernel


class LinearKernel(BaseElementaryKernel):
    def __init__(
        self,
        input_dimension: int,
        base_variance: float,
        base_offset: float,
        add_prior: bool,
        variance_prior_parameters: Tuple[float, float],
        offset_prior_parameters: Tuple[float, float],
        active_on_single_dimension: bool,
        active_dimension: int,
        name: str,
        **kwargs
    ):
        super().__init__(input_dimension, active_on_single_dimension, active_dimension, name)
        self.kernel = gpflow.kernels.Polynomial(degree=1, variance=f64([base_variance]), offset=f64([base_offset]))
        # self.kernel = gpflow.kernels.Linear(variance=f64([base_variance]))
        if add_prior:
            a_variance, b_variance = variance_prior_parameters
            a_offset, b_offset = offset_prior_parameters
            self.kernel.variance.prior = tfd.Gamma(f64([a_variance]), f64([b_variance]))
            self.kernel.offset.prior = tfd.Gamma(f64([a_offset]), f64([b_offset]))
