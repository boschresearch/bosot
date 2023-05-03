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
from typing import Tuple
import gpflow
from gpflow.utilities.traversal import print_summary

f64 = gpflow.utilities.to_default_float
import numpy as np
from tensorflow_probability import distributions as tfd
from bosot.kernels.base_elementary_kernel import BaseElementaryKernel


class PeriodicKernel(BaseElementaryKernel):
    def __init__(
        self,
        input_dimension: int,
        base_lengthscale: float,
        base_variance: float,
        base_period: float,
        add_prior: bool,
        lengthscale_prior_parameters: Tuple[float, float],
        variance_prior_parameters: Tuple[float, float],
        period_prior_parameters: Tuple[float, float],
        active_on_single_dimension: bool,
        active_dimension: int,
        name: str,
        **kwargs
    ):
        super().__init__(input_dimension, active_on_single_dimension, active_dimension, name)

        self.kernel = gpflow.kernels.Periodic(
            gpflow.kernels.RBF(
                lengthscales=f64(np.repeat(base_lengthscale, self.num_active_dimensions)),
                variance=f64([base_variance]),
            ),
            period=f64([base_period]),
        )
        if add_prior:
            a_lengthscale, b_lengthscale = lengthscale_prior_parameters
            a_variance, b_variance = variance_prior_parameters
            a_period, b_period = period_prior_parameters
            self.kernel.base_kernel.lengthscales.prior = tfd.Gamma(
                f64(np.repeat(a_lengthscale, self.num_active_dimensions)),
                f64(np.repeat(b_lengthscale, self.num_active_dimensions)),
            )
            self.kernel.base_kernel.variance.prior = tfd.Gamma(f64([a_variance]), f64([b_variance]))
            self.kernel.period.prior = tfd.Gamma(f64([a_period]), f64([b_period]))


if __name__ == "__main__":
    kernel = PeriodicKernel(3, 1.0, 1.0, 1.0, True, (1.0, 1.0), (1.0, 1.0), (1.0, 1.0))
