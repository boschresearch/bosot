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
from bosot.configs.kernels.base_kernel_config import BaseKernelConfig
from bosot.configs.kernels.base_elementary_kernel_config import BaseElementaryKernelConfig


class BasicRBFConfig(BaseElementaryKernelConfig):
    base_lengthscale: float = 1.0
    base_variance: float = 1.0
    add_prior: bool = False
    lengthscale_prior_parameters: Tuple[float, float] = (2.0, 2.0)
    variance_prior_parameters: Tuple[float, float] = (2.0, 3.0)
    name = "BasicRBF"


class RBFWithPriorConfig(BasicRBFConfig):
    add_prior: bool = True
    name = "RBFWithPrior"
