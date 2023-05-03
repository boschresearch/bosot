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
from gpflow import kernels
from scipy.stats.stats import mode
from bosot.configs.models.base_model_config import BaseModelConfig
from bosot.configs.models.gp_model_config import BasicGPModelConfig
from bosot.configs.models.gp_model_laplace_config import BasicGPModelLaplaceConfig
from bosot.models.gp_model import GPModel
from bosot.models.gp_model_laplace import GPModelLaplace
from bosot.kernels.kernel_factory import KernelFactory
from bosot.models.object_gp_model import ObjectGpModel
from bosot.configs.models.object_gp_model_config import BasicObjectGPModelConfig
import gpflow
import numpy as np


class ModelFactory:
    @staticmethod
    def build(model_config: BaseModelConfig):
        if isinstance(model_config, BasicGPModelConfig):
            kernel = KernelFactory.build(model_config.kernel_config)
            model = GPModel(kernel=kernel, **model_config.dict())
            return model
        elif isinstance(model_config, BasicGPModelLaplaceConfig):
            # @TODO: all kernels should be applicaple to laplace and gpmodelmarg
            kernel = KernelFactory.build(model_config.kernel_config)
            model = GPModelLaplace(kernel=kernel, **model_config.dict())
            return model
        elif isinstance(model_config, BasicObjectGPModelConfig):
            kernel = KernelFactory.build(model_config.kernel_config)
            model = ObjectGpModel(kernel=kernel, **model_config.dict())
            return model


if __name__ == "__main__":
    pass
