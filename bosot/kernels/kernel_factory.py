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
from bosot.configs.kernels.base_kernel_config import BaseKernelConfig
from bosot.configs.kernels.grammar_tree_kernel_kernel_configs import (
    OptimalTransportGrammarKernelConfig,
)
from bosot.configs.kernels.rational_quadratic_configs import BasicRQConfig
from bosot.kernels.kernel_kernel_grammar_tree import OptimalTransportKernelKernel
from bosot.configs.kernels.rbf_configs import BasicRBFConfig
from bosot.kernels.rational_quadratic_kernel import RationalQuadraticKernel
from bosot.kernels.rbf_kernel import RBFKernel
from bosot.kernels.linear_kernel import LinearKernel
from bosot.configs.kernels.linear_configs import BasicLinearConfig
from bosot.configs.kernels.hellinger_kernel_kernel_configs import BasicHellingerKernelKernelConfig
from bosot.kernels.kernel_kernel_hellinger import KernelKernelHellinger
from bosot.kernels.periodic_kernel import PeriodicKernel
from bosot.configs.kernels.periodic_configs import BasicPeriodicConfig, PeriodicWithPriorConfig


class KernelFactory:
    @staticmethod
    def build(kernel_config: BaseKernelConfig):
        if isinstance(kernel_config, BasicRBFConfig):
            kernel = RBFKernel(**kernel_config.dict())
            return kernel
        elif isinstance(kernel_config, BasicLinearConfig):
            kernel = LinearKernel(**kernel_config.dict())
            return kernel
        elif isinstance(kernel_config, BasicRQConfig):
            kernel = RationalQuadraticKernel(**kernel_config.dict())
            return kernel
        elif isinstance(kernel_config, BasicHellingerKernelKernelConfig):
            kernel = KernelKernelHellinger(**kernel_config.dict())
            return kernel
        elif isinstance(kernel_config, BasicPeriodicConfig):
            kernel = PeriodicKernel(**kernel_config.dict())
            return kernel
        elif isinstance(kernel_config, OptimalTransportGrammarKernelConfig):
            kernel = OptimalTransportKernelKernel(**kernel_config.dict())
            return kernel


if __name__ == "__main__":
    pass
