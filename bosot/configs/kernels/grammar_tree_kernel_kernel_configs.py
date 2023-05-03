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
from typing import List, Tuple
from bosot.configs.kernels.base_kernel_config import BaseKernelConfig
from bosot.kernels.kernel_grammar.kernel_grammar import KernelGrammarOperator
from bosot.kernels.kernel_kernel_grammar_tree import FeatureType


class OptimalTransportGrammarKernelConfig(BaseKernelConfig):
    feature_type_list: List[FeatureType]
    input_dimension: int = 0
    base_variance: float = 1.0
    base_lengthscale: float = 1.0
    base_alpha: float = 0.5
    alpha_trainable: bool = True
    parameters_trainable: bool = True
    transform_to_normal: bool = False
    name = "OptimalTransportGrammarKernel"


class OTWeightedDimsExtendedGrammarKernelConfig(OptimalTransportGrammarKernelConfig):
    feature_type_list: List[FeatureType] = [
        FeatureType.DIM_WISE_WEIGHTED_ELEMENTARY_COUNT,
        FeatureType.REDUCED_ELEMENTARY_PATHS,
        FeatureType.SUBTREES,
    ]
    name = "OTWeightedDimsExtendedGrammarKernel"
