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
from bosot.configs.kernels.kernel_grammar_generators.base_kernel_grammar_generator_config import BaseKernelGrammarGeneratorConfig
from bosot.configs.kernels.kernel_grammar_generators.cks_high_dim_generator_config import CKSHighDimGeneratorConfig
from bosot.configs.kernels.kernel_grammar_generators.cks_with_rq_generator_config import CKSWithRQGeneratorConfig
from bosot.configs.kernels.kernel_grammar_generators.compositional_kernel_search_configs import CompositionalKernelSearchGeneratorConfig
from bosot.configs.kernels.kernel_grammar_generators.n_dim_full_kernels_generators_configs import NDimFullKernelsGrammarGeneratorConfig
from bosot.kernels.kernel_grammar.kernel_grammar_search_spaces import (
    CKSHighDimSearchSpace,
    NDimFullKernelsSearchSpace,
    CKSWithRQSearchSpace,
    CompositionalKernelSearchSpace,
)
from bosot.kernels.kernel_grammar.kernel_grammar_candidate_generator import KernelGrammarCandidateGenerator


class GeneratorFactory:
    @staticmethod
    def build(generator_config: BaseKernelGrammarGeneratorConfig):
        if isinstance(generator_config, NDimFullKernelsGrammarGeneratorConfig):
            search_space = NDimFullKernelsSearchSpace(generator_config.input_dimension)
            generator = KernelGrammarCandidateGenerator(search_space=search_space, **generator_config.dict())
            return generator
        elif isinstance(generator_config, CompositionalKernelSearchGeneratorConfig):
            search_space = CompositionalKernelSearchSpace(generator_config.input_dimension)
            generator = KernelGrammarCandidateGenerator(search_space=search_space, **generator_config.dict())
            return generator
        elif isinstance(generator_config, CKSWithRQGeneratorConfig):
            search_space = CKSWithRQSearchSpace(generator_config.input_dimension)
            generator = KernelGrammarCandidateGenerator(search_space=search_space, **generator_config.dict())
            return generator
        elif isinstance(generator_config, CKSHighDimGeneratorConfig):
            search_space = CKSHighDimSearchSpace(generator_config.input_dimension)
            generator = KernelGrammarCandidateGenerator(search_space=search_space, **generator_config.dict())
            return generator
