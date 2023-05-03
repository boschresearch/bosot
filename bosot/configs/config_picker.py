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
from bosot.configs.bayesian_optimization.bayesian_optimizer_objects_configs import (
    ObjectBOExpectedImprovementConfig,
    ObjectBOExpectedImprovementEAConfig,
    ObjectBOExpectedImprovementEAFewerStepsConfig,
    ObjectBOExpectedImprovementPerSecondEAConfig,
)
from bosot.configs.bayesian_optimization.greedy_kernel_search_configs import (
    BaseGreedyKernelSearchConfig,
    GreedyKernelSearchBaseInitialConfig,
)
from bosot.configs.bayesian_optimization.treeGEP_optimizer_configs import TreeGEPEvolutionaryOptimizerConfig
from bosot.configs.kernels.grammar_tree_kernel_kernel_configs import (
    OTWeightedDimsExtendedGrammarKernelConfig,
    OptimalTransportGrammarKernelConfig,
)
from bosot.configs.kernels.hellinger_kernel_kernel_configs import BasicHellingerKernelKernelConfig
from bosot.configs.kernels.kernel_grammar_generators.cks_high_dim_generator_config import CKSHighDimGeneratorConfig
from bosot.configs.kernels.kernel_grammar_generators.cks_with_rq_generator_config import CKSWithRQGeneratorConfig
from bosot.configs.kernels.kernel_grammar_generators.compositional_kernel_search_configs import CompositionalKernelSearchGeneratorConfig
from bosot.configs.kernels.kernel_grammar_generators.n_dim_full_kernels_generators_configs import NDimFullKernelsGrammarGeneratorConfig
from bosot.configs.kernels.rational_quadratic_configs import BasicRQConfig, RQWithPriorConfig
from bosot.configs.models.gp_model_config import BasicGPModelConfig, GPModelExtenseOptimization, GPModelFastConfig, GPModelFixedNoiseConfig
from bosot.configs.models.gp_model_laplace_config import BasicGPModelLaplaceConfig
from bosot.configs.kernels.rbf_configs import BasicRBFConfig, RBFWithPriorConfig
from bosot.configs.kernels.linear_configs import BasicLinearConfig, LinearWithPriorConfig


class ConfigPicker:
    models_configs_dict = {
        c.__name__: c
        for c in [
            BasicGPModelConfig,
            GPModelFastConfig,
            GPModelExtenseOptimization,
            GPModelFixedNoiseConfig,
            BasicGPModelLaplaceConfig,
        ]
    }

    kernels_configs_dict = {
        c.__name__: c
        for c in [
            RBFWithPriorConfig,
            BasicRBFConfig,
            BasicLinearConfig,
            LinearWithPriorConfig,
            BasicRQConfig,
            RQWithPriorConfig,
            BasicHellingerKernelKernelConfig,
            OptimalTransportGrammarKernelConfig,
            OTWeightedDimsExtendedGrammarKernelConfig,
        ]
    }

    bayesian_optimization_configs_dict = {
        c.__name__: c
        for c in [
            ObjectBOExpectedImprovementConfig,
            ObjectBOExpectedImprovementEAConfig,
            ObjectBOExpectedImprovementPerSecondEAConfig,
            ObjectBOExpectedImprovementEAFewerStepsConfig,
        ]
    }

    greedy_kernel_seach_configs_dict = {
        c.__name__: c for c in [BaseGreedyKernelSearchConfig, GreedyKernelSearchBaseInitialConfig, TreeGEPEvolutionaryOptimizerConfig]
    }

    kernel_grammar_generator_configs_dict = {
        c.__name__: c
        for c in [
            NDimFullKernelsGrammarGeneratorConfig,
            CompositionalKernelSearchGeneratorConfig,
            CKSWithRQGeneratorConfig,
            CKSHighDimGeneratorConfig,
        ]
    }

    @staticmethod
    def pick_kernel_config(config_class_name):
        return ConfigPicker.kernels_configs_dict[config_class_name]

    @staticmethod
    def pick_model_config(config_class_name):
        return ConfigPicker.models_configs_dict[config_class_name]

    @staticmethod
    def pick_bayesian_optimization_config(config_class_name):
        return ConfigPicker.bayesian_optimization_configs_dict[config_class_name]

    @staticmethod
    def pick_kernel_grammar_generator_config(config_class_name):
        return ConfigPicker.kernel_grammar_generator_configs_dict[config_class_name]

    @staticmethod
    def pick_greedy_kernel_search_config(config_class_name):
        return ConfigPicker.greedy_kernel_seach_configs_dict[config_class_name]


if __name__ == "__main__":
    pass
