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
from bosot.configs.kernels.grammar_tree_kernel_kernel_configs import OTWeightedDimsExtendedGrammarKernelConfig
from bosot.configs.kernels.linear_configs import LinearWithPriorConfig
from bosot.configs.kernels.periodic_configs import PeriodicWithPriorConfig

from bosot.configs.kernels.rational_quadratic_configs import RQWithPriorConfig
from bosot.configs.kernels.rbf_configs import BasicRBFConfig, RBFWithPriorConfig
from bosot.configs.models.gp_model_config import BasicGPModelConfig
from bosot.configs.models.object_gp_model_config import BasicObjectGPModelConfig
from bosot.kernels.kernel_factory import KernelFactory
from bosot.models.gp_model import GPModel, PredictionQuantity
from bosot.models.model_factory import ModelFactory
from bosot.bayesian_optimization.bayesian_optimizer_objects import BayesianOptimizerObjects
from bosot.models.object_mean_functions import BICMean, ObjectConstant
from bosot.oracles.gp_model_bic_oracle import GPModelBICOracle
from bosot.configs.kernels.kernel_grammar_generators.cks_with_rq_generator_config import CKSWithRQGeneratorConfig
from bosot.configs.bayesian_optimization.bayesian_optimizer_objects_configs import ObjectBOExpectedImprovementEAConfig
import logging
import numpy as np
from bosot.bayesian_optimization.enums import AcquisitionFunctionType, ValidationType
from bosot.oracles.gp_model_evidence_oracle import GPModelEvidenceOracle
from bosot.kernels.kernel_grammar.generator_factory import GeneratorFactory
from bosot.oracles.test_oracle import TestOracle
from bosot.utils.plotter import Plotter
from bosot.utils.plotter2D import Plotter2D
from bosot.utils.utils import calculate_rmse

matplotlib_logger = logging.getLogger("matplotlib")
matplotlib_logger.setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

########### Start #########################

FAST_VERSION = True

PLOT_TOY_DATA = True

N_BO_STEPS = 40

N_DATA = 40

OBSERVATION_NOISE_DATA = 0.03

N_STEPS_EVOLUTIONARY_OPTIMIZER = 3

POPULATION_EV_OPTIMIZER = 100

############ Generate artifical dataset ####################

test_oracle = TestOracle(OBSERVATION_NOISE_DATA)

if PLOT_TOY_DATA:
    test_oracle.plot()

x_train, y_train = test_oracle.get_random_data(N_DATA)

x_test, y_test = test_oracle.get_random_data(200)

############ Configuration of Kernel seach via SOT ############

kernel_grammar_generator_config = CKSWithRQGeneratorConfig(input_dimension=2)

# build kernel grammar generator aka build the search space over kernels
kernel_grammar_generator = GeneratorFactory.build(kernel_grammar_generator_config)

# specify the oracle that should be optimized via BO aka the model selection criteria
oracle = GPModelEvidenceOracle(x_train, y_train, kernel_grammar_generator, fast_inference=FAST_VERSION, x_test=x_test, y_test=y_test)

# specfiy the kernel over kernels (define its config)
kernel_kernel_config = OTWeightedDimsExtendedGrammarKernelConfig(input_dimension=2)

# Build the GP Model over kernels
object_gp = ModelFactory.build(
    BasicObjectGPModelConfig(
        kernel_config=kernel_kernel_config, prediction_quantity=PredictionQuantity.PREDICT_F, perform_multi_start_optimization=False
    )
)

# set a constant mean function
object_gp.set_mean_function(ObjectConstant())

# specify BO variables like parameters for acquisition function optimizer or the kind of acquisition function
bo_config = ObjectBOExpectedImprovementEAConfig(
    n_steps_evolutionary=N_STEPS_EVOLUTIONARY_OPTIMIZER, population_evolutionary=POPULATION_EV_OPTIMIZER
)

# Initialize the Bayesian Optimizer
optimizer = BayesianOptimizerObjects(**bo_config.dict())

# Add all necessary objects to optimizer --> model, oracle, candidate_generator
optimizer.set_model(object_gp)

optimizer.set_oracle(oracle)

optimizer.set_candidate_generator(kernel_grammar_generator)

########### Start search procedure ################

# sample initial set of (kernel,oracle) pairs
optimizer.sample_train_set(24)

# Start BO aka start model selection
optimizer.maximize(N_BO_STEPS)

# Get the found kernel
best_kernel_expression = optimizer.get_current_best()

print("Best kernel:")
print(best_kernel_expression)

########## Use best kernel for prediction #############

best_kernel = best_kernel_expression.get_kernel()

gp_model_config = BasicGPModelConfig(kernel_config=BasicRBFConfig(input_dimension=0), set_prior_on_observation_noise=True)

gp_model = GPModel(best_kernel, **gp_model_config.dict())

gp_model.infer(x_train, y_train)

pred_mu, pred_sigma = gp_model.predictive_dist(x_test)

test_rmse = calculate_rmse(pred_mu, y_test)

print("Test-RMSE: " + str(test_rmse))
