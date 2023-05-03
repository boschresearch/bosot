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

import json
import logging
import os
from typing import Dict, List, Tuple

from bosot.bayesian_optimization.bayesian_optimizer_objects import BayesianOptimizerObjects
from bosot.bayesian_optimization.duration_time_predictors_objects import LinearTimePredictorKernelParameters
from bosot.bayesian_optimization.enums import AcquisitionFunctionType, AcquisitionOptimizationObjectBOType, ValidationType
from bosot.bayesian_optimization.greedy_kernel_search import GreedyKernelSearch
from bosot.configs.bayesian_optimization.greedy_kernel_search_configs import BaseGreedyKernelSearchConfig
from bosot.configs.bayesian_optimization.treeGEP_optimizer_configs import TreeGEPEvolutionaryOptimizerConfig
from bosot.configs.kernels.base_kernel_config import BaseKernelConfig
from bosot.configs.kernels.hellinger_kernel_kernel_configs import BasicHellingerKernelKernelConfig
from bosot.configs.models.gp_model_config import BasicGPModelConfig
from bosot.configs.models.object_gp_model_config import BasicObjectGPModelConfig
from bosot.data_sets.airfoil import Airfoil
from bosot.data_sets.airline_passenger import AirlinePassenger
from bosot.data_sets.lgbb import LGBB
from bosot.data_sets.power_plant import PowerPlant
from bosot.kernels.kernel_factory import KernelFactory
from bosot.kernels.kernel_grammar.generator_factory import GeneratorFactory
from bosot.kernels.kernel_grammar.kernel_grammar import BaseKernelGrammarExpression
from bosot.models.gp_model import GPModel, PredictionQuantity

from bosot.models.model_factory import ModelFactory
import numpy as np
import gpflow
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
import argparse
from bosot.utils.folder_handler import FolderHandler
from bosot.models.object_gp_model import ObjectGpModel
from bosot.models.object_mean_functions import BICMean, ObjectConstant
from bosot.oracles.gp_model_bic_oracle import GPModelBICOracle
from bosot.oracles.gp_model_cv_oracle import GPModelCVOracle
from bosot.oracles.gp_model_evidence_oracle import GPModelEvidenceOracle
from bosot.utils.plotter import Plotter
from bosot.utils.plotter2D import Plotter2D
from bosot.utils.projection_plotter import ProjectionPlotter
from bosot.bayesian_optimization.tree_gep_evolutionary_optimizer import TreeGEPEvoluationaryOptimizer

gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-4)
from gpflow.ci_utils import ci_niter

f64 = gpflow.utilities.to_default_float
import tensorflow as tf
from bosot.utils.utils import string2bool
from gpflow.utilities import print_summary, set_trainable, to_default_float
from bosot.configs.config_picker import ConfigPicker

matplotlib_logger = logging.getLogger("matplotlib")
matplotlib_logger.setLevel(logging.WARNING)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def write_config_to_working_dir(working_dir: str, name: str, config_dict: Dict):
    json_file_name = os.path.join(working_dir, name + "_config.json")
    with open(json_file_name, "w") as outfile:
        json.dump(json.loads(config_dict), outfile)


def add_experiment_sub_dir(experiment_output_dir, experiment_identifier):
    experiment_sub_dir = os.path.join(experiment_output_dir, experiment_identifier)
    if not os.path.exists(experiment_sub_dir):
        os.makedirs(experiment_sub_dir)
    return experiment_sub_dir


def write_sample_pairs_to_file(working_dir: str, pair_list: List[Tuple[BaseKernelGrammarExpression, float]], list_name: str):
    file_path = os.path.join(working_dir, list_name + ".txt")
    with open(file_path, "w") as textfile:
        for pair in pair_list:
            element, output = pair
            textfile.write(str(element) + " value=" + str(output) + "\n")


def write_test_metrics_to_file(working_dir: str, test_metrics_list: List):
    file_path = os.path.join(working_dir, "test_set_metrics.txt")
    with open(file_path, "w") as textfile:
        textfile.write("Index,TestRMSE,TestNLL\n")
        for index, test_rmse, test_nll in test_metrics_list:
            textfile.write(str(index) + "," + str(test_rmse) + "," + str(test_nll) + "\n")


def write_array_to_file(working_dir, array, file_name):
    file_path = os.path.join(working_dir, file_name)
    np.savetxt(file_path, array)


def parse_args():
    parser = argparse.ArgumentParser(
        description="This is a script for Structural Kernel Search via Bayesian Optimization and Symbolical Optimal Transport"
    )
    parser.add_argument("--experiment_output_dir")
    parser.add_argument("--data_dir")
    parser.add_argument("--n_steps")
    parser.add_argument("--run_name")
    parser.add_argument("--base_data_set_name", default="Powerplant")
    parser.add_argument("--target_function", default="Evidence")
    parser.add_argument("--base_data_n_train", default=500)
    parser.add_argument("--set_mean_function", default=True)
    parser.add_argument("--kernel_grammar_generator_config", default="CKSHighDimGeneratorConfig")
    parser.add_argument("--kernel_kernel_config", default="OTWeightedDimsExtendedGrammarKernelConfig")
    parser.add_argument("--use_heuristic", default=False)
    parser.add_argument("--num_stages_heuristics", default=5)
    parser.add_argument("--heuristic_search_config", default="GreedyKernelSearchBaseInitialConfig")
    parser.add_argument("--bayesian_optimizer_config", default="ObjectBOExpectedImprovementEAConfig")
    parser.add_argument("--override_run_identifier", default=True)
    parser.add_argument("--seed", default=1)
    parser.add_argument("--n_initial_kernels_factor", default=3)
    parser.add_argument("--base_data_seed", default=117)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    args = parse_args()

    experiment_output_dir = args.experiment_output_dir

    data_dir = args.data_dir

    kernel_kernel_config_name = args.kernel_kernel_config

    bayesian_optimizer_config_name = args.bayesian_optimizer_config

    kernel_grammar_generator_config_name = args.kernel_grammar_generator_config

    use_heuristic = string2bool(args.use_heuristic)

    num_stages_heuristics = int(args.num_stages_heuristics)

    greedy_kernel_search_config_name = args.heuristic_search_config

    target_function = args.target_function

    n_steps = int(args.n_steps)

    seed = int(args.seed)

    base_data_seed = int(args.base_data_seed)

    n_initial_kernels_factor = int(args.n_initial_kernels_factor)

    set_mean_function = string2bool(args.set_mean_function)

    base_data_n_train = int(args.base_data_n_train)

    base_data_set_name = args.base_data_set_name

    if target_function.lower() == "bic":
        selection_criteria_name = "BIC"
    elif target_function.lower() == "evidence":
        selection_criteria_name = "Evidence"
    elif target_function.lower() == "cv":
        selection_criteria_name = "CV"

    experiment_identifier = (
        base_data_set_name
        + "_"
        + str(base_data_seed)
        + "_ntrain_"
        + str(base_data_n_train)
        + "_"
        + selection_criteria_name
        + "_ninitfactor_"
        + str(n_initial_kernels_factor)
        + "_"
        + kernel_grammar_generator_config_name
    )

    experiment_sub_dir = add_experiment_sub_dir(experiment_output_dir, experiment_identifier)

    if use_heuristic:
        greedy_kernel_search_config = ConfigPicker.pick_greedy_kernel_search_config(greedy_kernel_search_config_name)()
        optimizer_name = greedy_kernel_search_config_name
    else:
        bayesian_optimizer_config = ConfigPicker.pick_bayesian_optimization_config(bayesian_optimizer_config_name)()
        if bayesian_optimizer_config.acquisition_function_type == AcquisitionFunctionType.EXPECTED_IMPROVEMENT:
            optimizer_name = "EI"
        elif bayesian_optimizer_config.acquisition_function_type == AcquisitionFunctionType.GP_UCB:
            optimizer_name = "UCB"
        elif bayesian_optimizer_config.acquisition_function_type == AcquisitionFunctionType.EXPECTED_IMPROVEMENT_PER_SECOND:
            optimizer_name = "EIperSec"
        if bayesian_optimizer_config.acquisiton_optimization_type == AcquisitionOptimizationObjectBOType.EVOLUTIONARY:
            optimizer_name = bayesian_optimizer_config_name + "EvOpt"

    override_run_identifier = string2bool(args.override_run_identifier)

    if override_run_identifier:
        run_identifier = str(args.run_name)
        assert len(run_identifier.split("_")) == 1
    else:
        if use_heuristic:
            run_identifier = optimizer_name
        else:
            run_identifier = kernel_kernel_config_name + optimizer_name

    folder_handler_object = FolderHandler(experiment_sub_dir, run_identifier, seed)

    folder_handler_object.initialize(False)

    workding_dir = folder_handler_object.get_working_dir_for_run()

    # Load base data

    possible_data_set_classes = [Airfoil, LGBB, PowerPlant, AirlinePassenger]

    DataSetClass = [
        data_set_class for data_set_class in possible_data_set_classes if data_set_class.__name__.lower() == base_data_set_name.lower()
    ][0]

    logger.info("Used base dataset class: " + str(DataSetClass))

    np.random.seed(base_data_seed)

    data_loader = DataSetClass(base_path=data_dir)

    data_loader.load_data_set()

    n_test = 400

    x_data, y_data, x_test, y_test = data_loader.sample_train_test(
        use_absolute=True, n_train=base_data_n_train, n_test=n_test, fraction_train=0.8
    )

    assert x_data.shape[0] == base_data_n_train

    # Main experiment - experiments vary over initial dataset (kernels, BIC pairs)

    n_dim = x_data.shape[1]

    kernel_grammar_generator_config = ConfigPicker.pick_kernel_grammar_generator_config(kernel_grammar_generator_config_name)(
        input_dimension=n_dim
    )

    kernel_grammar_generator = GeneratorFactory.build(kernel_grammar_generator_config)

    num_base_kernels = kernel_grammar_generator.search_space.get_num_base_kernels()

    n_data_initial = n_initial_kernels_factor * num_base_kernels

    if target_function.lower() == "bic":
        oracle = GPModelBICOracle(x_data, y_data, kernel_grammar_generator, fast_inference=False, x_test=x_test, y_test=y_test)
    elif target_function.lower() == "evidence":
        oracle = GPModelEvidenceOracle(x_data, y_data, kernel_grammar_generator, fast_inference=False, x_test=x_test, y_test=y_test)
    elif target_function.lower() == "cv":
        oracle = GPModelCVOracle(x_data, y_data, kernel_grammar_generator, fast_inference=False, x_test=x_test, y_test=y_test)

    if use_heuristic:
        write_config_to_working_dir(workding_dir, greedy_kernel_search_config_name, greedy_kernel_search_config.json())

        if isinstance(greedy_kernel_search_config, BaseGreedyKernelSearchConfig):
            optimizer = GreedyKernelSearch(**greedy_kernel_search_config.dict())
        elif isinstance(greedy_kernel_search_config, TreeGEPEvolutionaryOptimizerConfig):
            optimizer = TreeGEPEvoluationaryOptimizer(**greedy_kernel_search_config.dict())

        optimizer.set_oracle(oracle)

        optimizer.set_candidate_generator(kernel_grammar_generator)

        optimizer.sample_initial_dataset(n_data_initial, seed=seed, set_seed=True)

        metrics, query_list, best_list, test_set_metrics, iteration_time_list, oracle_time_list = optimizer.maximize(num_stages_heuristics)

        metric_name = ValidationType.get_name(ValidationType.MAX_OBSERVED)

        folder_handler_object.add_metric_for_run(metrics, metric_name, "KernelSpaceBO")

        write_sample_pairs_to_file(workding_dir, query_list, "QueryList")
        write_sample_pairs_to_file(workding_dir, best_list, "BestList")
        write_test_metrics_to_file(workding_dir, test_set_metrics)
        write_array_to_file(workding_dir, np.array(iteration_time_list), "iteration_time.txt")
        write_array_to_file(workding_dir, np.array(oracle_time_list), "oracle_time.txt")

    else:
        kernel_kernel_config = ConfigPicker.pick_kernel_config(kernel_kernel_config_name)(input_dimension=n_dim)

        object_gp_model_config = BasicObjectGPModelConfig(
            kernel_config=kernel_kernel_config, prediction_quantity=PredictionQuantity.PREDICT_F, perform_multi_start_optimization=False
        )

        if isinstance(kernel_kernel_config, BasicHellingerKernelKernelConfig):
            kernel = KernelFactory.build(kernel_kernel_config)
            kernel.set_virtual_x_from_dataset(x_data)
            object_gp = ObjectGpModel(kernel=kernel, **object_gp_model_config.dict())
        else:
            object_gp = ModelFactory.build(object_gp_model_config)

        if set_mean_function:
            object_gp.set_mean_function(ObjectConstant())

        duration_pediction_model = LinearTimePredictorKernelParameters()

        write_config_to_working_dir(workding_dir, kernel_kernel_config_name, kernel_kernel_config.json())

        write_config_to_working_dir(workding_dir, bayesian_optimizer_config_name, bayesian_optimizer_config.json())

        write_config_to_working_dir(workding_dir, kernel_grammar_generator_config_name, kernel_grammar_generator_config.json())

        optimizer = BayesianOptimizerObjects(**bayesian_optimizer_config.dict())

        optimizer.set_model(object_gp)

        optimizer.set_duration_time_predictor(duration_pediction_model)

        optimizer.set_oracle(oracle)

        optimizer.set_candidate_generator(kernel_grammar_generator)

        optimizer.sample_train_set(n_data_initial, seed=seed, set_seed=True)

        metrics, query_list, best_list, test_set_metrics, iteration_time_list, oracle_time_list, acquisition_time_list = optimizer.maximize(
            n_steps
        )

        metric_name = ValidationType.get_name(optimizer.validation_type)

        folder_handler_object.add_metric_for_run(metrics, metric_name, "KernelSpaceBO")

        write_sample_pairs_to_file(workding_dir, query_list, "QueryList")
        write_sample_pairs_to_file(workding_dir, best_list, "BestList")
        write_test_metrics_to_file(workding_dir, test_set_metrics)
        write_array_to_file(workding_dir, np.array(iteration_time_list), "iteration_time.txt")
        write_array_to_file(workding_dir, np.array(oracle_time_list), "oracle_time.txt")
        write_array_to_file(workding_dir, np.array(acquisition_time_list), "acquisition_time.txt")

    ## Plotting for final model

    gp_model_config = BasicGPModelConfig(kernel_config=BaseKernelConfig(name="dummy", input_dimension=0))
    kernel_expression = optimizer.get_current_best()
    model = GPModel(kernel_expression.get_kernel(), **gp_model_config.dict())
    model.infer(x_data, y_data)
    pred_mu, pred_sigma = model.predictive_dist(x_test)

    if n_dim == 1:
        plotter_object = Plotter(1)

        # plotter_object.add_gt_function(np.squeeze(x_test),np.squeeze(y_test),'black',0)

        plotter_object.add_datapoints(x_data, y_data, "green", 0)

        plotter_object.add_predictive_dist(np.squeeze(x_test), np.squeeze(pred_mu), np.squeeze(pred_sigma), 0)

        plotter_object.save_fig(workding_dir, "final_model_prediction.png")

    if n_dim == 2:
        plotter_object = Plotter2D(3)

        plotter_object.add_datapoints(x_data, "green", 0)

        plotter_object.add_gt_function(x_test, pred_sigma, "seismic", 40, 0)

        min_y = np.min(y_test)
        max_y = np.max(y_test)
        levels = np.linspace(min_y, max_y, 30)

        plotter_object.add_datapoints(x_data, "green", 1)

        plotter_object.add_gt_function(x_test, np.squeeze(pred_mu), "seismic", levels, 1)

        plotter_object.add_datapoints(x_data, "green", 2)
        plotter_object.add_gt_function(x_test, np.squeeze(y_test), "seismic", levels, 2)

        plotter_object.save_fig(workding_dir, "final_model_prediction.png")
