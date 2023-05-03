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
from abc import ABC, abstractmethod
from typing import List
from matplotlib.pyplot import axis
import numpy as np
from numpy import random
from bosot.bayesian_optimization.duration_time_predictors_objects import BaseDurationTimePredictorObjects
from bosot.bayesian_optimization.enums import AcquisitionFunctionType, AcquisitionOptimizationObjectBOType, ValidationType
from bosot.bayesian_optimization.evolutionary_optimizer_objects import EvolutionaryOptimizerObjects
from bosot.oracles.base_object_oracle import BaseObjectOracle
from bosot.models.object_gp_model import ObjectGpModel
import logging
from gpflow.utilities.traversal import tabulate_module_summary
from gpflow.config import default_summary_fmt
import time
from bosot.oracles.gp_model_bic_oracle import GPModelBICOracle
from bosot.oracles.gp_model_cv_oracle import GPModelCVOracle
from bosot.oracles.gp_model_evidence_oracle import GPModelEvidenceOracle
from bosot.bayesian_optimization.base_candidate_generator import CandidateGenerator
from scipy.stats import norm

logger = logging.getLogger(__name__)


class BayesianOptimizerObjects:
    """
    Class that implements Bayesian Optimization over structured objects. It uses CandidateGenerator objects that defines the
    search space and the object types. Furthermore it can use an EvolutionaryOptimizerObjects object for optimizing the acquisition function. It uses
    an GpModelObjects model as surrogate model (the kernel and mean function must be defined on the objects from the CandidateGenerator)

    Important Attributes:
        model : ObjectGpModel - surrogate GP model with a kernel and mean function that is defined on the same objects as the specified in the CandidateGenerator
        candidate_generator : CandidateGenerator - object that is used by the acquisiton optimization sheme to create lists of objects over which BO should take place
        oracle : BaseObjectOracle - oracle/function that should be maximized (should be defined over the objects specified in the CandidateGenerator)
        acquisition_function_type : AcquisitionFunctionType - kind of BO acquisition function specifed as Enum
        acquisiton_optimization_type : AcquisitionOptimizationObjectBOType - enum that decides the kind of acquisition function optimization e.g. Evolutionary
        population_evolutionary: int - population size in the evolutionary optimizer
        n_steps_evolutionary : int  - number of steps/generations in the evolutionary optimizer
        num_offspring_evolutionary : int - number of offspring in the evolutionary optimizer
    """

    def __init__(
        self,
        acquisition_function_type: AcquisitionFunctionType,
        validation_type: ValidationType,
        acquisiton_optimization_type: AcquisitionOptimizationObjectBOType,
        population_evolutionary: int,
        n_steps_evolutionary: int,
        num_offspring_evolutionary: int,
        n_prune_trailing: int,
        do_plotting: bool = False,
        **kwargs
    ):
        self.acquisition_function_type = acquisition_function_type
        self.validation_type = validation_type
        self.do_plotting = do_plotting
        self.acquisiton_optimization_type = acquisiton_optimization_type
        self.current_bests = []
        self.query_list = []
        self.additional_metrics = []
        self.iteration_time_list = []
        self.acquisition_time_list = []
        self.oracle_time_list = []
        self.query_durations_x_data = []
        self.additional_metrics_index_interval = 10
        self.n_prune_trailing = n_prune_trailing
        self.population_evolutionary = population_evolutionary
        self.n_steps_evolutionary = n_steps_evolutionary
        self.num_offspring_evolutionary = num_offspring_evolutionary
        self.validation_metrics = []
        self.oracle = None
        self.gp_ucb_beta = 3.0
        self.ei_xi = 0.01

    def set_oracle(self, oracle: BaseObjectOracle):
        self.oracle = oracle

    def set_model(self, model: ObjectGpModel):
        self.model = model

    def set_candidate_generator(self, generator: CandidateGenerator):
        self.candidate_generator = generator

    def set_duration_time_predictor(self, predictor: BaseDurationTimePredictorObjects):
        self.duration_time_predictor = predictor

    def set_evolutionary_opt_settings(self, population: int, n_steps: int, num_offspring: int):
        self.population_evolutionary = population
        self.n_steps_evolutionary = n_steps
        self.num_offspring_evolutionary = num_offspring

    def sample_train_set(self, n_data, seed=100, set_seed=False):
        """
        Samples the initial n_data datapoints (object_i,oracle(object_i)) via getting objects from the candidate generator and evaluating the oracle
        at the objects. This is used as the starting dataset for BO.

        Arguments:
            n_data : int - number of initial datapoints
        """
        self.x_data = self.candidate_generator.get_random_canditates(n_data, seed, set_seed)
        y_list = []
        logger.info("Sample train set")
        for x in self.x_data:
            logger.info("Initial datapoint: " + str(x))
            y, query_duration = self.oracle.query(x)
            logger.info("Output: " + str(y))
            y_list.append(y)
            self.query_durations_x_data.append(query_duration)
        self.y_data = np.expand_dims(np.array(y_list), axis=1)

    def set_train_set(self, x_train: List[object], y_train: np.array, query_durations_x_train: List[float]):
        """
        Method for setting the initial dataset manually
        """
        self.x_data = x_train
        self.y_data = y_train
        self.query_durations_x_data = query_durations_x_train
        assert len(self.query_durations_x_data) == len(self.x_data)

    def update(self, step: int):
        if self.acquisition_function_type == AcquisitionFunctionType.RANDOM:
            query = random.choice(self.candidates)
            return query, None

        if self.acquisition_function_type == AcquisitionFunctionType.EXPECTED_IMPROVEMENT_PER_SECOND:
            self.duration_time_predictor.fit(self.x_data, self.query_durations_x_data)

        self.model.reset_model()
        self.model.infer(self.x_data, self.y_data)
        if self.acquisiton_optimization_type == AcquisitionOptimizationObjectBOType.TRAILING_CANDIDATES:
            acquisition_values = self.acquisition_function(self.candidates)
            new_query = self.candidates[np.argmax(acquisition_values)]
            logger.info("Current best:")
            logger.info(self.get_current_best())
            return new_query, acquisition_values

        elif self.acquisiton_optimization_type == AcquisitionOptimizationObjectBOType.EVOLUTIONARY:
            optimizer = EvolutionaryOptimizerObjects(
                self.population_evolutionary, self.num_offspring_evolutionary, self.candidate_generator
            )
            new_query, _ = optimizer.maximize(self.acquisition_function, self.n_steps_evolutionary)
            return new_query, None

    def acquisition_function(self, x_grid: List[object]) -> np.array:
        if self.acquisition_function_type == AcquisitionFunctionType.GP_UCB:
            pred_mu, pred_sigma = self.model.predictive_dist(x_grid)
            acquisition_function = pred_mu + np.sqrt(self.gp_ucb_beta) * pred_sigma
        elif self.acquisition_function_type == AcquisitionFunctionType.EXPECTED_IMPROVEMENT:
            pred_mu_data, _ = self.model.predictive_dist(self.x_data)
            current_max = np.max(pred_mu_data)
            pred_mu_grid, pred_sigma_grid = self.model.predictive_dist(x_grid)
            d = pred_mu_grid - current_max - self.ei_xi
            acquisition_function = d * norm.cdf(d / pred_sigma_grid) + pred_sigma_grid * norm.pdf(d / pred_sigma_grid)
        elif self.acquisition_function_type == AcquisitionFunctionType.EXPECTED_IMPROVEMENT_PER_SECOND:
            pred_mu_data, _ = self.model.predictive_dist(self.x_data)
            current_max = np.max(pred_mu_data)
            pred_mu_grid, pred_sigma_grid = self.model.predictive_dist(x_grid)
            d = pred_mu_grid - current_max - self.ei_xi
            ei_acquisition_function = d * norm.cdf(d / pred_sigma_grid) + pred_sigma_grid * norm.pdf(d / pred_sigma_grid)
            duration_predictions = self.duration_time_predictor.predict(x_grid)
            assert len(ei_acquisition_function) == len(duration_predictions)
            acquisition_function = ei_acquisition_function / duration_predictions
        return acquisition_function

    def maximize(self, n_steps: int):
        """
        Main method - maximizes the oracle values over the search space via n_steps queries to the oracle

        Arguments:
            n_steps : int - number of BO steps
        """
        self.n_steps = n_steps
        if (
            self.acquisiton_optimization_type == AcquisitionOptimizationObjectBOType.TRAILING_CANDIDATES
            or self.acquisition_function_type == AcquisitionFunctionType.RANDOM
        ):
            self.candidates = self.candidate_generator.get_initial_candidates_trailing()
        self.add_additional_metrics(0)
        self.validate()
        for i in range(0, self.n_steps):
            time_before_iteration = time.perf_counter()
            time_before_acquisition = time.perf_counter()
            query, acquisition_values_candidates = self.update(i)
            time_after_acquisition = time.perf_counter()
            logger.info("Query:")
            logger.info(query)
            time_before_oracle = time.perf_counter()
            new_y, query_duration = self.oracle.query(query)
            time_after_oracle = time.perf_counter()
            logger.info("Oracle output:")
            logger.info(new_y)
            self.current_bests.append((self.get_current_best(), self.get_current_best_value()))
            self.query_list.append((query, np.float(new_y)))
            self.x_data.append(query)
            self.y_data = np.vstack((self.y_data, [new_y]))
            self.query_durations_x_data.append(query_duration)
            assert len(self.x_data) == len(self.query_durations_x_data)
            self.add_additional_metrics(i + 1)
            self.validate()
            if self.acquisiton_optimization_type == AcquisitionOptimizationObjectBOType.TRAILING_CANDIDATES:
                self.update_candidates(acquisition_values_candidates)
            if self.do_plotting:
                self.plot_validation_curve()
            time_after_iteration = time.perf_counter()
            time_diff_acquisition = time_after_acquisition - time_before_acquisition
            time_diff_oracle = time_after_oracle - time_before_oracle
            time_diff_iteration = time_after_iteration - time_before_iteration
            self.iteration_time_list.append(time_diff_iteration)
            self.acquisition_time_list.append(time_diff_acquisition)
            self.oracle_time_list.append(time_diff_oracle)
        return (
            np.array(self.validation_metrics),
            self.query_list,
            self.current_bests,
            self.additional_metrics,
            self.iteration_time_list,
            self.oracle_time_list,
            self.acquisition_time_list,
        )

    def update_candidates(self, acquisition_values):
        # First prune candidates to throw away candidates with low acquisition values and ones inside dataset
        self.candidates = self.get_pruned_candidates(acquisition_values)
        # Increase candidates with new random candidates and new candidates around the current best one
        self.candidates = self.candidates + self.candidate_generator.get_additional_candidates_trailing(self.get_current_best())

    def get_pruned_candidates(self, acquisition_values):
        pruned_candidates = []
        best_indexes = np.argsort(-1 * acquisition_values)[: self.n_prune_trailing]
        assert acquisition_values[best_indexes[0]] >= acquisition_values[best_indexes[1]]
        for index in best_indexes:
            candidate = self.candidates[index]
            if not self.check_if_in_list(candidate, pruned_candidates) and not self.check_if_in_list(candidate, self.x_data):
                pruned_candidates.append(self.candidates[index])
        return pruned_candidates

    def check_if_in_list(self, object_element, object_list):
        for object_list_element in object_list:
            if str(object_element) == str(object_list_element):
                return True
        return False

    def get_current_best(self):
        """
        Returns the object with highest oracle value found so far
        """
        return self.x_data[np.argmax(self.y_data)]

    def get_current_best_value(self):
        return np.max(self.y_data)

    def validate(self):
        if self.validation_type == ValidationType.MAX_OBSERVED:
            max_observed = np.max(self.y_data)
            self.validation_metrics.append(max_observed)

    def add_additional_metrics(self, index):
        if index % self.additional_metrics_index_interval == 0 or index == self.n_steps:
            if (
                isinstance(self.oracle, GPModelBICOracle)
                or isinstance(self.oracle, GPModelEvidenceOracle)
                or isinstance(self.oracle, GPModelCVOracle)
            ):
                additional_metrics = self.oracle.query_on_test_set(self.get_current_best())
                self.additional_metrics.append((index, *additional_metrics))
