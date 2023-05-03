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
import time
from bosot.kernels.kernel_grammar.kernel_grammar_candidate_generator import KernelGrammarCandidateGenerator
from bosot.oracles.base_object_oracle import BaseObjectOracle
import logging
import numpy as np

from bosot.oracles.gp_model_bic_oracle import GPModelBICOracle
from bosot.oracles.gp_model_cv_oracle import GPModelCVOracle
from bosot.oracles.gp_model_evidence_oracle import GPModelEvidenceOracle

logger = logging.getLogger(__name__)


class GreedyKernelSearch:
    def __init__(
        self, limit_num_visited_neighbours: bool, max_number_of_visited_neigbours: int, start_with_base_kernels_only: bool, **kwargs
    ):
        self.candidate_generator = None
        self.oracle = None
        self.max_number_of_visited_neigbours = max_number_of_visited_neigbours
        self.test_set_metrics_index_interval = 10
        self.limit_num_visited_neighbours = limit_num_visited_neighbours
        self.validation_metrics = []
        self.current_bests = []
        self.query_list = []
        self.iteration_time_list = []
        self.oracle_time_list = []
        self.test_set_metrics = []
        self.start_with_base_kernels_only = start_with_base_kernels_only
        self.n_diff_initial = 0

    def set_candidate_generator(self, candidate_generator: KernelGrammarCandidateGenerator):
        self.candidate_generator = candidate_generator

    def set_oracle(self, oracle: BaseObjectOracle):
        self.oracle = oracle

    def sample_initial_dataset(self, n_data, seed=100, set_seed=False):
        if self.start_with_base_kernels_only:
            if set_seed:
                np.random.seed(seed)
            logger.info("Start with base kernels only")
            self.x_data = self.candidate_generator.search_space.get_root_expressions()
            self.n_diff_initial = n_data - len(self.x_data)
            assert self.n_diff_initial >= 0
        else:
            self.x_data = self.candidate_generator.get_random_canditates(n_data, seed, set_seed)
        y_list = []
        logger.info("Sample initial data set")
        for x in self.x_data:
            logger.info("Sample: " + str(x))
            y, _ = self.oracle.query(x)
            logger.info("Output: " + str(y))
            y_list.append(y)
        self.y_data = np.expand_dims(np.array(y_list), axis=1)

    def get_current_best(self):
        return self.x_data[np.argmax(self.y_data)]

    def get_current_best_value(self):
        return np.max(self.y_data)

    def check_early_progress_to_next_stage(self, num_already_visited: int, max_at_stage: float):
        return (
            (num_already_visited > self.max_number_of_visited_neigbours)
            and self.limit_num_visited_neighbours
            and (self.get_current_best_value() > max_at_stage)
        )

    def maximize(self, depth: int):
        self.validation_metrics.append(self.get_current_best_value())
        self.current_bests.append((self.get_current_best(), self.get_current_best_value()))
        time_stamp_iteration = time.perf_counter()
        iteration_index = 0
        for step in range(0, depth):
            best_at_stage = self.get_current_best()
            max_at_stage = self.get_current_best_value()
            logger.info("Best: " + str(best_at_stage))
            neighbour_expressions = self.candidate_generator.search_space.get_neighbour_expressions(best_at_stage)
            np.random.shuffle(neighbour_expressions)
            for i, neighbour in enumerate(neighbour_expressions):
                logger.info("Query: " + str(neighbour))
                time_before_oracle = time.perf_counter()
                y_neighbour, _ = self.oracle.query(neighbour)
                self.query_list.append((neighbour, np.float(y_neighbour)))
                time_after_oracle = time.perf_counter()
                oracle_time = time_after_oracle - time_before_oracle
                self.oracle_time_list.append(oracle_time)
                logger.info("Output: " + str(y_neighbour))
                self.x_data.append(neighbour)
                self.y_data = np.vstack((self.y_data, [y_neighbour]))
                self.validation_metrics.append(self.get_current_best_value())
                self.current_bests.append((self.get_current_best(), self.get_current_best_value()))
                self.add_test_set_metrics(iteration_index)
                iteration_time = time.perf_counter() - time_stamp_iteration
                self.iteration_time_list.append(iteration_time)
                time_stamp_iteration = time.perf_counter()
                iteration_index += 1
                if self.check_early_progress_to_next_stage(i + 1, max_at_stage):
                    break
            # If no better kernel was found in the next stage the algorithm is done
            if str(best_at_stage) == str(self.get_current_best()) and not self.limit_num_visited_neighbours:
                break
        return (
            np.array(self.validation_metrics[self.n_diff_initial :]),
            self.query_list[self.n_diff_initial :],
            self.current_bests[self.n_diff_initial :],
            self.test_set_metrics,
            self.iteration_time_list[self.n_diff_initial :],
            self.oracle_time_list[self.n_diff_initial :],
        )

    def add_test_set_metrics(self, index):
        new_index = index - self.n_diff_initial
        if new_index % self.test_set_metrics_index_interval == 0 and new_index >= 0:
            if (
                isinstance(self.oracle, GPModelBICOracle)
                or isinstance(self.oracle, GPModelEvidenceOracle)
                or isinstance(self.oracle, GPModelCVOracle)
            ):
                test_set_metric_tuple = self.oracle.query_on_test_set(self.get_current_best())
                self.test_set_metrics.append((new_index, *test_set_metric_tuple))
