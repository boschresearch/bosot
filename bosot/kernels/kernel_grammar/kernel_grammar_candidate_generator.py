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
import logging
from typing import List

import numpy as np
from bosot.bayesian_optimization.base_candidate_generator import CandidateGenerator
from bosot.kernels.kernel_grammar.kernel_grammar_search_spaces import BaseKernelGrammarSearchSpace, CompositionalKernelSearchSpace

logger = logging.getLogger(__name__)


class KernelGrammarCandidateGenerator(CandidateGenerator):
    def __init__(
        self,
        search_space: BaseKernelGrammarSearchSpace,
        n_initial_factor_trailing: int,
        n_exploration_trailing: int,
        exploration_p_geometric: float,
        n_exploitation_trailing: int,
        walk_length_exploitation_trailing: int,
        do_random_walk_exploitation_trailing: bool,
        **kwargs
    ):
        self.search_space = search_space
        self.n_initial_trailing = n_initial_factor_trailing * self.search_space.get_num_base_kernels()
        assert n_exploitation_trailing % walk_length_exploitation_trailing == 0
        self.n_exploitation_trailing = n_exploitation_trailing
        self.walk_length_exploitation_trailing = walk_length_exploitation_trailing
        self.n_exploration_trailing = n_exploration_trailing
        self.exploration_p_geometric = exploration_p_geometric
        self.do_random_walk_exploitation_trailing = do_random_walk_exploitation_trailing

    def get_random_canditates(self, n_candidates: int, seed=100, set_seed=False) -> List[object]:
        """
        Generates random candidates by performing random walks from each root expression in the search space
        """
        assert n_candidates % self.search_space.get_num_base_kernels() == 0
        if set_seed:
            np.random.seed(seed)
        depth = int((n_candidates / self.search_space.get_num_base_kernels()) - 1)
        random_candidates = []
        for root_expression in self.search_space.get_root_expressions():
            random_candidates.append(root_expression)
            random_candidates += self.search_space.random_walk(depth, root_expression)
        return random_candidates

    def get_initial_candidates_trailing(self) -> List[object]:
        """
        Returns the initial candidates for the trailing optimization in the Object-BO procedure
        """
        return self.get_random_canditates(self.n_initial_trailing)

    def get_additional_candidates_trailing(self, best_current_candidate: object) -> List[object]:
        root_expressions = self.search_space.get_root_expressions()
        additional_candidates = []
        # Add random walks from root (exploration)
        for _ in range(0, self.n_exploration_trailing):
            initial_expression = np.random.choice(root_expressions)
            length = np.random.geometric(self.exploration_p_geometric)
            additional_candidates += self.search_space.random_walk(length, initial_expression)
        # Add candidates around current best - either random walks with a specified walk length or all direct neighbours (exploitation)
        if self.do_random_walk_exploitation_trailing:
            n_walks = int(self.n_exploitation_trailing / self.walk_length_exploitation_trailing)
            for _ in range(0, n_walks):
                additional_candidates += self.search_space.random_walk(self.walk_length_exploitation_trailing, best_current_candidate)
        else:
            additional_candidates += self.search_space.get_neighbour_expressions(best_current_candidate)
        return additional_candidates

    def get_around_candidate_for_evolutionary_opt(self, candidate: object, n_around_candidate: int):
        """
        Generates random walks in the search space from a given candidate - used in object evolutionary algorithm
        """
        expression_list = []
        for i in range(0, n_around_candidate):
            new_expression = self.search_space.get_random_neighbour_expression(candidate)
            expression_list.append(new_expression)
        return expression_list

    def get_initial_for_evolutionary_opt(self, n_initial):
        return self.get_dataset_recursivly_generated(n_initial, 1)

    def get_dataset_recursivly_generated(self, n_data, n_per_step, filter_out_equivalent_expressions=False):
        expression_list = self.search_space.get_root_expressions()
        # recursivly add n_per_step around a randomly chosen element of the list to the list until n_data is reached
        while len(expression_list) < n_data:
            chosen_expression = np.random.choice(expression_list)
            for i in range(0, n_per_step):
                if len(expression_list) < n_data:
                    new_expression = self.search_space.get_random_neighbour_expression(chosen_expression)
                    if filter_out_equivalent_expressions:
                        while self.check_if_equivalent_expression_in_list(new_expression, expression_list):
                            logger.info("Expression already in list - sample new neighbour")
                            new_expression = self.search_space.get_random_neighbour_expression(chosen_expression)
                    expression_list.append(new_expression)
        return expression_list

    def check_if_equivalent_expression_in_list(self, expression, expression_list):
        for expression_in_list in expression_list:
            if self.search_space.check_expression_equality(expression_in_list, expression):
                return True
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    search_space = CompositionalKernelSearchSpace(2)
    generator = KernelGrammarCandidateGenerator(search_space, 3, 10, 0.25, 10, 2, True)
    for expression in generator.get_dataset_recursivly_generated(100, 1, True):
        print(expression)
