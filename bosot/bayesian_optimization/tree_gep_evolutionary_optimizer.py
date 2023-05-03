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
import random
import time
from bosot.kernels.kernel_grammar.kernel_grammar import (
    BaseKernelGrammarExpression,
    ElementaryKernelGrammarExpression,
    KernelGrammarExpression,
)
from bosot.kernels.kernel_grammar.kernel_grammar_candidate_generator import KernelGrammarCandidateGenerator
from bosot.oracles.base_object_oracle import BaseObjectOracle
import logging
import numpy as np
from bosot.oracles.gp_model_bic_oracle import GPModelBICOracle
from bosot.oracles.gp_model_cv_oracle import GPModelCVOracle

from bosot.oracles.gp_model_evidence_oracle import GPModelEvidenceOracle

logger = logging.getLogger(__name__)


class TreeGEPEvoluationaryOptimizer:
    def __init__(
        self, population_size: int, reproduction_rate: float, tournament_fraction: float, mutation_max_n: int, max_depth: int, **kwargs
    ):
        self.population_size = population_size
        self.reproduction_rate = reproduction_rate
        self.tournament_fraction = tournament_fraction
        self.tournament_size = int(self.population_size * self.tournament_fraction)
        self.test_set_metrics_index_interval = 10
        self.mutation_max_n = mutation_max_n
        self.max_depth = max_depth
        self.validation_metrics = []
        self.current_bests = []
        self.query_list = []
        self.iteration_time_list = []
        self.oracle_time_list = []
        self.test_set_metrics = []

    def set_candidate_generator(self, candidate_generator: KernelGrammarCandidateGenerator):
        self.candidate_generator = candidate_generator

    def set_oracle(self, oracle: BaseObjectOracle):
        self.oracle = oracle

    def get_current_best(self):
        return self.x_data[np.argmax(self.y_data)]

    def get_current_best_value(self):
        return np.max(self.y_data)

    def sample_initial_dataset(self, n_data, seed=100, set_seed=False):
        self.x_data = self.candidate_generator.get_random_canditates(n_data, seed, set_seed)
        y_list = []
        logger.info("Sample initial data set")
        for x in self.x_data:
            logger.info("Sample: " + str(x))
            y, _ = self.oracle.query(x)
            logger.info("Output: " + str(y))
            y_list.append(y)
        self.y_data = np.expand_dims(np.array(y_list), axis=1)

    def maximize(self, rounds: int):
        self.validation_metrics.append(self.get_current_best_value())
        self.current_bests.append((self.get_current_best(), self.get_current_best_value()))
        self.iteration_index = 0
        population = self.get_initial_population()
        time_stamp_iteration = time.perf_counter()
        assert len(population) == self.population_size
        ### Main Genetic algorithm
        for i in range(0, rounds):
            max_value = self.get_current_best_value()
            selected = self.select(population)
            offspring, survivors = self.reproduce(selected)
            population = survivors
            for new_candidate in offspring:
                y_new_candidate = self.query(new_candidate)
                population.append((new_candidate, y_new_candidate))
                iteration_time = time.perf_counter() - time_stamp_iteration
                self.iteration_time_list.append(iteration_time)
                time_stamp_iteration = time.perf_counter()
            assert len(population) == self.population_size
            population = self.set_dynamical_depth(population, max_value)
        return (
            np.array(self.validation_metrics),
            self.query_list,
            self.current_bests,
            self.test_set_metrics,
            self.iteration_time_list,
            self.oracle_time_list,
        )

    def set_dynamical_depth(self, population, max_value):
        new_population = []
        for individual in population:
            expression, y_value = individual
            if expression.count_elementary_expressions() <= self.max_depth:
                new_population.append(individual)
                if y_value > max_value:
                    max_value = y_value
            else:
                if y_value > max_value:
                    new_population.append(individual)
                    max_value = y_value
                    self.max_depth = expression.count_elementary_expressions()
                    logger.info("New max depth: " + str(self.max_depth))
        return new_population

    def get_initial_population(self):
        time_stamp_iteration = time.perf_counter()
        assert len(self.x_data) <= self.population_size
        population = [(self.x_data[i], self.y_data[i]) for i in range(0, len(self.x_data))]
        if self.population_size > len(self.x_data):
            n_additional_initial = self.population_size - len(self.x_data)
            for i in range(0, n_additional_initial):
                base_candidate = random.choice(self.x_data)
                new_candidate = self.candidate_generator.get_around_candidate_for_evolutionary_opt(base_candidate, 1)[0]
                y_new_candidate = self.query(new_candidate)
                population.append((new_candidate, y_new_candidate))
                iteration_time = time.perf_counter() - time_stamp_iteration
                self.iteration_time_list.append(iteration_time)
                time_stamp_iteration = time.perf_counter()
        return population

    def query(self, new_candidate):
        logger.info("Query: " + str(new_candidate))
        time_before_oracle = time.perf_counter()
        y_new_candidate, _ = self.oracle.query(new_candidate)
        time_after_oracle = time.perf_counter()
        oracle_time = time_after_oracle - time_before_oracle
        self.oracle_time_list.append(oracle_time)
        logger.info("Output: " + str(y_new_candidate))
        self.x_data.append(new_candidate)
        self.y_data = np.vstack((self.y_data, [y_new_candidate]))
        self.validation_metrics.append(self.get_current_best_value())
        self.current_bests.append((self.get_current_best(), self.get_current_best_value()))
        self.add_test_set_metrics(self.iteration_index)
        self.iteration_index += 1
        return y_new_candidate

    def select(self, population):
        ### Tournament selection
        selected = []
        for i in range(0, self.population_size):
            tournament = random.sample(population, self.tournament_size)
            winner = sorted(tournament, key=lambda x: x[1], reverse=True)[0]
            selected.append(winner)
        return selected

    def reproduce(self, selected):
        num_survivors = int(self.reproduction_rate * self.population_size)
        num_offspring = self.population_size - num_survivors
        survivors = random.sample(selected, num_survivors)
        offspring = []
        while len(offspring) < num_offspring:
            mutate = np.random.randint(2)
            if mutate:
                chosen_candidate = random.choice(selected)[0]
                new_candidate = self.mutate(chosen_candidate)
                logger.info("Offspring - mutation: " + str(new_candidate))
                offspring.append(new_candidate)
            else:
                chosen_mother = random.choice(selected)[0]
                chosen_father = random.choice(selected)[0]
                new_candidate1, new_candidate2 = self.cross_over(chosen_mother, chosen_father)
                offspring.append(new_candidate1)
                logger.info("Offspring - cross-over: " + str(new_candidate1))
                if len(offspring) < num_offspring:
                    offspring.append(new_candidate2)
                    logger.info("Offspring - cross-over: " + str(new_candidate2))
        return offspring, survivors

    def mutate(self, candidate: BaseKernelGrammarExpression):
        assert isinstance(self.candidate_generator, KernelGrammarCandidateGenerator)
        length_mutation = np.random.choice(np.arange(1, self.mutation_max_n))
        new_subtree = self.candidate_generator.get_random_candidate_n_operations(length_mutation)
        if isinstance(candidate, ElementaryKernelGrammarExpression):
            return new_subtree
        elif isinstance(candidate, KernelGrammarExpression):
            new_candidate = candidate.deep_copy()
            random_expression_index = random.choice(new_candidate.get_indexes_of_subexpression())
            if random_expression_index == [-1]:
                return new_subtree
            else:
                new_candidate.set_expression_at_index(random_expression_index, new_subtree)
                return new_candidate

    def cross_over(self, candidate1: BaseKernelGrammarExpression, candidate2: BaseKernelGrammarExpression):
        cand1_copy = candidate1.deep_copy()
        cand2_copy = candidate2.deep_copy()
        assert isinstance(cand1_copy, BaseKernelGrammarExpression)
        assert isinstance(cand2_copy, BaseKernelGrammarExpression)
        index_subtree_cand1 = random.choice(cand1_copy.get_indexes_of_subexpression())
        index_subtree_cand2 = random.choice(cand2_copy.get_indexes_of_subexpression())
        subtree_cand1 = cand1_copy.get_expression_at_index(index_subtree_cand1)
        subtree_cand2 = cand2_copy.get_expression_at_index(index_subtree_cand2)
        if isinstance(cand1_copy, ElementaryKernelGrammarExpression) or index_subtree_cand1 == [-1]:
            offspring1 = subtree_cand2
        else:
            cand1_copy.set_expression_at_index(index_subtree_cand1, subtree_cand2)
            offspring1 = cand1_copy
        if isinstance(cand2_copy, ElementaryKernelGrammarExpression) or index_subtree_cand2 == [-1]:
            offspring2 = subtree_cand1
        else:
            cand2_copy.set_expression_at_index(index_subtree_cand2, subtree_cand1)
            offspring2 = cand2_copy
        return offspring1, offspring2

    def add_test_set_metrics(self, index):
        if index % self.test_set_metrics_index_interval == 0:
            if (
                isinstance(self.oracle, GPModelBICOracle)
                or isinstance(self.oracle, GPModelEvidenceOracle)
                or isinstance(self.oracle, GPModelCVOracle)
            ):
                test_set_metric_tuple = self.oracle.query_on_test_set(self.get_current_best())
                self.test_set_metrics.append((index, *test_set_metric_tuple))
