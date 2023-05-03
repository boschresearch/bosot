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
from typing import List

import numpy as np
from bosot.configs.kernels.base_kernel_config import BaseKernelConfig
from bosot.configs.kernels.linear_configs import LinearWithPriorConfig
from bosot.configs.kernels.periodic_configs import PeriodicWithPriorConfig
from bosot.configs.kernels.rational_quadratic_configs import RQWithPriorConfig
from bosot.configs.kernels.rbf_configs import RBFWithPriorConfig
from bosot.kernels.kernel_grammar.kernel_grammar import (
    BaseKernelGrammarExpression,
    ElementaryKernelGrammarExpression,
    KernelGrammarExpression,
    KernelGrammarOperator,
)
from bosot.kernels.kernel_factory import KernelFactory


class BaseKernelGrammarSearchSpace:
    """
    A search space through the space of KernelGrammarExpressions. Given a list of base_kernels and operators it defines which
    KernelGrammarExpressions are the neighbours of a given KernelGrammarExpressions.
    """

    def __init__(self, input_dimension: int):
        self.input_dimension = input_dimension
        self.base_kernel_config_list: List[BaseKernelConfig] = None
        self.operator_list: List[KernelGrammarOperator] = None
        self.name = None

    def get_root_expressions(self) -> List[ElementaryKernelGrammarExpression]:
        """
        Retrieves the elementary base kernels of the search space - the single expressions can be used as the starting points of the search space
        """
        root_expressions = []
        for base_kernel_config in self.base_kernel_config_list:
            root_expression = ElementaryKernelGrammarExpression(KernelFactory.build(base_kernel_config))
            root_expression.set_generator_name(self.name)
            root_expressions.append(root_expression)
        return root_expressions

    def get_num_base_kernels(self) -> int:
        return len(self.base_kernel_config_list)

    def get_neighbour_expressions(self, grammar_expression: BaseKernelGrammarExpression) -> List[BaseKernelGrammarExpression]:
        """
        Returns all neighbour expression of grammar_expression according to the change operations described in "Structure Discovery in Nonparametric Regression through
        Compositional Kernel Search" (2013). First it generates all neighbours where one elementary expressions is exchanged with another elementary expression. Secondly it
        generates all neighbours where each subexpression/subtree is expanded with an operator and an elementary expression. It returns a list containing all neighbours.
        """
        # Exchange elementary expressions
        neighbour_expressions = []
        elementary_indexes = grammar_expression.get_indexes_of_elementary_expressions()
        for index in elementary_indexes:
            for base_kernel_config in self.base_kernel_config_list:
                new_expression = grammar_expression.deep_copy()
                new_base_kernel = KernelFactory.build(base_kernel_config)
                if isinstance(grammar_expression, ElementaryKernelGrammarExpression):
                    assert len(elementary_indexes) == 1
                    new_expression.set_kernel(new_base_kernel)
                elif isinstance(grammar_expression, KernelGrammarExpression):
                    new_expression.change_elementary_expression_at_index(index, new_base_kernel)
                new_expression.set_generator_name(self.name)
                neighbour_expressions.append(new_expression)
        # Extend subexpressions with all operators
        subexpression_indexes = grammar_expression.get_indexes_of_subexpression()
        for index in subexpression_indexes:
            for operator in self.operator_list:
                for base_kernel_config in self.base_kernel_config_list:
                    copied_expression = grammar_expression.deep_copy()
                    if isinstance(grammar_expression, ElementaryKernelGrammarExpression):
                        assert len(subexpression_indexes) == 1
                        new_elementary_expression = ElementaryKernelGrammarExpression(KernelFactory.build(base_kernel_config))
                        new_expression = KernelGrammarExpression(copied_expression, new_elementary_expression, operator)
                    elif isinstance(grammar_expression, KernelGrammarExpression):
                        copied_expression.extend_sub_expression_at_index(index, operator, KernelFactory.build(base_kernel_config))
                        new_expression = copied_expression
                    new_expression.set_generator_name(self.name)
                    neighbour_expressions.append(new_expression)
        return neighbour_expressions

    def get_num_neighbour_expressions(self, grammar_expression: BaseKernelGrammarExpression) -> int:
        """
        Fast calculation of the number of neighbours that will be generated for grammar_expression
        """
        elementary_indexes = grammar_expression.get_indexes_of_elementary_expressions()
        subexpression_indexes = grammar_expression.get_indexes_of_subexpression()
        num_neighbours = len(elementary_indexes) * len(self.base_kernel_config_list) + len(subexpression_indexes) * len(
            self.operator_list
        ) * len(self.base_kernel_config_list)
        return num_neighbours

    def get_neighbour_at_index(self, grammar_expression: BaseKernelGrammarExpression, index: int) -> BaseKernelGrammarExpression:
        """
        Fast access to one neighbour without generation of the other neighbours - index is the index of
        neighbour expression in the list that would be generated by the self.get_neighbour_expressions method
        """
        elementary_indexes = grammar_expression.get_indexes_of_elementary_expressions()
        subexpression_indexes = grammar_expression.get_indexes_of_subexpression()
        num_neighbours = self.get_num_neighbour_expressions(grammar_expression)
        assert index < num_neighbours
        if index < len(elementary_indexes) * len(self.base_kernel_config_list):
            elementary_index = elementary_indexes[int(index / len(self.base_kernel_config_list))]
            base_kernel_config = self.base_kernel_config_list[int(index % len(self.base_kernel_config_list))]
            new_expression = grammar_expression.deep_copy()
            new_base_kernel = KernelFactory.build(base_kernel_config)
            if isinstance(grammar_expression, ElementaryKernelGrammarExpression):
                assert len(elementary_indexes) == 1
                new_expression.set_kernel(new_base_kernel)
            elif isinstance(grammar_expression, KernelGrammarExpression):
                new_expression.change_elementary_expression_at_index(elementary_index, new_base_kernel)
        else:
            index = index - len(elementary_indexes) * len(self.base_kernel_config_list)
            subexpression_index = subexpression_indexes[int(index / (len(self.operator_list) * len(self.base_kernel_config_list)))]
            subindex = int(index % (len(self.operator_list) * len(self.base_kernel_config_list)))
            operator = self.operator_list[int(subindex / len(self.base_kernel_config_list))]
            base_kernel_config = self.base_kernel_config_list[int(subindex % len(self.base_kernel_config_list))]
            copied_expression = grammar_expression.deep_copy()
            if isinstance(grammar_expression, ElementaryKernelGrammarExpression):
                assert len(subexpression_indexes) == 1
                new_elementary_expression = ElementaryKernelGrammarExpression(KernelFactory.build(base_kernel_config))
                new_expression = KernelGrammarExpression(copied_expression, new_elementary_expression, operator)
            elif isinstance(grammar_expression, KernelGrammarExpression):
                copied_expression.extend_sub_expression_at_index(subexpression_index, operator, KernelFactory.build(base_kernel_config))
                new_expression = copied_expression
        new_expression.set_generator_name(self.name)
        return new_expression

    def get_random_neighbour_expression(self, grammar_expression: BaseKernelGrammarExpression) -> BaseKernelGrammarExpression:
        """
        Returns a random neighbour expression of grammar_expression (uniform over all possible neighbours)
        """
        num_neighbours = self.get_num_neighbour_expressions(grammar_expression)
        random_neighbour_index = np.random.choice(num_neighbours)
        random_neighbour_expression = self.get_neighbour_at_index(grammar_expression, random_neighbour_index)
        return random_neighbour_expression

    def random_walk(self, length: int, initial_expression: BaseKernelGrammarExpression) -> List[BaseKernelGrammarExpression]:
        """
        Generates a random walk through the search space. It starts with an initial_expression, picks a random neigbour of that expression. It repeates this
        process length times and returns all expressions that were visited.
        """
        expressions = []
        next_expression = initial_expression
        for step in range(0, length):
            next_expression = self.get_random_neighbour_expression(next_expression)
            expressions.append(next_expression)
        return expressions

    def check_expression_equality(self, expression1: BaseKernelGrammarExpression, expression2: BaseKernelGrammarExpression) -> bool:
        """
        Checks if two expressions are equivalent - in case the only operators are ADD and MULT it checks for equivalence considering the
        rules of addition and multiplication - for arbitrary operators it only checks direct equivalence of the expression tree
        """
        if all([operator in [KernelGrammarOperator.ADD, KernelGrammarOperator.MULTIPLY] for operator in self.operator_list]):
            return expression1.get_add_mult_invariant_hash()[0] == expression2.get_add_mult_invariant_hash()[0]
        else:
            return expression1.get_hash()[0] == expression2.get_hash()[0]


class CompositionalKernelSearchSpace(BaseKernelGrammarSearchSpace):
    def __init__(self, input_dimension: int):
        base_kernel_config_class_list = [RBFWithPriorConfig, LinearWithPriorConfig, PeriodicWithPriorConfig]
        self.base_kernel_config_list = []
        for base_kernel_config_class in base_kernel_config_class_list:
            for i in range(0, input_dimension):
                kernel_config = base_kernel_config_class(
                    input_dimension=input_dimension, active_on_single_dimension=True, active_dimension=i
                )
                self.base_kernel_config_list.append(kernel_config)
        self.operator_list = [KernelGrammarOperator.ADD, KernelGrammarOperator.MULTIPLY]
        self.name = self.__class__.__name__


class CKSWithRQSearchSpace(BaseKernelGrammarSearchSpace):
    def __init__(self, input_dimension: int):
        base_kernel_config_class_list = [RBFWithPriorConfig, LinearWithPriorConfig, PeriodicWithPriorConfig, RQWithPriorConfig]
        self.base_kernel_config_list = []
        for base_kernel_config_class in base_kernel_config_class_list:
            for i in range(0, input_dimension):
                kernel_config = base_kernel_config_class(
                    input_dimension=input_dimension, active_on_single_dimension=True, active_dimension=i
                )
                self.base_kernel_config_list.append(kernel_config)
        self.operator_list = [KernelGrammarOperator.ADD, KernelGrammarOperator.MULTIPLY]
        self.name = self.__class__.__name__


class CKSHighDimSearchSpace(BaseKernelGrammarSearchSpace):
    def __init__(self, input_dimension: int):
        base_kernel_config_class_list = [RBFWithPriorConfig, RQWithPriorConfig]
        self.base_kernel_config_list = []
        for base_kernel_config_class in base_kernel_config_class_list:
            for i in range(0, input_dimension):
                kernel_config = base_kernel_config_class(
                    input_dimension=input_dimension, active_on_single_dimension=True, active_dimension=i
                )
                self.base_kernel_config_list.append(kernel_config)
        self.operator_list = [KernelGrammarOperator.ADD, KernelGrammarOperator.MULTIPLY]
        self.name = self.__class__.__name__


class NDimFullKernelsSearchSpace(BaseKernelGrammarSearchSpace):
    def __init__(self, input_dimension: int):
        self.base_kernel_config_list = [
            RBFWithPriorConfig(input_dimension=input_dimension),
            LinearWithPriorConfig(input_dimension=input_dimension),
            PeriodicWithPriorConfig(input_dimension=input_dimension),
        ]
        self.operator_list = [KernelGrammarOperator.ADD, KernelGrammarOperator.MULTIPLY]
        self.name = self.__class__.__name__


if __name__ == "__main__":
    search_space = CKSWithRQSearchSpace(2)
    root_expression = search_space.get_root_expressions()[0]
    random_walk = search_space.random_walk(20, root_expression)
    for expression in random_walk:
        print(expression)
