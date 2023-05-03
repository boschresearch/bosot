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
from typing import Dict, List, Optional, Tuple
import gpflow
from gpflow.utilities import print_summary
from enum import Enum
from abc import ABC, abstractmethod
import numpy as np
import pickle
import logging
from bosot.configs.kernels.linear_configs import LinearWithPriorConfig
from bosot.configs.kernels.periodic_configs import PeriodicWithPriorConfig
from bosot.configs.kernels.rational_quadratic_configs import BasicRQConfig
from bosot.configs.kernels.rbf_configs import BasicRBFConfig

from bosot.kernels.base_elementary_kernel import BaseElementaryKernel
from bosot.kernels.linear_kernel import LinearKernel
from bosot.kernels.periodic_kernel import PeriodicKernel
from bosot.kernels.rational_quadratic_kernel import RationalQuadraticKernel
from bosot.kernels.rbf_kernel import RBFKernel

logger = logging.getLogger(__name__)


class KernelGrammarOperator(Enum):
    ADD = 1
    MULTIPLY = 2


class BaseKernelGrammarExpression(ABC):
    """
    Base Object that represents a symbolic expression that corresponds to a kernel. It either represents a basic kernel (ElementaryKernelGrammarExpression)
    or a binary tree consisting of operator suchs as ADD and MULTIPLY as its nodes and base kernels as its children (KernelGrammarExpression). An instance of this class
    always correconds to a gpflow.kernel.Kernel instance which can be resolved via the get_kernel() method.
    """

    @abstractmethod
    def get_kernel(self) -> gpflow.kernels.Kernel:
        """
        Method to resolve/get the corresponding kernel object
        Return
            gpflow.kernels.Kernel - returns the associated kernel object
        """
        raise NotImplementedError

    @abstractmethod
    def deep_copy(self):
        """
        Returns a deep copy of itself - creates copies of all base kernels and subexpressions
        Return:
         BaseKernelGrammarExpression - a deep copy of its self
        """
        raise NotImplementedError

    @abstractmethod
    def count_elementary_expressions(self) -> int:
        """
        Counts number of elementary expressions (instances of ElementaryKernelGrammarExpression) inside the expression
        """
        raise NotImplementedError

    @abstractmethod
    def count_operators(self):
        """
        Counts the number of operators in the binary tree (only greater than 0 for instances of KernelGrammarExpression)
        """
        raise NotImplementedError

    @abstractmethod
    def get_name(self):
        """
        Returns a unique name based on the tree structure (operators) and the base kernels in the leaves
        """
        raise NotImplementedError

    @abstractmethod
    def get_hash(self) -> Tuple[int, List[Tuple[int, int, int]]]:
        """
        Calcuates hash of the expression based on hashes of all containing subexpressions -> makes it possible to count number
        of identical subexpressions - is used by the get_subtree_dict method
        Return:
         int - hash value of expression - hash is generated recursivly and is invariant to rotation of the nodes in the binary tree
         List[(int,int,int)] - list over all subexpressions with tuples containing: (hash values of subexpressions,number of elementary expressions inside subexpression,depth of the root of the subexpression)
        """
        raise NotImplementedError

    @abstractmethod
    def get_add_mult_invariant_hash(self) -> Tuple[int, List[Tuple[int, int, int]]]:
        raise NotImplementedError

    @abstractmethod
    def get_subtree_dict(self) -> Dict[int, List]:
        """
        Generates dict where each key is a hash of a subexpression and the value is a List containing the number how often the subexpression appeared inside this expression at index 0 -
        this method allows for example the tree grammar kernel kernel to count how often the same subexpression appeared in two different expressions - at index 1 a SubtreeMetaInformation
        object is stored associated to the subexpression
        """
        raise NotImplementedError

    @abstractmethod
    def get_add_mult_invariant_subtree_dict(self) -> Dict[int, List]:
        raise NotImplementedError

    @abstractmethod
    def get_input_dimension(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def get_operator(self) -> Optional[KernelGrammarOperator]:
        raise NotImplementedError

    @abstractmethod
    def push_down_one_mult(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def get_reduced_paths_to_elementaries(self, operators_to_reduce=[], use_name_as_key=False):
        raise NotImplementedError

    @abstractmethod
    def get_elementary_path_dict(self, operators_to_reduce=[]) -> Dict[int, List]:
        raise NotImplementedError

    @abstractmethod
    def get_elementary_count_dict(self) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def get_contains_elementary_dict(self) -> Dict:
        raise NotImplementedError

    @abstractmethod
    def get_elementary_below_operator_dict(self, operator: KernelGrammarOperator):
        raise NotImplementedError

    @abstractmethod
    def render_pickable(self):
        raise NotImplementedError

    @abstractmethod
    def set_generator_name(self, generator_name: str):
        """
        In case the grammar expression was generated by a kernel grammar generator the name of the generator could
        be set here - this allows downstream algorithms information of the possible elementary expressions and operators
        that can be present in the expression
        """
        raise NotImplementedError

    @abstractmethod
    def get_generator_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_indexes_of_subexpression(self) -> List[List[int]]:
        raise NotImplementedError

    @abstractmethod
    def get_indexes_of_elementary_expressions(self) -> List[List[int]]:
        raise NotImplementedError

    @abstractmethod
    def get_expression_at_index(self, index_list: List[int]):
        raise NotImplementedError

    def __str__(self) -> str:
        return self.get_name()


class SubtreeMetaInformation:
    def __init__(self, hash_value: int, num_elementary: int, depth_counter: int):
        self.hash_value = hash_value
        self.num_elementary = num_elementary
        self.depth_counter = depth_counter
        self.upstream_operators = []

    def increase_depth_count(self):
        self.depth_counter += 1

    def add_upstream_operator(self, operator: KernelGrammarOperator):
        self.upstream_operators.append(operator)

    def __str__(self):
        return (
            "[Hash: "
            + str(self.hash_value)
            + " num_elementary: "
            + str(self.num_elementary)
            + " depth_counter: "
            + str(self.depth_counter)
            + " upstream_operators: "
            + str(self.upstream_operators)
            + "]"
        )


class ElementarySubtreeMetaInformation(SubtreeMetaInformation):
    def __init__(self, hash_value: int, num_elementary: int, depth_counter: int, elementary_name: str):
        super().__init__(hash_value, num_elementary, depth_counter)
        self.elementary_name = elementary_name


class ElementaryPathMetaInformation:
    def __init__(self, elementary_hash: int, upstream_operator_list: List[KernelGrammarOperator]):
        self.elementary_hash = elementary_hash
        self.upstream_operator_list = upstream_operator_list
        self.hash = hash(str(self.upstream_operator_list + [self.elementary_hash]))

    def generate_operator_count_dict(self):
        operator_count_dict = {}
        for operator in self.upstream_operator_list:
            if operator in operator_count_dict:
                operator_count_dict[operator] += 1
            else:
                operator_count_dict[operator] = 1
        return operator_count_dict

    def get_hash(self):
        return self.hash


class ElementaryKernelGrammarExpression(BaseKernelGrammarExpression):
    def __init__(self, kernel: BaseElementaryKernel):
        assert isinstance(kernel, BaseElementaryKernel)
        self.kernel = kernel
        self.generator_name = None

    def set_kernel(self, kernel: BaseElementaryKernel):
        assert isinstance(kernel, BaseElementaryKernel)
        self.kernel = kernel

    def deep_copy(self):
        new_kernel = gpflow.utilities.deepcopy(self.kernel)
        assert isinstance(new_kernel, BaseElementaryKernel)
        return ElementaryKernelGrammarExpression(new_kernel)

    def count_elementary_expressions(self):
        return 1

    def count_operators(self):
        return 0

    def get_input_dimension(self):
        return self.kernel.get_input_dimension()

    def get_name(self):
        return self.kernel.name

    def render_pickable(self):
        gpflow.utilities.reset_cache_bijectors(self.kernel)

    def get_operator(self) -> Optional[KernelGrammarOperator]:
        return None

    def get_hash(self):
        hash_value = hash(self.kernel.name)
        subtree_meta_info = ElementarySubtreeMetaInformation(hash_value, 1, 0, self.get_name())
        return hash_value, [subtree_meta_info]

    def get_add_mult_invariant_hash(self):
        return self.get_hash()

    def get_kernel(self):
        return self.kernel

    def get_indexes_of_subexpression_internal(self) -> List[List[int]]:
        return [[None]]

    def get_indexes_of_subexpression(self) -> List[List[int]]:
        return [[-1]]

    def get_indexes_of_elementary_expressions(self) -> List[List[int]]:
        return [[-1]]

    def get_expression_at_index(self, index_list) -> BaseKernelGrammarExpression:
        assert len(index_list) == 1
        assert index_list[0] == -1
        return self

    def get_subtree_dict(self):
        subtree_meta_info = self.get_hash()[1][0]
        subtree_dict = {}
        subtree_dict[self.get_hash()[0]] = [1, subtree_meta_info]
        return subtree_dict

    def get_add_mult_invariant_subtree_dict(self):
        subtree_meta_info = self.get_add_mult_invariant_hash()[1][0]
        subtree_dict = {}
        subtree_dict[self.get_add_mult_invariant_hash()[0]] = [1, subtree_meta_info]
        return subtree_dict

    def get_reduced_paths_to_elementaries(self, operators_to_reduce=[], use_name_as_key=False):
        if use_name_as_key:
            return {self.get_name(): [[]]}
        else:
            return {self.get_hash()[0]: [[]]}

    def get_elementary_path_dict(self, operators_to_reduce=[]):
        path_meta_info_object = ElementaryPathMetaInformation(self.get_hash()[0], [])
        path_hash = path_meta_info_object.get_hash()
        final_dict = {}
        final_dict[path_hash] = [1, path_meta_info_object]
        return final_dict

    def get_elementary_count_dict(self) -> Dict:
        return {self.get_name(): 1}

    def get_contains_elementary_dict(self) -> Dict:
        return self.get_elementary_count_dict()

    def get_elementary_below_operator_dict(self, operator: KernelGrammarOperator):
        return {self.get_name(): 0}

    def push_down_one_mult(self) -> bool:
        return False

    def set_generator_name(self, generator_name: str):
        self.generator_name = generator_name

    def get_generator_name(self) -> str:
        return self.generator_name


class KernelGrammarExpression(BaseKernelGrammarExpression):
    def __init__(self, expression1: BaseKernelGrammarExpression, expression2: BaseKernelGrammarExpression, operator: KernelGrammarOperator):
        self.expression1: BaseKernelGrammarExpression = expression1
        self.expression2: BaseKernelGrammarExpression = expression2
        self.operator: KernelGrammarOperator = operator
        self.generator_name = None

    def get_kernel(self):
        if self.operator == KernelGrammarOperator.ADD:
            return self.expression1.get_kernel() + self.expression2.get_kernel()
        elif self.operator == KernelGrammarOperator.MULTIPLY:
            return self.expression1.get_kernel() * self.expression2.get_kernel()

    def get_operator(self) -> Optional[KernelGrammarOperator]:
        return self.operator

    def deep_copy(self):
        new_expression_1 = self.expression1.deep_copy()
        new_expression_2 = self.expression2.deep_copy()
        new_expression = KernelGrammarExpression(expression1=new_expression_1, expression2=new_expression_2, operator=self.operator)
        return new_expression

    def get_left_expression(self):
        return self.expression1

    def get_right_expression(self):
        return self.expression2

    def get_input_dimension(self):
        assert self.expression1.get_input_dimension() == self.expression2.get_input_dimension()
        return self.expression1.get_input_dimension()

    def render_pickable(self):
        self.expression1.render_pickable()
        self.expression2.render_pickable()

    def count_elementary_expressions(self):
        return self.expression1.count_elementary_expressions() + self.expression2.count_elementary_expressions()

    def count_operators(self):
        return self.expression1.count_operators() + self.expression2.count_operators() + 1

    def get_name(self):
        if self.operator == KernelGrammarOperator.ADD:
            return "(" + self.expression1.get_name() + " ADD " + self.expression2.get_name() + ")"
        elif self.operator == KernelGrammarOperator.MULTIPLY:
            return "(" + self.expression1.get_name() + " MULTIPLY " + self.expression2.get_name() + ")"

    def set_generator_name(self, generator_name: str):
        self.generator_name = generator_name

    def get_generator_name(self) -> str:
        return self.generator_name

    def get_indexes_of_subexpression_internal(self) -> List[List[int]]:
        list_left = [[0]]
        list_right = [[1]]
        list_expression_1 = self.expression1.get_indexes_of_subexpression_internal()
        for index_list in list_expression_1:
            extended_index_list = [0] + index_list
            list_left.append(extended_index_list)
        list_expression_2 = self.expression2.get_indexes_of_subexpression_internal()
        for index_list in list_expression_2:
            extended_index_list = [1] + index_list
            list_right.append(extended_index_list)
        return list_left + list_right

    def get_indexes_of_subexpression(self) -> List[List[int]]:
        """Returns lists of the form [0,0,1,0,...] for each subexpressions
        which indexes each subexpression with its own list -> it specifies the path in the tree to the subexpression
        [-1] is a special index list referencing to self/root - expressions corresponding to the index_list can be retrieved by
        get_expression_at_index
        """
        complete_list = self.get_indexes_of_subexpression_internal()
        reduced_list = [[-1]]
        for index_list in complete_list:
            if index_list[-1] is not None:
                reduced_list.append(index_list)
        return reduced_list

    def get_indexes_of_elementary_expressions(self) -> List[List[int]]:
        complete_list = self.get_indexes_of_subexpression_internal()
        reduced_list = []
        for index_list in complete_list:
            if index_list[-1] is None:
                reduced_list.append(index_list[:-1])
        return reduced_list

    def get_expression_at_index(self, index_list: List[int]) -> BaseKernelGrammarExpression:
        """
        Returns subexpression at index_list where index_list has the form [0,1,1,...] specifying the way down the tree
        0 go down expression1, 1 go down expression2. [-1] is the index_list for the expression itself
        """
        if len(index_list) == 1:
            if index_list[0] == 0:
                return self.expression1
            elif index_list[0] == 1:
                return self.expression2
            elif index_list[0] == -1:
                return self
        else:
            if index_list[0] == 0:
                assert isinstance(self.expression1, KernelGrammarExpression)
                return self.expression1.get_expression_at_index(index_list[1:])
            elif index_list[0] == 1:
                assert isinstance(self.expression2, KernelGrammarExpression)
                return self.expression2.get_expression_at_index(index_list[1:])

    def get_operator_hash(self):
        if self.operator == KernelGrammarOperator.ADD:
            hash_value = hash("ADD")
        elif self.operator == KernelGrammarOperator.MULTIPLY:
            hash_value = hash("MULTIPLY")
        return hash_value

    def get_hash(self):
        hash_collection = []
        hash_collection.append(self.get_operator_hash())
        hash_expr1, meta_info_list_expr1 = self.expression1.get_hash()
        hash_expr2, meta_info_list_expr2 = self.expression2.get_hash()
        for element in meta_info_list_expr1:
            element.increase_depth_count()
            element.add_upstream_operator(self.operator)
        for element in meta_info_list_expr2:
            element.increase_depth_count()
            element.add_upstream_operator(self.operator)
        hash_collection.append(hash_expr1)
        hash_collection.append(hash_expr2)
        hash_collection.sort()
        hash_value = hash(str(hash_collection))
        num_elementary_expressions = self.count_elementary_expressions()
        subtree_meta_info = SubtreeMetaInformation(hash_value, num_elementary_expressions, 0)
        subtree_meta_info_list = [subtree_meta_info] + meta_info_list_expr1 + meta_info_list_expr2
        return hash_value, subtree_meta_info_list

    def get_add_mult_invariant_hash(self):
        assert self.operator == KernelGrammarOperator.ADD or self.operator == KernelGrammarOperator.MULTIPLY
        hash_expr1, meta_info_list_expr1 = self.expression1.get_add_mult_invariant_hash()
        hash_expr2, meta_info_list_expr2 = self.expression2.get_add_mult_invariant_hash()
        for element in meta_info_list_expr1:
            element.increase_depth_count()
            element.add_upstream_operator(self.operator)
        for element in meta_info_list_expr2:
            element.increase_depth_count()
            element.add_upstream_operator(self.operator)
        if self.operator == KernelGrammarOperator.ADD:
            hash_value = hash_expr1 + hash_expr2
        elif self.operator == KernelGrammarOperator.MULTIPLY:
            hash_value = hash_expr1 * hash_expr2
        num_elementary_expressions = self.count_elementary_expressions()
        subtree_meta_info = SubtreeMetaInformation(hash_value, num_elementary_expressions, 0)
        subtree_meta_info_list = [subtree_meta_info] + meta_info_list_expr1 + meta_info_list_expr2
        return hash_value, subtree_meta_info_list

    def change_elementary_expression_at_index(self, index_list: List[int], new_kernel: gpflow.kernels.Kernel):
        if len(index_list) == 1:
            if index_list[0] == 0:
                assert isinstance(self.expression1, ElementaryKernelGrammarExpression)
                self.expression1 = ElementaryKernelGrammarExpression(new_kernel)
            elif index_list[0] == 1:
                assert isinstance(self.expression2, ElementaryKernelGrammarExpression)
                self.expression2 = ElementaryKernelGrammarExpression(new_kernel)
        else:
            if index_list[0] == 0:
                assert isinstance(self.expression1, KernelGrammarExpression)
                self.expression1.change_elementary_expression_at_index(index_list[1:], new_kernel)
            elif index_list[0] == 1:
                assert isinstance(self.expression2, KernelGrammarExpression)
                self.expression2.change_elementary_expression_at_index(index_list[1:], new_kernel)

    def change_random_elementary_expression(self, new_kernel: gpflow.kernels.Kernel):
        probability = 1.0
        if np.random.randint(2):
            current_expression = self.expression1
        else:
            current_expression = self.expression2
        probability = probability * 0.5

        while not isinstance(current_expression, ElementaryKernelGrammarExpression):
            if np.random.randint(2):
                current_expression = current_expression.expression1
            else:
                current_expression = current_expression.expression2
            probability = probability * 0.5

        assert isinstance(current_expression, ElementaryKernelGrammarExpression)
        current_expression.set_kernel(new_kernel)
        return probability

    def extend_sub_expression_at_index(self, index_list: List[int], operator: KernelGrammarOperator, new_kernel: gpflow.kernels.Kernel):
        if len(index_list) == 1:
            if index_list[0] == 0:
                self.expression1 = KernelGrammarExpression(self.expression1, ElementaryKernelGrammarExpression(new_kernel), operator)
            elif index_list[0] == 1:
                self.expression2 = KernelGrammarExpression(self.expression2, ElementaryKernelGrammarExpression(new_kernel), operator)
            elif index_list[0] == -1:
                self.expression1 = self.deep_copy()
                self.expression2 = ElementaryKernelGrammarExpression(new_kernel)
                self.operator = operator
        else:
            if index_list[0] == 0:
                assert isinstance(self.expression1, KernelGrammarExpression)
                self.expression1.extend_sub_expression_at_index(index_list[1:], operator, new_kernel)
            elif index_list[0] == 1:
                assert isinstance(self.expression2, KernelGrammarExpression)
                self.expression2.extend_sub_expression_at_index(index_list[1:], operator, new_kernel)

    def extend_random_sub_expression(
        self, operator: KernelGrammarOperator, new_kernel: gpflow.kernels.Kernel, skip_operator=False, operator_to_skip=None
    ):
        choose_first = np.random.randint(2)
        if choose_first:
            chosen_sub_expression = self.expression1
        else:
            chosen_sub_expression = self.expression2

        if isinstance(chosen_sub_expression, ElementaryKernelGrammarExpression):
            progress = False
        elif skip_operator and chosen_sub_expression.get_operator() == operator_to_skip:
            progress = True
        else:
            progress = np.random.randint(2)

        if progress:
            assert isinstance(chosen_sub_expression, KernelGrammarExpression)
            chosen_sub_expression.extend_random_sub_expression(operator, new_kernel)
        else:
            if choose_first:
                self.expression1 = KernelGrammarExpression(self.expression1, ElementaryKernelGrammarExpression(new_kernel), operator)
            else:
                self.expression2 = KernelGrammarExpression(self.expression2, ElementaryKernelGrammarExpression(new_kernel), operator)
            return

    def extend_top_level_operator(self, operator: KernelGrammarOperator, new_kernel: gpflow.kernels.Kernel):
        if self.operator == operator:
            choose_first = np.random.randint(2)
            if choose_first:
                chosen_sub_expression = self.expression1
            else:
                chosen_sub_expression = self.expression2

            if isinstance(chosen_sub_expression, ElementaryKernelGrammarExpression):
                progress = False
            elif not chosen_sub_expression.get_operator() == operator:
                progress = False
            else:
                progress = np.random.randint(2)

            if progress:
                assert isinstance(chosen_sub_expression, KernelGrammarExpression)
                chosen_sub_expression.extend_top_level_operator(operator, new_kernel)
            else:
                if choose_first:
                    self.expression1 = KernelGrammarExpression(self.expression1, ElementaryKernelGrammarExpression(new_kernel), operator)
                else:
                    self.expression2 = KernelGrammarExpression(self.expression2, ElementaryKernelGrammarExpression(new_kernel), operator)
                return
        else:
            self.expression1 = KernelGrammarExpression(self.expression1, self.expression2, self.operator)
            self.expression2 = ElementaryKernelGrammarExpression(new_kernel)
            self.operator = operator

    def get_subtree_dict(self):
        _, subtree_meta_info_list = self.get_hash()
        subtree_dict = {}
        for subtree_meta_info in subtree_meta_info_list:
            hash_value = subtree_meta_info.hash_value
            # num_elementary = subtree_meta_info.num_elementary
            if hash_value in subtree_dict:
                subtree_dict[hash_value][0] += 1
            else:
                subtree_dict[hash_value] = [1, subtree_meta_info]

        return subtree_dict

    def get_add_mult_invariant_subtree_dict(self):
        _, subtree_meta_info_list = self.get_add_mult_invariant_hash()
        subtree_dict = {}
        for subtree_meta_info in subtree_meta_info_list:
            hash_value = subtree_meta_info.hash_value
            # num_elementary = subtree_meta_info.num_elementary
            if hash_value in subtree_dict:
                subtree_dict[hash_value][0] += 1
            else:
                subtree_dict[hash_value] = [1, subtree_meta_info]
        return subtree_dict

    def get_paths_to_elementaries(self, use_name_as_key=False):
        _, subtree_meta_info_list = self.get_hash()
        paths = {}
        for subtree_meta_info in subtree_meta_info_list:
            hash_value = subtree_meta_info.hash_value
            num_elementary = subtree_meta_info.num_elementary
            if num_elementary == 1 and use_name_as_key:
                key = subtree_meta_info.elementary_name
            else:
                key = hash_value
            if key in paths:
                paths[key].append(subtree_meta_info.upstream_operators)
            elif num_elementary == 1:
                paths[key] = [subtree_meta_info.upstream_operators]
        return paths

    def get_reduced_paths_to_elementaries(self, operators_to_reduce=[], use_name_as_key=False):
        full_paths = self.get_paths_to_elementaries(use_name_as_key)
        reduced_paths = {}
        for elementary_hash in full_paths:
            reduced_paths[elementary_hash] = []
            for upstream_operator_list in full_paths[elementary_hash]:
                reduced_paths[elementary_hash].append(self.reduce_operator_list(upstream_operator_list, operators_to_reduce))
        return reduced_paths

    def reduce_operator_list(self, operator_list, operators_to_reduce):
        reduced_list = []
        previous_operator = None
        for operator in operator_list:
            if operator == previous_operator and operator in operators_to_reduce:
                pass
            else:
                reduced_list.append(operator)
            previous_operator = operator
        return reduced_list

    def get_elementary_count_dict(self) -> Dict:
        elementary_count_dict = {}
        elementary_count_dict_1 = self.expression1.get_elementary_count_dict()
        elementary_count_dict_2 = self.expression2.get_elementary_count_dict()
        for key in elementary_count_dict_1:
            elementary_count_dict[key] = elementary_count_dict_1[key]
        for key in elementary_count_dict_2:
            if key in elementary_count_dict:
                elementary_count_dict[key] += elementary_count_dict_2[key]
            else:
                elementary_count_dict[key] = elementary_count_dict_2[key]
        return elementary_count_dict

    def get_contains_elementary_dict(self) -> Dict:
        elementary_count_dict = self.get_elementary_count_dict()
        contains_elementary_dict = {}
        for key in elementary_count_dict:
            if elementary_count_dict[key] > 0:
                contains_elementary_dict[key] = 1
            else:
                contains_elementary_dict[key] = 0
        return contains_elementary_dict

    def get_elementary_path_dict(self, operators_to_reduce=[]):
        # Counts and indexes elementary paths e.g. ADD-> ADD-> MULTIPLY-> LINEAR
        # each path has a unique hash that is the key in this dict
        reduced_paths = self.get_reduced_paths_to_elementaries(operators_to_reduce)
        final_dict = {}
        for elementary_hash in reduced_paths:
            for path in reduced_paths[elementary_hash]:
                path_meta_info_object = ElementaryPathMetaInformation(elementary_hash, path)
                path_hash = path_meta_info_object.get_hash()
                if path_hash in final_dict:
                    final_dict[path_hash][0] += 1
                else:
                    final_dict[path_hash] = [1, path_meta_info_object]
        return final_dict

    def get_elementary_below_operator_dict(self, operator: KernelGrammarOperator):
        paths = self.get_paths_to_elementaries(True)
        final_dict = {}
        for elementary_name in paths:
            final_dict[elementary_name] = 0
            for upstream_operator_list in paths[elementary_name]:
                if operator in upstream_operator_list:
                    final_dict[elementary_name] += 1
        return final_dict

    def push_down_one_mult(self):
        """
        Pushes down multiplication and addition - does change the form of the expression! - does not change the resulting kernel
        Should only be used via the tansform to normal form method in the KernelGrammarExpressionTransformer
        """
        assert self.operator == KernelGrammarOperator.MULTIPLY or self.operator == KernelGrammarOperator.ADD
        if self.operator == KernelGrammarOperator.MULTIPLY:
            if self.get_left_expression().get_operator() == KernelGrammarOperator.ADD:
                self.operator = KernelGrammarOperator.ADD
                expression_right = self.get_right_expression()
                left_expression_left = self.get_left_expression().get_left_expression()
                left_expression_right = self.get_left_expression().get_right_expression()
                self.expression1 = KernelGrammarExpression(left_expression_left, expression_right, KernelGrammarOperator.MULTIPLY)
                self.expression2 = KernelGrammarExpression(left_expression_right, expression_right, KernelGrammarOperator.MULTIPLY)
                return True
            elif self.get_right_expression().get_operator() == KernelGrammarOperator.ADD:
                self.operator = KernelGrammarOperator.ADD
                expression_left = self.get_left_expression()
                right_expression_left = self.get_right_expression().get_left_expression()
                right_expression_right = self.get_right_expression().get_right_expression()
                self.expression1 = KernelGrammarExpression(right_expression_left, expression_left, KernelGrammarOperator.MULTIPLY)
                self.expression2 = KernelGrammarExpression(right_expression_right, expression_left, KernelGrammarOperator.MULTIPLY)
                return True

        pushed_down1 = self.expression1.push_down_one_mult()
        if pushed_down1:
            return True
        pushed_down2 = self.expression2.push_down_one_mult()
        if pushed_down2:
            return True
        return False


class KernelGrammarExpressionTransformer:
    @staticmethod
    def transform_to_normal_form(expression: BaseKernelGrammarExpression) -> BaseKernelGrammarExpression:
        """
        Produces a normal form representation - a sum over products representation - of a kernel grammar expression
        - resulting kernel grammar expression is associated with the same kernel.
        """
        new_expression = expression.deep_copy()
        new_expression.set_generator_name(expression.get_generator_name())
        assert isinstance(new_expression, BaseKernelGrammarExpression)
        mult_changed = new_expression.push_down_one_mult()
        while mult_changed:
            mult_changed = new_expression.push_down_one_mult()
        return new_expression


if __name__ == "__main__":
    base_expression_1 = ElementaryKernelGrammarExpression(RBFKernel(**BasicRBFConfig(input_dimension=2).dict()))
    base_expression_2 = ElementaryKernelGrammarExpression(LinearKernel(**LinearWithPriorConfig(input_dimension=2).dict()))
    base_expression_3 = ElementaryKernelGrammarExpression(LinearKernel(**LinearWithPriorConfig(input_dimension=2).dict()))
    base_expression_4 = ElementaryKernelGrammarExpression(RBFKernel(**BasicRBFConfig(input_dimension=2).dict()))
    extra_base_kernel = RationalQuadraticKernel(**BasicRQConfig(input_dimension=2).dict())
    expression1 = KernelGrammarExpression(base_expression_1, base_expression_2, operator=KernelGrammarOperator.ADD)
    expression2 = KernelGrammarExpression(base_expression_3, base_expression_4, operator=KernelGrammarOperator.ADD)
    expression3 = KernelGrammarExpression(expression1, expression2, operator=KernelGrammarOperator.MULTIPLY)
    expression4 = KernelGrammarExpressionTransformer.transform_to_normal_form(expression3)

    print("Expression1 ")
    for key, value in expression3.get_add_mult_invariant_subtree_dict().items():
        print("Hash: " + str(key))
        print(value)

    print("Expression2 ")
    for key, value in expression4.get_add_mult_invariant_subtree_dict().items():
        print("Hash: " + str(key))
        print(value)
