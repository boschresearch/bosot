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
from enum import Enum
from typing import Dict, List, Optional
import gpflow
from gpflow.config.__config__ import default_float
from gpflow.utilities.bijectors import positive
from tensorflow_probability import distributions as tfd
import numpy as np
from bosot.kernels.base_object_kernel import BaseObjectKernel
from bosot.kernels.kernel_grammar.kernel_grammar import (
    BaseKernelGrammarExpression,
    KernelGrammarExpressionTransformer,
    KernelGrammarOperator,
)
import tensorflow as tf
from bosot.kernels.kernel_grammar.optimal_transport_mappings import DimWiseWeightedDistanceExtractor

from bosot.utils.utils import manhatten_distance

gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-4)
from gpflow.ci_utils import ci_niter
import tensorflow_probability as tfp

f64 = gpflow.utilities.to_default_float


# Dict containing substructure information for one tree (one kernel in the kernel grammar)
# key=hash value of substructure
# List[0] - int - number of occurance of the substructure in the tree
# List[1] - object - meta info about substructure
StructuredDict = Dict[int, List]


class BaseKernelGrammarKernel(BaseObjectKernel):
    """
    Abstract class for a kernel over kernel expressions
    """

    def __init__(self, transform_to_normal: bool, **kwargs):
        self.transform_to_normal = transform_to_normal

    def K(self, X: List[BaseKernelGrammarExpression], X2: Optional[List[BaseKernelGrammarExpression]] = None):
        raise NotImplementedError

    def K_diag(self, X: List[BaseKernelGrammarExpression]):
        raise NotImplementedError

    def transform_X(self, X: List[BaseKernelGrammarExpression]) -> List[BaseKernelGrammarExpression]:
        if self.transform_to_normal:
            new_X = []
            for x in X:
                x_normal_form = KernelGrammarExpressionTransformer.transform_to_normal_form(x)
                new_X.append(x_normal_form)
            return new_X
        return X


class FeatureType(Enum):
    SUBTREES = 0
    REDUCED_ELEMENTARY_PATHS = 1
    DIM_WISE_WEIGHTED_ELEMENTARY_COUNT = 2


class OptimalTransportKernelKernel(BaseKernelGrammarKernel):
    """
    Kernel over kernels presented in paper. It uses optimial transport/wasserstein distance over features from the kernel grammar expressions

    Important Attributes:
        feature_type_list: List[FeatureType] - list of features that should be extracted from the kernel grammar tree e.g. subtree features (see paper for details)
        variance : gpflow.Parameter - variance parameter of the kernel-kernel
        lengthscale: gpflow.Parameter - lengthscale parameter of the kernel-kernel
        alphas : gpflow.Parameter - weighting parameters of the features in the OT distance
    """

    def __init__(
        self,
        feature_type_list: List[FeatureType],
        base_variance: float,
        base_lengthscale: float,
        base_alpha: float,
        alpha_trainable: bool,
        parameters_trainable: bool,
        transform_to_normal: bool,
        **kwargs
    ):
        super().__init__(transform_to_normal)
        transform = tfp.bijectors.Sigmoid(low=None, high=None, validate_args=False, name="sigmoid")
        train_alpha = parameters_trainable and alpha_trainable
        self.alphas = gpflow.Parameter(f64(np.repeat(base_alpha, len(feature_type_list))), transform=transform, trainable=train_alpha)
        self.lengthscale = gpflow.Parameter(f64(base_lengthscale), transform=positive(), trainable=parameters_trainable)
        self.variance = gpflow.Parameter(f64(base_variance), transform=positive(), trainable=parameters_trainable)
        self.feature_type_list = feature_type_list

    def K(self, X: List[BaseKernelGrammarExpression], X2: Optional[List[BaseKernelGrammarExpression]] = None):
        distance_matrix = self.wasserstein_distance(X, X2)
        K = self.variance * tf.math.exp(-1 * distance_matrix / tf.pow(self.lengthscale, 2.0))
        return K

    def K_diag(self, X: List[BaseKernelGrammarExpression]):
        distance_matrix = self.wasserstein_distance(X)
        diag_distance = tf.linalg.diag_part(distance_matrix)
        K_diag = self.variance * tf.math.exp(-1 * diag_distance / tf.pow(self.lengthscale, 2.0))
        return K_diag

    def get_distance_matrix(self, X: List[BaseKernelGrammarExpression], X2: Optional[List[BaseKernelGrammarExpression]] = None):
        return self.wasserstein_distance(X, X2)

    def wasserstein_distance(self, X: List[BaseKernelGrammarExpression], X2: Optional[List[BaseKernelGrammarExpression]] = None):
        """
        Main method to calculate wasserstein/OT distance between grammar expressions - it loops over the feature list
        calcuates the manhatten distances of the features and creates a weighted sum of these distances (see paper for relation to OT)
        """
        if X2 is None:
            manhatten_distances = []
            for feature_type in self.feature_type_list:
                X_feature = self.internal_transform_X(X, feature_type)
                index_dict_feature = self.create_hash_array_mapping(X_feature)
                feature_matrix = tf.convert_to_tensor(
                    self.feature_matrix(X_feature, index_dict_feature, normalize=self.normalize_features(feature_type)),
                    dtype=default_float(),
                )
                manhatten_distance_feature = manhatten_distance(feature_matrix, feature_matrix)
                manhatten_distances.append(manhatten_distance_feature)
            distance_weighting = tf.expand_dims(tf.expand_dims(self.alphas / tf.reduce_sum(self.alphas), axis=1), axis=2)
            manhatten_distances_stacked = tf.stack(manhatten_distances)
            distance_matrix = tf.reduce_sum(tf.multiply(distance_weighting, manhatten_distances_stacked), axis=0)
        else:
            manhatten_distances = []
            for feature_type in self.feature_type_list:
                X_feature = self.internal_transform_X(X, feature_type)
                X2_feature = self.internal_transform_X(X2, feature_type)
                index_dict_feature = self.create_hash_array_mapping(X_feature + X2_feature)
                X_feature_matrix = tf.convert_to_tensor(
                    self.feature_matrix(X_feature, index_dict_feature, normalize=self.normalize_features(feature_type)),
                    dtype=default_float(),
                )
                X2_feature_matrix = tf.convert_to_tensor(
                    self.feature_matrix(X2_feature, index_dict_feature, normalize=self.normalize_features(feature_type)),
                    dtype=default_float(),
                )
                manhatten_distance_feature = manhatten_distance(X_feature_matrix, X2_feature_matrix)
                manhatten_distances.append(manhatten_distance_feature)
            distance_weighting = tf.expand_dims(tf.expand_dims(self.alphas / tf.reduce_sum(self.alphas), axis=1), axis=2)
            manhatten_distances_stacked = tf.stack(manhatten_distances)
            distance_matrix = tf.reduce_sum(tf.multiply(distance_weighting, manhatten_distances_stacked), axis=0)
        return distance_matrix

    def internal_transform_X(self, X: List[BaseKernelGrammarExpression], feature_type: FeatureType) -> List[StructuredDict]:
        """
        Method for extrating the features from a list of kernel grammar expression - each feature collection for one kernel grammar expression
        is represented by a StructuredDict where the concrete feature value e.g. a subtree is encoded in the hash of the dict and number of occurances
        of the feature is the first entry in the dict value (see top) --> method returns a list of StructuredDicts
        """
        dict_list = []
        if feature_type == FeatureType.DIM_WISE_WEIGHTED_ELEMENTARY_COUNT:
            dim_wise_mapper = DimWiseWeightedDistanceExtractor(X[0].get_generator_name(), X[0].get_input_dimension())

        for kernel_grammar_expression in X:
            if feature_type == FeatureType.SUBTREES:
                feature_dict = kernel_grammar_expression.get_subtree_dict()
            elif feature_type == FeatureType.REDUCED_ELEMENTARY_PATHS:
                feature_dict = kernel_grammar_expression.get_elementary_path_dict(
                    [KernelGrammarOperator.ADD, KernelGrammarOperator.MULTIPLY]
                )
            elif feature_type == FeatureType.DIM_WISE_WEIGHTED_ELEMENTARY_COUNT:
                feature_dict = dim_wise_mapper.get_dim_wise_weighted_elementary_features(kernel_grammar_expression)
            self.check_feature_dict(feature_dict)
            dict_list.append(feature_dict)
        return dict_list

    def check_feature_dict(self, feature_dict):
        has_value = False
        for key in feature_dict:
            if isinstance(feature_dict[key], list):
                if feature_dict[key][0] > 0:
                    has_value = True
            else:
                if feature_dict[key] > 0:
                    has_value = True
        assert has_value

    def normalize_features(self, feature_type):
        if feature_type == FeatureType.DIM_WISE_WEIGHTED_ELEMENTARY_COUNT:
            return False
        return True

    def create_hash_array_mapping(self, tree_dict_list: List[StructuredDict]) -> StructuredDict:
        index_dict = {}
        index = 0
        for tree_dict in tree_dict_list:
            for hash in tree_dict:
                if not hash in index_dict:
                    index_dict[hash] = [index]
                    index += 1
        return index_dict

    def feature_matrix(self, X: List[StructuredDict], index_dict: StructuredDict, normalize: bool) -> np.array:
        feature_matrix = []
        for feature_dict in X:
            feature_vec = self.feature_dict_to_feature_vector(feature_dict, index_dict)
            if normalize:
                feature_vec = feature_vec / np.sum(feature_vec)
            feature_matrix.append(feature_vec)
        return np.array(feature_matrix)

    def feature_dict_to_feature_vector(self, feature_dict: StructuredDict, index_dict: StructuredDict):
        feature_vec = np.zeros(len(index_dict))
        for key in feature_dict:
            if isinstance(feature_dict[key], list):
                assert isinstance(feature_dict[key][0], int)
                feature_vec[index_dict[key][0]] = feature_dict[key][0]
            else:
                feature_vec[index_dict[key][0]] = feature_dict[key]
        return feature_vec

    def transform_X(self, X: List[BaseKernelGrammarExpression]) -> List[BaseKernelGrammarExpression]:
        return super().transform_X(X)


if __name__ == "__main__":
    pass
