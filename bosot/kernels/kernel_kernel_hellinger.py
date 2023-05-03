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
from gpflow.base import Transform
from gpflow.config.__config__ import default_float
from gpflow.utilities.bijectors import positive
from tensorflow.python.ops.gen_math_ops import exp
from bosot.configs import kernels
from bosot.configs.kernels.kernel_grammar_generators.cks_with_rq_generator_config import CKSWithRQGeneratorConfig
from bosot.configs.kernels.linear_configs import LinearWithPriorConfig
from bosot.configs.kernels.rbf_configs import RBFWithPriorConfig
from bosot.kernels.base_object_kernel import BaseObjectKernel
import tensorflow as tf
from tensorflow_probability import distributions as tfd
from typing import List, Optional, Tuple, Union
import gpflow
import numpy as np
import logging
from bosot.kernels.kernel_grammar.kernel_grammar import (
    BaseKernelGrammarExpression,
    ElementaryKernelGrammarExpression,
    KernelGrammarExpression,
    KernelGrammarOperator,
)
from bosot.kernels.linear_kernel import LinearKernel
from bosot.kernels.rbf_kernel import RBFKernel
from bosot.utils.utils import k_means

gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-4)
from gpflow.ci_utils import ci_niter

f64 = gpflow.utilities.to_default_float

logger = logging.getLogger(__name__)


class EvaluatedKernelWrapper:
    def __init__(self, gram_matrix_list, log_det_list, kernel_identifier: str):
        self.gram_matrix_list = gram_matrix_list
        self.log_det_list = log_det_list
        self.kernel_identifier = kernel_identifier

    def get_kernel_identifier(self):
        return self.kernel_identifier


class EvaluatedKernelCache:
    def __init__(self):
        self.cache_dict = {}

    def add_to_cache(self, kernel_identifier: str, evaluated_kernel_wrapper: EvaluatedKernelWrapper):
        self.cache_dict[kernel_identifier] = evaluated_kernel_wrapper

    def get_from_cache(self, kernel_identifier: str):
        if kernel_identifier in self.cache_dict:
            return self.cache_dict[kernel_identifier]
        return None

    def check_if_in_cache(self, kernel_identifier: str):
        return kernel_identifier in self.cache_dict

    def clear_cache(self):
        self.cache_dict = {}

    def get_number_of_cached_kernels(self):
        return len(self.cache_dict)


class ExpectedHellingerDistanceCache:
    def __init__(self) -> None:
        self.cache_dict = {}

    def add_to_cache(self, kernel_identifier_k1: str, kernel_identifier_k2: str, expected_hellinger_distance: np.float):
        if not kernel_identifier_k1 in self.cache_dict:
            self.cache_dict[kernel_identifier_k1] = {}
        if not kernel_identifier_k2 in self.cache_dict:
            self.cache_dict[kernel_identifier_k2] = {}
        self.cache_dict[kernel_identifier_k1][kernel_identifier_k2] = expected_hellinger_distance
        self.cache_dict[kernel_identifier_k2][kernel_identifier_k1] = expected_hellinger_distance

    def check_if_in_cache(self, kernel_identifier_k1: str, kernel_identifier_k2: str):
        return kernel_identifier_k1 in self.cache_dict and kernel_identifier_k2 in self.cache_dict[kernel_identifier_k1]

    def get_from_cache(self, kernel_identifier_k1: str, kernel_identifier_k2: str):
        if self.check_if_in_cache(kernel_identifier_k1, kernel_identifier_k2):
            return self.cache_dict[kernel_identifier_k1][kernel_identifier_k2]
        return None


class KernelKernelHellinger(BaseObjectKernel):
    def __init__(
        self,
        input_dimension: int,
        base_lengthscale: float,
        base_variance: float,
        num_param_samples: int,
        num_virtual_points: int,
        use_sobol_virtual_points: bool,
        use_hyperprior: bool,
        lengthscale_prior_parameters: Tuple[float, float],
        variance_prior_parameters: Tuple[float, float],
        **kwargs
    ):
        self.lengthscale = gpflow.Parameter(f64(base_lengthscale), transform=positive())
        self.variance = gpflow.Parameter(f64(base_variance), transform=positive())
        if use_hyperprior:
            ls_prior_mu, ls_prior_sigma = lengthscale_prior_parameters
            v_prior_mu, v_prior_sigma = variance_prior_parameters
            self.lengthscale.prior = tfd.LogNormal(loc=f64(ls_prior_mu), scale=f64(ls_prior_sigma))
            self.variance.prior = tfd.LogNormal(loc=f64(v_prior_mu), scale=f64(v_prior_sigma))
        self.num_param_samples = num_param_samples
        self.num_virtual_points = num_virtual_points
        if use_sobol_virtual_points:
            self.virtual_X = f64(tf.math.sobol_sample(input_dimension, self.num_virtual_points)).numpy()
        else:
            self.virtual_X = None  # needs to set via set_virtual_x_from_dataset
        self.evaluated_kernel_cache = EvaluatedKernelCache()
        self.expected_distance_cache = ExpectedHellingerDistanceCache()
        self.jitter = 1e-6

    def K(self, X: List[EvaluatedKernelWrapper], X2: Optional[List[EvaluatedKernelWrapper]] = None) -> tf.Tensor:
        hellinger_distances = self.get_hellinger_distance_matrix(
            X, X2
        )  # contains only numpy calculation to not slow down computations under tf.GradientTape
        K = self.variance * tf.math.exp(-0.5 * (hellinger_distances / tf.pow(self.lengthscale, 2.0)))
        return K

    def K_diag(self, X: List[gpflow.kernels.Kernel]):
        diag = self.variance * tf.ones(len(X), dtype=default_float())
        return diag

    def get_hellinger_distance_matrix(self, X: List[EvaluatedKernelWrapper], X2: Optional[List[EvaluatedKernelWrapper]] = None):
        n1 = len(X)
        if X2 is None:
            hellinger_distances = np.zeros((n1, n1))
            for i in range(0, n1):
                for j in range(i, n1):
                    calculated_hellinger_distance = self.expected_hellinger_distance(X[i], X[j])
                    if i == j:
                        hellinger_distances[i, i] = calculated_hellinger_distance
                    else:
                        hellinger_distances[i, j] = calculated_hellinger_distance
                        hellinger_distances[j, i] = calculated_hellinger_distance
        else:
            n2 = len(X2)
            hellinger_distances = np.zeros((n1, n2))
            for i in range(0, n1):
                for j in range(0, n2):
                    hellinger_distances[i, j] = self.expected_hellinger_distance(X[i], X2[j])
        return hellinger_distances

    def expected_hellinger_distance(
        self, evaluated_kernel_wrapper_k1: EvaluatedKernelWrapper, evaluated_kernel_wrapper_k2: EvaluatedKernelWrapper
    ):
        k1_identifier = evaluated_kernel_wrapper_k1.kernel_identifier
        k2_identifier = evaluated_kernel_wrapper_k2.kernel_identifier
        if self.expected_distance_cache.check_if_in_cache(k1_identifier, k2_identifier):
            return self.expected_distance_cache.get_from_cache(k1_identifier, k2_identifier)
        grams_kernel1 = evaluated_kernel_wrapper_k1.gram_matrix_list
        dets_kernel1 = evaluated_kernel_wrapper_k1.log_det_list
        grams_kernel2 = evaluated_kernel_wrapper_k2.gram_matrix_list
        dets_kernel2 = evaluated_kernel_wrapper_k2.log_det_list
        expected_distance_value = 0.0
        counter = 0
        for i in range(0, self.num_param_samples):
            K1 = grams_kernel1[i]
            K2 = grams_kernel2[i]
            log_det_K1 = dets_kernel1[i]
            log_det_K2 = dets_kernel2[i]
            hellinger_distance = self.hellinger_distance(K1, log_det_K1, K2, log_det_K2)
            # in the rare case of NaNs or negative distances because of numerical instabilities exclude these samples
            if not np.isnan(hellinger_distance).any() and hellinger_distance >= 0:
                expected_distance_value += hellinger_distance
                counter += 1
        expected_distance_value *= 1.0 / counter
        self.expected_distance_cache.add_to_cache(k1_identifier, k2_identifier, expected_distance_value)
        return expected_distance_value

    def hellinger_distance(self, K1, log_det_K1, K2, log_det_K2):
        avg_K = (K1 + K2) / 2.0
        _, log_det_avg_K = self.get_slog_det(avg_K)
        # We only consider GPs with zero mean functions thus the hellinger distance reduces to:
        hell_dist = 1.0 - np.math.exp(0.25 * log_det_K1 + 0.25 * log_det_K2 - 0.5 * log_det_avg_K)
        return hell_dist

    def sample_over_hps(self, kernel):
        gram_matrices = []
        determinants = []
        parameter_sizes = [tf.size(parameter) for parameter in kernel.trainable_parameters]
        parameter_shapes = [tf.shape(parameter) for parameter in kernel.trainable_parameters]
        num_params = tf.reduce_sum(parameter_sizes)
        sobol_sequence = f64(tf.math.sobol_sample(num_params, self.num_param_samples))
        for i in range(0, self.num_param_samples):
            sobol_sample_list = tf.split(sobol_sequence[i], parameter_sizes)
            reshaped_sobol_sample_list = [
                tf.reshape(sobol_sample_list_element, parameter_shape)
                for sobol_sample_list_element, parameter_shape in zip(sobol_sample_list, parameter_shapes)
            ]
            for j, parameters in enumerate(kernel.trainable_parameters):
                parameter_sample = parameters.prior.quantile(reshaped_sobol_sample_list[j])
                parameters.assign(parameter_sample)
            K = kernel.K(self.virtual_X)
            K_nump, log_det_K = self.get_slog_det(K.numpy())  # most expensive calculation
            gram_matrices.append(K_nump)
            determinants.append(log_det_K)
        logger.info("Precalculation of grams and log-dets done for kernel")
        return gram_matrices, determinants

    def get_slog_det(self, K: np.array) -> Tuple[np.array, np.float]:
        # add jitter to increase numerical stability of cholesky decomposition inside log determinant calculation
        K_with_jitter = K + np.eye(len(self.virtual_X)) * self.jitter
        _, log_det_K = np.linalg.slogdet(K_with_jitter)
        return K_with_jitter, log_det_K

    def unwrap_kernel_grammar_list(self, X: List[BaseKernelGrammarExpression]) -> List[gpflow.kernels.Kernel]:
        new_X = [kernel_expression.get_kernel() for kernel_expression in X]
        return new_X

    def transform_X(self, X: List[BaseKernelGrammarExpression]) -> List[EvaluatedKernelWrapper]:
        logger.info("-Check cache before computing gram matrices for new kernels")
        logger.info("-Cache currently holds " + str(self.evaluated_kernel_cache.get_number_of_cached_kernels()) + " evaluated kernels")
        evaluated_kernel_wrapper_list = []
        for x in X:
            assert isinstance(x, BaseKernelGrammarExpression)
            kernel_indentifier = str(x)
            if self.evaluated_kernel_cache.check_if_in_cache(kernel_indentifier):
                evaluated_kernel_wrapper_x = self.evaluated_kernel_cache.get_from_cache(kernel_indentifier)
            else:
                kernel = x.get_kernel()
                gram_matrices_x, log_determinants_x = self.sample_over_hps(kernel)
                evaluated_kernel_wrapper_x = EvaluatedKernelWrapper(gram_matrices_x, log_determinants_x, kernel_indentifier)
                self.evaluated_kernel_cache.add_to_cache(kernel_indentifier, evaluated_kernel_wrapper_x)
            evaluated_kernel_wrapper_list.append(evaluated_kernel_wrapper_x)
        return evaluated_kernel_wrapper_list

    def draw_from_kernel_hp_prior_and_assign(self, kernel) -> np.float:
        for parameter in kernel.trainable_parameters:
            new_value = parameter.prior.sample()
            parameter.assign(new_value)

    def set_virtual_x_from_dataset(self, x_data):
        indexes = np.random.choice(len(x_data), self.num_virtual_points, replace=False)
        self.virtual_X = x_data[indexes]
        logger.info("Virtual points")
        logger.info(self.virtual_X)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    base_expression_1 = ElementaryKernelGrammarExpression(RBFKernel(**RBFWithPriorConfig(input_dimension=2).dict()))
    base_expression_2 = ElementaryKernelGrammarExpression(LinearKernel(**LinearWithPriorConfig(input_dimension=2).dict()))
    expression = KernelGrammarExpression(base_expression_1, base_expression_2, KernelGrammarOperator.ADD)
    expressions = [base_expression_1, base_expression_2, expression]
    expressions2 = [base_expression_1, base_expression_2]
    kernel_kernel = KernelKernelHellinger(2, 1.0, 1.0, 100, 20, 0.0, 1.0)
    K = kernel_kernel.K(kernel_kernel.transform_X(expressions))
    print(K)
    K = kernel_kernel.K(kernel_kernel.transform_X(expressions))
    print(K)
    K = kernel_kernel.K(kernel_kernel.transform_X(expressions), kernel_kernel.transform_X(expressions2))
    print(K)
    K = kernel_kernel.K(kernel_kernel.transform_X(expressions), kernel_kernel.transform_X(expressions2))
    print(K)
