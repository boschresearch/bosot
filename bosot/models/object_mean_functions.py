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
import tensorflow as tf
from gpflow.config import default_float
from gpflow.mean_functions import Constant, MeanFunction
import gpflow

gpflow.config.set_default_float(np.float64)
f64 = gpflow.utilities.to_default_float
from bosot.kernels.kernel_grammar.kernel_grammar import BaseKernelGrammarExpression, ElementaryKernelGrammarExpression


class Zero(Constant):
    def __init__(self, output_dim=1):
        Constant.__init__(self)
        self.output_dim = output_dim
        del self.c

    def __call__(self, X):
        output_shape = tf.concat([[len(X)], [self.output_dim]], axis=0)
        return tf.zeros(output_shape, dtype=default_float())


class ObjectConstant(MeanFunction):
    def __init__(self, base_c=0.0, trainable=True, output_dim=1):
        super().__init__()
        self.output_dim = output_dim
        self.c = gpflow.Parameter(f64(base_c), trainable=trainable)

    def __call__(self, X):
        output_shape = tf.concat([[len(X)], [self.output_dim]], axis=0)
        return self.c * tf.ones(output_shape, dtype=default_float())


class BICMean(MeanFunction):
    def __init__(self, bic_n_data: int, divide_with_n_data=True, base_c=0.0, trainable=True):
        super().__init__()
        self.divide_with_n_data = divide_with_n_data
        self.bic_n_data = tf.convert_to_tensor(bic_n_data, dtype=default_float())
        self.c = gpflow.Parameter(f64(base_c), trainable=trainable)

    def __call__(self, X: List[BaseKernelGrammarExpression]):
        n_params = [
            tf.add_n([tf.size(tensor) for tensor in kernel_grammar_expression.get_kernel().trainable_variables])
            for kernel_grammar_expression in X
        ]

        n_params_tf = tf.expand_dims(tf.convert_to_tensor(n_params, dtype=default_float()), axis=1)
        if self.divide_with_n_data:
            return self.c - (n_params_tf * tf.math.log(self.bic_n_data)) / self.bic_n_data
        else:
            return self.c - (n_params_tf * tf.math.log(self.bic_n_data))

    def get_number_of_trainable_parameters(self, kernel):
        total_number = 0
        for variable in kernel.trainable_variables:
            total_number += tf.size(variable).numpy()
        return total_number


if __name__ == "__main__":
    bic_mean = BICMean(150)
    base_expression_1 = ElementaryKernelGrammarExpression(gpflow.kernels.RBF())
    base_expression_2 = ElementaryKernelGrammarExpression(gpflow.kernels.Matern52())
    base_expression_3 = ElementaryKernelGrammarExpression(gpflow.kernels.Linear())
    base_expression_4 = ElementaryKernelGrammarExpression(gpflow.kernels.Matern52())
    print(bic_mean([base_expression_1, base_expression_2, base_expression_3, base_expression_4]))
    kernel = base_expression_1.get_kernel()
    print(bic_mean.get_number_of_trainable_parameters(kernel))
