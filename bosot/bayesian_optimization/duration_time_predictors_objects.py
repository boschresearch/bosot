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

import numpy as np
import tensorflow as tf
from bosot.configs.kernels.linear_configs import LinearWithPriorConfig
from bosot.configs.kernels.rbf_configs import BasicRBFConfig
from bosot.kernels.kernel_grammar.kernel_grammar import (
    BaseKernelGrammarExpression,
    ElementaryKernelGrammarExpression,
    KernelGrammarExpression,
    KernelGrammarOperator,
)
from bosot.kernels.linear_kernel import LinearKernel
from bosot.kernels.rbf_kernel import RBFKernel
from sklearn.linear_model import LinearRegression
from bosot.utils.plotter import Plotter


class BaseDurationTimePredictorObjects(ABC):
    def fit(self, x_data: List[object], duration_times: List[float]):
        raise NotImplementedError

    def predict(self, x_test: List[object]):
        raise NotImplementedError


class LinearTimePredictorKernelParameters(BaseDurationTimePredictorObjects):
    def __init__(self) -> None:
        self.regressor = LinearRegression()
        self.do_plotting = True
        self.use_log = True

    def fit(self, x_data: List[BaseKernelGrammarExpression], duration_times: List[float]):
        duration_times = np.array(duration_times)
        n_params = self.get_n_params(x_data)
        if self.use_log:
            log_duration_times = np.log(duration_times)
            self.regressor.fit(n_params, log_duration_times)
        else:
            self.regressor.fit(n_params, duration_times)
        if self.do_plotting:
            pred_data = self.predict(x_data)
            plotter = Plotter(1)
            plotter.add_gt_function(np.squeeze(n_params), pred_data, "black", 0, True)
            plotter.add_datapoints(np.squeeze(n_params), duration_times, "green", 0)
            plotter.show()

    def predict(self, x_test: List[BaseKernelGrammarExpression]) -> np.array:
        n_params = self.get_n_params(x_test)
        prediction = self.regressor.predict(n_params)
        if self.use_log:
            prediction = np.exp(prediction)
        return prediction

    def get_n_params(self, X: List[BaseKernelGrammarExpression]):
        n_params = [
            tf.add_n([tf.size(tensor) for tensor in kernel_grammar_expression.get_kernel().trainable_variables])
            for kernel_grammar_expression in X
        ]
        return np.expand_dims(np.array(n_params), axis=1)


if __name__ == "__main__":
    predictor = LinearTimePredictorKernelParameters()
    base_expression_1 = ElementaryKernelGrammarExpression(RBFKernel(**BasicRBFConfig(input_dimension=2).dict()))
    base_expression_2 = ElementaryKernelGrammarExpression(LinearKernel(**LinearWithPriorConfig(input_dimension=2).dict()))
    base_expression_3 = ElementaryKernelGrammarExpression(LinearKernel(**LinearWithPriorConfig(input_dimension=2).dict()))
    base_expression_4 = ElementaryKernelGrammarExpression(RBFKernel(**BasicRBFConfig(input_dimension=2).dict()))
    expression1 = KernelGrammarExpression(base_expression_2, base_expression_2, operator=KernelGrammarOperator.ADD)
    expression2 = KernelGrammarExpression(base_expression_3, base_expression_4, operator=KernelGrammarOperator.SPLIT_CH)
    expression3 = KernelGrammarExpression(expression1, expression2, operator=KernelGrammarOperator.SPLIT_CH)
    print(predictor.get_n_params([expression1, expression2, expression3]))
    predictor.fit([expression1, expression2, expression3], [50, 100, 300])
