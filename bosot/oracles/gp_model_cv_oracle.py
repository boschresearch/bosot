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
from typing import Optional, Union

from tensorflow.python.ops.gen_math_ops import neg
from bosot.oracles.base_object_oracle import BaseObjectOracle
import numpy as np
import gpflow
from bosot.configs.models.gp_model_config import BasicGPModelConfig, GPModelFastConfig
from bosot.models.gp_model import GPModel
from bosot.configs.kernels.base_kernel_config import BaseKernelConfig
from bosot.kernels.kernel_grammar.kernel_grammar import BaseKernelGrammarExpression
from bosot.kernels.kernel_grammar.kernel_grammar_candidate_generator import KernelGrammarCandidateGenerator
from bosot.utils.utils import calculate_rmse
from sklearn.model_selection import KFold
import logging

logger = logging.getLogger(__name__)


class GPModelCVOracle(BaseObjectOracle):
    def __init__(
        self,
        x_data: np.array,
        y_data: np.array,
        grammar_generator: KernelGrammarCandidateGenerator,
        fast_inference=True,
        normalize=True,
        x_test: Optional[np.array] = None,
        y_test: Optional[np.array] = None,
    ) -> None:
        self.x_data = x_data
        self.y_data = y_data
        self.x_test = x_test
        self.y_test = y_test
        self.n_data = len(self.y_data)
        self.n_dim = x_data.shape[1]
        self.grammar_generator = grammar_generator
        self.k = 5
        self.splitter = KFold(n_splits=self.k)
        if fast_inference:
            self.gp_model_config = BasicGPModelConfig(
                kernel_config=BaseKernelConfig(name="dummy", input_dimension=0), n_starts_for_multistart_opt=2
            )
        else:
            self.gp_model_config = BasicGPModelConfig(
                kernel_config=BaseKernelConfig(name="dummy", input_dimension=0), n_starts_for_multistart_opt=10
            )

    def query(self, x: Union[gpflow.kernels.Kernel, BaseKernelGrammarExpression]) -> np.float:
        time_before_query = time.perf_counter()
        if isinstance(x, gpflow.kernels.Kernel):
            kernel = x
        elif isinstance(x, BaseKernelGrammarExpression):
            kernel = x.get_kernel()
        val_lls = []
        for train_index, val_index in self.splitter.split(self.x_data):
            x_data_j = self.x_data[train_index]
            y_data_j = self.y_data[train_index]
            x_val_j = self.x_data[val_index]
            y_val_j = self.y_data[val_index]
            model = GPModel(kernel, **self.gp_model_config.dict())
            model.infer(x_data_j, y_data_j)
            val_loglikeli = np.mean(model.predictive_log_likelihood(x_val_j, y_val_j))
            val_lls.append(val_loglikeli)
        cv_ll = np.mean(val_lls)
        time_after_query = time.perf_counter()
        duration = time_after_query - time_before_query
        return cv_ll, duration

    def query_on_test_set(self, x: Union[gpflow.kernels.Kernel, BaseKernelGrammarExpression]):
        assert self.x_test is not None
        if isinstance(x, gpflow.kernels.Kernel):
            kernel = x
        elif isinstance(x, BaseKernelGrammarExpression):
            kernel = x.get_kernel()
        model = GPModel(kernel, **self.gp_model_config.dict())
        model.infer(self.x_data, self.y_data)
        pred_mu, pred_sigma = model.predictive_dist(self.x_test)
        test_rmse = calculate_rmse(pred_mu, self.y_test)
        test_nll = np.mean(-1 * model.predictive_log_likelihood(self.x_test, self.y_test))
        return test_rmse, test_nll

    def get_random_data(self, n_data):
        x_out = self.grammar_generator.get_random_canditates(n_data)
        y_out = []
        for kernel_expression in x_out:
            logger.info(str(kernel_expression))
            y, _ = self.query(kernel_expression)
            y_out.append(y)
            logger.info("CV Loglikeli: " + str(y))
        return x_out, np.expand_dims(np.array(y_out), axis=1)

    def get_random_data_recursively(self, n_data, n_per_step: int = 5, filter_out_equivalent_expressions=False):
        x_out = self.grammar_generator.get_dataset_recursivly_generated(n_data, n_per_step, filter_out_equivalent_expressions)
        y_out = []
        for kernel_expression in x_out:
            logger.info(str(kernel_expression))
            y, _ = self.query(kernel_expression)
            y_out.append(y)
            logger.info("CV: " + str(y))
        return x_out, np.expand_dims(np.array(y_out), axis=1)
