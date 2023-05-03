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
import time
from typing import Optional, Union

from tensorflow.python.ops.gen_math_ops import neg
from bosot.configs.models.gp_model_config import BasicGPModelConfig, GPModelFastConfig
from bosot.kernels.kernel_grammar.kernel_grammar_candidate_generator import KernelGrammarCandidateGenerator
from bosot.models.gp_model import GPModel
from bosot.oracles.base_object_oracle import BaseObjectOracle
import numpy as np
import gpflow

from bosot.configs.kernels.base_kernel_config import BaseKernelConfig
from bosot.kernels.kernel_grammar.kernel_grammar import BaseKernelGrammarExpression
from bosot.configs.models.gp_model_laplace_config import BasicGPModelLaplaceConfig
from bosot.models.gp_model_laplace import GPModelLaplace
from bosot.utils.utils import calculate_rmse

logger = logging.getLogger(__name__)


class GPModelEvidenceOracle(BaseObjectOracle):
    def __init__(
        self,
        x_data: np.array,
        y_data: np.array,
        grammar_generator: KernelGrammarCandidateGenerator,
        fast_inference=False,
        normalize=True,
        x_test: Optional[np.array] = None,
        y_test: Optional[np.array] = None,
    ) -> None:
        self.x_data = x_data
        self.y_data = y_data
        self.x_test = x_test
        self.y_test = y_test
        self.n_data = len(self.y_data)
        self.normalize = normalize
        self.grammar_generator = grammar_generator

        if fast_inference:
            self.model_config = BasicGPModelLaplaceConfig(
                kernel_config=BaseKernelConfig(name="dummy", input_dimension=0), n_starts_for_multistart_opt=3
            )
        else:
            self.model_config = BasicGPModelLaplaceConfig(
                kernel_config=BaseKernelConfig(name="dummy", input_dimension=0), n_starts_for_multistart_opt=10
            )

        if fast_inference:
            self.prediction_model_config = BasicGPModelConfig(
                kernel_config=BaseKernelConfig(name="dummy", input_dimension=0), n_starts_for_multistart_opt=3
            )
        else:
            self.prediction_model_config = BasicGPModelConfig(
                kernel_config=BaseKernelConfig(name="dummy", input_dimension=0), n_starts_for_multistart_opt=20
            )

    def query(self, x: Union[gpflow.kernels.Kernel, BaseKernelGrammarExpression]) -> np.float:
        time_before_query = time.perf_counter()
        if isinstance(x, gpflow.kernels.Kernel):
            kernel = x
        elif isinstance(x, BaseKernelGrammarExpression):
            kernel = x.get_kernel()
        model = GPModelLaplace(kernel, **self.model_config.dict())
        model.infer(self.x_data, self.y_data)
        log_evidence = model.estimate_log_model_evidence()
        if self.normalize:
            log_evidence = log_evidence / self.n_data
        logger.info("Log-model-evidence: " + str(log_evidence))
        time_after_query = time.perf_counter()
        duration = time_after_query - time_before_query
        return log_evidence, duration

    def query_on_test_set(self, x: Union[gpflow.kernels.Kernel, BaseKernelGrammarExpression]):
        assert self.x_test is not None
        if isinstance(x, gpflow.kernels.Kernel):
            kernel = x
        elif isinstance(x, BaseKernelGrammarExpression):
            kernel = x.get_kernel()
        model = GPModel(kernel, **self.prediction_model_config.dict())
        model.infer(self.x_data, self.y_data)
        pred_mu, pred_sigma = model.predictive_dist(self.x_test)
        test_rmse = calculate_rmse(pred_mu, self.y_test)
        test_nll = np.mean(-1 * model.predictive_log_likelihood(self.x_test, self.y_test))
        return test_rmse, test_nll

    def get_random_data(self, n_data):
        x_out = self.grammar_generator.get_random_canditates(n_data)
        y_out = []
        for kernel_expression in x_out:
            y, _ = self.query(kernel_expression)
            y_out.append(y)
        return x_out, np.expand_dims(np.array(y_out), axis=1)

    def get_random_data_recursively(self, n_data, n_per_step: int = 5, filter_out_equivalent_expressions=False):
        x_out = self.grammar_generator.get_dataset_recursivly_generated(n_data, n_per_step, filter_out_equivalent_expressions)
        y_out = []
        for kernel_expression in x_out:
            logger.info(str(kernel_expression))
            y, _ = self.query(kernel_expression)
            y_out.append(y)
            logger.info("Evidence: " + str(y))
        return x_out, np.expand_dims(np.array(y_out), axis=1)
