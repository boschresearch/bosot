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
from typing import List, Tuple
from bosot.models.gp_model import GPModel, PredictionQuantity
import gpflow
import numpy as np
from gpflow.utilities import print_summary, set_trainable
from tensorflow_probability import distributions as tfd
from bosot.kernels.base_object_kernel import BaseObjectKernel
from bosot.models.object_gpr import ObjectGPR
from bosot.utils.utils import twod_array_to_list_over_arrays
from gpflow.mean_functions import MeanFunction


class ObjectGpModel(GPModel):
    """
    GPModel defined over objects - kernel and mean function need to be defined over objects - over which objects
    exactly depends on the implementation of the kernel and mean function - method uses an adapted gpflow.GPR to objects
    and uses/calls all main prediction and inference methods from GPModel
    """

    def __init__(
        self,
        kernel: BaseObjectKernel,
        observation_noise: float,
        optimize_hps: bool,
        train_likelihood_variance: bool,
        pertube_parameters_at_start=False,
        perform_multi_start_optimization=False,
        set_prior_on_observation_noise=False,
        n_starts_for_multistart_opt: int = 5,
        expected_observation_noise: float = 0.1,
        prediction_quantity: PredictionQuantity = PredictionQuantity.PREDICT_Y,
        **kwargs
    ):
        super().__init__(
            kernel,
            observation_noise,
            optimize_hps,
            train_likelihood_variance,
            pertube_parameters_at_start=pertube_parameters_at_start,
            perform_multi_start_optimization=perform_multi_start_optimization,
            set_prior_on_observation_noise=set_prior_on_observation_noise,
            n_starts_for_multistart_opt=n_starts_for_multistart_opt,
            expected_observation_noise=expected_observation_noise,
            prediction_quantity=prediction_quantity,
            **kwargs
        )

    def build_model(self, x_data: List[object], y_data: np.array):
        if self.use_mean_function:
            self.model = ObjectGPR(
                data=(x_data, y_data),
                kernel=self.kernel,
                noise_variance=np.power(self.observation_noise, 2.0),
                mean_function=self.mean_function,
            )
        else:
            self.model = ObjectGPR(data=(x_data, y_data), kernel=self.kernel, noise_variance=np.power(self.observation_noise, 2.0))
        set_trainable(self.model.likelihood.variance, self.train_likelihood_variance)
        if self.set_prior_on_observation_noise:
            self.model.likelihood.variance.prior = tfd.Exponential(1 / np.power(self.expected_observation_noise, 2.0))

    def set_mean_function(self, mean_function: MeanFunction):
        self.use_mean_function = True
        self.mean_function = mean_function

    def infer(self, x_data: List[object], y_data: np.array):
        x_data = self.transform_input(x_data)
        super().infer(x_data, y_data)

    def predictive_dist(self, x_test: List[object]) -> Tuple[np.array, np.array]:
        x_test = self.transform_input(x_test)
        return super().predictive_dist(x_test)

    def predictive_log_likelihood(self, x_test: List[object], y_test: np.array) -> np.array:
        x_test = self.transform_input(x_test)
        return super().predictive_log_likelihood(x_test, y_test)

    def entropy_predictive_dist(self, x_test: List[object]) -> np.array:
        x_test = self.transform_input(x_test)
        return super().entropy_predictive_dist(x_test)

    def transform_input(self, input):
        if isinstance(input, np.ndarray):
            input = twod_array_to_list_over_arrays(input)
        else:
            input = self.kernel.transform_X(input)
        return input
