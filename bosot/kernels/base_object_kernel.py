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
from typing import List, Optional
import gpflow
from abc import ABC, abstractmethod
import tensorflow as tf


class BaseObjectKernel(gpflow.kernels.Kernel, ABC):
    @abstractmethod
    def K(self, X: List[object], X2: Optional[List[object]] = None) -> tf.Tensor:
        raise NotImplementedError

    @abstractmethod
    def K_diag(self, X: List[object]):
        raise NotImplementedError

    def transform_X(self, X: List[object]):
        return X

    def __call__(self, X, X2=None, *, full_cov=True):
        if (not full_cov) and (X2 is not None):
            raise ValueError("Ambiguous inputs: `not full_cov` and `X2` are not compatible.")

        if not full_cov:
            assert X2 is None
            return self.K_diag(X)

        else:
            return self.K(X, X2)
