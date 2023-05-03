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
from typing import Tuple
import numpy as np


class BaseObjectOracle:
    def query(self, x: object) -> Tuple[np.float, np.float]:
        """
        Queries the oracle at input object x and gets back the oracle value

        Arguments:
            x : object - General oracle -> input can be any object (not restricted to numbers/arrays)
        Returns:
            np.float - value of oracle at location x
            np.float - time duration of query call in seconds
        """
        raise NotImplementedError
