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
from pydantic import BaseSettings


class TreeGEPEvolutionaryOptimizerConfig(BaseSettings):
    population_size: int = 200
    reproduction_rate: float = 0.1
    tournament_fraction: float = 0.1
    mutation_max_n: int = 4
    max_depth: int = 10
    name = "TreeGEPEvolutionaryOptimizer"
