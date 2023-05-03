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
from bosot.bayesian_optimization.enums import AcquisitionFunctionType, AcquisitionOptimizationObjectBOType, ValidationType


class BaseObjectBOConfig(BaseSettings):
    acquisition_function_type: AcquisitionFunctionType
    validation_type: ValidationType = ValidationType.MAX_OBSERVED
    acquisiton_optimization_type = AcquisitionOptimizationObjectBOType.TRAILING_CANDIDATES
    population_evolutionary: int = 100
    n_steps_evolutionary: int = 10
    num_offspring_evolutionary: int = 9  # population % (n_offspring+1)==0
    n_prune_trailing: int = 600
    do_plotting: bool = False
    name = "BaseObjectBO"


class ObjectBOExpectedImprovementConfig(BaseObjectBOConfig):
    acquisition_function_type: AcquisitionFunctionType = AcquisitionFunctionType.EXPECTED_IMPROVEMENT
    name = "ExpectedImprovementBO"


class ObjectBOExpectedImprovementEAConfig(ObjectBOExpectedImprovementConfig):
    acquisiton_optimization_type = AcquisitionOptimizationObjectBOType.EVOLUTIONARY
    population_evolutionary: int = 100
    n_steps_evolutionary: int = 10
    num_offspring_evolutionary: int = 4
    name = "ExpectedImprovementBOEA"


class ObjectBOExpectedImprovementEAFewerStepsConfig(ObjectBOExpectedImprovementEAConfig):
    n_steps_evolutionary: int = 6
    name = "ObjectBOExpectedImprovementEAFewerSteps"


class ObjectBOExpectedImprovementPerSecondEAConfig(BaseObjectBOConfig):
    acquisition_function_type: AcquisitionFunctionType = AcquisitionFunctionType.EXPECTED_IMPROVEMENT_PER_SECOND
    acquisiton_optimization_type = AcquisitionOptimizationObjectBOType.EVOLUTIONARY
    population_evolutionary: int = 200
    n_steps_evolutionary: int = 10
    num_offspring_evolutionary: int = 9
    name = "ExpectedImprovementPerSecondBOEA"
