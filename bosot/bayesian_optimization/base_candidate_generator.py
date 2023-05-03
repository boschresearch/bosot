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


class CandidateGenerator(ABC):
    """
    CandidateGenerator objects are used inside both acquisition function optimization methods of the BO procedure.
    They can be used as wrapper around the object generation process e.g. to generate abstract objects like kernel grammar expressions
    All methods return lists of the respective objects that are used inside the different acquisition function optimization shemes.
    """

    @abstractmethod
    def get_initial_candidates_trailing(self) -> List[object]:
        """
        Returns initial list of objects used for trailing acquisition function optimization
        """
        raise NotImplementedError

    @abstractmethod
    def get_random_canditates(self, n_candidates: int, seed=100, set_seed=False) -> List[object]:
        """
        General method for returning n_candidates random object
        """
        raise NotImplementedError

    @abstractmethod
    def get_additional_candidates_trailing(self, best_current_candidate: object) -> List[object]:
        """
        This method returns a list of objects based on the best_current_candidate object (mostly the object with highest function value in BO). Should generate
        a list of related/close objects to the best_current_candidate object
        """
        raise NotImplementedError

    @abstractmethod
    def get_initial_for_evolutionary_opt(self, n_initial) -> List[object]:
        """
        Returns initial list of objects used for evolutionary acquisition function optimization
        """
        raise NotImplementedError

    @abstractmethod
    def get_around_candidate_for_evolutionary_opt(self, candidate: object, n_around_candidate: int) -> List[object]:
        """
        Returns list of objects based on candidate object (around object) - similar to get_additional_candidates_trailing however it might be implemented differently
        for the evolutionary optimizer
        """
        raise NotImplementedError

    @abstractmethod
    def get_dataset_recursivly_generated(self, n_data: int, n_per_step: int) -> List[object]:
        raise NotImplementedError
