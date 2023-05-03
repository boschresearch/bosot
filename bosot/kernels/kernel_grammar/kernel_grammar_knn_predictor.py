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
from typing import Dict, List

import numpy as np
from sklearn import neighbors
from bosot.kernels.kernel_grammar.kernel_grammar import BaseKernelGrammarExpression
from bosot.kernels.kernel_grammar.kernel_grammar_search_spaces import BaseKernelGrammarSearchSpace, CKSWithRQSearchSpace
from sklearn.model_selection import KFold
from bosot.utils.utils import calculate_rmse
import logging

logger = logging.getLogger(__name__)


class GrammarNeighbour:
    def __init__(self, expression_identifier: str):
        self.expression_identifier = expression_identifier

    def get_expression_identifier(self):
        return self.expression_identifier

    def __str__(self) -> str:
        return self.expression_identifier


class KernelGrammarKNNPredictor:
    def __init__(self, search_space: BaseKernelGrammarSearchSpace) -> None:
        self.search_space = search_space
        self.neighbours_dict = {}
        self.cvv_k = 5

    def build_neighbour_dict(self, x_complete: List[BaseKernelGrammarExpression]):
        self.neighbours_dict = {}
        for expression in x_complete:
            logger.info("Check for neighbours of: " + str(expression))
            self.initialize_in_neighbour_dict(str(expression))
            expression_neighbours = self.search_space.get_neighbour_expressions(expression)
            reverse_neighbour_object = GrammarNeighbour(str(expression))
            for expression_2 in x_complete:
                self.initialize_in_neighbour_dict(str(expression_2))
                if self.check_if_expression_in_list(expression_2, expression_neighbours) and not str(expression_2) == str(expression):
                    neighbour_object = GrammarNeighbour(str(expression_2))
                    self.neighbours_dict[str(expression)].append(neighbour_object)
                    self.neighbours_dict[str(expression_2)].append(reverse_neighbour_object)

    def initialize_in_neighbour_dict(self, expression_identifier: str):
        if not expression_identifier in self.neighbours_dict:
            self.neighbours_dict[expression_identifier] = []

    def cross_validation_k(self, x_train: List[BaseKernelGrammarExpression], y_train: np.array):
        logger.info("Start knn cross-validation for k")
        splitter = KFold(n_splits=self.cvv_k, shuffle=True)
        ks = np.arange(1, 11)
        cv_rmse_for_ks = []
        for k in ks:
            rmses = []
            for train_indexes, val_indexes in splitter.split(x_train):
                x_train_cv = []
                y_train_cv = y_train[train_indexes]
                for train_index in train_indexes:
                    x_train_cv.append(x_train[train_index])
                x_val_cv = []
                y_val_cv = y_train[val_indexes]
                for val_index in val_indexes:
                    x_val_cv.append(x_train[val_index])
                pred_y_val_cv = self.predict(x_val_cv, x_train_cv, y_train_cv, k)
                rmse = calculate_rmse(pred_y_val_cv, y_val_cv)
                rmses.append(rmse)
            cv_rmse_for_k = np.mean(rmses)
            cv_rmse_for_ks.append(cv_rmse_for_k)
        logger.info(str(cv_rmse_for_ks))
        best_index = np.argmin(cv_rmse_for_ks)
        best_k = ks[best_index]
        logger.info("Best k: " + str(best_k))
        return best_k

    def predict(self, x_test: List[BaseKernelGrammarExpression], x_train: List[BaseKernelGrammarExpression], y_train: np.array, k: int):
        assert len(self.neighbours_dict) > 0
        pred_ys = []
        train_dict = self.get_train_dict(x_train, y_train)
        for test_expression in x_test:
            pred_y = self.predict_expression_knn(str(test_expression), train_dict, k)
            pred_ys.append(pred_y)
        return np.array(pred_ys)

    def get_train_dict(self, x_train: List[BaseKernelGrammarExpression], y_train: np.array):
        train_dict = {}
        for i, expression in enumerate(x_train):
            train_dict[str(expression)] = y_train[i]
        return train_dict

    def predict_expression_knn(self, expression_identifier: str, train_dict: Dict, k: int):
        train_neighbour_identifiers = []
        all_neighbours = self.neighbours_dict[expression_identifier]
        while len(train_neighbour_identifiers) < k:
            new_all_neighbours = []
            for neighbor_object in all_neighbours:
                new_all_neighbours += self.neighbours_dict[neighbor_object.expression_identifier]
                if neighbor_object.expression_identifier in train_dict:
                    train_neighbour_identifiers.append(neighbor_object.expression_identifier)
            all_neighbours = new_all_neighbours
        k_neareset_neighbours = train_neighbour_identifiers[0:k]
        logger.debug("Expression: " + str(expression_identifier))
        logger.debug("Neihgbours:")
        logger.debug(k_neareset_neighbours)
        train_neighbour_values = []
        for expression_identifier in k_neareset_neighbours:
            train_neighbour_values.append(train_dict[expression_identifier])
        logger.debug("Values:")
        logger.debug(train_neighbour_values)
        prediction = np.mean(train_neighbour_values)
        logger.debug("Prediction:")
        logger.debug(prediction)
        return prediction

    def check_if_expression_in_list(self, expression, expression_list):
        for expression_in_list in expression_list:
            if str(expression) == str(expression_in_list):
                return True
        return False

    def show_neighbour_dict(self):
        for key in self.neighbours_dict:
            print("Test Expression: " + key)
            print("Num Neighbours: " + str(len(self.neighbours_dict[key])))
            print("Neighbours:")
            for neighbour in self.neighbours_dict[key]:
                print(neighbour)
            print("")


if __name__ == "__main__":
    pass
