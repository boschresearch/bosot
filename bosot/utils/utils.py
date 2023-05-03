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
import gpflow
import numpy as np
from scipy import integrate
from scipy.stats import norm
from sklearn.cluster import KMeans
import math
import tensorflow as tf


def gmm_density(y, mus, sigmas, weights):
    normalized_weights = weights / np.sum(weights)
    density = 0.0
    for i in range(0, mus.shape[0]):
        density += normalized_weights[i] * norm.pdf(y, mus[i], sigmas[i])
    return density


def gmm_entropy_integrand(y, mus, sigmas, weights):
    p = gmm_density(y, mus, sigmas, weights)
    # print(p)
    if math.isclose(p, 0.0):
        return 0.0
    else:
        return -1 * p * np.log(p)


def entropy_of_gmm(mus, sigmas, weights, uniform_weights):
    if weights is None or uniform_weights:
        weights = np.repeat(1.0, mus.shape[0])
    f = lambda y: gmm_entropy_integrand(y, mus, sigmas, weights)
    int_f = integrate.quad(f, -np.infty, np.infty)
    return int_f[0]


def calculate_multioutput_rmse(pred_y, y):
    M = pred_y.shape[1]
    assert y.shape[1] == M
    rmses = []
    for m in range(0, M):
        rmse = np.sqrt(np.mean(np.power(pred_y[:, m] - np.squeeze(y[:, m]), 2.0)))
        rmses.append(rmse)
    return np.array(rmses)


def create_grid(a, b, n_per_dim, dimensions):
    grid_points = np.linspace(a, b, n_per_dim)
    n = int(np.power(n_per_dim, dimensions))
    X = np.zeros((n, dimensions))
    for i in range(0, dimensions):
        repeats_per_item = int(np.power(n_per_dim, i))
        block_size = repeats_per_item * n_per_dim
        block_repeats = int(n / block_size)
        for block in range(0, block_repeats):
            for j in range(0, n_per_dim):
                point = grid_points[j]
                for l in range(0, repeats_per_item):
                    index = block * block_size + j * repeats_per_item + l
                    X[index, i] = point
    return X


def filter_safety(X, y, safety_threshold, safety_is_upper_bound):
    if safety_is_upper_bound:
        safe_indexes = np.squeeze(y) < safety_threshold
    else:
        safe_indexes = np.squeeze(y) > safety_threshold
    return X[safe_indexes], y[safe_indexes]


def one_fold_cross_validation(model, x_data, y_data, only_use_subset=False, subset_indexes=[]):
    n = len(x_data)
    true_ys = []
    pred_ys = []
    if only_use_subset:
        val_indexes = subset_indexes
    else:
        val_indexes = list(range(0, n))
    for val_index in val_indexes:
        train_indexes = list(range(0, n))
        train_indexes.pop(val_index)
        train_data_x = x_data[train_indexes]
        train_data_y = y_data[train_indexes]
        test_point_x = np.expand_dims(x_data[val_index], axis=0)
        test_point_y = y_data[val_index]
        true_ys.append(test_point_y)
        model.reset_model()
        print(train_data_x.shape)
        print(train_data_y.shape)
        model.infer(train_data_x, train_data_y)
        predicted_y, _ = model.predictive_dist(test_point_x)
        pred_ys.append(predicted_y)
    true_ys = np.array(true_ys)
    pred_ys = np.array(pred_ys)
    rmse = np.sqrt(np.mean(np.power(pred_ys - np.squeeze(true_ys), 2.0)))
    return rmse


def calculate_rmse(y_pred, y_true):
    return np.sqrt(np.mean(np.power(np.squeeze(y_pred) - np.squeeze(y_true), 2.0)))


def extract_grid_lines(grid, point):
    dimensions = grid.shape[1]
    lines = []
    for i in range(0, dimensions):
        grid_buffer = grid.copy()
        for j in range(0, dimensions):
            if i != j:
                grid_buffer = grid_buffer[grid_buffer[:, j] == point[j]]
        lines.append(grid_buffer)

    return lines


def normal_entropy(sigma):
    entropy = np.log(sigma * np.sqrt(2 * np.pi * np.exp(1)))
    return entropy


def string2bool(b):
    if isinstance(b, bool):
        return b
    if b.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif b.lower() in ("no", "false", "f", "n", "0"):
        return False


def normalize_data(x: np.array):
    assert len(x.shape) == 2
    x_normalized = (x - np.expand_dims(np.mean(x, axis=0), axis=0)) / np.expand_dims(np.std(x, axis=0), axis=0)
    return x_normalized


def min_max_normalize_data(x: np.array):
    assert len(x.shape) == 2
    x_normalized = (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))
    return x_normalized


def k_means(num_clusters: int, x_data: np.array):
    assert len(x_data.shape) == 2
    kmeans = KMeans(n_clusters=num_clusters).fit(x_data)
    return kmeans.cluster_centers_


def twod_array_to_list_over_arrays(array):
    list_over_arrays = [array[i, :] for i in range(0, array.shape[0])]
    return list_over_arrays


def manhatten_distance(X: np.array, X2: np.array) -> tf.Tensor:
    differences = gpflow.utilities.ops.difference_matrix(X, X2)
    return tf.reduce_sum(tf.math.abs(differences), axis=2)


def draw_from_hp_prior_and_assign(kernel):
    print("-Draw from hyperparameter prior")
    for parameter in kernel.trainable_parameters:
        new_value = parameter.prior.sample()
        parameter.assign(new_value)


if __name__ == "__main__":
    list_of_arrays = twod_array_to_list_over_arrays(np.array([[1, 2, 3], [2, 2, 2]]))
    print(list_of_arrays)
