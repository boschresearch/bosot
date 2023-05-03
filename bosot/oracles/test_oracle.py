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
import numpy as np
import matplotlib.pyplot as plt


class TestOracle:
    def __init__(self, observation_noise):
        self.a = 0
        self.b = 1
        self.dimension = 2
        self.observation_noise = observation_noise

    def f(self, x1, x2):
        return self.exp2d(x1, x2) + 0.05 * x1 - 0.3 * np.power(x2, 2.0) + 0.05 * np.sin(self.scale(x1) * 2)

    def exp2d(self, x1, x2):
        return self.scale(x1) * np.exp(-1 * np.power(self.scale(x1), 2.0) - np.power(self.scale(x2), 2.0)) * 0.5

    def scale(self, x):
        return x * 7 - 2.0

    def query(self, x, noisy=True, scale_factor=1.0):
        function_value = self.f(x[0], x[1]) * scale_factor
        if noisy:
            epsilon = np.random.normal(0, self.observation_noise, 1)[0]
            function_value += epsilon
        return function_value

    def get_random_data(self, n, noisy=True):
        X = np.random.uniform(low=self.a, high=self.b, size=(n, self.dimension))
        function_values = []
        for x in X:
            function_value = self.query(x, noisy)
            function_values.append(function_value)
        return X, np.expand_dims(np.array(function_values), axis=1)

    def get_box_bounds(self):
        return self.a, self.b

    def get_dimension(self):
        return self.dimension

    def plot(self):
        xs, ys = self.get_random_data(2000, True)
        # x_safe,y_safe = self.get_random_data_in_random_box_with_safety(10,2.0,-0.1)

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1, projection="3d")
        ax.scatter(xs[:, 0], xs[:, 1], ys, marker=".", color="black")
        # ax.scatter(x_safe[:,0],x_safe[:,1],y_safe,marker='o',color='green')
        plt.show()


if __name__ == "__main__":
    function = TestOracle(0.03)
    function.plot()
