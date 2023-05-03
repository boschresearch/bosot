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
import os
from datetime import datetime
import numpy as np
import random
import json
import time


class FolderHandler:
    def __init__(self, base_dir, run_name, run_index):
        self.base_dir = base_dir
        self.run_name = run_name
        self.run_index = run_index

    def initialize(self, add_figure_folder):
        self.metric_path, self.figure_base_path, self.working_dirs_base_path = self.create_base_folders(add_figure_folder)
        self.add_working_dir_for_run()
        if add_figure_folder:
            self.add_figure_folder_for_run()

    def write_metrics_to_file(self, metric_array, run_name, metric_name, save_folder, run_index, experiment_name):
        now = datetime.now()
        date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
        print(run_name)
        print(metric_name)
        print(date_time)
        print(experiment_name)
        file_name = run_name + "_" + str(run_index) + "_" + metric_name + "_" + date_time + "_" + experiment_name + ".txt"
        np.savetxt(os.path.join(save_folder, file_name), metric_array)

    def write_run_configs_to_working_dir(self, model_config_json, active_learner_config_json, run_starting_iteration=0):
        json_file_name = os.path.join(
            self.get_working_dir_for_run(), self.run_name + "_step_" + str(run_starting_iteration) + "_model_config.json"
        )
        with open(json_file_name, "w") as outfile:
            json.dump(json.loads(model_config_json), outfile)

        json_file_name2 = os.path.join(
            self.get_working_dir_for_run(), self.run_name + "_step_" + str(run_starting_iteration) + "_al_config.json"
        )
        with open(json_file_name2, "w") as outfile2:
            json.dump(json.loads(active_learner_config_json), outfile2)

    def create_base_folders(self, add_figure_folder):
        metric_path = os.path.join(self.base_dir, "metrics")
        working_dirs_base_path = os.path.join(self.base_dir, "working_dirs")
        figure_base_path = None
        if not os.path.exists(metric_path):
            os.makedirs(metric_path)
        if not os.path.exists(working_dirs_base_path):
            os.makedirs(working_dirs_base_path)
        if add_figure_folder:
            figure_base_path = os.path.join(self.base_dir, "figures")
            if not os.path.exists(figure_base_path):
                os.makedirs(figure_base_path)
        return metric_path, figure_base_path, working_dirs_base_path

    def add_metric_for_run(self, metric_array, metric_name, experiment_name):
        self.write_metrics_to_file(metric_array, self.run_name, metric_name, self.metric_path, self.run_index, experiment_name)

    def add_figure_folder_for_run(self):
        folder_name = self.run_name + "_" + str(self.run_index)
        figure_folder = os.path.join(self.figure_base_path, folder_name)
        if not os.path.exists(figure_folder):
            os.makedirs(figure_folder)

    def add_working_dir_for_run(self):
        folder_name = self.run_name + "_" + str(self.run_index)
        working_dir = os.path.join(self.working_dirs_base_path, folder_name)
        if not os.path.exists(working_dir):
            os.makedirs(working_dir)

    def get_working_dir_for_run(self):
        folder_name = self.run_name + "_" + str(self.run_index)
        working_dir = os.path.join(self.working_dirs_base_path, folder_name)
        return working_dir

    def get_figure_folder_for_run(self):
        folder_name = self.run_name + "_" + str(self.run_index)
        figure_folder = os.path.join(self.figure_base_path, folder_name)
        return figure_folder
