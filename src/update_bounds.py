#!/usr/bin/env python
import rospy
import torch
from torch import nn, sigmoid
from torch.nn import Parameter, ParameterDict

import csv
import os
import numpy as np
import time
import pandas as pd
import math
import random

from std_msgs.msg import (Int32, Float32)
from std_srvs.srv import (Empty)
from Scripts.srv import (ReceiveInt)

# Set data type based on CUDA availability
dtype = torch.cuda.FloatTensor if torch.cuda.is_available(
) else torch.DoubleTensor


class PerformanceModel(nn.Module):

    def __init__(self, operator, lr, weight_decay):
        super(PerformanceModel, self).__init__()

        self.OPERATOR = operator
        self.__local_operators = nn.ModuleDict()
        self.__remote_operators = nn.ModuleDict()

        self.__t_count = 0
        self.LR = lr
        self.DECAY = weight_decay

        # Changed initialization of local and remote operators

        operator_params = {
            'lower_bound_1':
                Parameter(dtype(-10.0 * np.ones(1)), requires_grad=True),
            'upper_bound_1':
                Parameter(dtype(10.0 * np.ones(1)), requires_grad=True),
            'lower_bound_2':
                Parameter(dtype(-10.0 * np.ones(1)), requires_grad=True),
            'upper_bound_2':
                Parameter(dtype(10.0 * np.ones(1)), requires_grad=True),
            'lower_bound_3':
                Parameter(dtype(-10.0 * np.ones(1)), requires_grad=True),
            'upper_bound_3':
                Parameter(dtype(10.0 * np.ones(1)), requires_grad=True)
        }

        if self.OPERATOR % 2 == 0:
            self.__local_operators[f'{self.OPERATOR}'] = ParameterDict(
                operator_params
            )
        else:
            self.__remote_operators[f'{self.OPERATOR}'] = ParameterDict(
                operator_params
            )

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.LR, weight_decay=self.DECAY
        )

    def get_parameters(self, operator_number):

        if operator_number % 2 == 0:
            params = self.__local_operators[f'{operator_number}']
        else:
            params = self.__remote_operators[f'{operator_number}']

        if params['lower_bound_1'] > params['upper_bound_1']:
            params['lower_bound_1'], params['upper_bound_1'] = params[
                'upper_bound_1'], params['lower_bound_1']

        if params['lower_bound_2'] > params['upper_bound_2']:
            params['lower_bound_2'], params['upper_bound_2'] = params[
                'upper_bound_2'], params['lower_bound_2']

        if params['lower_bound_3'] > params['upper_bound_3']:
            params['lower_bound_3'], params['upper_bound_3'] = params[
                'upper_bound_3'], params['lower_bound_3']

        norm_lower_bound_1 = sigmoid(params['lower_bound_1'])
        norm_upper_bound_1 = sigmoid(params['upper_bound_1'])

        norm_lower_bound_2 = sigmoid(params['lower_bound_2'])
        norm_upper_bound_2 = sigmoid(params['upper_bound_2'])

        norm_lower_bound_3 = sigmoid(params['lower_bound_3'])
        norm_upper_bound_3 = sigmoid(params['upper_bound_3'])

        norms = [
            (norm_lower_bound_1, norm_upper_bound_1),
            (norm_lower_bound_2, norm_upper_bound_2),
            (norm_lower_bound_3, norm_upper_bound_3)
        ]
        return norms

    def forward(
        self, bin_centers, observation_probability_index, operator_number
    ):

        n_diffs = observation_probability_index.shape[0]
        performance = torch.zeros(n_diffs)

        for i in range(n_diffs):
            bin_center_idx_1, bin_center_idx_2, bin_center_idx_3 = observation_probability_index[
                i]

            performance[i] = (
                self.calculate_performance(
                    1, operator_number, bin_centers[bin_center_idx_1]
                ) * self.calculate_performance(
                    2, operator_number, bin_centers[bin_center_idx_2]
                ) * self.calculate_performance(
                    3, operator_number, bin_centers[bin_center_idx_3]
                )
            )

        return performance.cuda() if torch.cuda.is_available() else performance

    def calculate_performance(self, number, operator_number, task_requirement):

        if operator_number % 2 == 0:
            params = self.__local_operators[f'{operator_number}']
        else:
            params = self.__remote_operators[f'{operator_number}']

        upper_bound = params[f'upper_bound_{number}']
        lower_bound = params[f'lower_bound_{number}']

        x = (upper_bound - math.log(task_requirement / (1 - task_requirement)
                                   )) / (upper_bound - lower_bound + 0.0001)

        return torch.sigmoid(x)

    def __check_convergence(self, operator):

        norms = self.get_parameters(operator)

        bounds = [
            'upper_bound_1', 'lower_bound_1', 'upper_bound_2', 'lower_bound_2',
            'upper_bound_3', 'lower_bound_3'
        ]

        counter = 0

        for i, (norm_l, norm_u) in enumerate(norms):
            if abs(norm_u - norm_l) < 0.005:
                operators = self.__local_operators if operator % 2 == 0 else self.__remote_operators
                upper_bound = operators[f'{operator}'][bounds[2 * i]]
                lower_bound = operators[f'{operator}'][bounds[2 * i + 1]]

                if upper_bound.grad is not None and lower_bound.grad is not None:
                    operators[f'{operator}'][bounds[2 * i
                                                   ]].requires_grad = False
                    operators[f'{operator}'][bounds[2 * i
                                                    + 1]].requires_grad = False

                print(
                    "Capability", i, "converged with values (" + lower_bound
                    + "," + upper_bound + ")"
                )

                counter += 1

        if counter == 3:

            print("All capabilities have converged!")
            print(norms)

    def update(self, bin_centers, obs_probs, obs_probs_idxs, operator_number):

        predicted_values = self.forward(
            bin_centers, obs_probs_idxs, operator_number
        )
        obs_probs_vect = torch.tensor(
            [obs_probs[j, k, z] for j, k, z in obs_probs_idxs],
            dtype=torch.float64,
            requires_grad=True
        )

        obs_probs = dtype(obs_probs)

        loss = torch.mean(torch.pow((predicted_values - obs_probs_vect), 2.0))

        if loss.item() < 0.0005:
            self.__t_count += 1
            return  # Early return if the loss is already below the threshold

        # Iterative optimization
        t = 0

        while t < 2200:

            def closure():

                diff = self(
                    bin_centers, obs_probs_idxs, operator_number
                ) - obs_probs_vect
                loss = torch.mean(torch.pow(diff, 2.0))
                self.optimizer.zero_grad()
                loss.backward()

                return loss

            self.optimizer.step(closure)

            predicted_values = self.forward(
                bin_centers, obs_probs_idxs, operator_number
            )
            loss = torch.mean(
                torch.pow((predicted_values - obs_probs_vect), 2.0)
            )

            self.__check_convergence(operator_number)

            if loss.item() < 0.0005:
                self.__t_count += 1
                break

            t += 1
            self.__t_count += 1


class AllocationFramework:

    def __init__(self, num_operator, num_bins, threshold, lr, weight_decay):

        np.seterr(divide='ignore', invalid='ignore')

        self.OPERATOR = num_operator
        self.NUM_BINS = num_bins
        self.THRESHOLD = threshold
        self.LR = lr
        self.DECAY = weight_decay
        self.RATE = rospy.Rate(100)

        self.__model = PerformanceModel(self.OPERATOR, self.LR, self.DECAY)
        self.__model = self.__model.cuda() if torch.cuda.is_available(
        ) else self.__model
        self.__data_recorder = DataRecorder(
            self.OPERATOR,
            self.THRESHOLD,
            self.LR,
            self.DECAY,
        )

        self.__bin_limits = dtype(
            np.concatenate(
                [[0], np.linspace(1 / self.NUM_BINS, 1.0, self.NUM_BINS)]
            )
        )
        self.bin_centers = dtype(
            (self.__bin_limits[:-1] + self.__bin_limits[1:]) / 2.0
        )

        self.__total_observations_local = {}
        self.__total_observations_remote = {}

        self.__total_success_local = {}
        self.__total_success_remote = {}

        self.__success_local = {}
        self.__success_remote = {}
        self.observations_probabilities_local = {}
        self.observations_probabilities_remote = {}
        self.success_matrix = {}

        self.probabilities_index_local = []
        self.probabilities_index_remote = []

        self.__failures_assigned = {}

        self.__start_time_failure = 0
        self.failure_counter = 0
        self.task_allocation = []
        self.__adjusted_threshold = 0
        self.__is_failure_resolved = False
        self.__counter = None

        self.__failures = {
            'start_time': [],
            'end_time': [],
            'duration': [],
            'resolution_start_time': []
        }

        self.__operator_time_failures = {operator: 0}
        self.__operator_speed = {operator: []}
        self.total = {operator: 0}
        self.success = {operator: 0}

        self.__failures_assigned[self.OPERATOR] = 0
        self.success_matrix[self.OPERATOR] = np.ones(
            (self.NUM_BINS, self.NUM_BINS, self.NUM_BINS)
        )

        # Local operators have even number indexes
        if self.OPERATOR % 2 == 0:
            self.observations_probabilities_local[self.OPERATOR] = np.zeros(
                (self.NUM_BINS, self.NUM_BINS, self.NUM_BINS)
            )
            self.__total_observations_local[self.OPERATOR] = np.zeros(
                (self.NUM_BINS, self.NUM_BINS, self.NUM_BINS)
            )
            self.__total_success_local[self.OPERATOR] = np.zeros(
                (self.NUM_BINS, self.NUM_BINS)
            )
            self.__success_local[self.OPERATOR] = np.zeros(
                (self.NUM_BINS, self.NUM_BINS, self.NUM_BINS)
            )

        else:
            self.observations_probabilities_remote[self.OPERATOR] = np.zeros(
                (self.NUM_BINS, self.NUM_BINS, self.NUM_BINS)
            )
            self.__total_observations_remote[self.OPERATOR] = np.zeros(
                (self.NUM_BINS, self.NUM_BINS, self.NUM_BINS)
            )
            self.__total_success_remote[self.OPERATOR] = np.zeros(
                (self.NUM_BINS, self.NUM_BINS)
            )
            self.__success_remote[self.OPERATOR] = np.zeros(
                (self.NUM_BINS, self.NUM_BINS, self.NUM_BINS)
            )

        load_file_name = '/home/fetch/catkin_workspaces/hololens_ws/src/failure_allocation/src/failure_requirements.csv'
        data_frame = pd.read_csv(load_file_name, header=None)

        self.__task_requirements_list = (data_frame.iloc[:, :18].values)

        self.__task_requirements = []

        self.__capabilities = {}

        rospy.Subscriber(
            '/task_manager/target_counter',
            Int32,
            self.__target_counter,
        )

        rospy.Service(
            '/assign_failure',
            ReceiveInt,
            self.__failure_service,
        )

        rospy.Service(
            '/failure_resolved',
            Empty,
            self.__failure_resolved,
        )

        rospy.Service(
            '/task_started',
            Empty,
            self.__task_started,
        )

        self.__update_target_service = rospy.ServiceProxy(
            '/update_target',
            Empty,
        )

        self.__lower_1_publisher = rospy.Publisher(
            '/lower_bound_1',
            Float32,
            queue_size=1,
        )
        self.__lower_2_publisher = rospy.Publisher(
            '/lower_bound_2',
            Float32,
            queue_size=1,
        )
        self.__lower_3_publisher = rospy.Publisher(
            '/lower_bound_3',
            Float32,
            queue_size=1,
        )
        self.__upper_1_publisher = rospy.Publisher(
            '/upper_bound_1',
            Float32,
            queue_size=1,
        )
        self.__upper_2_publisher = rospy.Publisher(
            '/upper_bound_2',
            Float32,
            queue_size=1,
        )
        self.__upper_3_publisher = rospy.Publisher(
            '/upper_bound_3',
            Float32,
            queue_size=1,
        )

    def __target_counter(self, message):

        self.__counter = message.data

    def __failure_service(self, request):

        error_code = request.request

        if error_code == 1:

            if self.failure_counter in [5, 10]:
                # Medicine is misplaced
                physical_requirement = round(random.uniform(0.4, 0.6), 1)
                cognitive_requirement = round(random.uniform(0.8, 1.0), 1)
                urgency = round(random.uniform(0.1, 1.0), 1)
            else:
                physical_requirement = round(random.uniform(0.2, 0.4), 1)
                cognitive_requirement = round(random.uniform(0.4, 0.6), 1)
                urgency = round(random.uniform(0.1, 1.0), 1)

        elif error_code == 2:
            physical_requirement = round(random.uniform(0.3, 0.5), 1)
            cognitive_requirement = round(random.uniform(0.2, 0.3), 1)
            urgency = round(random.uniform(0.1, 1.0), 1)

        elif error_code == 3:
            physical_requirement = round(random.uniform(0.9, 1.0), 1)
            cognitive_requirement = round(random.uniform(0.3, 0.4), 1)
            urgency = round(random.uniform(0.1, 1.0), 1)

        elif error_code == 4:
            physical_requirement = round(random.uniform(0.1, 0.3), 1)
            cognitive_requirement = round(random.uniform(0.6, 0.7), 1)
            urgency = round(random.uniform(0.1, 1.0), 1)

        self.__task_requirements = [
            physical_requirement,
            cognitive_requirement,
            urgency,
        ]

        self.__start_time_failure = time.time() - self.__start_time
        self.__failures['start_time'].append(self.__start_time_failure)

        # Remote only
        self.__assigned_to = self.OPERATOR

        return self.__assigned_to

    def __task_started(self, request):

        self.__start_time = time.time()

        return []

    def __failure_resolved(self, request):

        self.__end_time_failure = time.time() - self.__start_time
        self.__failures['end_time'].append(self.__end_time_failure)

        self.__failures['duration'].append(
            self.__end_time_failure - self.__start_time_failure
        )

        self.__is_failure_resolved = True

        return []

    def __record_failure(self, request):

        # Add a ridiculous number to record it as a failure
        self.__end_time_failure = (time.time() + 10000) - self.__start_time

        self.__failures['end_time'].append(self.__end_time_failure)

        self.__failures['duration'].append(
            self.__end_time_failure - self.__start_time_failure
        )

        self.__is_failure_resolved = True

        return []

    def main_loop(self):

        norms = self.__model.get_parameters(self.OPERATOR)

        self.__capabilities[self.OPERATOR] = {
            'lower_bound_1': norms[0][0],
            'upper_bound_1': norms[0][1],
            'lower_bound_2': norms[1][0],
            'upper_bound_2': norms[1][1],
            'lower_bound_3': norms[2][0],
            'upper_bound_3': norms[2][1]
        }

        lower_bound_1 = Float32()
        lower_bound_1.data = norms[0][0]
        self.__lower_1_publisher.publish(lower_bound_1)

        upper_bound_1 = Float32()
        upper_bound_1.data = norms[0][1]
        self.__upper_1_publisher.publish(upper_bound_1)

        lower_bound_2 = Float32()
        lower_bound_2.data = norms[1][0]
        self.__lower_2_publisher.publish(lower_bound_2)

        upper_bound_2 = Float32()
        upper_bound_2.data = norms[1][1]
        self.__upper_2_publisher.publish(upper_bound_2)

        lower_bound_3 = Float32()
        lower_bound_3.data = norms[2][0]
        self.__lower_3_publisher.publish(lower_bound_3)

        upper_bound_3 = Float32()
        upper_bound_3.data = norms[2][1]
        self.__upper_3_publisher.publish(upper_bound_3)

        if self.__is_failure_resolved:

            self.__adjusted_threshold = self.THRESHOLD / (
                1 + self.__task_requirements[2]
            )

            if self.__failures['duration'][-1] < self.THRESHOLD:
                speed = 1 - (self.__failures['duration'][-1] / self.THRESHOLD)
            else:
                speed = 0

            self.__operator_speed[self.__assigned_to].append(speed)

            self.__update_bounds()
            self.__update_target_service()

            self.__is_failure_resolved = False

            self.__data_recorder.save_results(
                self.__counter, self.__assigned_to,
                self.__failures['start_time'][-1],
                self.__failures['duration'][-1], self.__adjusted_threshold,
                self.__task_requirements, self.__capabilities,
                self.__operator_time_failures
            )

            if self.__counter == 5:
                self.__data_recorder.write_results()

    def __update_bounds(self):

        for j in range(self.NUM_BINS):
            for k in range(self.NUM_BINS):
                for z in range(self.NUM_BINS):

                    if (
                        (self.__bin_limits[j]) <
                        (self.__task_requirements[0]) <=
                        (self.__bin_limits[j + 1])
                    ) and (
                        (self.__bin_limits[k]) <
                        (self.__task_requirements[1]) <=
                        (self.__bin_limits[k + 1])
                    ) and (
                        (self.__bin_limits[z]) <
                        (self.__task_requirements[2]) <=
                        (self.__bin_limits[z + 1])
                    ):

                        if self.__assigned_to % 2 == 0:

                            if self.__failures['duration'][
                                -1] < self.__adjusted_threshold:
                                self.__success_local[self.__assigned_to][j, k,
                                                                         z] += 1
                                self.success[self.__assigned_to] += 1

                            self.__total_observations_local[self.__assigned_to][
                                j, k, z] += 1

                        else:

                            if self.__failures['duration'][
                                -1] < self.__adjusted_threshold:
                                self.__success_remote[self.__assigned_to][
                                    j, k, z] += 1
                                self.success[self.__assigned_to] += 1

                            self.__total_observations_remote[self.__assigned_to
                                                            ][j, k, z] += 1

                        self.total[self.__assigned_to] += 1

        if self.__assigned_to % 2 == 0:
            self.observations_probabilities_local[
                self.__assigned_to
            ] = np.divide(
                self.__success_local[self.__assigned_to],
                self.__total_observations_local[self.__assigned_to],
                where=self.__total_observations_local[self.__assigned_to] != 0
            )

            self.probabilities_index_local = np.array(
                [
                    [j, k, z] for j in range(self.NUM_BINS)
                    for k in range(self.NUM_BINS)
                    for z in range(self.NUM_BINS)
                    if self.__total_observations_local[self.__assigned_to][
                        j, k, z] > 0
                ]
            )
            self.__model.update(
                self.bin_centers,
                self.observations_probabilities_local[self.__assigned_to],
                self.probabilities_index_local, self.__assigned_to
            )

        else:
            self.observations_probabilities_remote[
                self.__assigned_to
            ] = np.divide(
                self.__success_remote[self.__assigned_to],
                self.__total_observations_remote[self.__assigned_to],
                where=self.__total_observations_remote[self.__assigned_to] != 0
            )

            self.probabilities_index_remote = np.array(
                [
                    [j, k, z] for j in range(self.NUM_BINS)
                    for k in range(self.NUM_BINS)
                    for z in range(self.NUM_BINS)
                    if self.__total_observations_remote[self.__assigned_to][
                        j, k, z] > 0
                ]
            )
            self.__model.update(
                self.bin_centers,
                self.observations_probabilities_remote[self.__assigned_to],
                self.probabilities_index_remote, self.__assigned_to
            )


class DataRecorder:

    def __init__(self, operator, threshold, lr, weight_decay):

        self.OPERATOR = operator
        self.THRESHOLD = threshold
        self.LR = lr
        self.DECAY = weight_decay

        keys = [
            'failure_id', 'self.__assigned_to', 'start_time', 'duration',
            'success', 'requirement_1', 'requirement_2', 'requirement_3',
            'operator', 'threshold', 'learning_rate', 'weight_decay'
        ]

        keys.append(f'{self.OPERATOR}_lower_bound_1')
        keys.append(f'{self.OPERATOR}_upper_bound_1')
        keys.append(f'{self.OPERATOR}_lower_bound_2')
        keys.append(f'{self.OPERATOR}_upper_bound_2')
        keys.append(f'{self.OPERATOR}_lower_bound_3')
        keys.append(f'{self.OPERATOR}_upper_bound_3')
        keys.append(f'{self.OPERATOR}_cost')
        keys.append(f'{self.OPERATOR}_reward_capability_1')
        keys.append(f'{self.OPERATOR}_reward_capability_2')
        keys.append(f'{self.OPERATOR}_reward_capability_3')
        keys.append(f'{self.OPERATOR}_total_reward')
        keys.append(f'{self.OPERATOR}_assigned_failures')
        keys.append(f'{self.OPERATOR}_time_spent_failures')

        self.__results = {key: [] for key in keys}

    def save_results(
        self,
        failure_id,
        assigned_to,
        start_time,
        duration,
        threshold,
        task_requirements,
        norms,
        operator_time_failures,
    ):

        self.__results['failure_id'].append(failure_id)
        self.__results['self.__assigned_to'].append(assigned_to)
        self.__results['start_time'].append(start_time)
        self.__results['duration'].append(duration)
        self.__results['requirement_1'].append(task_requirements[0])
        self.__results['requirement_2'].append(task_requirements[1])
        self.__results['requirement_3'].append(task_requirements[2])
        self.__results['operator'].append(self.OPERATOR)
        self.__results['threshold'].append(threshold)
        self.__results['learning_rate'].append(self.LR)
        self.__results['weight_decay'].append(self.DECAY)

        if duration < threshold:
            self.__results['success'].append(True)
        else:
            self.__results['success'].append(False)

        self.__results[f'{self.OPERATOR}_lower_bound_1'].append(
            norms[self.OPERATOR]['lower_bound_1'].item()
        )
        self.__results[f'{self.OPERATOR}_upper_bound_1'].append(
            norms[self.OPERATOR]['upper_bound_1'].item()
        )
        self.__results[f'{self.OPERATOR}_lower_bound_2'].append(
            norms[self.OPERATOR]['lower_bound_2'].item()
        )
        self.__results[f'{self.OPERATOR}_upper_bound_2'].append(
            norms[self.OPERATOR]['upper_bound_2'].item()
        )
        self.__results[f'{self.OPERATOR}_lower_bound_3'].append(
            norms[self.OPERATOR]['lower_bound_3'].item()
        )
        self.__results[f'{self.OPERATOR}_upper_bound_3'].append(
            norms[self.OPERATOR]['upper_bound_3'].item()
        )
        self.__results[f'{self.OPERATOR}_time_spent_failures'].append(
            operator_time_failures[self.OPERATOR]
        )

    def write_results(self):

        # Determine the filename
        base_filename = f'operator_{self.OPERATOR}.csv'
        results_dir = 'results'  # Directory to save results
        filename = os.path.join(results_dir, base_filename)

        # Ensure the results directory exists
        os.makedirs(results_dir, exist_ok=True)

        # Check if the file already exists
        if os.path.exists(filename):
            # Find the next available filename
            suffix = 1
            while True:
                new_filename = f'{filename.split(".csv")[0]} ({suffix}).csv'
                if not os.path.exists(new_filename):
                    filename = new_filename
                    break
                suffix += 1

        # Generate headers from self.__results keys
        headers = list(self.__results.keys())

        # Write to CSV
        with open(filename, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(headers)  # Write headers

            # Write each key's values as separate rows
            max_length = max(len(self.__results[key]) for key in headers)
            for i in range(max_length):
                row = [
                    self.__results[key][i]
                    if i < len(self.__results[key]) else None for key in headers
                ]
                writer.writerow(row)


if __name__ == "__main__":

    rospy.init_node(
        'failure_allocation',
        log_level=rospy.INFO,
    )

    operator = rospy.get_param(param_name=f'{rospy.get_name()}/operator')

    bins = 25
    max_threshold = 100
    learning_rate = 0.001
    decay = 0.001

    performance_model_allocation = AllocationFramework(
        num_operator=operator,
        num_bins=bins,
        threshold=max_threshold,
        lr=learning_rate,
        weight_decay=decay
    )

    while not rospy.is_shutdown():

        performance_model_allocation.main_loop()
        performance_model_allocation.RATE.sleep()
