#!/usr/bin/env python
import rospy
import torch
from torch import nn, sigmoid
from torch.nn import Parameter, ParameterDict

import numpy as np
import time
import pandas as pd
import math
import random

from std_msgs.msg import (Int32, Bool, Float32, Float32MultiArray)
from std_srvs.srv import (Empty)
from Scripts.srv import (ReceiveInt)

# Set data type based on CUDA availability
dtype = torch.cuda.FloatTensor if torch.cuda.is_available(
) else torch.DoubleTensor


class PerformanceModel(nn.Module):

    def __init__(
        self,
        num_operators,
        lr,
        weight_decay,
    ):
        super(PerformanceModel, self).__init__()

        self.OPERATORS = range(1, num_operators + 1)
        self.__local_operators = nn.ModuleDict()
        self.__remote_operators = nn.ModuleDict()

        self.__t_count = 0
        self.LR = lr
        self.DECAY = weight_decay

        # Changed initialization of local and remote operators
        for operator_number in self.OPERATORS:

            if operator_number % 2 == 0:
                operator_params = {
                    'lower_bound_1':
                        Parameter(
                            dtype(-10.0 * np.ones(1)), requires_grad=True
                        ),
                    'upper_bound_1':
                        Parameter(dtype(10.0 * np.ones(1)), requires_grad=True),
                    'lower_bound_2':
                        Parameter(
                            dtype(-10.0 * np.ones(1)), requires_grad=True
                        ),
                    'upper_bound_2':
                        Parameter(dtype(10.0 * np.ones(1)), requires_grad=True),
                    'lower_bound_3':
                        Parameter(
                            dtype(-10.0 * np.ones(1)), requires_grad=True
                        ),
                    'upper_bound_3':
                        Parameter(dtype(10.0 * np.ones(1)), requires_grad=True)
                }
            else:
                operator_params = {
                    'lower_bound_1':
                        Parameter(dtype(10.0 * np.ones(1)), requires_grad=True),
                    'upper_bound_1':
                        Parameter(
                            dtype(-10.0 * np.ones(1)), requires_grad=True
                        ),
                    'lower_bound_2':
                        Parameter(dtype(10.0 * np.ones(1)), requires_grad=True),
                    'upper_bound_2':
                        Parameter(
                            dtype(-10.0 * np.ones(1)), requires_grad=True
                        ),
                    'lower_bound_3':
                        Parameter(dtype(10.0 * np.ones(1)), requires_grad=True),
                    'upper_bound_3':
                        Parameter(
                            dtype(-10.0 * np.ones(1)), requires_grad=True
                        ),
                }

            if operator_number % 2 == 0:
                self.__local_operators[f'{operator_number}'] = ParameterDict(
                    operator_params
                )
            else:
                self.__remote_operators[f'{operator_number}'] = ParameterDict(
                    operator_params
                )

        self.optimizer = torch.optim.Adam(
            self.parameters(), lr=self.LR, weight_decay=self.DECAY
        )

    def params_no_sigmoid(self, operator_number):

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

        bounds = [
            params['lower_bound_1'].item(), params['upper_bound_1'].item(),
            params['lower_bound_2'].item(), params['upper_bound_2'].item(),
            params['lower_bound_3'].item(), params['upper_bound_3'].item()
        ]

        return bounds

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
            'upper_bound_1',
            'lower_bound_1',
            'upper_bound_2',
            'lower_bound_2',
            'upper_bound_3',
            'lower_bound_3',
        ]

        counter = 0

        for i, (norm_l, norm_u) in enumerate(norms):
            if abs(norm_u - norm_l) < 0.05:
                operators = self.__local_operators if operator % 2 == 0 else self.__remote_operators
                upper_bound = operators[f'{operator}'][bounds[2 * i]]
                lower_bound = operators[f'{operator}'][bounds[2 * i + 1]]

                if upper_bound.grad is not None and lower_bound.grad is not None:
                    operators[f'{operator}'][bounds[2 * i
                                                   ]].requires_grad = False
                    operators[f'{operator}'][bounds[2 * i
                                                    + 1]].requires_grad = False

                # print(
                #     "Capability {}, converged with values ({})".format(
                #         i, str(lower_bound)
                #     )
                # )

                counter += 1

        if counter == 3:

            print(f"All capabilities have converged for operator {operator}:")
            print(self.params_no_sigmoid(operator))

            # print("Bound 1:")
            # print("Bound 2:")
            # print("Bound 3:")

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

    def __init__(
        self,
        num_operators,
        num_bins,
        threshold,
        lr,
        weight_decay,
    ):

        np.seterr(divide='ignore', invalid='ignore')

        self.OPERATORS = range(1, num_operators + 1)
        self.NUM_BINS = num_bins
        self.THRESHOLD = threshold
        self.LR = lr
        self.DECAY = weight_decay
        self.RATE = rospy.Rate(100)

        self.__model = PerformanceModel(num_operators, self.LR, self.DECAY)
        self.__model = self.__model.cuda() if torch.cuda.is_available(
        ) else self.__model

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
        self.__failure_state = False
        self.__is_success = False
        self.__assigned_to = 0
        self.__task_allocation = []

        self.__failures = {
            'start_time': [],
            'end_time': [],
            'duration': [],
            'resolution_start_time': []
        }

        self.__operator_time_failures = {i: 0 for i in self.OPERATORS}
        self.__operator_speed = {i: [] for i in self.OPERATORS}
        self.total = {i: 0 for i in self.OPERATORS}
        self.success = {i: 0 for i in self.OPERATORS}

        self.__failures_assigned = {}

        for i in self.OPERATORS:
            operator_number = i

            self.__failures_assigned[operator_number] = 0
            self.success_matrix[operator_number] = np.ones(
                (self.NUM_BINS, self.NUM_BINS, self.NUM_BINS)
            )

            # Local operators have even number indexes
            if operator_number % 2 == 0:
                self.observations_probabilities_local[
                    operator_number] = np.zeros(
                        (self.NUM_BINS, self.NUM_BINS, self.NUM_BINS)
                    )
                self.__total_observations_local[operator_number] = np.zeros(
                    (self.NUM_BINS, self.NUM_BINS, self.NUM_BINS)
                )
                self.__total_success_local[operator_number] = np.zeros(
                    (self.NUM_BINS, self.NUM_BINS)
                )
                self.__success_local[operator_number] = np.zeros(
                    (self.NUM_BINS, self.NUM_BINS, self.NUM_BINS)
                )

            else:
                self.observations_probabilities_remote[
                    operator_number] = np.zeros(
                        (self.NUM_BINS, self.NUM_BINS, self.NUM_BINS)
                    )
                self.__total_observations_remote[operator_number] = np.zeros(
                    (self.NUM_BINS, self.NUM_BINS, self.NUM_BINS)
                )
                self.__total_success_remote[operator_number] = np.zeros(
                    (self.NUM_BINS, self.NUM_BINS)
                )
                self.__success_remote[operator_number] = np.zeros(
                    (self.NUM_BINS, self.NUM_BINS, self.NUM_BINS)
                )

        self.__task_requirements = [0.0, 0.0, 0.0]

        self.__capabilities = {}

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
            '/record_failure',
            Empty,
            self.__record_failure,
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

        self.__failure_state_publisher = rospy.Publisher(
            '/update_bounds/failure_state',
            Bool,
            queue_size=1,
        )

        self.__threshold_publisher = rospy.Publisher(
            '/update_bounds/threshold',
            Int32,
            queue_size=1,
        )
        self.__learning_rate_publisher = rospy.Publisher(
            '/update_bounds/learning_rate',
            Float32,
            queue_size=1,
        )
        self.__weight_decay_publisher = rospy.Publisher(
            '/update_bounds/weight_decay',
            Float32,
            queue_size=1,
        )
        self.__remote_bound_1_publisher = rospy.Publisher(
            '/update_bounds/remote_bound_1',
            Float32MultiArray,
            queue_size=1,
        )
        self.__local_bound_1_publisher = rospy.Publisher(
            '/update_bounds/local_bound_1',
            Float32MultiArray,
            queue_size=1,
        )
        self.__remote_bound_2_publisher = rospy.Publisher(
            '/update_bounds/remote_bound_2',
            Float32MultiArray,
            queue_size=1,
        )
        self.__local_bound_2_publisher = rospy.Publisher(
            '/update_bounds/local_bound_2',
            Float32MultiArray,
            queue_size=1,
        )
        self.__remote_bound_3_publisher = rospy.Publisher(
            '/update_bounds/remote_bound_3',
            Float32MultiArray,
            queue_size=1,
        )
        self.__local_bound_3_publisher = rospy.Publisher(
            '/update_bounds/local_bound_3',
            Float32MultiArray,
            queue_size=1,
        )
        self.__failure_id_publisher = rospy.Publisher(
            '/update_bounds/failure_id',
            Int32,
            queue_size=1,
        )
        self.__failure_start_publisher = rospy.Publisher(
            '/update_bounds/failure_start_time',
            Float32MultiArray,
            queue_size=1,
        )
        self.__failure_end_publisher = rospy.Publisher(
            '/update_bounds/failure_end_time',
            Float32MultiArray,
            queue_size=1,
        )
        self.__failure_durations_publisher = rospy.Publisher(
            '/update_bounds/failure_durations',
            Float32MultiArray,
            queue_size=1,
        )
        self.__adjusted_threshold_publisher = rospy.Publisher(
            '/update_bounds/adjusted_threshold',
            Float32,
            queue_size=1,
        )
        self.__is_success_publisher = rospy.Publisher(
            '/update_bounds/success',
            Bool,
            queue_size=1,
        )
        self.__requirements_publisher = rospy.Publisher(
            '/update_bounds/',
            Float32MultiArray,
            queue_size=1,
        )
        self.__operator_times_publisher = rospy.Publisher(
            '/update_bounds/operator_failure_times',
            Float32MultiArray,
            queue_size=1,
        )
        self.__failure_assigned_publisher = rospy.Publisher(
            '/update_bounds/assigned_to',
            Int32,
            queue_size=1,
        )
        self.__bounds_remote_publisher = rospy.Publisher(
            '/update_bounds/remote_no_sigmoid_bounds',
            Float32MultiArray,
            queue_size=1,
        )
        self.__bounds_local_publisher = rospy.Publisher(
            '/update_bounds/remote_no_sigmoid_bounds',
            Float32MultiArray,
            queue_size=1,
        )

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

        self.__failure_state = True

        # Remote only
        if not self.__task_allocation:

            self.__assigned_to = 1

        else:
            if self.__task_allocation[-1] == 1:

                self.__assigned_to = 2

            elif self.__task_allocation[-1] == 2:

                self.__assigned_to = 1

        self.__task_allocation.append(self.__assigned_to)

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

        self.__operator_time_failures[self.__assigned_to
                                     ] += self.__failures['duration'][-1]

        self.__failure_state = False
        self.__is_failure_resolved = True

        return []

    def __record_failure(self, request):

        # Merge this with failure_resolved?

        # Add a ridiculous number to record it as a failure
        self.__end_time_failure = (time.time() + 10000) - self.__start_time

        self.__failures['end_time'].append(self.__end_time_failure)

        self.__failures['duration'].append(
            self.__end_time_failure - self.__start_time_failure
        )

        self.__failure_state = False
        self.__is_failure_resolved = True

        return []

    def main_loop(self):

        self.__publish_data()

        if self.__is_failure_resolved:

            self.__adjusted_threshold = self.THRESHOLD / (
                1 + self.__task_requirements[2]
            )

            if self.__failures['duration'][-1] < self.THRESHOLD:
                speed = 1 - (self.__failures['duration'][-1] / self.THRESHOLD)
                self.__is_success = True
            else:
                speed = 0
                self.__is_success = False

            self.__operator_speed[self.__assigned_to].append(speed)

            self.__update_bounds()
            self.__update_target_service()

            self.__assigned_to = 0
            self.__is_failure_resolved = False

    def __publish_data(self):

        for operator in self.OPERATORS:
            norms = self.__model.get_parameters(operator)

            self.__capabilities[operator] = {
                'lower_bound_1': norms[0][0],
                'upper_bound_1': norms[0][1],
                'lower_bound_2': norms[1][0],
                'upper_bound_2': norms[1][1],
                'lower_bound_3': norms[2][0],
                'upper_bound_3': norms[2][1]
            }

        bounds_remote = Float32MultiArray()
        bounds_remote.data = self.__model.params_no_sigmoid(1)
        self.__bounds_remote_publisher.publish(bounds_remote)

        bounds_local = Float32MultiArray()
        bounds_local.data = self.__model.params_no_sigmoid(2)
        self.__bounds_local_publisher.publish(bounds_local)

        remote_bound_1 = Float32MultiArray()
        remote_bound_1.data = [
            self.__capabilities[1]['lower_bound_1'].item(),
            self.__capabilities[1]['upper_bound_1'].item(),
        ]
        self.__remote_bound_1_publisher.publish(remote_bound_1)

        local_bound_1 = Float32MultiArray()
        local_bound_1.data = [
            self.__capabilities[2]['lower_bound_1'].item(),
            self.__capabilities[2]['upper_bound_1'].item(),
        ]
        self.__local_bound_1_publisher.publish(local_bound_1)

        remote_bound_2 = Float32MultiArray()
        remote_bound_2.data = [
            self.__capabilities[1]['lower_bound_2'].item(),
            self.__capabilities[1]['upper_bound_2'].item(),
        ]
        self.__remote_bound_2_publisher.publish(remote_bound_2)

        local_bound_2 = Float32MultiArray()
        local_bound_2.data = [
            self.__capabilities[2]['lower_bound_2'].item(),
            self.__capabilities[2]['upper_bound_2'].item(),
        ]
        self.__local_bound_2_publisher.publish(local_bound_2)

        remote_bound_3 = Float32MultiArray()
        remote_bound_3.data = [
            self.__capabilities[1]['lower_bound_3'].item(),
            self.__capabilities[1]['upper_bound_3'].item(),
        ]
        self.__remote_bound_3_publisher.publish(remote_bound_3)

        local_bound_3 = Float32MultiArray()
        local_bound_3.data = [
            self.__capabilities[2]['lower_bound_3'].item(),
            self.__capabilities[2]['upper_bound_3'].item(),
        ]
        self.__local_bound_3_publisher.publish(local_bound_3)

        threshold = Int32()
        threshold.data = self.THRESHOLD
        self.__threshold_publisher.publish(threshold)

        learning_rate = Float32()
        learning_rate.data = self.LR
        self.__learning_rate_publisher.publish(learning_rate)

        weight_decay = Float32()
        weight_decay.data = self.DECAY
        self.__weight_decay_publisher.publish(weight_decay)

        failure_id = Int32()
        failure_id.data = self.failure_counter
        self.__failure_id_publisher.publish(failure_id)

        failure_start = Float32MultiArray()
        failure_start.data = self.__failures['start_time']
        self.__failure_start_publisher.publish(failure_start)

        failure_end = Float32MultiArray()
        failure_end.data = self.__failures['end_time']
        self.__failure_end_publisher.publish(failure_end)

        failure_durations = Float32MultiArray()
        failure_durations.data = self.__failures['duration']
        self.__failure_durations_publisher.publish(failure_durations)

        task_requirements = Float32MultiArray()
        task_requirements.data = self.__task_requirements
        self.__requirements_publisher.publish(task_requirements)

        adjusted_threshold = Float32()
        adjusted_threshold.data = self.__adjusted_threshold
        self.__adjusted_threshold_publisher.publish(adjusted_threshold)

        is_success = Bool()
        is_success.data = self.__is_success
        self.__is_success_publisher.publish(is_success)

        failure_state = Bool()
        failure_state.data = self.__failure_state
        self.__failure_state_publisher.publish(failure_state)

        time_failures = Float32MultiArray()
        time_failures.data = [
            self.__operator_time_failures[1],
            self.__operator_time_failures[2],
        ]
        self.__operator_times_publisher.publish(time_failures)

        assigned_to = Int32()
        assigned_to.data = self.__assigned_to
        self.__failure_assigned_publisher.publish(assigned_to)

    def __update_bounds(self):

        rospy.loginfo(f'\033[92mUpdating bounds ...\033[0m',)

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
                self.probabilities_index_local,
                self.__assigned_to,
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

        rospy.loginfo(f'\033[92mBounds updated.\033[0m',)


if __name__ == "__main__":

    rospy.init_node(
        'failure_allocation',
        log_level=rospy.INFO,
    )

    operators = 2
    bins = 25
    max_threshold = 100
    learning_rate = 0.001
    decay = 0.001

    performance_model_allocation = AllocationFramework(
        num_operators=operators,
        num_bins=bins,
        threshold=max_threshold,
        lr=learning_rate,
        weight_decay=decay
    )

    while not rospy.is_shutdown():

        performance_model_allocation.main_loop()
        performance_model_allocation.RATE.sleep()
