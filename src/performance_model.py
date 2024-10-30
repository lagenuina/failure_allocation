#!/usr/bin/env python
import rospy
import torch
from torch import nn, sigmoid
from torch.nn import Parameter, ParameterDict

import numpy as np
import time
import pandas as pd
import random

import os
from glob import glob
from ast import (literal_eval)
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
        remote_1,
        remote_2,
        remote_3,
        local_1,
        local_2,
        local_3,
    ):
        super(PerformanceModel, self).__init__()

        self.OPERATORS = range(1, num_operators + 1)
        self.__local_operators = nn.ModuleDict()
        self.__remote_operators = nn.ModuleDict()

        self.LR = lr
        self.DECAY = weight_decay

        # Changed initialization of local and remote operators
        for i in self.OPERATORS:
            operator_number = i

            if i % 2 == 0:
                operator_params = {
                    'lower_bound_1':
                        Parameter(
                            dtype(local_1[0] * np.ones(1)), requires_grad=True
                        ),
                    'upper_bound_1':
                        Parameter(
                            dtype(local_1[1] * np.ones(1)), requires_grad=True
                        ),
                    'lower_bound_2':
                        Parameter(
                            dtype(local_2[0] * np.ones(1)), requires_grad=True
                        ),
                    'upper_bound_2':
                        Parameter(
                            dtype(local_2[1] * np.ones(1)), requires_grad=True
                        ),
                    'lower_bound_3':
                        Parameter(
                            dtype(local_3[0] * np.ones(1)), requires_grad=True
                        ),
                    'upper_bound_3':
                        Parameter(
                            dtype(local_3[1] * np.ones(1)), requires_grad=True
                        )
                }

            else:
                operator_params = {
                    'lower_bound_1':
                        Parameter(
                            dtype(remote_1[0] * np.ones(1)), requires_grad=True
                        ),
                    'upper_bound_1':
                        Parameter(
                            dtype(remote_1[1] * np.ones(1)), requires_grad=True
                        ),
                    'lower_bound_2':
                        Parameter(
                            dtype(remote_2[0] * np.ones(1)), requires_grad=True
                        ),
                    'upper_bound_2':
                        Parameter(
                            dtype(remote_2[1] * np.ones(1)), requires_grad=True
                        ),
                    'lower_bound_3':
                        Parameter(
                            dtype(remote_3[0] * np.ones(1)), requires_grad=True
                        ),
                    'upper_bound_3':
                        Parameter(
                            dtype(remote_3[1] * np.ones(1)), requires_grad=True
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
            self.parameters(),
            lr=self.LR,
            weight_decay=self.DECAY,
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


class AllocationFramework:

    def __init__(
        self,
        num_operators,
        num_bins,
        num_failures,
        threshold,
        lr,
        weight_decay,
        mode,
        remote_1,
        remote_2,
        remote_3,
        local_1,
        local_2,
        local_3,
    ):

        np.seterr(divide='ignore', invalid='ignore')

        self.OPERATORS = range(1, num_operators + 1)
        self.NUM_BINS = num_bins
        self.NUM_FAILURES = num_failures
        self.THRESHOLD = threshold
        self.LR = lr
        self.DECAY = weight_decay
        self.RATE = rospy.Rate(100)
        self.MODE = mode

        self.__model = PerformanceModel(
            num_operators,
            self.LR,
            self.DECAY,
            remote_1,
            remote_2,
            remote_3,
            local_1,
            local_2,
            local_3,
        )

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
        self.__task_allocation = []
        self.__adjusted_threshold = 0
        self.__is_failure_resolved = False
        self.__counter = None

        self.__failures = {
            'start_time': [],
            'end_time': [],
            'duration': [],
        }

        self.__reward_capability_1 = {i: 0 for i in self.OPERATORS}
        self.__reward_capability_2 = {i: 0 for i in self.OPERATORS}
        self.__reward_capability_3 = {i: 0 for i in self.OPERATORS}
        self.__reward_speed = {i: 0 for i in self.OPERATORS}

        self.__cost = {i: 0 for i in self.OPERATORS}
        self.__operator_time_failures = {i: 0 for i in self.OPERATORS}
        self.__operator_speed = {i: [] for i in self.OPERATORS}
        self.total = {i: 0 for i in self.OPERATORS}
        self.success = {i: 0 for i in self.OPERATORS}

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

        load_file_name = '/home/fetch/catkin_workspaces/hololens_ws/src/failure_allocation/src/failure_requirements.csv'
        data_frame = pd.read_csv(load_file_name, header=None)
        self.__task_requirements_list = (
            data_frame.iloc[:, :self.NUM_FAILURES].values
        )
        self.__task_requirements = [0.0, 0.0, 0.0]

        self.__assigned_to = 0
        self.__is_success = False
        self.__failure_state = False

        self.__capabilities = {i: 0 for i in self.OPERATORS}

        self.__reward_operators = {i: 0 for i in self.OPERATORS}
        self.__expected_reward = {i: 0 for i in self.OPERATORS}

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
            '/performance_model/failure_state',
            Bool,
            queue_size=1,
        )

        self.__total_assigned_publisher = rospy.Publisher(
            '/performance_model/total_failures_assigned',
            Float32MultiArray,
            queue_size=1,
        )

        self.__failure_allocation_publisher = rospy.Publisher(
            '/performance_model/failure_allocation',
            Float32MultiArray,
            queue_size=1,
        )

        self.__threshold_publisher = rospy.Publisher(
            '/performance_model/threshold',
            Int32,
            queue_size=1,
        )

        self.__remote_bound_1_publisher = rospy.Publisher(
            '/performance_model/remote_bound_1',
            Float32MultiArray,
            queue_size=1,
        )
        self.__local_bound_1_publisher = rospy.Publisher(
            '/performance_model/local_bound_1',
            Float32MultiArray,
            queue_size=1,
        )
        self.__remote_bound_2_publisher = rospy.Publisher(
            '/performance_model/remote_bound_2',
            Float32MultiArray,
            queue_size=1,
        )
        self.__local_bound_2_publisher = rospy.Publisher(
            '/performance_model/local_bound_2',
            Float32MultiArray,
            queue_size=1,
        )
        self.__remote_bound_3_publisher = rospy.Publisher(
            '/performance_model/remote_bound_3',
            Float32MultiArray,
            queue_size=1,
        )
        self.__local_bound_3_publisher = rospy.Publisher(
            '/performance_model/local_bound_3',
            Float32MultiArray,
            queue_size=1,
        )
        self.__failure_id_publisher = rospy.Publisher(
            '/performance_model/failure_id',
            Int32,
            queue_size=1,
        )
        self.__failure_start_publisher = rospy.Publisher(
            '/performance_model/failure_start_time',
            Float32MultiArray,
            queue_size=1,
        )
        self.__failure_end_publisher = rospy.Publisher(
            '/performance_model/failure_end_time',
            Float32MultiArray,
            queue_size=1,
        )
        self.__failure_durations_publisher = rospy.Publisher(
            '/performance_model/failure_durations',
            Float32MultiArray,
            queue_size=1,
        )
        self.__adjusted_threshold_publisher = rospy.Publisher(
            '/performance_model/adjusted_threshold',
            Float32,
            queue_size=1,
        )
        self.__is_success_publisher = rospy.Publisher(
            '/performance_model/success',
            Bool,
            queue_size=1,
        )
        self.__requirements_publisher = rospy.Publisher(
            '/performance_model/task_requirements',
            Float32MultiArray,
            queue_size=1,
        )
        self.__mode_publisher = rospy.Publisher(
            '/mode',
            Int32,
            queue_size=1,
        )
        self.__cost_publisher = rospy.Publisher(
            '/performance_model/cost',
            Float32MultiArray,
            queue_size=1,
        )
        self.__reward_capability_1_publisher = rospy.Publisher(
            '/performance_model/reward_capability_1',
            Float32MultiArray,
            queue_size=1,
        )
        self.__reward_capability_2_publisher = rospy.Publisher(
            '/performance_model/reward_capability_2',
            Float32MultiArray,
            queue_size=1,
        )
        self.__reward_capability_3_publisher = rospy.Publisher(
            '/performance_model/reward_capability_3',
            Float32MultiArray,
            queue_size=1,
        )
        self.__total_reward_publisher = rospy.Publisher(
            '/performance_model/total_reward',
            Float32MultiArray,
            queue_size=1,
        )
        self.__expected_reward_publisher = rospy.Publisher(
            '/performance_model/expected_reward',
            Float32MultiArray,
            queue_size=1,
        )
        self.__failure_assigned_publisher = rospy.Publisher(
            '/performance_model/assigned_to',
            Int32,
            queue_size=1,
        )
        self.__operator_times_publisher = rospy.Publisher(
            '/performance_model/operator_failure_times',
            Float32MultiArray,
            queue_size=1,
        )
        self.__reward_speed_publisher = rospy.Publisher(
            '/performance_model/reward_speed',
            Float32MultiArray,
            queue_size=1,
        )

    def __target_counter(self, message):

        self.__counter = message.data

    def __failure_service(self, request):

        error_code = request.request

        if self.__counter not in [1, 3, 5, 6, 9, 10, 11, 15, 16, 18]:

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
        else:
            self.__task_requirements = (
                self.__task_requirements_list[:, self.failure_counter]
            )

        self.__calculate_expected_reward()

        self.__assigned_to = self.__assign_failure(self.__expected_reward)

        self.__failure_state = True

        self.__start_time_failure = time.time() - self.__start_time
        self.__failures['start_time'].append(self.__start_time_failure)

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

        print("Failure resolved!")

        return []

    def __record_failure(self, request):

        # Add a ridiculous number to record it as a failure
        self.__end_time_failure = (time.time() + 10000) - self.__start_time

        self.__failures['end_time'].append(self.__end_time_failure)

        self.__failures['duration'].append(
            self.__end_time_failure - self.__start_time_failure
        )

        self.__is_failure_resolved = True
        self.__failure_state = False

        print("Failure resolved!")

        return []

    def __assign_failure(self, expected_reward):

        if self.MODE == 1:
            failure_allocated_to = np.random.randint(1, 3)

        elif self.MODE == 2:

            reward_keys = list(expected_reward.keys())
            reward_tensors = list(expected_reward.values())

            # Concatenate tensors into a single tensor
            concatenated_tensor = torch.cat(reward_tensors)

            # Find the maximum value in the tensor
            highest_reward = concatenated_tensor.max().item()

            # Find all indices with the maximum value
            operators_highest_reward = (concatenated_tensor == highest_reward
                                       ).nonzero(as_tuple=True)[0].tolist()

            if len(operators_highest_reward) > 1:

                # Assign to the operator with least amount of failures
                failure_allocated_to = min(
                    self.__failures_assigned,
                    key=lambda k: self.__failures_assigned[k]
                )

            else:
                highest_reward_operator_index = operators_highest_reward[0]
                failure_allocated_to = reward_keys[highest_reward_operator_index
                                                  ]

        self.__failures_assigned[failure_allocated_to] += 1

        self.__task_allocation.append(failure_allocated_to)

        return failure_allocated_to

    def calculate_reward(self, norms, task_requirement, operator):

        performance = []

        for i, (lower_bound, upper_bound) in enumerate(norms):

            if task_requirement[i] <= lower_bound:
                performance.append(torch.tensor([1.0]))
            elif task_requirement[i] > upper_bound:
                performance.append(torch.tensor([0.0]))
            else:
                performance.append(
                    (upper_bound - task_requirement[i]) /
                    (upper_bound - lower_bound + 0.0001)
                )

        self.__reward_capability_1[operator] = performance[0]
        self.__reward_capability_2[operator] = performance[1]
        self.__reward_capability_3[operator] = performance[2]

        reward_capability = 0.3 * self.__reward_capability_1[
            operator] + 0.3 * self.__reward_capability_2[
                operator] + 0.4 * self.__reward_capability_3[operator]

        return reward_capability

    def calculate_cost(self):

        cost = []
        total_time_failures = 0

        for k in self.__failures['duration']:

            if k < 1000:
                total_time_failures += k

        for i in self.OPERATORS:
            if self.failure_counter != 0:
                ratio_failures = (
                    self.__operator_time_failures[i] / total_time_failures
                )
                cost.append(ratio_failures)

            else:
                cost = np.zeros(len(self.OPERATORS))

        self.failure_counter += 1

        return cost

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

        mode = Int32()
        mode.data = self.MODE
        self.__mode_publisher.publish(mode)

        cost = Float32MultiArray()
        cost.data = [
            self.__cost[1],
            self.__cost[2],
        ]
        self.__cost_publisher.publish(cost)

        reward_1 = Float32MultiArray()
        reward_1.data = [
            self.__reward_capability_1[1],
            self.__reward_capability_1[2],
        ]
        self.__reward_capability_1_publisher.publish(reward_1)

        reward_2 = Float32MultiArray()
        reward_2.data = [
            self.__reward_capability_2[1],
            self.__reward_capability_2[2],
        ]
        self.__reward_capability_2_publisher.publish(reward_2)

        reward_3 = Float32MultiArray()
        reward_3.data = [
            self.__reward_capability_3[1],
            self.__reward_capability_3[2],
        ]
        self.__reward_capability_3_publisher.publish(reward_3)

        total_reward = Float32MultiArray()
        total_reward.data = [
            self.__reward_operators[1],
            self.__reward_operators[2],
        ]
        self.__total_reward_publisher.publish(total_reward)

        expected_reward = Float32MultiArray()
        expected_reward.data = [
            self.__expected_reward[1],
            self.__expected_reward[2],
        ]
        self.__expected_reward_publisher.publish(expected_reward)

        assigned_to = Int32()
        assigned_to.data = self.__assigned_to
        self.__failure_assigned_publisher.publish(assigned_to)

        time_failures = Float32MultiArray()
        time_failures.data = [
            self.__operator_time_failures[1],
            self.__operator_time_failures[2],
        ]
        self.__operator_times_publisher.publish(time_failures)

        reward_speed = Float32MultiArray()
        reward_speed.data = [
            self.__reward_speed[1],
            self.__reward_speed[2],
        ]
        self.__reward_speed_publisher.publish(reward_speed)

        total_failures = Float32MultiArray()
        total_failures.data = [
            self.__failures_assigned[1],
            self.__failures_assigned[2],
        ]
        self.__total_assigned_publisher.publish(total_failures)

        task_allocation = Float32MultiArray()
        task_allocation.data = self.__task_allocation
        self.__failure_allocation_publisher.publish(task_allocation)

    def main_loop(self):

        self.__adjusted_threshold = self.THRESHOLD / (
            1 + self.__task_requirements[2]
        )

        self.__publish_data()

        if self.__is_failure_resolved:

            if self.__failures['duration'][-1] < self.THRESHOLD:
                speed = 1 - (self.__failures['duration'][-1] / self.THRESHOLD)
                self.__is_success = True
            else:
                speed = 0
                self.__is_success = False

            self.__operator_speed[self.__assigned_to].append(speed)

            self.__update_target_service()

            self.__assigned_to = 0
            self.__is_failure_resolved = False

    def __calculate_expected_reward(self):

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

            self.__reward_operators[operator] = self.calculate_reward(
                norms,
                self.__task_requirements,
                operator,
            )

        if torch.cuda.is_available():
            self.__reward_operators = {
                key: value.cuda()
                for key, value in self.__reward_operators.items()
            }

        cost = self.calculate_cost()
        self.__cost[1] = cost[0]
        self.__cost[2] = cost[1]

        for operator in self.OPERATORS:

            speed = sum(self.__operator_speed[operator]
                       ) / (len(self.__operator_speed[operator]) + 0.0000001)

            self.__reward_speed[
                operator] = (self.__task_requirements[2] *
                             (1 + speed)) / (1 + self.__task_requirements[2])

            self.__expected_reward[
                operator] = self.__reward_operators[operator] * (
                    self.__reward_speed[operator] - self.__cost[operator]
                )


if __name__ == "__main__":

    rospy.init_node(
        'failure_allocation',
        log_level=rospy.INFO,
    )

    operators = 2
    bins = 25
    failures = 18
    max_threshold = 80

    learning_rate = 0.001
    decay = 0.001

    mode = rospy.get_param(
        param_name=f'{rospy.get_name()}/mode',
        default='',
    )

    participant = rospy.get_param(
        param_name=f'{rospy.get_name()}/participant',
        default='',
    )

    folder_path = f'/home/fetch/data/failure-allocation/p{participant}/m0/t0'
    csv_files = glob(os.path.join(folder_path, '*.csv'))

    # Sort files by last modified time in descending order (most recent first)
    csv_files = sorted(csv_files, key=os.path.getmtime, reverse=True)

    if csv_files:
        latest_csv_file = csv_files[0]

        # Load the data from the latest CSV file
        data = pd.read_csv(latest_csv_file)

        local_bounds = list(
            literal_eval(data['local_no_sigmoid_bounds'].iloc[-1])
        )
        remote_bounds = list(
            literal_eval(data['remote_no_sigmoid_bounds'].iloc[-1])
        )

        remote_capability_1 = [remote_bounds[0], remote_bounds[1]]
        remote_capability_2 = [remote_bounds[2], remote_bounds[3]]
        remote_capability_3 = [remote_bounds[4], remote_bounds[5]]

        local_capability_1 = [local_bounds[0], local_bounds[1]]
        local_capability_2 = [local_bounds[2], local_bounds[3]]
        local_capability_3 = [local_bounds[4], local_bounds[5]]

        print("Remote bounds:")
        print(remote_capability_1)
        print(remote_capability_2)
        print(remote_capability_3)

        print("")
        print("Local bounds:")
        print(local_capability_1)
        print(local_capability_2)
        print(local_capability_3)

    performance_model_allocation = AllocationFramework(
        num_operators=operators,
        num_bins=bins,
        num_failures=failures,
        threshold=max_threshold,
        lr=learning_rate,
        weight_decay=decay,
        mode=mode,
        remote_1=remote_capability_1,
        remote_2=remote_capability_2,
        remote_3=remote_capability_3,
        local_1=local_capability_1,
        local_2=local_capability_2,
        local_3=local_capability_3,
    )

    while not rospy.is_shutdown():

        performance_model_allocation.main_loop()
        performance_model_allocation.RATE.sleep()
