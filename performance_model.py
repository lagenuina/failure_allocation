import torch
from torch import nn, sigmoid
from torch.nn import Parameter, ParameterDict
from torch.optim.lr_scheduler import CosineAnnealingLR

import numpy as np
import random
import time
import pandas as pd
import csv
import os
import math

# Set data type based on CUDA availability
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.DoubleTensor

class PerformanceModel(nn.Module):
    def __init__(self, NUM_OPERATORS, lr, weight_decay):
        super(PerformanceModel, self).__init__()
        
        self.NUM_OPERATORS = NUM_OPERATORS
        self.__local_operators = nn.ModuleDict()
        self.__remote_operators = nn.ModuleDict()

        self.__t_count = 0
        self.LR = lr
        self.DECAY = weight_decay

        # Changed initialization of local and remote operators
        for i in range(self.NUM_OPERATORS):
            operator_number = i

            if i % 2 == 0:
                operator_params = {
                    'lower_bound_1': Parameter(dtype(-10.0 * np.ones(1)), requires_grad=True),
                    'upper_bound_1': Parameter(dtype(10.0 * np.ones(1)), requires_grad=True),
                    'lower_bound_2': Parameter(dtype(-10.0 * np.ones(1)), requires_grad=True),
                    'upper_bound_2': Parameter(dtype(10.0 * np.ones(1)), requires_grad=True),
                    'lower_bound_3': Parameter(dtype(-10.0 * np.ones(1)), requires_grad=True),
                    'upper_bound_3': Parameter(dtype(10.0 * np.ones(1)), requires_grad=True)
                }
            else:
                operator_params = {
                    'lower_bound_1': Parameter(dtype(-10.0 * np.ones(1)), requires_grad=True),
                    'upper_bound_1': Parameter(dtype(10.0 * np.ones(1)), requires_grad=True),
                    'lower_bound_2': Parameter(dtype(-10.0 * np.ones(1)), requires_grad=True),
                    'upper_bound_2': Parameter(dtype(10.0 * np.ones(1)), requires_grad=True),
                    'lower_bound_3': Parameter(dtype(-10.0 * np.ones(1)), requires_grad=True),
                    'upper_bound_3': Parameter(dtype(10.0 * np.ones(1)), requires_grad=True)
                }
            if operator_number % 2 == 0:
                self.__local_operators[f'{operator_number}'] = ParameterDict(operator_params)
            else:
                self.__remote_operators[f'{operator_number}'] = ParameterDict(operator_params)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.LR, weight_decay=self.DECAY)

    def get_parameters(self, operator_number):

        if operator_number % 2 == 0:
            params = self.__local_operators[f'{operator_number}']
        else:
            params = self.__remote_operators[f'{operator_number}']

        if params['lower_bound_1'] > params['upper_bound_1']:
            params['lower_bound_1'], params['upper_bound_1'] = params['upper_bound_1'], params['lower_bound_1']

        if params['lower_bound_2'] > params['upper_bound_2']:
            params['lower_bound_2'], params['upper_bound_2'] = params['upper_bound_2'], params['lower_bound_2']

        if params['lower_bound_3'] > params['upper_bound_3']:
            params['lower_bound_3'], params['upper_bound_3'] = params['upper_bound_3'], params['lower_bound_3']

        norm_lower_bound_1 = sigmoid(params['lower_bound_1'])
        norm_upper_bound_1 = sigmoid(params['upper_bound_1'])

        norm_lower_bound_2 = sigmoid(params['lower_bound_2'])
        norm_upper_bound_2 = sigmoid(params['upper_bound_2'])

        norm_lower_bound_3 = sigmoid(params['lower_bound_3'])
        norm_upper_bound_3 = sigmoid(params['upper_bound_3'])

        return norm_lower_bound_1, norm_upper_bound_1, norm_lower_bound_2, norm_upper_bound_2, norm_lower_bound_3, norm_upper_bound_3
    
    def forward(self, bin_centers, observation_probability_index, operator_number):
        
        n_diffs = observation_probability_index.shape[0]
        performance = torch.zeros(n_diffs)
        norm_lower_bound_1, norm_upper_bound_1, norm_lower_bound_2, norm_upper_bound_2, norm_lower_bound_3, norm_upper_bound_3 = self.get_parameters(operator_number)

        for i in range(n_diffs):
            bin_center_idx_1, bin_center_idx_2, bin_center_idx_3 = observation_probability_index[i]

            performance[i] = (
                self.calculate_performance(1, operator_number, bin_centers[bin_center_idx_1]) *
                self.calculate_performance(2, operator_number, bin_centers[bin_center_idx_2]) *
                self.calculate_performance(3, operator_number, bin_centers[bin_center_idx_3]))
            
            # performance[i] = (
            #     self.calculate_performance(norm_lower_bound_1, norm_upper_bound_1, bin_centers[bin_center_idx_1]) *
            #     self.calculate_performance(norm_lower_bound_2, norm_upper_bound_2, bin_centers[bin_center_idx_2]) *
            #     self.calculate_performance(norm_lower_bound_3, norm_upper_bound_3, bin_centers[bin_center_idx_3]))

        return performance.cuda() if torch.cuda.is_available() else performance


    def calculate_performance(self, number, operator_number, task_requirement):
        
        if operator_number % 2 == 0:
            params = self.__local_operators[f'{operator_number}']
        else:
            params = self.__remote_operators[f'{operator_number}']

        upper_bound = params[f'upper_bound_{number}']
        lower_bound = params[f'lower_bound_{number}']
        
        x = (upper_bound - math.log(task_requirement / (1 - task_requirement))) / (upper_bound - lower_bound + 0.0001)
        
        return torch.sigmoid(x)
    
    def __check_convergence(self, operator):
        norm_l1, norm_u1, norm_l2, norm_u2, norm_l3, norm_u3 = self.get_parameters(operator)
        
        if abs(norm_u1 - norm_l1) < 0.005:
            
            if operator % 2 == 0:
                if self.__local_operators[f'{operator}']['upper_bound_1'].grad is not None and self.__local_operators[f'{operator}']['lower_bound_1'].grad is not None:
                    self.__local_operators[f'{operator}']['upper_bound_1'].requires_grad = False
                    self.__local_operators[f'{operator}']['lower_bound_1'].requires_grad = False
            else:
                if self.__remote_operators[f'{operator}']['upper_bound_1'].grad is not None and self.__remote_operators[f'{operator}']['lower_bound_1'].grad is not None:
                    self.__remote_operators[f'{operator}']['upper_bound_1'].requires_grad = False
                    self.__remote_operators[f'{operator}']['lower_bound_1'].requires_grad = False
        if abs(norm_u2 - norm_l2) < 0.005:
            if operator % 2 == 0:
                if self.__local_operators[f'{operator}']['upper_bound_2'].grad is not None and self.__local_operators[f'{operator}']['lower_bound_2'].grad is not None:
                    self.__local_operators[f'{operator}']['upper_bound_2'].requires_grad = False
                    self.__local_operators[f'{operator}']['lower_bound_2'].requires_grad = False
            else:
                if self.__remote_operators[f'{operator}']['upper_bound_2'].grad is not None and self.__remote_operators[f'{operator}']['lower_bound_2'].grad is not None:
                    self.__remote_operators[f'{operator}']['upper_bound_2'].requires_grad = False
                    self.__remote_operators[f'{operator}']['lower_bound_2'].requires_grad = False

        if abs(norm_u3 - norm_l3) < 0.005:
            if operator % 2 == 0:
                if self.__local_operators[f'{operator}']['upper_bound_3'].grad is not None and self.__local_operators[f'{operator}']['lower_bound_3'].grad is not None:
                    self.__local_operators[f'{operator}']['upper_bound_3'].requires_grad = False
                    self.__local_operators[f'{operator}']['lower_bound_3'].requires_grad = False
            else:
                if self.__remote_operators[f'{operator}']['upper_bound_3'].grad is not None and self.__remote_operators[f'{operator}']['lower_bound_3'].grad is not None:
                    self.__remote_operators[f'{operator}']['upper_bound_3'].requires_grad = False
                    self.__remote_operators[f'{operator}']['lower_bound_3'].requires_grad = False

    def update(self, bin_centers, obs_probs, obs_probs_idxs, operator_number):

        predicted_values = self.forward(bin_centers, obs_probs_idxs, operator_number)
        obs_probs_vect = torch.tensor([obs_probs[j, k, z] for j, k, z in obs_probs_idxs], dtype=torch.float64, requires_grad=True)
        
        obs_probs = dtype(obs_probs)

        loss = torch.mean(torch.pow((predicted_values - obs_probs_vect), 2.0))

        if loss.item() < 0.0005:
            self.__t_count += 1
            return  # Early return if the loss is already below the threshold

        # Iterative optimization
        t = 0

        while t < 2200:
            def closure():

                diff = self(bin_centers, obs_probs_idxs, operator_number) - obs_probs_vect
                loss = torch.mean(torch.pow(diff, 2.0))
                self.optimizer.zero_grad()
                loss.backward()

                return loss
            
            self.optimizer.step(closure)            

            predicted_values = self.forward(bin_centers, obs_probs_idxs, operator_number)
            loss = torch.mean(torch.pow((predicted_values - obs_probs_vect), 2.0))
            self.__loss_to_save.append(loss.item())
            
            self.__check_convergence(operator_number)

            if loss.item() < 0.0005:
                self.__t_count += 1
                break

            t += 1
            self.__t_count += 1

class TaskAllocation:
    def __init__(self, num_operators, threshold):

        self.NUM_OPERATORS = num_operators
        self.THRESHOLD = threshold

        self.__start_time_failure = 0
        self.failure_counter = 0
        self.task_allocation = []

        self.failures = {
            'start_time': [],
            'end_time': [],
            'duration': [], 
            'resolution_start_time': []
        }

        self.__primary_task_start_time = {i: [] for i in range(self.NUM_OPERATORS)}

        #FOR SIMULATING
        self.__start_time = time.time()
        for i in range(self.NUM_OPERATORS):
            self.__primary_task_start_time[i].append(time.time() - self.__start_time)

    def failure_started(self):

        self.__start_time_failure = time.time() - self.__start_time
        self.failures['start_time'].append(self.__start_time_failure)

    def failure_ended(self, failure_allocated):

        if failure_allocated % 2 == 0:
            duration = random.randint(1, 3)
            # duration = 3
        else:
            # duration = 6
            duration = random.randint(4, 6)

        # duration = (time.time() - self.__start_time) - self.failures['start_time'][-1]
        
        self.failures['duration'].append(duration)
        self.failures['end_time'].append(time.time() - self.__start_time)

    def calculate_reward(self, urgency, responsiveness):

        reward = (urgency * (1 + responsiveness)) / (1 + urgency)

        return reward
    
    def calculate_reward_capability(self, lower_bound, upper_bound, task_requirement):

        if task_requirement <= lower_bound:
            performance = torch.tensor([1.0])
        elif task_requirement > upper_bound:
            performance = torch.tensor([0.0])
        else:
            performance = (upper_bound - task_requirement) / (upper_bound - lower_bound + 0.0001)

        return performance.cuda() if torch.cuda.is_available() else performance
    
    def calculate_cost(self):

        cost = []
        total_time_failures = sum(self.failures['duration'])

        for i in range(self.NUM_OPERATORS):
            operator_time_failures = 0
            
            if self.failure_counter != 0:
                for j in range(self.failure_counter):
                    if self.task_allocation[j] == i:
                        operator_time_failures += self.failures['duration'][j]

                ratio_failures = (operator_time_failures/total_time_failures)
                cost.append(ratio_failures)
            else:
                cost = np.zeros(self.NUM_OPERATORS)

        self.failure_counter += 1

        return cost
    
class AllocationFramework:
    def __init__(self, num_operators, num_bins, num_failures, threshold, lr, weight_decay):
        
        np.seterr(divide='ignore', invalid='ignore')

        self.NUM_OPERATORS = num_operators
        self.NUM_BINS = num_bins
        self.NUM_FAILURES = num_failures
        self.THRESHOLD = threshold
        self.LR = lr
        self.DECAY = weight_decay

        self.__task_allocation = TaskAllocation(self.NUM_OPERATORS, self.THRESHOLD)
        self.__model = PerformanceModel(self.NUM_OPERATORS, self.LR, self.DECAY)
        self.__model = self.__model.cuda() if torch.cuda.is_available() else self.__model

        self.__bin_limits = dtype(np.concatenate([[0], np.linspace(1 / self.NUM_BINS, 1.0, self.NUM_BINS)]))
        self.bin_centers = dtype((self.__bin_limits[:-1] + self.__bin_limits[1:]) / 2.0)

        self.__workload = {}

        keys = []
        for i in range(self.NUM_OPERATORS):
            self.__workload[i] = 0
            keys.append(f'{i}_lower_bound_1') 
            keys.append(f'{i}_upper_bound_1') 
            keys.append(f'{i}_lower_bound_2') 
            keys.append(f'{i}_upper_bound_2')
            keys.append(f'{i}_lower_bound_3') 
            keys.append(f'{i}_upper_bound_3')
            keys.append(f'{i}_reward_urgency')
            keys.append(f'{i}_reward_capabilities')
            keys.append(f'{i}_cost')
            keys.append(f'{i}_total_reward')
        
        self.__results = {key: [] for key in keys}
        self.__total_observations_local = {}
        self.__total_observations_remote = {}

        self.__total_success_local = {}
        self.__total_success_remote = {}

        self.__success_local = {}
        self.__success_remote = {}
        self.observations_probabilities_local = {}
        self.observations_probabilities_remote = {}

        self.probabilities_index_local = []
        self.probabilities_index_remote = []

        self.__failures_assigned = {}
        

        for i in range(self.NUM_OPERATORS):  
            operator_number = i

            self.__failures_assigned[operator_number] = 0

            # Local operators have even number indexes
            if operator_number % 2 == 0:
                self.observations_probabilities_local[operator_number] = np.zeros((self.NUM_BINS, self.NUM_BINS,  self.NUM_BINS)) 
                self.__total_observations_local[operator_number] = np.zeros((self.NUM_BINS, self.NUM_BINS, self.NUM_BINS)) 
                self.__total_success_local[operator_number] = np.zeros((self.NUM_BINS, self.NUM_BINS)) 
                self.__success_local[operator_number] = np.zeros((self.NUM_BINS, self.NUM_BINS, self.NUM_BINS)) 

            else: 
                self.observations_probabilities_remote[operator_number] = np.zeros((self.NUM_BINS, self.NUM_BINS,  self.NUM_BINS)) 
                self.__total_observations_remote[operator_number] = np.zeros((self.NUM_BINS, self.NUM_BINS, self.NUM_BINS)) 
                self.__total_success_remote[operator_number] = np.zeros((self.NUM_BINS, self.NUM_BINS)) 
                self.__success_remote[operator_number] = np.zeros((self.NUM_BINS, self.NUM_BINS, self.NUM_BINS))


        load_file_name = "failure_requirements.csv"
        data_frame = pd.read_csv(load_file_name, header=None)
        self.__task_requirements = data_frame.iloc[:, :self.NUM_FAILURES].values

        self.__capabilities_local = {}
        self.__capabilities_remote = {}

        self.__performance_index = {}

    def __assign_failure(self, expected_reward):

        expected_rewards_tensor = torch.stack(expected_reward)
        
        # Find the maximum value in the tensor
        highest_reward = expected_rewards_tensor.max().item()
        
        # Find all indices with the maximum value
        operators_highest_reward = (expected_rewards_tensor == highest_reward).nonzero(as_tuple=True)[0].tolist()
        if len(operators_highest_reward) > 1:

            # Assign to the operator with least amount of failures
            failure_allocated_to = min(self.__failures_assigned, key=lambda k: self.__failures_assigned[k])

        else:
            failure_allocated_to = operators_highest_reward[0]

        self.__failures_assigned[failure_allocated_to] += 1

        self.__task_allocation.task_allocation.append(failure_allocated_to)

        self.__task_allocation.failure_ended(failure_allocated_to)
        print("\n")

        return failure_allocated_to

    def main_loop(self):

        for i in range(self.NUM_FAILURES):
            print("Failure", i)
            print(self.__failures_assigned)

            self.__task_allocation.failure_started() #SIMULATION PURPOSES
            
            reward_urgency = []

            for operator in range(self.NUM_OPERATORS):
                norm_l1, norm_u1, norm_l2, norm_u2, norm_l3, norm_u3, = self.__model.get_parameters(operator)
                
                if operator % 2 == 0:
                    self.__capabilities_local[operator] = {'lower_bound_1': norm_l1,
                                                    'upper_bound_1': norm_u1,
                                                    'lower_bound_2': norm_l2,
                                                    'upper_bound_2': norm_u2,
                                                    'lower_bound_3': norm_l3,
                                                    'upper_bound_3': norm_u3,}
                    
                else:
                    self.__capabilities_remote[operator] = {'lower_bound_1': norm_l1,
                                                'upper_bound_1': norm_u1,
                                                'lower_bound_2': norm_l2,
                                                'upper_bound_2': norm_u2,
                                                'lower_bound_3': norm_l3,
                                                'upper_bound_3': norm_u3,}
                    
                performance_capability_1 = self.__model.calculate_performance(1, operator, self.__task_requirements[0, i])
                performance_capability_2 = self.__model.calculate_performance(2, operator, self.__task_requirements[1, i])
                performance_capability_3 = self.__model.calculate_performance(3, operator, self.__task_requirements[2, i])
                # performance_capability_1 = self.__model.calculate_performance(norm_l1, norm_u1, self.__task_requirements[0, i])
                # performance_capability_2 = self.__model.calculate_performance(norm_l2, norm_u2, self.__task_requirements[1, i])
                # performance_capability_3 = self.__model.calculate_performance(norm_l3, norm_u3, self.__task_requirements[2, i])

                reward = self.__task_allocation.calculate_reward(self.__task_requirements[2, i], performance_capability_3)
                
                reward_urgency.append(reward)
                self.__performance_index[operator] = performance_capability_1 * performance_capability_2 * reward

            if torch.cuda.is_available():
                self.__performance_index = {key: value.cuda() for key, value in self.__performance_index.items()}

            cost =  self.__task_allocation.calculate_cost()

            expected_reward = []
            for operator in range(self.NUM_OPERATORS):
                expected_reward.append(self.__performance_index[operator] - cost[operator])

            assigned_to = self.__assign_failure(expected_reward)

            adjusted_threshold = self.THRESHOLD/(1 + self.__task_requirements[2, i])
            # adjusted_threshold = self.THRESHOLD

            for j in range(self.NUM_BINS):
                for k in range(self.NUM_BINS):
                    for z in range(self.NUM_BINS):

                        if self.__bin_limits[j] < self.__task_requirements[0, i] <= self.__bin_limits[j + 1] and self.__bin_limits[k] < self.__task_requirements[1, i] <= self.__bin_limits[k + 1] and self.__bin_limits[z] < self.__task_requirements[2, i] <= self.__bin_limits[z + 1]:

                            
                            if assigned_to % 2 == 0:

                                if self.__task_allocation.failures['duration'][-1] < adjusted_threshold:
                                    self.__success_local[assigned_to][j, k, z] += 1

                                self.__total_observations_local[assigned_to][j, k, z] += 1

                            else:

                                if self.__task_allocation.failures['duration'][-1] < adjusted_threshold:
                                    self.__success_remote[assigned_to][j, k, z] += 1

                                self.__total_observations_remote[assigned_to][j, k, z] += 1


            if assigned_to % 2 == 0:
                self.observations_probabilities_local[assigned_to] = np.divide(self.__success_local[assigned_to], self.__total_observations_local[assigned_to], where=self.__total_observations_local[assigned_to] != 0)

                self.probabilities_index_local = np.array([[j, k, z] for j in range(self.NUM_BINS) for k in range(self.NUM_BINS) for z in range(self.NUM_BINS) if self.__total_observations_local[assigned_to][j, k, z] > 0])
                self.__model.update(self.bin_centers, self.observations_probabilities_local[assigned_to], self.probabilities_index_local, assigned_to)
            else:
                self.observations_probabilities_remote[assigned_to] = np.divide(self.__success_remote[assigned_to], self.__total_observations_remote[assigned_to], where=self.__total_observations_remote[assigned_to] != 0)

                self.probabilities_index_remote = np.array([[j, k, z] for j in range(self.NUM_BINS) for k in range(self.NUM_BINS) for z in range(self.NUM_BINS) if self.__total_observations_remote[assigned_to][j, k, z] > 0])
                self.__model.update(self.bin_centers, self.observations_probabilities_remote[assigned_to], self.probabilities_index_remote, assigned_to)

            self.save_results(reward_urgency, self.__performance_index, cost, expected_reward)

        self.save_other_results()
        self.write_results()
    
    def save_results(self, reward_urgency, reward_capabilities, cost, total_reward):

        for operator in range(self.NUM_OPERATORS):
            norm_l1, norm_u1, norm_l2, norm_u2, norm_l3, norm_u3 = self.__model.get_parameters(operator)

            self.__results[f'{operator}_lower_bound_1'].append(norm_l1.item())
            self.__results[f'{operator}_upper_bound_1'].append(norm_u1.item())
            self.__results[f'{operator}_lower_bound_2'].append(norm_l2.item())
            self.__results[f'{operator}_upper_bound_2'].append(norm_u2.item())
            self.__results[f'{operator}_lower_bound_3'].append(norm_l3.item())
            self.__results[f'{operator}_upper_bound_3'].append(norm_u3.item())
            self.__results[f'{operator}_reward_urgency'].append(reward_urgency[operator].item())
            self.__results[f'{operator}_reward_capabilities'].append((reward_capabilities[operator]).item())
            self.__results[f'{operator}_cost'].append(cost[operator])
            self.__results[f'{operator}_total_reward'].append(total_reward[operator].item())

    def save_other_results(self):

        self.__results['failures_duration'] = self.__task_allocation.failures['duration']
        self.__results['assigned'] = self.__task_allocation.task_allocation

    def write_results(self):

        # Determine the filename
        # base_filename = f'{self.NUM_OPERATORS}_operators_{self.NUM_FAILURES}_failures.csv'
        base_filename = f'lr_{self.LR}_w_{self.DECAY}_operators_{self.NUM_OPERATORS}_failures_{self.NUM_FAILURES}.csv'
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
                row = [self.__results[key][i] if i < len(self.__results[key]) else None for key in headers]
                writer.writerow(row)
                    

if __name__ == "__main__":

    num_operators = 2
    num_bins = 25
    num_failures = 40
    threshold = 6
    # lr = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    # weight_decay = [0.0, 0.000001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01]
    
    lr = [0.001]
    weight_decay = [0.001]

    # for i in range(10):
    for learning_rate in lr:
        for decay in weight_decay:

            performance_model_allocation = AllocationFramework(num_operators=2, num_bins=25, num_failures=20, threshold=10, lr=learning_rate, weight_decay=decay)

            performance_model_allocation.main_loop()







