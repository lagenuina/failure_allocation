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

        norms = [(norm_lower_bound_1, norm_upper_bound_1), (norm_lower_bound_2, norm_upper_bound_2), (norm_lower_bound_3, norm_upper_bound_3)]
        return norms
    
    def forward(self, bin_centers, observation_probability_index, operator_number):
        
        n_diffs = observation_probability_index.shape[0]
        performance = torch.zeros(n_diffs)

        for i in range(n_diffs):
            bin_center_idx_1, bin_center_idx_2, bin_center_idx_3 = observation_probability_index[i]

            performance[i] = (
                self.calculate_performance(1, operator_number, bin_centers[bin_center_idx_1]) *
                self.calculate_performance(2, operator_number, bin_centers[bin_center_idx_2]) *
                self.calculate_performance(3, operator_number, bin_centers[bin_center_idx_3]))
          
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
        norms = self.get_parameters(operator)
        
        bounds = ['upper_bound_1', 'lower_bound_1', 'upper_bound_2', 'lower_bound_2', 'upper_bound_3', 'lower_bound_3']
        
        for i, (norm_l, norm_u) in enumerate(norms):
            if abs(norm_u - norm_l) < 0.005:
                operators = self.__local_operators if operator % 2 == 0 else self.__remote_operators
                upper_bound = operators[f'{operator}'][bounds[2 * i]]
                lower_bound = operators[f'{operator}'][bounds[2 * i + 1]]
                
                if upper_bound.grad is not None and lower_bound.grad is not None:
                    operators[f'{operator}'][bounds[2 * i]].requires_grad = False
                    operators[f'{operator}'][bounds[2 * i + 1]].requires_grad = False

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
            
            self.__check_convergence(operator_number)

            if loss.item() < 0.0005:
                self.__t_count += 1
                break

            t += 1
            self.__t_count += 1

class AllocationFramework:
    def __init__(self, num_operators, num_bins, num_failures, threshold, lr, weight_decay):
        
        np.seterr(divide='ignore', invalid='ignore')

        self.NUM_OPERATORS = num_operators
        self.NUM_BINS = num_bins
        self.NUM_FAILURES = num_failures
        self.THRESHOLD = threshold
        self.LR = lr
        self.DECAY = weight_decay

        self.__model = PerformanceModel(self.NUM_OPERATORS, self.LR, self.DECAY)
        self.__model = self.__model.cuda() if torch.cuda.is_available() else self.__model
        self.__data_recorder = DataRecorder(self.NUM_OPERATORS, self.NUM_FAILURES, self.THRESHOLD, self.LR, self.DECAY)


        self.__bin_limits = dtype(np.concatenate([[0], np.linspace(1 / self.NUM_BINS, 1.0, self.NUM_BINS)]))
        self.bin_centers = dtype((self.__bin_limits[:-1] + self.__bin_limits[1:]) / 2.0)

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
        
        self.__start_time_failure = 0
        self.failure_counter = 0
        self.task_allocation = []
        self.__adjusted_threshold = 0

        self.__failures = {
            'start_time': [],
            'end_time': [],
            'duration': [], 
            'resolution_start_time': []
        }

        self.__primary_task_start_time = {i: [] for i in range(self.NUM_OPERATORS)}
        self.__reward_capability_1 = {i: 0 for i in range(self.NUM_OPERATORS)}
        self.__reward_capability_2 = {i: 0 for i in range(self.NUM_OPERATORS)}
        self.__reward_capability_3 = {i: 0 for i in range(self.NUM_OPERATORS)}
        self.__cost = {i: 0 for i in range(self.NUM_OPERATORS)}
        self.__operator_time_failures = {i: 0 for i in range(self.NUM_OPERATORS)}

        #FOR SIMULATING
        self.__start_time = time.time()
        for i in range(self.NUM_OPERATORS):
            self.__primary_task_start_time[i].append(time.time() - self.__start_time)

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

        self.__capabilities = {}

        self.__reward_operators = {}
        self.__expected_reward = {}


    def __assign_failure(self, expected_reward):

        reward_tensors = list(expected_reward.values())

        # Concatenate tensors into a single tensor
        concatenated_tensor = torch.cat(reward_tensors)
        
        # Find the maximum value in the tensor
        highest_reward = concatenated_tensor.max().item()
        
        # Find all indices with the maximum value
        operators_highest_reward = (concatenated_tensor == highest_reward).nonzero(as_tuple=True)[0].tolist()
        if len(operators_highest_reward) > 1:

            # Assign to the operator with least amount of failures
            failure_allocated_to = min(self.__failures_assigned, key=lambda k: self.__failures_assigned[k])

        else:
            failure_allocated_to = operators_highest_reward[0]

        self.__failures_assigned[failure_allocated_to] += 1

        self.task_allocation.append(failure_allocated_to)

        self.__failure_ended(failure_allocated_to)
        print("\n")

        return failure_allocated_to

    def __failure_started(self):

        self.__start_time_failure = time.time() - self.__start_time
        self.__failures['start_time'].append(self.__start_time_failure)

    def __failure_ended(self, failure_allocated):

        if failure_allocated % 2 == 0:
            duration = random.randint(4, 6)
            # duration = 3
        else:
            # duration = 6
            duration = random.randint(6, 8)

        # duration = (time.time() - self.__start_time) - self.__failures['start_time'][-1]
        
        self.__failures['duration'].append(duration)
        self.__failures['end_time'].append(time.time() - self.__start_time)

    def calculate_reward(self, norms, task_requirement, operator):
        
        performance = []

        for i, (lower_bound, upper_bound) in enumerate(norms):

            # if i != 2:
            if task_requirement[i] <= lower_bound:
                performance.append(torch.tensor([1.0]))
            elif task_requirement[i] > upper_bound:
                performance.append(torch.tensor([0.0]))
            else:
                performance.append((upper_bound - task_requirement[i]) / (upper_bound - lower_bound + 0.0001))
            # else:
            #     performance.append((upper_bound + lower_bound) / 2)
                
        urgency = task_requirement[2]
        responsiveness = performance[2]

        self.__reward_capability_1[operator] = performance[0]
        self.__reward_capability_2[operator] = performance[1]
        # self.__reward_capability_3[operator] = (urgency * (1 + responsiveness)) / (1 + urgency)
        self.__reward_capability_3[operator] = performance[2]

        return 0.3 * self.__reward_capability_1[operator] + 0.3 * self.__reward_capability_2[operator] + 0.4 * self.__reward_capability_3[operator]
    
    def calculate_cost(self):

        cost = []
        total_time_failures = sum(self.__failures['duration'])

        for i in range(self.NUM_OPERATORS):
            self.__operator_time_failures[i] = 0
            
            if self.failure_counter != 0:
                for j in range(self.failure_counter):
                    if self.task_allocation[j] == i:
                        self.__operator_time_failures[i] += self.__failures['duration'][j]

                ratio_failures = (self.__operator_time_failures[i]/total_time_failures)
                cost.append(ratio_failures)
            else:
                cost = np.zeros(self.NUM_OPERATORS)

        self.failure_counter += 1

        return [i * 0.7 for i in cost]
    
    def main_loop(self):

        for i in range(self.NUM_FAILURES):
            print("Failure", i)
            print(self.__failures_assigned)

            self.__failure_started() #SIMULATION PURPOSES
            
            for operator in range(self.NUM_OPERATORS):
                norms = self.__model.get_parameters(operator)
                
                self.__capabilities[operator] = {'lower_bound_1': norms[0][0],
                                                'upper_bound_1': norms[0][1],
                                                'lower_bound_2': norms[1][0],
                                                'upper_bound_2': norms[1][1],
                                                'lower_bound_3': norms[2][0],
                                                'upper_bound_3': norms[2][1]}
                    
                self.__reward_operators[operator] = self.calculate_reward(norms, self.__task_requirements[:,i], operator)
                
            if torch.cuda.is_available():
                self.__reward_operators = {key: value.cuda() for key, value in self.__reward_operators.items()}

            self.__cost =  self.calculate_cost()

            for operator in range(self.NUM_OPERATORS):
                self.__expected_reward[operator] = self.__reward_operators[operator] - self.__cost[operator]

            assigned_to = self.__assign_failure(self.__expected_reward)

            self.__adjusted_threshold = self.THRESHOLD/(1 + self.__task_requirements[2, i])

            for j in range(self.NUM_BINS):
                for k in range(self.NUM_BINS):
                    for z in range(self.NUM_BINS):

                        if self.__bin_limits[j] < self.__task_requirements[0, i] <= self.__bin_limits[j + 1] and self.__bin_limits[k] < self.__task_requirements[1, i] <= self.__bin_limits[k + 1] and self.__bin_limits[z] < self.__task_requirements[2, i] <= self.__bin_limits[z + 1]:

                            
                            if assigned_to % 2 == 0:

                                if self.__failures['duration'][-1] < self.__adjusted_threshold:
                                    self.__success_local[assigned_to][j, k, z] += 1

                                self.__total_observations_local[assigned_to][j, k, z] += 1

                            else:

                                if self.__failures['duration'][-1] < self.__adjusted_threshold:
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

            self.__data_recorder.save_results(i, 
                                            assigned_to,
                                            self.__failures['start_time'][-1],
                                            self.__failures['duration'][-1],
                                            self.__adjusted_threshold,
                                            self.__task_requirements[:,i],
                                            self.__capabilities,
                                            self.__reward_capability_1, 
                                            self.__reward_capability_2, 
                                            self.__reward_capability_3,
                                            self.__cost,
                                            self.__expected_reward,
                                            self.__failures_assigned,
                                            self.__operator_time_failures
                                            )
            
        self.__data_recorder.write_results()
        
class DataRecorder:
    def __init__(self, num_operators, num_failures, threshold, lr, weight_decay):
    
        self.NUM_OPERATORS = num_operators
        self.NUM_FAILURES = num_failures
        self.THRESHOLD = threshold
        self.LR = lr
        self.DECAY = weight_decay

        keys = ['failure_id', 
                'assigned_to', 
                'start_time', 
                'duration',
                'success',
                'requirement_1',
                'requirement_2',
                'requirement_3',
                'num_operators', 
                'num_failures',
                'threshold',
                'learning_rate',
                'weight_decay']
        
        for i in range(self.NUM_OPERATORS):
            keys.append(f'{i}_lower_bound_1') 
            keys.append(f'{i}_upper_bound_1') 
            keys.append(f'{i}_lower_bound_2') 
            keys.append(f'{i}_upper_bound_2')
            keys.append(f'{i}_lower_bound_3') 
            keys.append(f'{i}_upper_bound_3')
            keys.append(f'{i}_cost')
            keys.append(f'{i}_reward_capability_1')
            keys.append(f'{i}_reward_capability_2')
            keys.append(f'{i}_reward_capability_3')
            keys.append(f'{i}_total_reward')
            keys.append(f'{i}_assigned_failures')
            keys.append(f'{i}_time_spent_failures')
        
        self.__results = {key: [] for key in keys}

    def save_results(self, failure_id, assigned_to, start_time, duration, threshold, task_requirements, norms, reward_capability_1, reward_capability_2, reward_capability_3, cost, reward_operators, failures_assigned, operator_time_failures):

        self.__results['failure_id'].append(failure_id)
        self.__results['assigned_to'].append(assigned_to)
        self.__results['start_time'].append(start_time)
        self.__results['duration'].append(duration)
        self.__results['requirement_1'].append(task_requirements[0])
        self.__results['requirement_2'].append(task_requirements[1])
        self.__results['requirement_3'].append(task_requirements[2])
        self.__results['num_operators'].append(self.NUM_OPERATORS)
        self.__results['num_failures'].append(self.NUM_FAILURES)
        self.__results['threshold'].append(threshold)
        self.__results['learning_rate'].append(self.LR)
        self.__results['weight_decay'].append(self.DECAY)

        if duration < threshold:
            self.__results['success'].append(True)
        else:
            self.__results['success'].append(False)

        for operator in range(self.NUM_OPERATORS):

            self.__results[f'{operator}_lower_bound_1'].append(norms[operator]['lower_bound_1'].item())
            self.__results[f'{operator}_upper_bound_1'].append(norms[operator]['upper_bound_1'].item())
            self.__results[f'{operator}_lower_bound_2'].append(norms[operator]['lower_bound_2'].item())
            self.__results[f'{operator}_upper_bound_2'].append(norms[operator]['upper_bound_2'].item())
            self.__results[f'{operator}_lower_bound_3'].append(norms[operator]['lower_bound_3'].item())
            self.__results[f'{operator}_upper_bound_3'].append(norms[operator]['upper_bound_3'].item())
            self.__results[f'{operator}_cost'].append(cost[operator])
            self.__results[f'{operator}_reward_capability_1'].append((reward_capability_1[operator]).item())
            self.__results[f'{operator}_reward_capability_2'].append((reward_capability_2[operator]).item())
            self.__results[f'{operator}_reward_capability_3'].append((reward_capability_3[operator]).item())
            self.__results[f'{operator}_total_reward'].append(reward_operators[operator].item())
            self.__results[f'{operator}_assigned_failures'].append(failures_assigned[operator])
            self.__results[f'{operator}_time_spent_failures'].append(operator_time_failures[operator])

    def write_results(self):

        # Determine the filename
        base_filename = f'{self.NUM_OPERATORS}_operators_{self.NUM_FAILURES}_failures.csv'
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

    operators_array = [2, 2, 2, 2]
    bins = 25
    failures = 50
    max_threshold = 10

    learning_rate = 0.001
    decay = 0.001

    for i in range(3):
        for operators in operators_array:
            performance_model_allocation = AllocationFramework(num_operators=operators, num_bins=bins, num_failures=failures, threshold=max_threshold, lr=learning_rate, weight_decay=decay)
            performance_model_allocation.main_loop()







