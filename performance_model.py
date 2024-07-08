import torch
from torch import nn, sigmoid
from torch.nn import Parameter, ParameterDict
import numpy as np
import scipy.io as sio
import random
import time

# Set data type based on CUDA availability
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.DoubleTensor

class PerformanceModel(nn.Module):
    def __init__(self, NUM_OPERATORS):
        super(PerformanceModel, self).__init__()
        
        self.NUM_OPERATORS = NUM_OPERATORS
        self.__local_operators = nn.ModuleDict()
        self.__remote_operators = nn.ModuleDict()

        self.__loss_to_save = []
        self.__t_count = 0

        for i in range(NUM_OPERATORS):
            operator_number = i

            # Local operators have even number indexes
            if operator_number % 2 == 0:
                
                #TO DO: automate based on the number of capabilities
                self.__local_operators[f'{operator_number}'] = ParameterDict({'lower_bound_1': Parameter(dtype(-10.0 * np.ones(1)), requires_grad=True),
                                                           'upper_bound_1': Parameter(dtype(10.0 * np.ones(1)), requires_grad=True),
                                                           'lower_bound_2': Parameter(dtype(-10.0 * np.ones(1)), requires_grad=True),
                                                           'upper_bound_2': Parameter(dtype(10.0 * np.ones(1)), requires_grad=True),
                                                           'responsivness': Parameter(dtype(10.0 * np.ones(1)), requires_grad=True),
                })
            
            else: 
                self.__remote_operators[f'{operator_number}'] = ParameterDict({'lower_bound_1': Parameter(dtype(-10.0 * np.ones(1)), requires_grad=True),
                                                           'upper_bound_1': Parameter(dtype(10.0 * np.ones(1)), requires_grad=True),
                                                           'lower_bound_2': Parameter(dtype(-10.0 * np.ones(1)), requires_grad=True),
                                                           'upper_bound_2': Parameter(dtype(10.0 * np.ones(1)), requires_grad=True),
                                                           'responsivness': Parameter(dtype(7.0 * np.ones(1)), requires_grad=True),
                })

        self.optimizer = torch.optim.Adam(self.parameters(), lr=0.001, weight_decay=0.0001)


    def get_parameters(self, operator_number):

        if operator_number % 2 == 0:
            params = self.__local_operators[f'{operator_number}']
        else:
            params = self.__remote_operators[f'{operator_number}']

        if params['lower_bound_1'] > params['upper_bound_1']:
            params['lower_bound_1'], params['upper_bound_1'] = params['upper_bound_1'], params['lower_bound_1']

        if params['lower_bound_2'] > params['upper_bound_2']:
            params['lower_bound_2'], params['upper_bound_2'] = params['upper_bound_2'], params['lower_bound_2']

        norm_lower_bound_1 = sigmoid(params['lower_bound_1'])
        norm_upper_bound_1 = sigmoid(params['upper_bound_1'])

        norm_lower_bound_2 = sigmoid(params['lower_bound_2'])
        norm_upper_bound_2 = sigmoid(params['upper_bound_2'])

        return norm_lower_bound_1, norm_upper_bound_1, norm_lower_bound_2, norm_upper_bound_2, sigmoid(params['responsivness'])
    
    def forward(self, bin_centers, observation_probability_index, operator_number):
        
        norm_lower_bound_1, norm_upper_bound_1, norm_lower_bound_2, norm_upper_bound_2, responsivness = self.get_parameters(operator_number)

        n_diffs = observation_probability_index.shape[0]
        performance = torch.zeros(n_diffs)

        for i in range(n_diffs):
            bin_center_idx_1, bin_center_idx_2 = observation_probability_index[i]
            performance[i] = (
                self.calculate_performance(norm_lower_bound_1, norm_upper_bound_1, bin_centers[bin_center_idx_1]) *
                self.calculate_performance(norm_lower_bound_2, norm_upper_bound_2, bin_centers[bin_center_idx_2])
            )

        return performance.cuda() if torch.cuda.is_available() else performance

    def calculate_performance(self, lower_bound, upper_bound, task_requirement):
        
        if task_requirement <= lower_bound:
            performance = torch.tensor([1.0])
        elif task_requirement > upper_bound:
            performance = torch.tensor([0.0])
        else:
            performance = (upper_bound - task_requirement) / (upper_bound - lower_bound + 0.0001)

        return performance.cuda() if torch.cuda.is_available() else performance
    
    def calculate_performance_to_urgency(self, task_urgency, responsivness):
        
        performance_urgency = (task_urgency * (1 + responsivness)) / (1 + task_urgency)
        
        return performance_urgency.cuda() if torch.cuda.is_available() else performance_urgency

    def update(self, bin_centers, obs_probs, obs_probs_idxs, operator_number):

        predicted_values = self.forward(bin_centers, obs_probs_idxs, operator_number)
        obs_probs_vect = torch.tensor([obs_probs[j, k] for j, k in obs_probs_idxs], dtype=torch.float64, requires_grad=True)
        
        obs_probs = dtype(obs_probs)

        loss = torch.mean(torch.pow((predicted_values - obs_probs_vect), 2.0))

        if loss.item() < 0.0005:
            self.__t_count += 1
            return  # Early return if the loss is already below the threshold

        # Iterative optimization
        t = 0
        loss_200_iters_ago, current_loss = 0, 0

        while t < 2200:
            def closure():

                diff = self(bin_centers, obs_probs_idxs, operator_number) - obs_probs_vect
                loss = torch.mean(torch.pow(diff, 2.0))
                self.optimizer.zero_grad()
                loss.backward()

                return loss
            
            self.optimizer.step(closure)

            norm_lower_bound_1, norm_upper_bound_1, norm_lower_bound_2, norm_upper_bound_2, responsivness = self.get_parameters(operator_number)

            predicted_values = self.forward(bin_centers, obs_probs_idxs, operator_number)
            loss = torch.mean(torch.pow((predicted_values - obs_probs_vect), 2.0))

            self.__loss_to_save.append(loss.item())

        
            if loss.item() < 0.0005:
                self.__t_count += 1
                break

            if t % 200 == 0:
                if t == 0:
                    loss_200_iters_ago = -1000
                    current_loss = self.__loss_to_save[-1]
                else:
                    loss_200_iters_ago = current_loss
                    current_loss = self.__loss_to_save[-1]

            if abs(current_loss - loss_200_iters_ago) < 1e-8:
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
            'duration': [],  # Duration of each failure
            'resolution_start_time': []
        }

        self.__primary_task_start_time = {i: [] for i in range(self.NUM_OPERATORS)}

        #FOR SIMULATING
        self.__start_time = time.time()
        self.__primary_task_start_time[0].append(time.time() - self.__start_time)
        self.__primary_task_start_time[1].append(time.time() - self.__start_time)

    def failure_started(self):

        self.__start_time_failure = time.time() - self.__start_time
        self.failures['start_time'].append(self.__start_time_failure)

    def failure_ended(self):

        duration = (time.time() - self.__start_time) - self.failures['start_time'][-1]
        
        self.failures['duration'].append(duration)
        self.failures['end_time'].append(time.time() - self.__start_time)

    def normalize(self, arr, t_min, t_max):
        norm_arr = []
        diff = t_max - t_min
        
        diff_arr = max(arr) - min(arr)    
        for i in arr:
            temp = (((i - min(arr))*diff)/(diff_arr+0.0000001)) + t_min
            norm_arr.append(temp)
        return norm_arr

    def calculate_reward(self):

        reward = []
        deviations = []

        average_workload = sum(self.failures['duration'])/(len(self.failures['duration']) + 0.00000001)

        for failure_number in range(self.failure_counter):
            deviations.append(self.failures['duration'][failure_number] - average_workload)
        
        if len(deviations) != 0:
            normalized_deviations = self.normalize(deviations, 0, 1)

        for i in range(self.NUM_OPERATORS):

            average_speed = 0
            deviation_from_mean = 0
            r = []

            if self.failure_counter != 0:

                for j in range(self.failure_counter):
                    if self.task_allocation[j] == i:
                        
                        deviation_from_mean += normalized_deviations[j]
                        
                        if self.failures['duration'][j] > self.THRESHOLD:
                            r.append(0)
                        else:
                            r.append(1 - (self.failures['duration'][j]/self.THRESHOLD))

                average_speed = sum(r)/(len(r) + 0.00001)
            
            if self.failure_counter != 0:
                print("Normalized deviations:", normalized_deviations)
            print("Average speed:", average_speed)
            print("Deviation from mean", deviation_from_mean)
            print("\n")
            reward.append(average_speed - deviation_from_mean)
        
        self.failure_counter += 1

        return reward
    
class AllocationFramework:
    def __init__(self, num_operators, num_bins, num_failures, threshold):
        
        np.seterr(divide='ignore', invalid='ignore')

        self.NUM_OPERATORS = num_operators
        self.NUM_BINS = num_bins
        self.NUM_FAILURES = num_failures
        self.THRESHOLD = threshold

        self.__task_allocation = TaskAllocation(self.NUM_OPERATORS, self.THRESHOLD)
        self.__model = PerformanceModel(self.NUM_OPERATORS)
        self.__model = self.__model.cuda() if torch.cuda.is_available() else self.__model

        self.__bin_limits = dtype(np.concatenate([[0], np.linspace(1 / self.NUM_BINS, 1.0, self.NUM_BINS)]))
        self.bin_centers = dtype((self.__bin_limits[:-1] + self.__bin_limits[1:]) / 2.0)

        self.__total_observations_local = {}
        self.__total_observations_remote = {}

        self.__total_success_local = {}
        self.__total_success_remote = {}

        self.observations_probabilities_local = {}
        self.observations_probabilities_remote = {}

        self.probabilities_index_local = []
        self.probabilities_index_remote = []

        for i in range(self.NUM_OPERATORS):  
            operator_number = i

            # Local operators have even number indexes
            if operator_number % 2 == 0:

                self.__total_observations_local[operator_number] = np.zeros((self.NUM_BINS, self.NUM_BINS)) 
                self.__total_success_local[operator_number] = np.zeros((self.NUM_BINS, self.NUM_BINS)) 

            else: 
                self.__total_observations_remote[operator_number] = np.zeros((self.NUM_BINS, self.NUM_BINS)) 
                self.__total_success_remote[operator_number] = np.zeros((self.NUM_BINS, self.NUM_BINS)) 


        load_file_name = "./artificial-trust-task-allocation-main/code/results/tasks/TA_normaldist_tasks_0.mat"
        fixed_tasks_mat = sio.loadmat(load_file_name)
        self.__task_requirements = fixed_tasks_mat["p"][:, :self.NUM_FAILURES]

        # Add row for urgency
        self.__task_requirements = np.vstack((fixed_tasks_mat["p"][:, :self.NUM_FAILURES], np.round(np.random.uniform(0.1, 1, self.NUM_FAILURES), 2)))

        self.__capabilities_local = {}
        self.__capabilities_remote = {}

        self.__performance_index = {}

    def __assign_failure(self, expected_reward_local, expected_reward_remote):

        expected_reward_difference = abs(expected_reward_local - expected_reward_remote)
        alpha_tolerance = 0.0

        failures_assigned_to_local = sum(1 for number in self.__task_allocation.task_allocation if number % 2 == 0)
        failures_assigned_to_remote = sum(1 for number in self.__task_allocation.task_allocation if number % 2 != 0)

        if expected_reward_difference <= alpha_tolerance:

            if failures_assigned_to_local <= failures_assigned_to_remote:

                failure_allocated_to = 0
                print("Task assigned to local (within tolerance).")

            else:

                failure_allocated_to = 1
                print("Task assigned to remote (within tolerance).")
        else:
            if expected_reward_local > expected_reward_remote:

                failure_allocated_to = 0
                print("Task assigned to local.")
            else:

                failure_allocated_to = 1
                print("Task assigned to remote.")

        self.__task_allocation.task_allocation.append(failure_allocated_to)
        
        ## Simulation purposes
        if failure_allocated_to == 0:
            time.sleep(random.randint(0, 4))
        else:
            time.sleep(random.randint(0, 10))


        self.__task_allocation.failure_ended()
        print("\n")

        return failure_allocated_to

    def main_loop(self):

        print(f"Running ATTA Iter {iter}")
        for i in range(self.NUM_FAILURES):

            self.__task_allocation.failure_started() #SIMULATION PURPOSES
            
            for operator in range(self.NUM_OPERATORS):
                norm_l1, norm_u1, norm_l2, norm_u2, responsivness = self.__model.get_parameters(operator)
                
                if i % 2 == 0:
                    self.__capabilities_local[operator] = {'lower_bound_1': norm_l1,
                                                    'upper_bound_1': norm_u1,
                                                    'lower_bound_2': norm_l2,
                                                    'upper_bound_2': norm_u2,
                                                    'responsivness': responsivness}
                    
                else:
                    self.__capabilities_remote[operator] = {'lower_bound_1': norm_l1,
                                                'upper_bound_1': norm_u1,
                                                'lower_bound_2': norm_l2,
                                                'upper_bound_2': norm_u2,
                                                'responsivness': responsivness}
                    
                performance_capability_1 = self.__model.calculate_performance(norm_l1, norm_u1, self.__task_requirements[0, i])
                performance_capability_2 = self.__model.calculate_performance(norm_l2, norm_u2, self.__task_requirements[1, i])
                performance_urgency = self.__model.calculate_performance_to_urgency(self.__task_requirements[2, i], responsivness)
            
                print("Task urgency:", self.__task_requirements[2, i])
                print("Responsivness:", responsivness)
                print("Performance urgency:", performance_urgency)

                self.__performance_index[operator] = performance_capability_1 * performance_capability_2 * performance_urgency

            if torch.cuda.is_available():
                self.__performance_index = {key: value.cuda() for key, value in self.__performance_index.items()}

            reward = self.__task_allocation.calculate_reward()

            expected_reward_local = reward[0] * self.__performance_index[0]
            expected_reward_remote = reward[1] * self.__performance_index[1]

            assigned_to = self.__assign_failure(expected_reward_local, expected_reward_remote)

            for j in range(self.NUM_BINS):
                for k in range(self.NUM_BINS):
                    if self.__bin_limits[j] < self.__task_requirements[0, i] <= self.__bin_limits[j + 1] and self.__bin_limits[k] < self.__task_requirements[1, i] <= self.__bin_limits[k + 1]:
                        if assigned_to == 0:

                            #TO DO: Change to generalize to any local
                            self.__total_observations_local[0][j, k] += 1

                            if self.__task_allocation.failures['duration'][i] < self.THRESHOLD:

                                
                                self.__total_success_local[0][j, k] += 1

                        elif assigned_to == 1:
                            self.__total_observations_remote[1][j, k] += 1
                            if self.__task_allocation.failures['duration'][i] < self.THRESHOLD:
                                self.__total_success_remote[1][j, k] += 1

            # Calculate observed probabilities for local and remote
            self.observations_probabilities_local[0] = dtype(np.divide(self.__total_success_local[0], self.__total_observations_local[0], where=self.__total_observations_local[0] != 0))
            self.observations_probabilities_remote[1] = dtype(np.divide(self.__total_success_remote[1], self.__total_observations_remote[1], where=self.__total_observations_remote[1] != 0))

            # Prepare data for trust model update for local
            self.probabilities_index_local = np.array([[j, k] for j in range(self.NUM_BINS) for k in range(self.NUM_BINS) if self.__total_observations_local[0][j, k] > 0])
            
            # Prepare data for trust model update for remote
            self.probabilities_index_remote = np.array([[j, k] for j in range(self.NUM_BINS) for k in range(self.NUM_BINS) if self.__total_observations_remote[1][j, k] > 0])

            if assigned_to == 1:
                self.__model.update(self.bin_centers, self.observations_probabilities_remote[1], self.probabilities_index_remote, 1)
            else:
                self.__model.update(self.bin_centers, self.observations_probabilities_local[0], self.probabilities_index_local, 0)

if __name__ == "__main__":

    num_operators = 2
    num_bins = 25
    num_failures = 40
    threshold = 5
    performance_model_allocation = AllocationFramework(num_operators=2, num_bins=25, num_failures=50, threshold=5)

    performance_model_allocation.main_loop()







