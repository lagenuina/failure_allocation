import pandas as pd
import matplotlib.pyplot as plt

num_operators = 2
num_failures = 30

# Step 1: Read the CSV file into a pandas DataFrame
df = pd.read_csv(f'results/{num_operators}_operators_{num_failures}_failures (36).csv')

# Step 2: Extract data for plotting
operators = range(num_operators)  # Assuming you want to plot for operators 0, 1, and 2
colors = ['blue', 'green', 'red']  # Colors for different operators

# Create subplots for each operator
fig, axs = plt.subplots(len(operators), 1, figsize=(10, 6*len(operators)), sharex=True)

# Plot each operator's data in separate subplots
for i, operator in enumerate(operators):
    lower_bound_1_col = f'{operator}_lower_bound_1'
    upper_bound_1_col = f'{operator}_upper_bound_1'
    lower_bound_2_col = f'{operator}_lower_bound_2'
    upper_bound_2_col = f'{operator}_upper_bound_2'
    # responsiveness_col = f'{operator}_responsiveness'
    lower_bound_3_col = f'{operator}_lower_bound_3'
    upper_bound_3_col = f'{operator}_upper_bound_3'

    lower_bound_1_data = df[lower_bound_1_col]
    upper_bound_1_data = df[upper_bound_1_col]
    lower_bound_2_data = df[lower_bound_2_col]
    upper_bound_2_data = df[upper_bound_2_col]
    lower_bound_3_data = df[lower_bound_3_col]
    upper_bound_3_data = df[upper_bound_3_col]

    # Plot lower and upper bounds
    axs[i].plot(lower_bound_1_data, label=f'Lower Bound 1', color=colors[0], linestyle='-')
    axs[i].plot(upper_bound_1_data, label=f'Upper Bound 1', color=colors[0], linestyle='-')
    axs[i].plot(lower_bound_2_data, label=f'Lower Bound 2', color=colors[1], linestyle='-')
    axs[i].plot(upper_bound_2_data, label=f'Upper Bound 2', color=colors[1], linestyle='-')

    # Plot responsiveness
    # axs[i].plot(responsiveness_data, label=f'Responsiveness', color=colors[2])
    axs[i].plot(lower_bound_3_data, label=f'Lower Bound 3', color=colors[2], linestyle='-')
    axs[i].plot(upper_bound_3_data, label=f'Upper Bound 3', color=colors[2], linestyle='-')

    # Customize subplot
    axs[i].set_title(f'Operator {operator}')
    axs[i].set_xlabel('Data Points')
    axs[i].set_ylabel('Values')
    axs[i].legend()
    axs[i].grid(True)
    axs[i].set_ylim(0, 1)  # Set y-axis limits to [0, 1]


columns = ['reward_urgency', 'reward_capabilities', 'cost', 'total_reward']
plot_data = {}

for column in columns:
    plot_data[column] = [df[f'{operator}_{column}'] for operator in operators]

# Step 3: Create plots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()

for i, (column, ax) in enumerate(zip(columns, axs)):
    for operator_data, operator in zip(plot_data[column], operators):
        ax.plot(operator_data, label=operator)
    ax.set_xlabel('Iterations')
    ax.set_ylabel(column.replace('_', ' ').title())
    ax.legend()

assignment_column = 'assigned'
if assignment_column in df:
    assignment_counts = df[assignment_column].value_counts().sort_index()
    print(assignment_counts)

# Adjust layout and show plot
plt.tight_layout()
plt.show()