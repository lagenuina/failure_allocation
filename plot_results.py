import pandas as pd
import matplotlib.pyplot as plt
import glob

csv_files = glob.glob('results/*.csv')
colors = [
    'red', 'purple', 'orange', 'blue', 'green', 'cyan', 'magenta', 
    'yellow', 'black', 'pink', 'brown', 'gray', 'olive', 'lime', 
    'indigo', 'maroon', 'navy', 'teal', 'violet', 'gold', 'coral', 
    'turquoise', 'salmon', 'plum', 'orchid', 'crimson', 'khaki', 
    'lavender', 'sienna', 'tan']

for file_index, csv_file in enumerate(csv_files):
    # Step 2: Read the CSV file into a pandas DataFrame
    df = pd.read_csv(csv_file)

    # Step 2: Extract data for plotting
    operators = int(df['num_operators'].iloc[0]) # Assuming you want to plot for operators 0, 1, and 2
    # colors = colors[:operators]  # Colors for different operators

    # Create subplots for each operator
    fig, axs = plt.subplots(operators, 1, figsize=(10, 6*operators), sharex=True)

    # Plot each operator's data in separate subplots
    for operator in range(operators):
        lower_bound_1_col = f'{operator}_lower_bound_1'
        upper_bound_1_col = f'{operator}_upper_bound_1'
        lower_bound_2_col = f'{operator}_lower_bound_2'
        upper_bound_2_col = f'{operator}_upper_bound_2'
        lower_bound_3_col = f'{operator}_lower_bound_3'
        upper_bound_3_col = f'{operator}_upper_bound_3'

        lower_bound_1_data = df[lower_bound_1_col]
        upper_bound_1_data = df[upper_bound_1_col]
        lower_bound_2_data = df[lower_bound_2_col]
        upper_bound_2_data = df[upper_bound_2_col]
        lower_bound_3_data = df[lower_bound_3_col]
        upper_bound_3_data = df[upper_bound_3_col]

        # Plot lower and upper bounds
        axs[operator].plot(lower_bound_1_data, label=f'Lower Bound 1', color=colors[0], linestyle='-')
        axs[operator].plot(upper_bound_1_data, label=f'Upper Bound 1', color=colors[0], linestyle='-')
        axs[operator].plot(lower_bound_2_data, label=f'Lower Bound 2', color=colors[1], linestyle='-')
        axs[operator].plot(upper_bound_2_data, label=f'Upper Bound 2', color=colors[1], linestyle='-')
        axs[operator].plot(lower_bound_3_data, label=f'Lower Bound 3', color=colors[2], linestyle='-')
        axs[operator].plot(upper_bound_3_data, label=f'Upper Bound 3', color=colors[2], linestyle='-')

        # Customize subplot
        axs[operator].set_title(f'Operator {operator}')
        axs[operator].set_xlabel('Data Points')
        axs[operator].set_ylabel('Values')
        axs[operator].legend()
        axs[operator].grid(True)
        axs[operator].set_ylim(0, 1)  # Set y-axis limits to [0, 1]
        
        # Save the figure
        output_filename = f'{csv_file.split("\\")[-1]}.png'

        plt.savefig(output_filename)

columns = ['reward_capability_1', 'reward_capability_2', 'reward_capability_3', 'cost', 'total_reward']
plot_data = {}

for column in columns:
    plot_data[column] = [df[f'{operator}_{column}'] for operator in range(operators)]

# Step 3: Create plots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
axs = axs.flatten()

for i, (column, ax) in enumerate(zip(columns, axs)):
    for operator_data, operator in zip(plot_data[column], range(operators)):
        ax.plot(operator_data, label=operator)
    ax.set_xlabel('Iterations')
    ax.set_ylabel(column.replace('_', ' ').title())
    ax.legend()

# fig, ax = plt.subplots(figsize=(12, 8))

# for operator in range(operators):
#     total_reward_col = f'{operator}_total_reward'
#     total_reward_data = df[total_reward_col]
#     ax.plot(total_reward_data, label=f'Operator {operator}', color=colors[operator])

# ax.set_title('Total Reward for Operators')
# ax.set_xlabel('Iterations')
# ax.set_ylabel('Total Reward')
# ax.legend()
# ax.grid(True)

# Adjust layout and show plot
plt.tight_layout()
fig.canvas.manager.set_window_title(f'File: {csv_file}')
plt.show()