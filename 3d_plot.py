import plotly.graph_objs as go
import glob
import pandas as pd
import numpy as np

csv_files = glob.glob('results/*.csv')
colors = [
    'blue', 'green', 'red', 'purple', 'orange', 'cyan', 'magenta', 
    'yellow', 'black', 'pink', 'brown', 'gray', 'olive', 'lime', 
    'indigo', 'maroon', 'navy', 'teal', 'violet', 'gold', 'coral', 
    'turquoise', 'salmon', 'plum', 'orchid', 'crimson', 'khaki', 
    'lavender', 'sienna', 'tan']

for file_index, csv_file in enumerate(csv_files):
    
    # Load the CSV file
    data = pd.read_csv(csv_file)

    # Select relevant columns
    base_columns = ['requirement_1', 'requirement_2', 'requirement_3', 'assigned_to', 'success']
    operator_capability_columns = [col for col in data.columns if 'reward_capability' in col]
    requirements = data[base_columns + operator_capability_columns]

    # Determine unique operators
    operators = int(data['num_operators'].iloc[0])

    # Create a 3D scatter plot using Plotly
    fig = go.Figure()

    for operator in range(operators):
        color = colors[operator % len(colors)]
        
        operator_success_full = requirements[(requirements['assigned_to'] == operator) & (requirements['success'] == True) & 
                                             ~(requirements[f'{operator}_reward_capability_1'] == 0) & 
                                             ~(requirements[f'{operator}_reward_capability_2'] == 0) & 
                                             ~(requirements[f'{operator}_reward_capability_3'] == 0)]

        operator_failure_full = requirements[(requirements['assigned_to'] == operator) & (requirements['success'] == False) & 
                                             ~(requirements[f'{operator}_reward_capability_1'] == 0) & 
                                             ~(requirements[f'{operator}_reward_capability_2'] == 0) & 
                                             ~(requirements[f'{operator}_reward_capability_3'] == 0)]

        operator_success_missing = requirements[(requirements['assigned_to'] == operator) & (requirements['success'] == True) & 
                                                ((requirements[f'{operator}_reward_capability_1'] == 0) | 
                                                 (requirements[f'{operator}_reward_capability_2'] == 0) | 
                                                 (requirements[f'{operator}_reward_capability_3'] == 0))]

        operator_failure_missing = requirements[(requirements['assigned_to'] == operator) & (requirements['success'] == False) & 
                                                ((requirements[f'{operator}_reward_capability_1'] == 0) | 
                                                 (requirements[f'{operator}_reward_capability_2'] == 0) | 
                                                 (requirements[f'{operator}_reward_capability_3'] == 0))]

        # Operator Success with Full Capability
        fig.add_trace(go.Scatter3d(
            x=operator_success_full['requirement_1'],
            y=operator_success_full['requirement_2'],
            z=operator_success_full['requirement_3'],
            mode='markers',
            marker=dict(
                size=5,
                color=color,
                symbol='circle',
                opacity=0.8
            ),
            name=f'Operator {operator} Success (Full Capability)'
        ))

        # Operator Failure with Full Capability
        fig.add_trace(go.Scatter3d(
            x=operator_failure_full['requirement_1'],
            y=operator_failure_full['requirement_2'],
            z=operator_failure_full['requirement_3'],
            mode='markers',
            marker=dict(
                size=5,
                color=color,
                symbol='x',
                opacity=0.8
            ),
            name=f'Operator {operator} Failure (Full Capability)'
        ))

        # Operator Success with Missing Capability
        fig.add_trace(go.Scatter3d(
            x=operator_success_missing['requirement_1'],
            y=operator_success_missing['requirement_2'],
            z=operator_success_missing['requirement_3'],
            mode='markers',
            marker=dict(
                size=5,
                color=color,
                symbol='circle',
                opacity=0.2,
                line=dict(
                    color=color,
                    width=3
                )
            ),
            name=f'Operator {operator} Success (Missing Capability)'
        ))

        # Operator Failure with Missing Capability
        fig.add_trace(go.Scatter3d(
            x=operator_failure_missing['requirement_1'],
            y=operator_failure_missing['requirement_2'],
            z=operator_failure_missing['requirement_3'],
            mode='markers',
            marker=dict(
                size=5,
                color='white',
                symbol='x',
                line=dict(
                    color=color,
                    width=3
                )
            ),
            name=f'Operator {operator} Failure (Missing Capability)'
        ))

    # Update layout
    fig.update_layout(
        scene=dict(
            xaxis_title='Capability 1',
            yaxis_title='Capability 2',
            zaxis_title='Capability 3'
        ),
        legend=dict(
            x=0.8,
            y=0.9
        )
    )

    # Show plot
    fig.show()
