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
    requirements = data[['requirement_1', 'requirement_2', 'requirement_3', 'assigned_to', 'success', 
                         '0_reward_capability_1', '0_reward_capability_2', '0_reward_capability_3', 
                         '1_reward_capability_1', '1_reward_capability_2', '1_reward_capability_3']]

    # Determine if any capability is zero
    operator_0_missing_capability = (
        (requirements['0_reward_capability_1'] == 0) | 
        (requirements['0_reward_capability_2'] == 0) | 
        (requirements['0_reward_capability_3'] == 0)
    )
    
    operator_1_missing_capability = (
        (requirements['1_reward_capability_1'] == 0) | 
        (requirements['1_reward_capability_2'] == 0) | 
        (requirements['1_reward_capability_3'] == 0)
    )

    # Filter data based on operator, success, and missing capability
    operator_0_success_missing = requirements[(requirements['assigned_to'] == 0) & (requirements['success'] == True) & operator_0_missing_capability]
    operator_0_failure_missing = requirements[(requirements['assigned_to'] == 0) & (requirements['success'] == False) & operator_0_missing_capability]
    
    operator_0_success_full = requirements[(requirements['assigned_to'] == 0) & (requirements['success'] == True) & ~operator_0_missing_capability]
    operator_0_failure_full = requirements[(requirements['assigned_to'] == 0) & (requirements['success'] == False) & ~operator_0_missing_capability]
    
    operator_1_success_missing = requirements[(requirements['assigned_to'] == 1) & (requirements['success'] == True) & operator_1_missing_capability]
    operator_1_failure_missing = requirements[(requirements['assigned_to'] == 1) & (requirements['success'] == False) & operator_1_missing_capability]
    
    operator_1_success_full = requirements[(requirements['assigned_to'] == 1) & (requirements['success'] == True) & ~operator_1_missing_capability]
    operator_1_failure_full = requirements[(requirements['assigned_to'] == 1) & (requirements['success'] == False) & ~operator_1_missing_capability]

    # Create a 3D scatter plot using Plotly
    fig = go.Figure()

    # Operator 0 Success with Full Capability
    fig.add_trace(go.Scatter3d(
        x=operator_0_success_full['requirement_1'],
        y=operator_0_success_full['requirement_2'],
        z=operator_0_success_full['requirement_3'],
        mode='markers',
        marker=dict(
            size=5,
            color='blue',
            symbol='circle',
            opacity=0.8
        ),
        name='Operator 0 Success (Full Capability)'
    ))

    # Operator 0 Failure with Full Capability
    fig.add_trace(go.Scatter3d(
        x=operator_0_failure_full['requirement_1'],
        y=operator_0_failure_full['requirement_2'],
        z=operator_0_failure_full['requirement_3'],
        mode='markers',
        marker=dict(
            size=5,
            color='blue',
            symbol='x',
            opacity=0.8
        ),
        name='Operator 0 Failure (Full Capability)'
    ))

    # Operator 0 Success with Missing Capability
    fig.add_trace(go.Scatter3d(
        x=operator_0_success_missing['requirement_1'],
        y=operator_0_success_missing['requirement_2'],
        z=operator_0_success_missing['requirement_3'],
        mode='markers',
        marker=dict(
            size=5,
            color='blue',
            symbol='circle',
            opacity=0.2,
            line=dict(
                color='blue',
                width=3
            )
        ),
        name='Operator 0 Success (Missing Capability)'
    ))

    # Operator 0 Failure with Missing Capability
    fig.add_trace(go.Scatter3d(
        x=operator_0_failure_missing['requirement_1'],
        y=operator_0_failure_missing['requirement_2'],
        z=operator_0_failure_missing['requirement_3'],
        mode='markers',
        marker=dict(
            size=5,
            color='white',
            symbol='x',
            # opacity=0.2,
            line=dict(
                color='blue',
                width=3
            )
        ),
        name='Operator 0 Failure (Missing Capability)'
    ))

    # Operator 1 Success with Full Capability
    fig.add_trace(go.Scatter3d(
        x=operator_1_success_full['requirement_1'],
        y=operator_1_success_full['requirement_2'],
        z=operator_1_success_full['requirement_3'],
        mode='markers',
        marker=dict(
            size=5,
            color='red',
            symbol='circle',
            opacity=0.8
        ),
        name='Operator 1 Success (Full Capability)'
    ))

    # Operator 1 Failure with Full Capability
    fig.add_trace(go.Scatter3d(
        x=operator_1_failure_full['requirement_1'],
        y=operator_1_failure_full['requirement_2'],
        z=operator_1_failure_full['requirement_3'],
        mode='markers',
        marker=dict(
            size=5,
            color='red',
            symbol='x',
            opacity=0.8
        ),
        name='Operator 1 Failure (Full Capability)'
    ))

    # Operator 1 Success with Missing Capability
    fig.add_trace(go.Scatter3d(
        x=operator_1_success_missing['requirement_1'],
        y=operator_1_success_missing['requirement_2'],
        z=operator_1_success_missing['requirement_3'],
        mode='markers',
        marker=dict(
            size=5,
            color='red',
            symbol='circle',
            opacity=0.2,
            line=dict(
                color='red',
                width=3
            )
        ),
        name='Operator 1 Success (Missing Capability)'
    ))

    # Operator 1 Failure with Missing Capability
    fig.add_trace(go.Scatter3d(
        x=operator_1_failure_missing['requirement_1'],
        y=operator_1_failure_missing['requirement_2'],
        z=operator_1_failure_missing['requirement_3'],
        mode='markers',
        marker=dict(
            size=5,
            color='red',
            symbol='x',
            opacity=0.2,
            line=dict(
                color='red',
                width=3
            )
        ),
        name='Operator 1 Failure (Missing Capability)'
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