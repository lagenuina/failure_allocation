import csv
import random

# Function to generate a list of random numbers
def generate_random_numbers(n):
    return [round(random.uniform(0, 1), 2) for _ in range(n)]

# Define the CSV file name
csv_file = 'failure_requirements.csv'

# Generate data
data = [generate_random_numbers(500) for _ in range(3)]

# Write the random numbers to the CSV file
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)

print(f'{csv_file} has been created with 3 rows and 500 columns of random numbers.')
