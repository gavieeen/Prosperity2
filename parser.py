import pandas as pd
import os

# Get the current working directory
current_directory = os.getcwd()

# Construct the file path to the CSV file
file_path = os.path.join(current_directory, 'round-1-island-data-bottle', 'prices_round_1_day_-1.csv')

# Read CSV file into a pandas DataFrame
df = pd.read_csv(file_path)

# Display the DataFrame
print(df.head()) # type: ignore