"""Cleans csvs to remove all missining dates and replaces missing tags with 'unknown'"""

import pandas as pd

# Load the CSV file with ; delimiter
file_path = 'Kaggle_final_test.csv'  
df = pd.read_csv(file_path, delimiter=';')

# Function to clean the 4th column
def clean_column(value):
    if pd.isna(value):  # Skip if the value is missing
        return value
    if ';' in str(value):  # Check if the value contains multiple parts
        parts = str(value).split(';')  # Split by ;
        if len(parts) >= 2:
            return parts[1]  # Return the middle value
    return value  # Return the original value if no cleaning is needed

# Apply the cleaning function to the 4th column
fourth_column = df.columns[3]  # Assuming the 4th column is at index 3
df[fourth_column] = df[fourth_column].apply(clean_column)

# Check the second last column and replace missing values with 'unknown'
second_last_column = df.columns[-2]  # Get the name of the second last column
df[second_last_column] = df[second_last_column].fillna('unknown')
df = df.dropna()

# Save the modified CSV
output_file_path = 'Kaggle_clean_metadata.csv'  # Replace with your desired output file path
df.to_csv(output_file_path, sep=';', index=False)

print(f"Modified CSV saved to {output_file_path}")
print("end")
