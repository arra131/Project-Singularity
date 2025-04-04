"""This scripts downloads Kaggle datasets, inspects them for metadata, and saves the metadata to a CSV file"""

import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from dateutil import parser

# Authenticate Kaggle API and download datasets
# Ensure the Kaggle API credentials are set up in ~/.kaggle/kaggle.json
def download_kaggle_dataset(kaggle_dataset, download_path):
    """
    Downloads the Kaggle datasets
    """
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(kaggle_dataset, path=download_path, unzip=True)
    print(f"Downloaded and unzipped dataset: {kaggle_dataset}")

def parse_date_column(df, col):
    """
    Ensures a column is converted to proper datetime format (YYYY-MM-DD HH:MM:SS).
    - If only a year is present, it sets month & day to "01".
    - If separate date and time columns exist, they are combined.
    - If non-standard date formats are found, it attempts parsing.
    """
    try:
        # Try direct conversion
        df[col] = pd.to_datetime(df[col], errors='coerce')

        # If the column is still object type, manually parse dates
        if df[col].dtype == "object":
            df[col] = df[col].apply(lambda x: parser.parse(x, fuzzy=True) if pd.notna(x) else pd.NaT)

        # Ensure format YYYY-MM-DD HH:MM:SS
        df[col] = df[col].dt.strftime('%Y-%m-%d %H:%M:%S')

        return df
    except Exception as e:
        print(f"Error parsing {col}: {e}")
        return df

def inspect_dataset(file_path):
    """
    Inspects a dataset file to extract metadata, including variance and row count.
    Now correctly detects 'year' and standard 'date' columns.
    """
    filename = os.path.basename(file_path)
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return {"error": f"Failed to read {filename}: {e}"}

    df.replace("", pd.NA, inplace=True)  # Replace empty strings with NaN
    df.dropna(how='all', axis=1, inplace=True)  # Drop columns if all values are NaN

    date_columns = []
    data_columns = df.select_dtypes(include=['number']).columns.tolist()

    for col in df.columns:
        col_lower = col.lower()

        # If column is named 'date' or similar, process it
        if "date" in col_lower or "year" in col_lower or "time" in col_lower:
            df = parse_date_column(df, col)
            date_columns.append(col)

    # Remove detected date columns from data columns
    data_columns = [col for col in data_columns if col not in date_columns]

    # Calculate variance, ignoring NaN values
    variance = {col: df[col].var(skipna=True) for col in data_columns}
    variance_str = ",".join([f"{round(v, 6)}" if pd.notnull(v) else "None" for v in variance.values()])

    return {
        "file_name": filename,
        "date_column": ";".join(date_columns) if date_columns else None,
        "data_column": ",".join(data_columns) if data_columns else None,
        "multivariate": len(data_columns) > 1,
        "variance": variance_str if variance else None,
        "DataPoints": df.shape[0],  # Number of rows in dataset
    }

def read_kaggle_datasets_from_excel(file_path, dataset_col="datasetID", domain_col="Tags"):
    """
    Reads Kaggle dataset names and corresponding tags from an Excel file.
    Returns a list of dataset names and a mapping of dataset names to tags.
    """
    try:
        df = pd.read_excel(file_path)  
        kaggle_datasets = df[dataset_col].dropna().tolist()
        domain_mapping = df.set_index(dataset_col)[domain_col].dropna().to_dict()
        return kaggle_datasets, domain_mapping
    except Exception as e:
        print(f"Error reading Excel file: {e}")
        return [], {}

def process_kaggle_datasets(kaggle_datasets, download_base_path, domain_mapping):
    """
    Processes multiple Kaggle datasets: downloads, inspects, and returns metadata.
    """
    all_metadata = []
    count = 0
    for kaggle_dataset in kaggle_datasets:
        dataset_name = kaggle_dataset.split("/")[-1]
        domain = domain_mapping.get(kaggle_dataset, None)
        download_path = os.path.join(download_base_path, dataset_name)

        print(count, f"Processing dataset: {kaggle_dataset}")
        download_kaggle_dataset(kaggle_dataset, download_path)
        count += 1
        for root, _, files in os.walk(download_path):
            for file in files:
                if file.endswith(('.csv', '.xlsx')):  
                    file_path = os.path.join(root, file)
                    metadata = inspect_dataset(file_path)

                    metadata["name"] = dataset_name
                    metadata["datasetID"] = kaggle_dataset
                    metadata["Tags"] = domain
                    all_metadata.append(metadata)

    return all_metadata

def save_metadata_to_csv(metadata_list, output_csv):
    """
    Saves metadata to a CSV file in the specified column order.
    """
    df = pd.DataFrame(metadata_list)

    # Reorder columns
    column_order = [
        "name", "datasetID", "file_name", "date_column", "data_column",
        "multivariate", "variance", "Tags", "DataPoints"
    ]
    
    df = df.reindex(columns=column_order, fill_value=None)
    df.to_csv(output_csv, sep=';', index=False)
    print(f"Metadata saved to {output_csv}")

# Main Script
if __name__ == "__main__":
    excel_file_path = "Kaggle_dataset_list.xlsx"
    dataset_col_name = "datasetID"  
    domain_col_name = "Tags"

    kaggle_datasets, domain_mapping = read_kaggle_datasets_from_excel(
        excel_file_path, dataset_col=dataset_col_name, domain_col=domain_col_name
    )

    if not kaggle_datasets:
        print("No datasets found in the Excel file.")
    else:
        print(f"Found {len(kaggle_datasets)} datasets from Excel.")

    download_base_path = "./kaggle_datasets"
    output_csv = "Kaggle_metadata.csv"

    metadata_list = process_kaggle_datasets(kaggle_datasets, download_base_path, domain_mapping)
    save_metadata_to_csv(metadata_list, output_csv)