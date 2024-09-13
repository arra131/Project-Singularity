import os
import pandas as pd
from pathlib import Path
import kaggle

# Authenticate with Kaggle API
kaggle.api.authenticate()

def download_dataset(kaggle_dataset_name: str, file_name: str, download_path: str):
    """Download a Kaggle dataset and return the file path."""
    download_dir = Path(download_path)
    
    # Create the download directory if it doesn't exist
    if not download_dir.exists():
        os.makedirs(download_dir)
    
    # Download the dataset from Kaggle and unzip it
    print(f"Downloading {kaggle_dataset_name} to {download_path}...")
    kaggle.api.dataset_download_files(
        dataset=kaggle_dataset_name, 
        path=download_path, 
        unzip=True  # Make sure to unzip the dataset
    )

    # Check if the file exists after downloading
    file_path = str(download_dir / file_name)
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File '{file_name}' not found in the directory '{download_path}'")
    
    return file_path

def process_dataset(filepath: str, date_column: str, multivariate: bool, target_columns: list):
    """Process both univariate and multivariate datasets."""
    # Read the dataset
    df = pd.read_csv(filepath)
    
    # Ensure the date column exists
    if date_column not in df.columns:
        raise ValueError(f"Expected '{date_column}' column in the dataset.")

    # Convert date to standard format
    df[date_column] = pd.to_datetime(df[date_column]).dt.strftime('%Y-%m-%d %H:%M:%S')
    dates = df[date_column].tolist()

    # Process multivariate or univariate data based on configuration
    if multivariate:
        # Multivariate: Ensure all columns exist
        missing_columns = [col for col in target_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing columns in the dataset: {', '.join(missing_columns)}")
        
        # Extract multivariate values
        values = [df[col].tolist() for col in target_columns]  # Each column forms a separate list
    else:
        # Univariate: Ensure target column exists
        if target_columns[0] not in df.columns:
            raise ValueError(f"Expected column '{target_columns[0]}' in the dataset for univariate processing.")
        
        # Extract univariate values (wrap in a list of lists for consistency)
        values = [df[target_columns[0]].tolist()]

    return {
        "date": dates,
        "value": values
    }

def main():
    # Configuration for datasets
    datasets_config = [
        {
            "kaggle_dataset_name": "kapatsa/modelled-time-series",
            "file_name": "GDPUS_nsa.csv",
            "date_column": "DATE",
            "multivariate": False,  # Univariate dataset
            "target_columns": ["NA000334Q"]
        },
        {
            "kaggle_dataset_name": "vitthalmadane/ts-temp-1",
            "file_name": "MLTempDataset1.csv",
            "date_column": "Datetime",
            "multivariate": False, 
            "target_columns": ["Hourly_Temp"]
        },
        {
            "kaggle_dataset_name": "arashnic/time-series-forecasting-with-yahoo-stock-price",
            "file_name": "yahoo_stock.csv",
            "date_column": "Date",
            "multivariate": True,  # Multivariate dataset
            "target_columns": ["High", "Open", "Close", "Low"]
        }
    ]
    
    # Directory where datasets will be downloaded
    download_dir = "./datasets"

    # Process each dataset
    for config in datasets_config:
        try:
            file_path = download_dataset(
                kaggle_dataset_name=config['kaggle_dataset_name'],
                file_name=config['file_name'],
                download_path=download_dir
            )

            # Process the dataset
            dataset = process_dataset(
                filepath=file_path,
                date_column=config['date_column'],
                multivariate=config['multivariate'],
                target_columns=config['target_columns']
            )

            # Print the first record for each dataset as a sample
            print(f"Dataset from {config['kaggle_dataset_name']}")
            print("Date:", dataset['date'])  
            print("Values:", dataset['value'])  

        except Exception as e:
            print(f"Error processing {config['kaggle_dataset_name']}: {e}")

if __name__ == "__main__":
    main()
