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

import pandas as pd

def process_dataset(filepath: str, date_column: str, multivariate: bool, target_columns: list):
    """Process both univariate and multivariate datasets."""
    # Read the dataset
    df = pd.read_csv(filepath)
    
    # Ensure the date column exists
    if date_column not in df.columns:
        raise ValueError(f"Expected '{date_column}' column in the dataset.")
    
    # Convert the date column to standard format
    def convert_date_format(date_str):
        """Try different date formats."""
        try:
            # Convert the input to string first
            date_str = str(date_str)
            # Try full date format (MM/DD/YYYY)
            return pd.to_datetime(date_str, format='%m/%d/%Y').strftime('%Y-%m-%d %H:%M:%S')
        except ValueError:
            try:
                # Try format 'DD-MMM' assuming the current or default year
                return pd.to_datetime(date_str + '-2023', format='%d-%b-%Y').strftime('%Y-%m-%d %H:%M:%S')
            except ValueError:
                try:
                    # Try year-only format (e.g., '2019')
                    return pd.to_datetime(date_str, format='%Y').strftime('%Y-01-01 %H:%M:%S')
                except ValueError:
                    try:
                        # Fallback: Parse other formats or ISO format
                        return pd.to_datetime(date_str).strftime('%Y-%m-%d %H:%M:%S')
                    except ValueError:
                        print(f"Unrecognized date format: {date_str}")
                        return None

    # Apply the date conversion logic
    df[date_column] = df[date_column].apply(convert_date_format)

    # Drop rows with invalid dates
    df = df.dropna(subset=[date_column])
    
    # Convert date column to list
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
            "kaggle_dataset_name": "kandij/electric-production",
            "file_name": "Electric_Production.csv",
            "date_column": "DATE",
            "multivariate": False,  # Univariate dataset
            "target_columns": ["Value"]
        },
        {
            "kaggle_dataset_name": "rakannimer/air-passengers",
            "file_name": "AirPassengers.csv",
            "date_column": "Month",
            "multivariate": False,  # Univariate dataset
            "target_columns": ["#Passengers"]
        },
        {
            "kaggle_dataset_name": "mukeshmanral/univariate-time-series",
            "file_name": "date_count.csv",
            "date_column": "Date",
            "multivariate": False,  # Univariate dataset
            "target_columns": ["count"]
        },
        {
            "kaggle_dataset_name": "arashnic/learn-time-series-forecasting-from-gold-price",
            "file_name": "gold_price_data.csv",
            "date_column": "Date",
            "multivariate": False,  # Univariate dataset
            "target_columns": ["Value"]
        },
        {
            "kaggle_dataset_name": "vikramamin/holt-winters-forecasting-for-sales-data",
            "file_name": "MonthlySales.csv",
            "date_column": "month",
            "multivariate": False,  # Univariate dataset
            "target_columns": ["sales"]
        },
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
            "kaggle_dataset_name": "ranja7/electricity-consumption",
            "file_name": "daily_consumption.csv",
            "date_column": "Date",
            "multivariate": False, 
            "target_columns": ["Energy Consumption (kWh)"]
        },
        {
            "kaggle_dataset_name": "prakharmkaushik/airline-passengers-tsa",
            "file_name": "AirPassengers.csv",
            "date_column": "Timeline",
            "multivariate": False, 
            "target_columns": ["Number_of_Passengers"]
        },
        {
            "kaggle_dataset_name": "billykal/monthly-sunspots",
            "file_name": "monthly-sunspots.csv",
            "date_column": "Month",
            "multivariate": False, 
            "target_columns": ["Sunspots"]
        },
        {
            "kaggle_dataset_name": "artemig/time-series-sample-001",
            "file_name": "time_series_sample_001.csv",
            "date_column": "timestamp",
            "multivariate": False, 
            "target_columns": ["value"]
        },
        {
            "kaggle_dataset_name": "mohamedharris/customers-of-beauty-parlour-time-series",
            "file_name": "Customers_Parlour.csv",
            "date_column": "date",
            "multivariate": False, 
            "target_columns": ["Customers"]
        },
        {
            "kaggle_dataset_name": "joebeachcapital/dow-jones-and-s-and-p500-indices-daily-update",
            "file_name": "DJCA.csv",
            "date_column": "DATE",
            "multivariate": False, 
            "target_columns": ["DJCA"]
        },
        {
            "kaggle_dataset_name": "joebeachcapital/dow-jones-and-s-and-p500-indices-daily-update",
            "file_name": "DJIA.csv",
            "date_column": "DATE",
            "multivariate": False, 
            "target_columns": ["DJIA"]
        },
        {
            "kaggle_dataset_name": "joebeachcapital/dow-jones-and-s-and-p500-indices-daily-update",
            "file_name": "DJTA.csv",
            "date_column": "DATE",
            "multivariate": False, 
            "target_columns": ["DJTA"]
        },
        {
            "kaggle_dataset_name": "joebeachcapital/dow-jones-and-s-and-p500-indices-daily-update",
            "file_name": "DJUA.csv",
            "date_column": "DATE",
            "multivariate": False, 
            "target_columns": ["DJUA"]
        },
        {
            "kaggle_dataset_name": "joebeachcapital/dow-jones-and-s-and-p500-indices-daily-update",
            "file_name": "SP500.csv",
            "date_column": "DATE",
            "multivariate": False, 
            "target_columns": ["SP500"]
        },
        {
            "kaggle_dataset_name": "rassiem/monthly-car-sales",
            "file_name": "monthly-car-sales.csv",
            "date_column": "Month",
            "multivariate": False, 
            "target_columns": ["Sales"]
        },
        {
            "kaggle_dataset_name": "jylim21/malaysia-public-data",
            "file_name": "births.csv",
            "date_column": "date",
            "multivariate": False, 
            "target_columns": ["births"]
        },
        {
            "kaggle_dataset_name": "ankitkalauni/tps-jan22-google-trends-kaggle-search-dataset",
            "file_name": "multiTimeline.csv",
            "date_column": "Month",
            "multivariate": False, 
            "target_columns": ["kaggle"]
        },
        {
            "kaggle_dataset_name": "ekayfabio/immigration-apprehended",
            "file_name": "immigration_apprehended.csv",
            "date_column": "Year",
            "multivariate": False, 
            "target_columns": ["Number"]
        },
        {
            "kaggle_dataset_name": "gokcegok/falls-mortality-dataset",
            "file_name": "falls_mortality__dataset.csv",
            "date_column": "year",
            "multivariate": False, 
            "target_columns": ["death"]
        },
        {
            "kaggle_dataset_name": "nekoslevin/spydataa",
            "file_name": "SPYdata.csv",
            "date_column": "Trade_date",
            "multivariate": False, 
            "target_columns": ["SPY"]
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
