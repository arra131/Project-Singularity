import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi
from datasets import Dataset, DatasetDict

def download_kaggle_dataset(kaggle_dataset, download_path):
    """
    Downloads a Kaggle dataset.
    """
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files(kaggle_dataset, path=download_path, unzip=True)
    print(f"Downloaded and unzipped dataset: {kaggle_dataset}")

def inspect_dataset(file_path):
    """
    Inspects a dataset file to extract metadata, including variance for data columns.
    """
    filename = os.path.basename(file_path)
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        return {"error": f"Failed to read {filename}: {e}"}

    # Remove unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed:')]

    # Detect date and data columns
    date_columns = []
    data_columns = []

    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            try:
                pd.to_datetime(df[col], errors='raise')  # Test conversion
                date_columns.append(col)
            except:
                pass

    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    data_columns = numeric_cols

    # Calculate variance
    if data_columns:
        if len(data_columns) == 1:
            # Univariate: Compute variance for the single column
            variance = df[data_columns[0]].var()
            variance = round(variance, 6) if pd.notnull(variance) else None
        else:
            # Multivariate: Compute variance for each column and join as a string
            variance = ",".join(
                [str(round(df[col].var(), 6)) if pd.notnull(df[col].var()) else "None" for col in data_columns]
            )
    else:
        variance = None

    # Join data columns with commas instead of semicolons
    formatted_data_columns = ",".join(data_columns) if data_columns else None

    return {
        "file_name": filename,
        "date_column": ";".join(date_columns) if date_columns else None,  # Use semicolon for date columns
        "data_column": formatted_data_columns,  # Use comma for data columns
        "multivariate": len(data_columns) > 1,
        "variance": variance,  # Include variance in metadata
    }

def process_kaggle_datasets(kaggle_datasets, download_base_path, domain_mapping):
    """
    Processes multiple Kaggle datasets: downloads, inspects, and returns metadata.
    """
    all_metadata = []

    for kaggle_dataset in kaggle_datasets:
        dataset_name = kaggle_dataset.split("/")[-1]
        domain = domain_mapping.get(kaggle_dataset, None)
        download_path = os.path.join(download_base_path, dataset_name)

        print(f"Processing dataset: {kaggle_dataset}")
        download_kaggle_dataset(kaggle_dataset, download_path)

        # Inspect downloaded files
        for root, _, files in os.walk(download_path):
            for file in files:
                if file.endswith(('.csv', '.xlsx')):  # Adjust for dataset formats
                    file_path = os.path.join(root, file)
                    metadata = inspect_dataset(file_path)
                    # Add other fields in the correct order
                    metadata["name"] = dataset_name
                    metadata["kaggle_dataset"] = kaggle_dataset
                    metadata["domain"] = domain
                    # Append to the metadata list
                    all_metadata.append(metadata)

    return all_metadata

def save_metadata_to_csv(metadata_list, output_csv):
    """
    Saves metadata to a CSV file.
    """
    df = pd.DataFrame(metadata_list)
    df.to_csv(output_csv, sep=';', index=False)  # Use semicolon as the separator
    print(f"Metadata saved to {output_csv}")

# Main Script
if __name__ == "__main__":
    # Parameters
    kaggle_datasets = [
        "vitthalmadane/ts-temp-1",
        "mukeshmanral/univariate-time-series",
        "vitthalmadane/ts-temp-1",
        "mukeshmanral/univariate-time-series",
        "artemig/time-series-sample-001",
        "kandij/electric-production",
        "vikramamin/holt-winters-forecasting-for-sales-data",
        "prakharmkaushik/airline-passengers-tsa",
        "billykal/monthly-sunspots",
        "ashfakyeafi/air-passenger-data-for-time-series-analysis",
        "mohamedharris/customers-of-beauty-parlour-time-series",
        "rassiem/monthly-car-sales",
        "jylim21/malaysia-public-data",
        "ankitkalauni/tps-jan22-google-trends-kaggle-search-dataset",
        "ekayfabio/immigration-apprehended",
        "nekoslevin/spydataa",
        "kapatsa/modelled-time-series",
        "arashnic/time-series-forecasting-with-yahoo-stock-price",
        "meetnagadia/apple-stock-price-from-19802021",
        "pdquant/sp500-daily-19862018",
        "pritsheta/netflix-stock-data-from-2002-to-2021",
		"jillanisofttech/tesla-stock-price",
        "hananxx/gamestop-historical-stock-prices",
		"meetnagadia/coco-cola-stock-data-19622021",
		"asimislam/30-yrs-stock-market-data",
        "meetnagadia/us-dollar-inr-rupee-dataset20032021",
		"andrewmvd/sp-500-stocks",
        "yash16jr/snp500-dataset",
		"thedevastator/analyzing-credit-card-spending-habits-in-india",
		"adilbhatti/dollar-exchange-rates-asian-countries",
        "lydia70/skechers-historical-stock-data",
        "meetnagadia/dogecoin-inr-dataset-20172020",
		"kanchana1990/futurocoin-saga-a-200-year-cryptocurrency-odyssey",
        "varpit94/tesla-stock-data-updated-till-28jun2021",
		"guillemservera/grains-and-cereals-futures",
        "arashnic/learn-time-series-forecasting-from-gold-price",
    ]  
    download_base_path = "./kaggle_datasets"
    output_csv = "kaggle_dataset_metadata.csv"
    
    # Optional: Domain Mapping (can be empty if domain isn't specified)
    domain_mapping = {
        "vitthalmadane/ts-temp-1": "Temperature",
        "mukeshmanral/univariate-time-series": "Time Series",
        "vitthalmadane/ts-temp-1": "Temperature",
        "mukeshmanral/univariate-time-series": "Time Series",
        "artemig/time-series-sample-001": "test domain",
        "kandij/electric-production": "energry",
        "vikramamin/holt-winters-forecasting-for-sales-data": "sales",
        "prakharmkaushik/airline-passengers-tsa": "transport",
        "billykal/monthly-sunspots": "time series",
        "ashfakyeafi/air-passenger-data-for-time-series-analysis": "transport",
        "mohamedharris/customers-of-beauty-parlour-time-series": "sales",
        "rassiem/monthly-car-sales": "sales",
        "jylim21/malaysia-public-data": "govt",
        "ankitkalauni/tps-jan22-google-trends-kaggle-search-dataset": "social",
        "ekayfabio/immigration-apprehended": "govt",
        "nekoslevin/spydataa": "govt",
        "kapatsa/modelled-time-series": "time series",
        "arashnic/time-series-forecasting-with-yahoo-stock-price": "finance",
        "meetnagadia/apple-stock-price-from-19802021": "finance",
        "marquis03/afac2023-time-series-prediction-in-finance": "finance",
        "pdquant/sp500-daily-19862018": "finance",
        "pritsheta/netflix-stock-data-from-2002-to-2021": "finance",
		"jillanisofttech/tesla-stock-price": "finance",
        "hananxx/gamestop-historical-stock-prices": "finance",
		"meetnagadia/coco-cola-stock-data-19622021": "finance",
		"asimislam/30-yrs-stock-market-data": "finance",
        "meetnagadia/us-dollar-inr-rupee-dataset20032021": "finance",
		"andrewmvd/sp-500-stocks": "finance",
        "yash16jr/snp500-dataset": "finance",
		"thedevastator/analyzing-credit-card-spending-habits-in-india": "finance",
		"adilbhatti/dollar-exchange-rates-asian-countries": "finance",
        "lydia70/skechers-historical-stock-data" : "finance",
		"meetnagadia/dogecoin-inr-dataset-20172020" : "finance",
		"kanchana1990/futurocoin-saga-a-200-year-cryptocurrency-odyssey" : "finance",
        "varpit94/tesla-stock-data-updated-till-28jun2021" : "finance",
		"guillemservera/grains-and-cereals-futures" : "finance",
        "arashnic/learn-time-series-forecasting-from-gold-price" : "finance",
		
    }

    # Step 1: Process all Kaggle datasets
    metadata_list = process_kaggle_datasets(kaggle_datasets, download_base_path, domain_mapping)

    # Step 2: Save metadata to CSV
    save_metadata_to_csv(metadata_list, output_csv)

   