import os
import pandas as pd
import requests
from huggingface_hub import HfApi, Repository

def download_dataset(url, save_dir="datasets"):
    """Download dataset from a URL."""
    os.makedirs(save_dir, exist_ok=True)
    file_name = os.path.basename(url.split("?")[0])  # Extract file name from URL
    file_path = os.path.join(save_dir, file_name)
    
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded {file_name} from {url}")
        return file_path
    else:
        print(f"Failed to download {url}. Status code: {response.status_code}")
        return None

def map_domain(name):
    """Map dataset name to a domain."""
    print(f"Mapping domain for name: {name}")  # Log the name being passed
    domain_mapping = {
        "bus": "transport",
        "airline-passengers": "transport",
        "videostats": "social media",
        "data": "general",
        "Test set 1": "climate"
        # Add more mappings as needed
    }
    for keyword, domain in domain_mapping.items():
        if keyword.lower() in name.lower():
            return domain
    return "Unknown"  # Default domain if no match is found

def clean_column_list(column_list):
    """Format column list as a clean string with no spaces after commas."""
    return ",".join(column_list)

def extract_metadata(file_path, url):
    """Extract metadata from the dataset."""
    try:
        # Load the dataset to analyze
        df = pd.read_csv(file_path)
        columns = df.columns.tolist()

        # Detect date columns
        date_column = [col for col in columns if "date" in col.lower() or "time" in col.lower()]
        date_column = date_column[0] if date_column else None

        # Detect data columns (numeric columns for simplicity)
        data_columns = df.select_dtypes(include=["number"]).columns.tolist()
        data_columns = [col for col in data_columns if not col.startswith("Unnamed:")]  # Remove "Unnamed:" columns

        # Check if it's multivariate (more than one numeric column)
        multivariate = len(data_columns) > 1

        # Calculate variance
        if data_columns:
            if multivariate:
                # Calculate variance for each data column, join as a string
                variance = ",".join(
                    [str(round(df[col].var(), 6)) if pd.notnull(df[col].var()) else "None" for col in data_columns]
                )
            else:
                # Calculate variance for the single data column
                variance = round(df[data_columns[0]].var(), 6) if pd.notnull(df[data_columns[0]].var()) else "None"
        else:
            variance = None

        # Extract file name and name
        file_name = os.path.basename(file_path)
        name = file_name.replace(".csv", "").replace("_", " ").title()

        # Map domain
        domain = map_domain(name)

        return {
            "name": name,
            "url": url,
            "file_name": file_name,
            "date_column": date_column,
            "data_column": clean_column_list(data_columns),
            "multivariate": multivariate,
            "variance": variance,
            "domain": domain,
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def save_metadata_to_csv(metadata_list, output_csv):
    """Save metadata to a CSV file."""
    df = pd.DataFrame(metadata_list)
    df.to_csv(output_csv, index=False, sep=";")
    print(f"Metadata saved to {output_csv}")

def upload_to_huggingface(csv_path, repo_name, token):
    """Upload metadata CSV to Hugging Face."""
    api = HfApi()
    try:
        repo_url = api.create_repo(repo_id=repo_name, exist_ok=True, private=False, repo_type='dataset', token=token)
        print(f"Repository created or already exists: {repo_url}")
        
        repo = Repository(local_dir=".", clone_from=repo_name, token=token)
        os.rename(csv_path, os.path.join(repo.local_dir, os.path.basename(csv_path)))
        repo.push_to_hub(commit_message="Uploaded dataset metadata")
        print(f"File successfully uploaded to {repo_name}")
    except Exception as e:
        print(f"Error uploading to Hugging Face: {e}")

# Main Script
if __name__ == "__main__":
    # List of URLs
    urls = [
        "https://zenodo.org/records/12665355/files/bus.csv?download=1",
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv",
        "https://huggingface.co/datasets/jettisonthenet/timeseries_trending_youtube_videos_2019-04-15_to_2020-04-15/resolve/main/videostats.csv",
        "https://huggingface.co/datasets/zaai-ai/time_series_datasets/resolve/main/data.csv",
        "https://huggingface.co/datasets/Oumar199/Nalohou_climatic_time_series/resolve/main/test_set_1.csv"
    ]

    # Output CSV file
    output_csv = "dataset_metadata_url.csv"

    # Process datasets
    metadata_list = []
    for url in urls:
        dataset_path = download_dataset(url)
        if dataset_path:
            metadata = extract_metadata(dataset_path, url)
            if metadata:
                metadata_list.append(metadata)

    # Save metadata to CSV
    save_metadata_to_csv(metadata_list, output_csv)
