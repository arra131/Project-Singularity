import os
import pandas as pd
import requests
import zipfile
from huggingface_hub import HfApi, Repository

def download_dataset(url, save_dir="datasets"):
    """Download dataset from a URL and handle ZIP files to retain original file names (CSV and Excel)."""
    os.makedirs(save_dir, exist_ok=True)
    file_name = os.path.basename(url.split("?")[0])  # Temporary name for the downloaded file
    file_path = os.path.join(save_dir, file_name)

    # Download the file
    response = requests.get(url)
    if response.status_code == 200:
        with open(file_path, "wb") as f:
            f.write(response.content)
        print(f"Downloaded {file_name} from {url}")

        # Extract if the file is a valid ZIP
        if zipfile.is_zipfile(file_path):
            print(f"Extracting {file_name}...")
            extract_dir = os.path.join(save_dir, os.path.splitext(file_name)[0])  # Folder for extracted files
            os.makedirs(extract_dir, exist_ok=True)
            with zipfile.ZipFile(file_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
                extracted_files = zip_ref.namelist()  # Get the list of files in the archive

            # Filter for Excel files first, then CSV if no Excel files exist
            excel_files = [file for file in extracted_files if file.endswith(".xlsx")]
            csv_files = [file for file in extracted_files if file.endswith(".csv")]

            if excel_files:
                print(f"Found Excel files: {excel_files}")
                return [os.path.join(extract_dir, file) for file in excel_files]
            elif csv_files:
                print(f"No Excel files found. Considering CSV files: {csv_files}")
                return [os.path.join(extract_dir, file) for file in csv_files]
            else:
                print("No valid Excel or CSV files found in the archive.")
                return None
        else:
            print(f"{file_name} is not a valid ZIP file. No extraction performed.")
            return None

    """except requests.exceptions.RequestException as e:
        print(f"Failed to download {url}: {e}")
        return None
    except zipfile.BadZipFile:
        print(f"{file_name} is not a valid ZIP file.")
        return None"""

def extract_metadata(file_path, url):
    """Extract metadata from a dataset (CSV or Excel)."""
    try:
        # Determine file type and load data
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        elif file_path.endswith(".xlsx"):
            df = pd.read_excel(file_path)
        else:
            print(f"Unsupported file type: {file_path}")
            return None

        # Extract metadata
        columns = df.columns.tolist()
        date_column = [col for col in columns if "date" in col.lower() or "time" in col.lower()]
        date_column = date_column[0] if date_column else None
        data_columns = df.select_dtypes(include=["number"]).columns.tolist()
        data_columns = [col for col in data_columns if not col.startswith("Unnamed:")]
        multivariate = len(data_columns) > 1

        # Calculate variance
        if data_columns:
            if multivariate:
                variance = ",".join(
                    [str(round(df[col].var(), 6)) if pd.notnull(df[col].var()) else "None" for col in data_columns]
                )
            else:
                variance = round(df[data_columns[0]].var(), 6) if pd.notnull(df[data_columns[0]].var()) else "None"
        else:
            variance = None

        file_name = os.path.basename(file_path)
        name = file_name.replace(".csv", "").replace(".xlsx", "").replace("_", " ").title()

        return {
            "name": name,
            "url": url,
            "file_name": file_name,
            "date_column": date_column,
            "data_column": ",".join(data_columns),
            "multivariate": multivariate,
            "variance": variance,
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def save_metadata_to_csv(metadata_list, output_csv):
    """Save metadata to a CSV file."""
    df = pd.DataFrame(metadata_list)
    df.to_csv(output_csv, index=False, sep=";")
    print(f"Metadata saved to {output_csv}")

"""def upload_to_huggingface(csv_path, repo_name, token):
    # Upload metadata CSV to Hugging Face.
    api = HfApi()
    try:
        repo_url = api.create_repo(repo_id=repo_name, exist_ok=True, private=False, repo_type='dataset', token=token)
        print(f"Repository created or already exists: {repo_url}")
        
        repo = Repository(local_dir=".", clone_from=repo_name, token=token)
        os.rename(csv_path, os.path.join(repo.local_dir, os.path.basename(csv_path)))
        repo.push_to_hub(commit_message="Uploaded dataset metadata")
        print(f"File successfully uploaded to {repo_name}")
    except Exception as e:
        print(f"Error uploading to Hugging Face: {e}")"""

# Main Script
if __name__ == "__main__":
    # Read input Excel
    input_excel = "UCI_datasets_selenium.xlsx"  # Update with your Excel file path
    output_csv = "test_uci3.csv"  # Output metadata file
    metadata_list = []

    # Load dataset info from Excel
    df = pd.read_excel(input_excel)

    for _, row in df.iterrows():
        name, url, domain, tags = row["Name"], row["Download_Link"], row["Domain"], row["Tags"]
        downloaded_paths = download_dataset(url)

        if downloaded_paths:
            for file_path in downloaded_paths:
                metadata = extract_metadata(file_path, url)
                if metadata:
                    metadata.update({
                        "name": name,
                        "url": url,
                        "domain": domain,
                        "tags": tags,
                    })
                    metadata_list.append(metadata)

    # Save metadata to CSV
    save_metadata_to_csv(metadata_list, output_csv)

    """# Upload to Hugging Face
    token = "your_huggingface_token_here"  # Replace with your Hugging Face token
    repo_name = "your_repo_name_here"  # Replace with your repository name
    upload_to_huggingface(output_csv, repo_name, token)
"""