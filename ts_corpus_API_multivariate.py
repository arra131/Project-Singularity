from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any, List
import pandas as pd
import datasets
import subprocess
from tqdm import tqdm
from huggingface_hub import hf_hub_download

_VERSION = "1.0.0"
_DESCRIPTION = "Time Series Corpus containing multiple univariate and multivariate datasets"
_CITATION = "Provide a suitable citation here"

# Define the base directory for dataset downloads
BASE_DIR = Path("./KaggleData")
BASE_DIR.mkdir(parents=True, exist_ok=True)  # Ensure the base directory exists

# Load dataset configuration from Hugging Face
def load_datasets_config():
    # Download the config_datasets.csv from the Hugging Face Hub
    csv_file_path = hf_hub_download(repo_id="ddrg/kaggle-time-series-datasets", filename="kaggle_data_config.csv", repo_type="dataset")
    config_df = pd.read_csv(csv_file_path, delimiter=';')
    config_dict = {}

    for _, row in config_df.iterrows():
        # Create the file path within the BASE_DIR
        file_path = BASE_DIR / Path(row['file_name']).name  # Ensure all files are saved to BASE_DIR

        data_columns = [col.strip() for col in row['data_column'].split(',')] if ',' in row['data_column'] else row['data_column'].strip()
        multivariate = str(row['multivariate']).strip().upper() == 'TRUE'

        config_dict[row['name'].strip()] = {
            "kaggle_dataset": row['kaggle_dataset'].strip(),
            "file_name": str(file_path),
            "date_column": row['date_column'].strip(),
            "data_column": data_columns,
            "multivariate": multivariate
        }
    return config_dict

@dataclass
class TimeSeriesDatasetConfig(datasets.BuilderConfig):
    datasets_config: Optional[Dict[str, Dict[str, Any]]] = None

class TimeSeriesDataset(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version(_VERSION)

    # Load dataset configuration from the CSV file in the Hugging Face repository
    BUILDER_CONFIGS = [
        TimeSeriesDatasetConfig(
            name="UNIVARIATE",
            version=datasets.Version(_VERSION),
            description="Multiple univariate datasets",
            datasets_config=load_datasets_config()
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            version=self.VERSION
        )

    def _split_generators(self, dl_manager):
        """Download all datasets and return the train split."""
        downloaded_files = {}
    
        for dataset_name, dataset_info in tqdm(self.config.datasets_config.items(), desc="Downloading datasets", unit="dataset"):
            dest = Path(dataset_info["file_name"])
            dest.parent.mkdir(parents=True, exist_ok=True)  # Ensure the directory exists
    
            # Download the dataset using Kaggle API (without unzipping)
            kaggle_command = f"kaggle datasets download -d {dataset_info['kaggle_dataset']} -p {dest.parent} --force --unzip"
            try:
                result = subprocess.run(kaggle_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                print(result.stdout.decode())  # Print the output of the download command
                
                # Check if CSV exists directly; if not, skip
                if dest.with_suffix('.csv').exists():
                    downloaded_files[dataset_name] = str(dest.with_suffix('.csv'))
                    print(f"Successfully downloaded {dataset_name} to {dest.with_suffix('.csv')}.")
                else:
                    print(f"No CSV found for {dataset_name} at expected location: {dest.with_suffix('.csv')}")
                    continue  # Skip if no CSV file found

            except subprocess.CalledProcessError as e:
                print(f"Failed to download {dataset_name}: {e}\n{e.stderr.decode()}")
                continue  # Skip to the next dataset if download fails
    
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepaths": downloaded_files}
            )
        ]

    def _generate_examples(self, filepaths):
        """Generate examples in the requested format."""
        all_datasets = []

        for dataset_name, filepath in filepaths.items():
            dataset_info = self.config.datasets_config[dataset_name]

            try:
                if Path(filepath).exists():
                    df = pd.read_csv(filepath, on_bad_lines='skip')
                else:
                    print(f"File {filepath} does not exist.")
                    continue

                if dataset_info["date_column"] in df.columns:
                    df[dataset_info["date_column"]] = pd.to_datetime(df[dataset_info["date_column"]]).dt.strftime('%Y-%m-%d %H:%M:%S')
                    dates = df[dataset_info["date_column"]].tolist()
                else:
                    raise ValueError(f"Specified date column '{dataset_info['date_column']}' not found in the dataset.")

                data_columns = dataset_info["data_column"]
                values = []

                if isinstance(data_columns, list):
                    for col in data_columns:
                        if col in df.columns:
                            values.append(df[col].tolist())
                        else:
                            raise ValueError(f"Specified data column '{col}' not found in the dataset.")
                else:
                    if data_columns in df.columns:
                        values.append(df[data_columns].tolist())
                    else:
                        raise ValueError(f"Specified data column '{data_columns}' not found in the dataset.")

                # Store the dataset information in the desired format
                all_datasets.append({
                    "name": dataset_name,
                    "date": dates,
                    "value": values  # Store values as a list of lists
                })

            except Exception as e:
                print(f"Error processing {dataset_name} ({filepath}): {e}")
                continue

        # Yield all datasets
        for idx, data in enumerate(all_datasets):
            yield idx, data
