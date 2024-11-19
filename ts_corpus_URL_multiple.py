# Import necessary dependencies

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union, Any
import pandas as pd
import datasets
import requests
from tqdm import tqdm
from huggingface_hub import hf_hub_download

_VERSION = "1.0.0"
_DESCRIPTION = "Time Series Corpus containing multiple univariate and multivariate datasets from various web sources"
_CITATION = "Provide a suitable citation here - Later"

# Load the csv file from Hugging Face repo 
def load_datasets_config():
    # Specify repo_type="dataset" to ensure it looks in the dataset repository
    csv_file_path = hf_hub_download(repo_id="ddrg/time-series-datasets", filename="config_datasets_URL.csv", repo_type="dataset")
    config_df = pd.read_csv(csv_file_path, delimiter=';')  # Use semicolon as delimiter
    config_dict = {}

    for _, row in config_df.iterrows():
        # Handle multivariate data columns by splitting on commas and stripping whitespace
        data_columns = [col.strip() for col in row['data_column'].split(',')] if ',' in row['data_column'] else row['data_column'].strip()
        multivariate = str(row['multivariate']).strip().upper() == 'TRUE'

        config_dict[row['name'].strip()] = {
            "url": row['url'].strip(),
            "file_name": row['file_name'].strip(),
            "date_column": row['date_column'].strip(),
            "data_column": data_columns,
            "multivariate": multivariate,
            "domain": row['domain'].strip()
        }

    return config_dict

@dataclass
class TSCorpusBuilderConfig(datasets.BuilderConfig):
    datasets_config: Optional[Dict[str, Dict[str, Any]]] = None  # Dictionary to hold dataset information

class TSCorpus(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version(_VERSION)

    # Use CSV configuration loaded from Hugging Face
    BUILDER_CONFIGS = [
        TSCorpusBuilderConfig(
            name="UNIVARIATE",
            version=datasets.Version(_VERSION),
            description="Multiple univariate and multivariate datasets",
            datasets_config=load_datasets_config()
        )
    ]

    def _info(self):
        """Returns the dataset metadata"""
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            version=self.VERSION,
            features=datasets.Features({
                "dataset_name": datasets.Value("string"),
                "date": datasets.Sequence(datasets.Value("string")),  # list of date strings
                "value": datasets.Sequence(datasets.Sequence(datasets.Value("float32"))),  # for multivariate data
                "domain": datasets.Value("string")  
            })
        )

    def _split_generators(self, dl_manager):
        """Download all datasets and split them"""
        downloaded_files = {}
        for dataset_name, dataset_info in tqdm(self.config.datasets_config.items(), desc="Downloading datasets", unit="dataset"):
            url = dataset_info["url"]
            dest = Path(dataset_info["file_name"])
            dest.parent.mkdir(parents=True, exist_ok=True)

            # Download the file
            response = requests.get(url)
            response.raise_for_status()  # Ensure the request was successful
            dest.write_bytes(response.content)
            downloaded_files[dataset_name] = dest

        return [datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepaths": downloaded_files})]

    def _generate_examples(self, filepaths):
        """Generate examples in the requested format"""
        for dataset_name, filepath in filepaths.items():
            dataset_info = self.config.datasets_config[dataset_name]
    
            try:
                if Path(filepath).exists():
                    df = pd.read_csv(filepath, on_bad_lines='skip')
                else:
                    print(f"File {filepath} does not exist.")
                    continue
    
                # Process dates in a standard format
                if dataset_info["date_column"] in df.columns:
                    df[dataset_info["date_column"]] = pd.to_datetime(df[dataset_info["date_column"]], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
                    dates = df[dataset_info["date_column"]].tolist()    # Create a list of dates
                else:
                    raise ValueError(f"Specified date column '{dataset_info['date_column']}' not found in {dataset_name}.")
    
                # Process values for univariate and multivariate data
                data_columns = dataset_info["data_column"]
    
                if isinstance(data_columns, list):  # Multivariate case
                    # Create a list of lists, one list per column
                    values = [
                        df[col].dropna().astype(float).tolist() if col in df.columns else []
                        for col in data_columns
                    ]
                else:  # Univariate case
                    if data_columns in df.columns:
                        # Single column as a list of values wrapped in another list
                        values = [df[data_columns].dropna().astype(float).tolist()]
                    else:
                        raise ValueError(f"Specified data column '{data_columns}' not found in {dataset_name}.")
    
                # Ensure `dates` and `values` align
                if len(dates) != len(values[0]):  # only check the length of the first column
                    print(f"Warning: Mismatch in dates and values length for {dataset_name}. Skipping this dataset.")
                    continue

                domain = dataset_info["domain"]  # include domain field in the output
    
                # Yield the processed data for each dataset
                yield dataset_name, {               
                    "dataset_name": dataset_name,   # String
                    "date": dates,                  # List
                    "value": values,                # List of lists: one per column
                    "domain": domain                # String
                }
    
            except Exception as e:
                print(f"Error processing {dataset_name} ({filepath}): {e}")
                continue