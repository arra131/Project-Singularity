from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import datasets
import requests

_VERSION = "1.0.0"
_DESCRIPTION = "Time Series Corpus"
_CITATION = "Suitable citation"

@dataclass
class TSCorpusBuilderConfig(datasets.BuilderConfig):
    file_name: Optional[str] = None                         # name of the dataset file
    url: Optional[str] = None                               # source URL of the dataset
    prediction_length: Optional[str] = None                 # length of prediction horizon
    item_id_column: Optional[str] = None                    # column with unique identifiers
    data_column: Optional[str] = None                       # column with actual time series data points
    date_column: Optional[str] = None                       # column containing the date information
    target_fields: Optional[List[str]] = None               # target variables for prediction
    feat_dynamic_real_fields: Optional[List[str]] = None    # list of additional time varying features
    multivariate: bool = False                              # univariate or multivariate
    rolling_evaluations: int = 1    

class TSCorpus(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version('1.0.0')
    BUILDER_CONFIG_CLASS = TSCorpusBuilderConfig
    
    BUILDER_CONFIGS = [
        TSCorpusBuilderConfig(
            name="Bus route identification",
            version=VERSION,
            description="Univariate data on unique identification of bus routes",
            url="https://zenodo.org/records/12665355/files/bus.csv?download=1",
            file_name="./tmp/bus.csv",
            data_column="validations_per_hour",  # specify the column for univariate data
            date_column="date",  # specify the date column
            multivariate=False
        ),      
		TSCorpusBuilderConfig(
            name="timeseries_trending_youtube_videos",
            version=VERSION,
            description="timeseries_trending_youtube_videos_2019-04-15_to_2020-04-15", url="https://huggingface.co/datasets/jettisonthenet/timeseries_trending_youtube_videos_2019-04-15_to_2020-04-15/resolve/main/videostats.csv",
            file_name="./tmp/videostats.csv",
            target_fields=["likes", "dislikes", "views"],  # specify target fields for multivariate data
            date_column="timestamp",  # specify the date column
            multivariate=True
        ),
        TSCorpusBuilderConfig(
            name="Inflation and visits dataset",
            version=VERSION,
            description="Multivariate dataset on Inflation Rate and Visits",
            url="https://huggingface.co/datasets/zaai-ai/time_series_datasets/resolve/main/data.csv",
            file_name="./tmp/inflation_visits.csv",
            target_fields=["Inflation_Rate", "visits"],  # specify target fields
            date_column="Date",  # specify the date column
            multivariate=True
        ),     
		TSCorpusBuilderConfig(
            name="Controlled anomalies time series",
            version=VERSION,
            description="Multivariate dataset with columns arnd, bso1, and cso1",
            url="https://huggingface.co/datasets/patrickfleith/controlled-anomalies-time-series-dataset/resolve/main/data.csv",
            file_name="./tmp/controlled_anomalies.csv",
            target_fields=["arnd", "bso1", "cso1"],  # specify the target fields for multivariate data
            date_column="timestamp",  # specify the date column
            multivariate=True
        ),
        TSCorpusBuilderConfig(
            name="Dynamical System",
            version=VERSION,
            description="Dynamical System Multivariate Time Series",
            url="https://zenodo.org/records/11526904/files/data.csv?download=1",
            file_name="./tmp/data.csv",
            target_fields=["arnd", "asin2"],  # specify the target fields for multivariate data
            date_column="timestamp",  # specify the date column
            multivariate=True
        ),
        TSCorpusBuilderConfig(
            name="S&P500 Mean Correlation",
            version=VERSION,
            description="S&P500 Mean Correlation Time Series (1992-2012)",
            url="https://zenodo.org/records/8167592/files/Financial_Time_Series_Centered_Interval.csv?download=1",
            file_name="./tmp/Financial_Time_Series_Centered_Interval.csv",
            target_fields=["Mean_Correlation"],  # specify the target fields for multivariate data
            date_column="Centre_of_Interval",  # specify the date column
            multivariate=False
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            version=self.config.version
        )
    
    def _split_generators(self, dl_manager):
        # Download the file
        url = self.config.url
        dest = Path(self.config.file_name)
        dest.parent.mkdir(parents=True, exist_ok=True)

        response = requests.get(url)
        response.raise_for_status()  # Ensure the request was successful
        dest.write_bytes(response.content)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": str(dest)}
            )
        ]

    def _generate_examples(self, filepath):
        df = pd.read_csv(filepath)

        # Extract date values
        if self.config.date_column and self.config.date_column in df.columns:
            df[self.config.date_column] = pd.to_datetime(df[self.config.date_column]).dt.strftime('%Y-%m-%d %H:%M:%S')
            dates = df[self.config.date_column].tolist()
        else:
            raise ValueError(f"Specified date column '{self.config.date_column}' not found in the dataset.")
        
        # Extract values for target fields or data column
        if self.config.multivariate or self.config.target_fields:
            # Collect values for each target field separately
            values = [df[field].tolist() for field in self.config.target_fields]
        else:
            # Handle univariate data
            if self.config.data_column and self.config.data_column in df.columns:
                values = [df[self.config.data_column].tolist()]  # Wrap in a list to keep it consistent
            else:
                raise ValueError(f"Specified data column '{self.config.data_column}' not found in the dataset.")

        # Prepare the final dictionary
        result = {
            "date": dates,
            "value": values  # List of lists for each feature
        }

        # Yield the result as requested
        yield 0, result



if __name__ == "__main__":
    from datasets import load_dataset
    # Load all datasets
    for config in TSCorpus.BUILDER_CONFIGS:
        dataset = load_dataset('ts_corpus_URL_multiple.py', name=config.name, trust_remote_code=True)
        print(f"Dataset {config.name}:")
        #print(dataset['train'][0])
        df = pd.DataFrame(dataset['train'])
        # Use to_string to print the entire DataFrame without truncation
        print(df.to_string())