# data source - Kaggle, downloaded using Kaggle API

"""Copyright
    License"""

""" Downloading single time series dataset from kaggle website """

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import datasets
from pandas.tseries.frequencies import to_offset

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
    target_fields: Optional[List[str]] = None               # target variables for prediction
    feat_dynamic_real_fields: Optional[List[str]] = None    # list of additional time varying features
    multivariate: bool = False                              # univariaate or multivariate
    rolling_evaluations: int = 1    

class TSCorpus(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version('1.0.0')
    BUILDER_CONFIG_CLASS = TSCorpusBuilderConfig
    BUILDER_CONFIGS = [
        TSCorpusBuilderConfig(
            name = "Time series analysis",
            version=VERSION,
            description="univariate data on electricity production",
            url=None,
            file_name="electric-production\Electric_Production.csv",
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            version=self.config.version
        )
    
    def _split_generators(self, dl_manager):
        # Use local file directly
        file_path = self.config.file_name
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": file_path}
            )
        ]

    """Process 1 - Iterates over each row and extracts the date and value"""
    
    '''def _generate_examples(self, filepath):
        # Step 1: Load the data
        df = pd.read_csv(filepath)
        
        # Step 2: Process the data
        for idx, row in df.iterrows():
            # Transforming the data: parsing date and extracting value
            yield idx, {
                "date": datetime.strptime(row["DATE"], "%d-%m-%Y"),  # column is named "DATE"
                "value": row["Value"],  # column is named "VALUE"
            }'''

    """Process 2 - Normalizes the values and adds a feature to indicate day of the week for each row"""

    def _generate_examples(self, filepath):
        # Step 1: Load the data
        df = pd.read_csv(filepath)
        
        # Step 2: Normalize the "Value" column
        min_value = df["Value"].min()
        max_value = df["Value"].max()
        df["Normalized_Value"] = (df["Value"] - min_value) / (max_value - min_value)
        
        # Step 3: Add a new feature for the day of the week
        df["Day_of_Week"] = df["DATE"].apply(lambda x: datetime.strptime(x, "%d-%m-%Y").strftime('%A'))
        
        # Step 4: Process the data
        for idx, row in df.iterrows():
            # Transforming the data: parsing date, normalized value, and extracting day of the week
            yield idx, {
                "date": datetime.strptime(row["DATE"], "%d-%m-%Y"),
                "value": row["Value"],
                "normalized_value": row["Normalized_Value"],
                "day_of_week": row["Day_of_Week"],
            }

if __name__ == "__main__":
    from datasets import load_dataset
    first_dataset = load_dataset('ts_corpus_API.py', name="Time series analysis", trust_remote_code=True)
    print(first_dataset)

    print(first_dataset['train'][:5])