# data source - Kaggle, downloaded using Kaggle API to local and imported from local to process

"""Copyright
    License"""

""" Downloading single univariate time series dataset from kaggle """

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd
import datasets
from pandas.tseries.frequencies import to_offset
import kaggle

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
    kaggle.api.authenticate()
    name = 'kandij/electric-production'
    dest = 'D:\Master Thesis\Code'
    kaggle.api.dataset_download_files(name, path=dest, unzip=True)
    VERSION = datasets.Version('1.0.0')
    BUILDER_CONFIG_CLASS = TSCorpusBuilderConfig
    BUILDER_CONFIGS = [
        TSCorpusBuilderConfig(
            name = "Time series analysis",
            version=VERSION,
            description="univariate data on electricity production",
            url=None,
            file_name="Electric_Production.csv",
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            version=self.config.version
        )
    
    def _split_generators(self, dl_manager):
        file_path = self.config.file_name
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": file_path}
            )
        ]

    def _generate_examples(self, filepath):
        df = pd.read_csv(filepath)
        
        df['DATE'] = pd.to_datetime(df['DATE']).dt.strftime('%Y-%m-%d %H:%M:%S')

        dates = df['DATE']
        values = df['Value']
        
        # data as a single row
        yield 0, {
            "date": dates,
            "value": values
        }

if __name__ == "__main__":
    from datasets import load_dataset
    first_dataset = load_dataset('ts_corpus_API_v2.py', name="Time series analysis", trust_remote_code=True)
    # print(first_dataset)

    print(first_dataset['train'][:1])