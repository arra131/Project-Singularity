# data source - URL

"""Copyright
    License"""

""" Downloading single univariate time series dataset from Zenodo """

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
    target_fields: Optional[List[str]] = None               # target variables for prediction
    feat_dynamic_real_fields: Optional[List[str]] = None    # list of additional time varying features
    multivariate: bool = False                              # univariate or multivariate
    rolling_evaluations: int = 1    

class TSCorpus(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version('1.0.0')
    BUILDER_CONFIG_CLASS = TSCorpusBuilderConfig
    BUILDER_CONFIGS = [
        TSCorpusBuilderConfig(
            name = "Bus route identification",
            version=VERSION,
            description="univariate data on unique identification of bus routes",
            url="https://zenodo.org/records/12665355/files/bus.csv?download=1",
            file_name="./tmp/bus.csv",
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
        dest.write_bytes(response.content)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": dest}
            )
        ]

    def _generate_examples(self, filepath):
        df = pd.read_csv(filepath)
    
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d %H:%M:%S')

        dates = df['date']
        values = df['validations_per_hour']  # Assuming 'validations_per_hour' is the correct column name
    
        # Convert values to a list of lists
        values = [values.tolist()]
    
        # Convert dates to a list
        dates = dates.tolist()
    
        # Yield the result with values as a list of lists
        yield 0, {
            "date": dates,
            "value": values
    }

if __name__ == "__main__":
    from datasets import load_dataset
    first_dataset = load_dataset('ts_corpus_URL.py', name="Bus route identification", trust_remote_code=True)

    print(first_dataset['train'][0])