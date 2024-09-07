from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import pandas as pd
import datasets
import kaggle

_VERSION = "1.0.0"
_DESCRIPTION = "Time Series Corpus"
_CITATION = "Suitable citation"

@dataclass
class TSCorpusBuilderConfig(datasets.BuilderConfig):
    file_name: Optional[str] = None                         # name of the dataset file
    kaggle_dataset_name: Optional[str] = None               # Kaggle dataset name in 'username/dataset-name' format
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
            name="Electric Production",
            version=_VERSION,
            description="univariate data on electricity production",
            kaggle_dataset_name="kandij/electric-production",
            file_name="./tmp/electric_production/Electric_Production.csv",
        ),
        TSCorpusBuilderConfig(
            name="Air Passengers",
            version=_VERSION,
            description="univariate data on air passengers",
            kaggle_dataset_name="rakannimer/air-passengers",
            file_name="./tmp/air_passengers/AirPassengers.csv",
        ),
        TSCorpusBuilderConfig(
            name="Train Count",
            version=_VERSION,
            description="univariate data on hourly train count",
            kaggle_dataset_name="wolfmedal/time-series-notes-data-set",
            file_name="./tmp/time_series_notes/Train_SU63ISt.csv",
        ),
        TSCorpusBuilderConfig(
            name="Ad Interval",
            version=_VERSION,
            description="Hourly interval of Ads",
            kaggle_dataset_name="wolfmedal/time-series-notes-data-set",
            file_name="./tmp/time_series_notes/ads.csv",
        ),
        TSCorpusBuilderConfig(
            name="Currency",
            version=_VERSION,
            description="Daily expenditure of virtual currency",
            kaggle_dataset_name="wolfmedal/time-series-notes-data-set",
            file_name="./tmp/time_series_notes/currency.csv",
        ),
        TSCorpusBuilderConfig(
            name="Air Temperature",
            version=_VERSION,
            description="Hourly room air temperature",
            kaggle_dataset_name="vitthalmadane/ts-temp-1",
            file_name="./tmp/ts-temp-1/MLTempDataset1.csv",
        )
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            version=self.config.version
        )

    def _split_generators(self, dl_manager):
        # Authenticate and download the dataset from Kaggle
        kaggle.api.authenticate()

        if self.config.file_name is None:
            raise ValueError("file_name must be specified in the configuration.")

        download_path = str(Path(self.config.file_name).parent)
        kaggle.api.dataset_download_files(
            dataset=self.config.kaggle_dataset_name,
            path=download_path,
            unzip=True
        )

        file_path = self.config.file_name
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": file_path}
            )
        ]

    def _generate_examples(self, filepath):
        df = pd.read_csv(filepath)

        # Adjust column names according to the dataset
        if 'DATE' in df.columns and 'Value' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE']).dt.strftime('%Y-%m-%d %H:%M:%S')
            dates = df['DATE']
            values = df['Value']
        elif 'Month' in df.columns and '#Passengers' in df.columns:
            df['Month'] = pd.to_datetime(df['Month']).dt.strftime('%Y-%m-%d %H:%M:%S')
            dates = df['Month']
            values = df['#Passengers']
        elif 'Datetime' in df.columns and 'Count' in df.columns:
            df['Datetime'] = pd.to_datetime(df["Datetime"], dayfirst=True).dt.strftime('%Y-%m-%d %H:%M:%S')
            dates = df['Datetime']
            values = df['Count']
        elif 'Time' in df.columns and 'Ads' in df.columns:
            df['Time'] = pd.to_datetime(df["Time"]).dt.strftime('%Y-%m-%d %H:%M:%S')
            dates = df['Time']
            values = df['Ads']
        elif 'Time' in df.columns and 'GEMS_GEMS_SPENT' in df.columns:
            df['Time'] = pd.to_datetime(df["Time"], format='%m/%d/%y').dt.strftime('%Y-%m-%d %H:%M:%S')
            dates = df['Time']
            values = df['GEMS_GEMS_SPENT']
        elif 'Datetime' in df.columns and 'Hourly_Temp' in df.columns:
            df['Datetime'] = pd.to_datetime(df["Datetime"]).dt.strftime('%Y-%m-%d %H:%M:%S')
            dates = df['Datetime']
            values = df['Hourly_Temp']
        else:
            raise ValueError("Unexpected column names in the dataset")

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

    # Load each dataset
    datasets_to_load = ["Electric Production", "Air Passengers", "Train Count", "Ad Interval", "Currency", "Air Temperature"]
    
    print(datasets_to_load)

    for dataset_name in datasets_to_load:
        dataset = load_dataset('ts_corpus_API_multiple.py', name=dataset_name, trust_remote_code=True)
        print(dataset)
        print(dataset['train'])