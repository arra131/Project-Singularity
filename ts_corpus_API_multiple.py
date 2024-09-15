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
            name="Air Temperature",
            version=_VERSION,
            description="Hourly room air temperature",
            kaggle_dataset_name="vitthalmadane/ts-temp-1",
            file_name="./tmp/ts-temp-1/MLTempDataset1.csv",
        ),
        TSCorpusBuilderConfig(
            name="Date Count",
            version=_VERSION,
            description="Daily count",
            kaggle_dataset_name="mukeshmanral/univariate-time-series",
            file_name="./tmp/univariate-time-series/date_count.csv",
        ),
        TSCorpusBuilderConfig(
            name="Gold Price",
            version=_VERSION,
            description="Daily gold prices",
            kaggle_dataset_name="arashnic/learn-time-series-forecasting-from-gold-price",
            file_name="./tmp/gold-price/gold_price_data.csv",
        ),
        TSCorpusBuilderConfig(
            name="Holt Sales Data",
            version=_VERSION,
            description="Monthly sales data for Holt Winters",
            kaggle_dataset_name="vikramamin/holt-winters-forecasting-for-sales-data",
            file_name="./tmp/holt-sales-data/MonthlySales.csv",
        ),
        TSCorpusBuilderConfig(
            name="Energy Consumption",
            version=_VERSION,
            description="Daily energy consumption",
            kaggle_dataset_name="ranja7/electricity-consumption",
            file_name="./tmp/electricity-consumption/daily_consumption.csv",
        ),
        TSCorpusBuilderConfig(
            name="Airline Passengers TSA",
            version=_VERSION,
            description="Monthly passenger count",
            kaggle_dataset_name="prakharmkaushik/airline-passengers-tsa",
            file_name="./tmp/airline-passengers-tsa/AirPassengers.csv",
        ),
        TSCorpusBuilderConfig(
            name="Monthly Sunspots",
            version=_VERSION,
            description="Monthly sunspots",
            kaggle_dataset_name="billykal/monthly-sunspots",
            file_name="./tmp/monthly-sunspots/monthly-sunspots.csv",
        ),
        TSCorpusBuilderConfig(
            name="Time Series Sample",
            version=_VERSION,
            description="time_series_sample_001",
            kaggle_dataset_name="artemig/time-series-sample-001",
            file_name="./tmp/time-series-sample/time_series_sample_001.csv",
        ),
        TSCorpusBuilderConfig(
            name="Air Passenger Data",
            version=_VERSION,
            description="Monthly increase in passengers",
            kaggle_dataset_name="ashfakyeafi/air-passenger-data-for-time-series-analysis",
            file_name="./tmp/air-passenger-data/AirPassengers.csv",
        ),
        TSCorpusBuilderConfig(
            name="Beauty Parlour",
            version=_VERSION,
            description="Daily customers of beauty parlour",
            kaggle_dataset_name="mohamedharris/customers-of-beauty-parlour-time-series",
            file_name="./tmp/beauty-parlour/Customers_Parlour.csv",
        ),
        TSCorpusBuilderConfig(
            name="Dow Jones Composite Average",
            version=_VERSION,
            description="Daily updates of the Dow Jones Composite Average (DCIA)",
            kaggle_dataset_name="joebeachcapital/dow-jones-and-s-and-p500-indices-daily-update",
            file_name="./tmp/DJCA/DJCA.csv",
        ),
        TSCorpusBuilderConfig(
            name="Dow Jones Industrial Average",
            version=_VERSION,
            description="Dow Jones Industrial Average (DJIA)",
            kaggle_dataset_name="joebeachcapital/dow-jones-and-s-and-p500-indices-daily-update",
            file_name="./tmp/DJIA/DJIA.csv",
        ),
        TSCorpusBuilderConfig(
            name="Dow Jones Transportation Average",
            version=_VERSION,
            description="Dow Jones Transportation Average (DJTA)",
            kaggle_dataset_name="joebeachcapital/dow-jones-and-s-and-p500-indices-daily-update",
            file_name="./tmp/DJTA/DJTA.csv",
        ),
        TSCorpusBuilderConfig(
            name="Dow Jones Utility Average",
            version=_VERSION,
            description="Dow Jones Utility Average (DJUA)",
            kaggle_dataset_name="joebeachcapital/dow-jones-and-s-and-p500-indices-daily-update",
            file_name="./tmp/DJUA/DJUA.csv",
        ),
        TSCorpusBuilderConfig(
            name="Standard & Poor 500",
            version=_VERSION,
            description="Standard & Poor 500 (SP500)",
            kaggle_dataset_name="joebeachcapital/dow-jones-and-s-and-p500-indices-daily-update",
            file_name="./tmp/SP500/SP500.csv",
        ),
        TSCorpusBuilderConfig(
            name="Car Sales",
            version=_VERSION,
            description="Monthly car sales",
            kaggle_dataset_name="rassiem/monthly-car-sales",
            file_name="./tmp/car-sales/monthly-car-sales.csv",
        ),
        TSCorpusBuilderConfig(
            name="Malaysia Births",
            version=_VERSION,
            description="Daily number of births in Malaysia",
            kaggle_dataset_name="jylim21/malaysia-public-data",
            file_name="./tmp/births/births.csv",
        ),
        TSCorpusBuilderConfig(
            name="Kaggle search DataSet",
            version=_VERSION,
            description="Kaggle keyword web search",
            kaggle_dataset_name="ankitkalauni/tps-jan22-google-trends-kaggle-search-dataset",
            file_name="./tmp/kaggle/multiTimeline.csv",
        ),
        TSCorpusBuilderConfig(
            name="Fall Mortality",
            version=_VERSION,
            description="Mumber of deaths due to falls in Japan",
            kaggle_dataset_name="gokcegok/falls-mortality-dataset",
            file_name="./tmp/mortality/falls_mortality__dataset.csv",
        ),
        TSCorpusBuilderConfig(
            name="Gold SP",
            version=_VERSION,
            description="Mumber of deaths due to falls in Japan",
            kaggle_dataset_name="yudifaturohman/emas-batangan-antam",
            file_name="./tmp/gold-sp/gold.csv",
        ),
        TSCorpusBuilderConfig(
            name="Undocumented Immigrants",
            version=_VERSION,
            description="Yearly number of immigrants apprehended",
            kaggle_dataset_name="ekayfabio/immigration-apprehended",
            file_name="./tmp/immigration/immigration_apprehended.csv",
        ),
        TSCorpusBuilderConfig(
            name="SPY Data",
            version=_VERSION,
            description="Stock market price",
            kaggle_dataset_name="nekoslevin/spydataa",
            file_name="./tmp/SPY-Data/SPYdata.csv",
        ),
        TSCorpusBuilderConfig(
            name="US GDP",
            version=_VERSION,
            description="Yearly US GDP values",
            kaggle_dataset_name="kapatsa/modelled-time-series",
            file_name="./tmp/USGDP/GDPUS_nsa.csv",
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
        elif 'Time' in df.columns and 'GEMS_GEMS_SPENT' in df.columns:
            df['Time'] = pd.to_datetime(df["Time"], format='%m/%d/%y').dt.strftime('%Y-%m-%d %H:%M:%S')
            dates = df['Time']
            values = df['GEMS_GEMS_SPENT']
        elif 'Datetime' in df.columns and 'Hourly_Temp' in df.columns:
            df['Datetime'] = pd.to_datetime(df["Datetime"]).dt.strftime('%Y-%m-%d %H:%M:%S')
            dates = df['Datetime']
            values = df['Hourly_Temp']
        elif 'Date' in df.columns and 'count' in df.columns:
            df['Date'] = pd.to_datetime(df["Date"]).dt.strftime('%Y-%m-%d %H:%M:%S')
            dates = df['Date']
            values = df['count']
        elif 'Date' in df.columns and 'Value' in df.columns:
            df['Date'] = pd.to_datetime(df["Date"]).dt.strftime('%Y-%m-%d %H:%M:%S')
            dates = df['Date']
            values = df['Value']
        elif 'month' in df.columns and 'sales' in df.columns:
            df['month'] = pd.to_datetime(df["month"]).dt.strftime('%Y-%m-%d %H:%M:%S')
            dates = df['month']
            values = df['sales']
        elif 'Date' in df.columns and 'Energy Consumption (kWh)' in df.columns:
            # To accept both date types - %m/%d/%Y and %m-%d-%Y
            df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y', errors='coerce').fillna(
              pd.to_datetime(df['Date'], format='%m-%d-%Y', errors='coerce'))
            df['Date'] = df['Date'].dt.strftime('%Y-%m-%d %H:%M:%S')
            dates = df['Date']
            values = df['Energy Consumption (kWh)']
        elif 'Timeline' in df.columns and 'Number_of_Passengers' in df.columns:
            df['Timeline'] = pd.to_datetime(df["Timeline"]).dt.strftime('%Y-%m-%d %H:%M:%S')
            dates = df['Timeline']
            values = df['Number_of_Passengers']
        elif 'Month' in df.columns and 'Sunspots' in df.columns:
            df['Month'] = pd.to_datetime(df["Month"]).dt.strftime('%Y-%m-%d %H:%M:%S')
            dates = df['Month']
            values = df['Sunspots']
        elif 'timestamp' in df.columns and 'value' in df.columns:
            df['timestamp'] = pd.to_datetime(df["timestamp"]).dt.strftime('%Y-%m-%d %H:%M:%S')
            dates = df['timestamp']
            values = df['value']
        elif 'Month' in df.columns and '#Passengers' in df.columns:
            df['Month'] = pd.to_datetime(df["Month"]).dt.strftime('%Y-%m-%d %H:%M:%S')
            dates = df['Month']
            values = df['#Passengers']   
        elif 'date' in df.columns and 'Customers' in df.columns:
            df['date'] = pd.to_datetime(df["date"]).dt.strftime('%Y-%m-%d %H:%M:%S')
            dates = df['date']
            values = df['Customers']  
        elif 'DATE' in df.columns and 'DJCA' in df.columns:
            df['DATE'] = pd.to_datetime(df["DATE"]).dt.strftime('%Y-%m-%d %H:%M:%S')
            dates = df['DATE']
            values = df['DJCA']  
        elif 'DATE' in df.columns and 'DJIA' in df.columns:
            df['DATE'] = pd.to_datetime(df["DATE"]).dt.strftime('%Y-%m-%d %H:%M:%S')
            dates = df['DATE']
            values = df['DJIA']
        elif 'DATE' in df.columns and 'DJTA' in df.columns:
            df['DATE'] = pd.to_datetime(df["DATE"]).dt.strftime('%Y-%m-%d %H:%M:%S')
            dates = df['DATE']
            values = df['DJTA']
        elif 'DATE' in df.columns and 'DJUA' in df.columns:
            df['DATE'] = pd.to_datetime(df["DATE"]).dt.strftime('%Y-%m-%d %H:%M:%S')
            dates = df['DATE']
            values = df['DJUA']
        elif 'DATE' in df.columns and 'SP500' in df.columns:
            df['DATE'] = pd.to_datetime(df["DATE"]).dt.strftime('%Y-%m-%d %H:%M:%S')
            dates = df['DATE']
            values = df['SP500']  
        elif 'Month' in df.columns and 'Sales' in df.columns:
            df['Month'] = pd.to_datetime(df["Month"]).dt.strftime('%Y-%m-%d %H:%M:%S')
            dates = df['Month']
            values = df['Sales'] 
        elif 'date' in df.columns and 'births' in df.columns:
            df['date'] = pd.to_datetime(df["date"]).dt.strftime('%Y-%m-%d %H:%M:%S')
            dates = df['date']
            values = df['births'] 
        elif 'Month' in df.columns and 'kaggle' in df.columns:
            df['Month'] = pd.to_datetime(df["Month"]).dt.strftime('%Y-%m-%d %H:%M:%S')
            dates = df['Month']
            values = df['kaggle']   
        elif 'year' in df.columns and 'death' in df.columns:
            df['year'] = pd.to_datetime(df["year"]).dt.strftime('%Y-%m-%d %H:%M:%S')
            dates = df['year']
            values = df['death'] 
        elif 'Year' in df.columns and 'Number' in df.columns:
            df['Year'] = pd.to_datetime(df['Year']).dt.strftime('%Y-%m-%d %H:%M:%S')
            dates = df['Year']
            values = df['Number'] 
        elif 'Trade_date' in df.columns and 'SPY' in df.columns:
            df['Trade_date'] = pd.to_datetime(df['Trade_date']).dt.strftime('%Y-%m-%d %H:%M:%S')
            dates = df['Trade_date']
            values = df['SPY']
        elif 'DATE' in df.columns and 'NA000334Q' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE']).dt.strftime('%Y-%m-%d %H:%M:%S')
            dates = df['DATE']
            values = df['NA000334Q']
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
    datasets_to_load = ["Electric Production", "Air Passengers", "Air Temperature", "Date Count", 
                        "Gold Price", "Holt Sales Data", "Energy Consumption", "Airline Passengers TSA", 
                        "Monthly Sunspots", "Time Series Sample", "Air Passenger Data", "Beauty Parlour", 
                        "Dow Jones Composite Average", "Dow Jones Industrial Average", 
                        "Dow Jones Transportation Average", "Dow Jones Utility Average", 
                        "Standard & Poor 500", "Car Sales", "Malaysia Births", "Kaggle search DataSet",
                        "Fall Mortality", "Undocumented Immigrants", "SPY Data", "US GDP"]
    
    print(datasets_to_load)

    for dataset_name in datasets_to_load:
        dataset = load_dataset('ts_corpus_API_multiple.py', name=dataset_name, trust_remote_code=True)
        print(f"Dataset {dataset_name}:")
        print(dataset['train'])
