## Creating a Large Corpus of Time-Series Datasets and Publishing an Interface on Hugging Face ##

This project aims to create a comprehensive corpus of time-series datasets and provide an easy-to-use interface on Hugging Face. The datasets are collected from Kaggle, preprocessed, and made accessible via Hugging Face's `datasets` library, enabling researchers and developers to load them effortlessly for machine learning and statistical analysis.

### Repository Structure ###
- `requirement.txt`           # Dependencies required to download the corpus
- `Kaggle_API_setup.md`       # Instructions to setup Kaggle API to download the datasets
- `selenium_kaggle.py`        # Selenium script to automate the scraping of dataset information from Kaggle
- `selenium_uci.edu.py`       # Selenium script to automate the scraping of dataset information from uci.edu
- `CSVgenerationAPI.py`       # This scripts downloads Kaggle datasets, inspects them for metadata, and saves the metadata to a CSV file
- `CSVcleaning.py`            # Cleans CSVs to remove all missining dates and replaces missing tags with 'unknown'
- `DataLoader_Builder.py`     # Dataloader builder script to automate dataset retrieval, processing, and structuring of the downloaded datasets
- `README.md`                 # Documentation file

### Respoitory Usage

The dataset can be loaded using the Hugging Face `datasets` library:

```python

from datasets import load_dataset

dataset = load_dataset("ddrg/kaggle-time-series-datasets", "TIME_SERIES", trust_remote_code = TRUE)


```
