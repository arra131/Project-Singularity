# Instructions to setup Kaggle API 

Follow the below instructions to setup the Kaggle API for downloading datasets from Kaggle 

## Step 1: Install the dependencies

Ensure the kaggle and datasets libraries are installed

`pip install kaggle datasets`

## Step 2: Create .kaggle folder

Create a folder in the root directory in order to save the API 

`mkdir ~/.kaggle`

## Step 3: Get kaggle API token 

* Create an account on kaggle.com. 
* Login to your account and go to settings.
* Click on 'Create New API Token'. This will download a kaggle.json file with your username and API key. 

## Step 4: Place the file in correct location

Move the downloaded kaggle.json file from the current location to ~/.kaggle folder. 

#### Your Kaggle API is set and ready to be used. Run the script `<final_script.py>` to download the datasets from Kaggle.
