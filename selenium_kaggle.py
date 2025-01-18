# Ready and working

"""Selenium script to automate the collection of dataset names and URLs from Kaggle"""

"""
    1. Open kaggle website and navigate to datasets tab
    2. Filter the time series datasets
    3. Get names and URLs of all datasets
    4. Open every URL and get their tags
    5. Save Names, URLs and tags (Metadata) in an excel file"""

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException, TimeoutException
from webdriver_manager.chrome import ChromeDriverManager
import pandas as pd
import time
import random

start_time = time.time()

options = Options()
# options.add_experimental_option("detach", True)  # Keeps the browser open after the script ends

# Initialize the WebDriver
driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

# Step 1: Open Kaggle and navigate to datasets
driver.get("https://www.kaggle.com/")
driver.maximize_window()
time.sleep(5)

# Step 2: Click on "Datasets"
input_element = driver.find_element(By.LINK_TEXT, "Datasets")
input_element.click()
time.sleep(5)

# Step 3: Search for "time series"
search_element = driver.find_element(By.XPATH, "//input[@placeholder = 'Search datasets']")
search_element.send_keys("time series")
search_element.send_keys("\n")  # Press Enter
time.sleep(5)

# Step 4: Scrape dataset names, links, and tags (limit to 3 pages)
all_datasets = []
page_counter = 0
count = 0
# while page_counter < 2:  # Limit to 1 page
while page_counter<15:  # For all pages
    # Find all dataset containers
    dataset_containers = driver.find_elements(By.CSS_SELECTOR, "div.sc-kLJHhQ.ithYPd.km-listitem--large")
    for container in dataset_containers:
        try:
            name = container.find_element(By.CSS_SELECTOR, "div.sc-eauhAA.sc-fXwCOG").text
            link = container.find_element(By.CSS_SELECTOR, "a.sc-lgprfV").get_attribute("href")
            all_datasets.append({"Name": name, "Link": link, "datasetID":"", "Tags": ""})
            count += 1
            print(count, f"Dataset: {name} - {link}")
        except Exception as e:
            print(f"Error occurred while extracting dataset details: {e}")
    page_counter += 1
    # Move to the next page
    try:
        next_button_element = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button[aria-label='Go to next page']"))
        )
        driver.execute_script("arguments[0].scrollIntoView(true);", next_button_element)
        next_button_element.click()
        time.sleep(5)  # Wait for the next page to load
    except Exception as e:
        print(f"No more pages or error occurred: {e}")
        break

# Step 5: Visit each dataset link and scrape tags
count = 0
for dataset in all_datasets:
    try:
        driver.get(dataset["Link"])
        time.sleep(5)

        # Locate "Tags" section and extract tags
        tags_header = driver.find_element(By.XPATH, "//h2[text()='Tags']")
        tags_elements = tags_header.find_elements(By.XPATH, "./following-sibling::div//span[contains(@class, 'sc-eUlrpB')]")
        tags = ", ".join([tag.text.strip() for tag in tags_elements])
        dataset["Tags"] = tags if tags else "No tags found"
        count += 1
        time.sleep(random.uniform(2, 5))
        print(count, f"Tags for '{dataset['Name']}': {dataset['Tags']}")

    except (NoSuchElementException, TimeoutException):
        dataset["Tags"] = "No tags found"
        print(f"Tags for '{dataset['Name']}': No tags found")
    except Exception as e:
        dataset["Tags"] = f"Error: {e}"
        print(f"Tags for '{dataset['Name']}': Error: {e}")

# Step 6: Change the links to dataset IDs
df = pd.DataFrame(all_datasets)
df['datasetID'] = df['Link'].str.replace('https://www.kaggle.com/datasets/', '', regex=False)
 
# Step 7: Save the dataset metadata to an Excel file
df.to_excel("kaggle_datasets_with_tags.xlsx", index=False, columns=["Name", "Link", "datasetID", "Tags"])

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.2f} seconds")

print("Scraping complete. Data saved to 'kaggle_datasets_with_tags.xlsx'.")
