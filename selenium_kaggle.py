"""Selenium script to automate the collection of dataset names and URLs from Kaggle"""

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By     # to locate an element
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import StaleElementReferenceException
import pandas as pd
import time

options = Options()
options.add_experimental_option("detach", True)     # leaves the browser open after the task is done

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()),
                          options=options)

# Step 1: Open the website in the browser
driver.get("https://www.kaggle.com/") # website link; opens the browser 
driver.maximize_window()    # to maximize the window

time.sleep(5) # pauses execution, gives time to load the webpage

# Step 2: Find and Click on datasets 
input_element = driver.find_element(By.LINK_TEXT, "Datasets")
input_element.click()
time.sleep(5) # time to load the page

# Step 3: Input 'time series' in the search field
search_element = driver.find_element(By.XPATH, "//input[@placeholder = 'Search datasets']")
search_element.send_keys("time series")
search_element.send_keys("\n")  # Press Enter to start the search
time.sleep(5)  # Wait for search results to load

"""# Step 4: Get first 5 Dataset names and links from the Kaggle website # THIS WORKS
dataset_names = driver.find_elements(By.CSS_SELECTOR, "div.sc-kLJHhQ.ithYPd.km-listitem--large div.sc-eauhAA.sc-fXwCOG") # combo of outer and inner container
dataset_links = driver.find_elements(By.CSS_SELECTOR, "a[href*='/datasets/']")

for i in range(min(5, len(dataset_links))):
    name = dataset_names[i].text  # Dataset name
    link = dataset_links[i].get_attribute("href")  # Dataset link
    print(f"Dataset {i + 1}:")
    print(f"Name: {name}")
    print(f"Link: {link}")
    print("-" * 40)"""

# Step 4: Get all Dataset names and links across all pages
all_datasets = []

while True:
    # Extract names and links from the current page
    dataset_containers = driver.find_elements(By.CSS_SELECTOR, "div.sc-kLJHhQ.ithYPd.km-listitem--large")
    for container in dataset_containers:
        name = container.find_element(By.CSS_SELECTOR, "div.sc-eauhAA.sc-fXwCOG").text
        print(name)
        link = container.find_element(By.CSS_SELECTOR, "a.sc-lgprfV").get_attribute("href")
        print(link)
        all_datasets.append({"Name": name, "Link": link})

    # Check if next button exists and is clickable
    time.sleep(5)
    try:
        # Wait for the "Next" button to be clickable
        next_button_element = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button[aria-label='Go to next page']"))
        )
        
        # Scroll into view and click
        driver.execute_script("arguments[0].scrollIntoView(true);", next_button_element)
        time.sleep(1)  # Allow time for the page to adjust
        next_button_element.click()
        
        # Wait for the next page to load
        time.sleep(3)
    except StaleElementReferenceException:
        # Re-locate the "Next" button if it's stale
        print("Stale element, retrying...")
        continue
    except Exception as e:
        print(f"No more pages or error occurred: {e}")
        break

# Step 5: Save all collected datasets to an Excel file
df = pd.DataFrame(all_datasets)  # Create a DataFrame from the list of dictionaries
df.to_excel("automated_kaggle_datasets.xlsx", index=False, columns=["Name", "Link"])

print('done')
