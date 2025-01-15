# Ready and working

"""UCI Machine Learning Repository - Selenium script to automate the collection of dataset names and URLs and tags"""

""" Step 1: open direct link to website with time series data
    Step 2: Get every name
    Step 3: Get every URL
    Step 4: Go to every page and get the tag under 'Subject Area'
    Step 5: Save it in excel """

from selenium import webdriver
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By     
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import StaleElementReferenceException, NoSuchElementException, TimeoutException
import pandas as pd
import time

start_time = time.time()

options = Options()
# options.add_experimental_option("detach", True)     # leaves the browser open after the task is done

driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()),
                          options=options)

# Step 1: Open the website in the browser
driver.get("https://archive.ics.uci.edu/") # website link; opens the browser 
driver.maximize_window()    # to maximize the window

time.sleep(3) # pauses execution, gives time to load the webpage

# Step 2: Find and Click on datasets 
input_element = driver.find_element(By.LINK_TEXT, "Datasets")
input_element.click()
time.sleep(3) # time to load the page

# Step 3: Find 'Data Type' dropdown and click on it 
data_type = driver.find_element(By.XPATH, "//div[@role='button' and .//span[text()='Data Type']]")
data_type.click()
time.sleep(2)

# Step 4: Select 'Time Series' from the dropdown
ts_button = driver.find_element(By.XPATH, "//span[@class = 'label-text' and text() = 'Time-Series']")
# Scroll into view
driver.execute_script("arguments[0].scrollIntoView(true);", ts_button)
action = ActionChains(driver)
action.move_to_element(ts_button).click().perform()
time.sleep(2)
# Scroll to the top of the page
driver.execute_script("window.scrollTo(0, 0);")
time.sleep(2)

# Step 5: Get dataset names across all pages
all_datasets = []
page_counter = 0

while page_counter<1:
    # Find all datasets
    dataset_containers = driver.find_elements(By.XPATH, "//div[contains(@class, 'relative') and contains(@class, 'col-span-8')]")
    print(f"Found {len(dataset_containers)} dataset containers.")

    # Loop through all dataset containers and extract links
    for index, container in enumerate(dataset_containers):
        try:
            # Locate the <a> tag with the dataset name inside each container
            link = container.find_element(By.XPATH, ".//a[contains(@class, 'link-hover') and contains (@class, 'text-xl')]")
            
            # Extract the dataset name and URL
            name = link.text
            url = link.get_attribute("href")
            
            # Debugging output to check the extracted data
            print(f"Name: {name}")
            print(f"URL: {url}")
            
            # Append to the list of datasets
            all_datasets.append({"Name": name, "Link": url, "Domain": ""})
        except Exception as e:
            print(f"Error processing container {index + 1}: {e}")
    page_counter += 1
    # Move to the next page
    try:
        next_button_element = driver.find_element(By.CSS_SELECTOR, "button[aria-label='Next Page']")
        driver.execute_script("arguments[0].scrollIntoView(true);", next_button_element)
        next_button_element.click()
        time.sleep(5)  # Wait for the next page to load
    except Exception as e:
        print(f"No more pages or error occurred: {e}")
        break

# Loop through the links to get their metadata
counter = 0
for dataset in all_datasets:
    try:
        driver.get(dataset["Link"])
        time.sleep(5)

        # Locate the "Domain - Subject Area" header
        dom_header = driver.find_element(By.XPATH, "//h1[text()='Subject Area']")
        # Locate all elements following the "Domain - Subject Area" header
        dom_elements = driver.find_elements(By.XPATH, "//h1[text()='Subject Area']/following-sibling::p[@class='text-md']")
        # Extract text from each <p> element
        subject_areas = [elem.text.strip() for elem in dom_elements]

        # Add the subject areas to the dataset dictionary
        dataset["Domain"] = ", ".join(subject_areas) if subject_areas else "No subject areas found"
        counter += 1
        print(counter, f"Subject Area for '{dataset['Name']}', {dataset['Domain']}")

    except (NoSuchElementException, TimeoutException):
        dataset["Domain"] = "No subject areas found"
        print(f"Subject Area for '{dataset['Name']}': No subject areas found")
        

    except Exception as e:
        dataset["Domain"] = f"Error: {e}"
        print(f"Subject Area for '{dataset['Name']}': Error: {e}")
        

    # Wait before moving to the next dataset
    time.sleep(2)

# Step 6: Save all collected datasets to an Excel file
df = pd.DataFrame(all_datasets)  # Create a DataFrame from the list of dictionaries
df.to_excel("UCI_datasets_selenium.xlsx", index=False)

end_time = time.time()
execution_time = end_time - start_time
print(f"Execution Time: {execution_time:.2f} seconds")

print('done')