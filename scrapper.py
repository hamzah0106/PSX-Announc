import os
import json
import time
from datetime import datetime
import requests
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from pdf2image import convert_from_path
import pytesseract
import fitz  # PyMuPDF
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up the web driver
def set_up_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=chrome_options)
    return driver

# Retry mechanism for table loading
def wait_for_table(driver, retries=3, delay=5):
    for i in range(retries):
        try:
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'table#announcementsTable tbody tr'))
            )
            logging.info("Table loaded successfully.")
            return True
        except Exception as e:
            logging.warning(f"Retry {i+1}/{retries} - Waiting failed: {e}")
            time.sleep(delay)
    logging.error("Failed to load the table after retries.")
    return False

# Download the file from the URL
def download_file(url, save_path):
    if not url.startswith(('http://', 'https://')):
        logging.warning(f"Invalid URL skipped: {url}")
        return False
    response = requests.get(url, allow_redirects=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        logging.info(f"File downloaded: {save_path}")
        return True
    else:
        logging.error(f"Failed to download file from {url}, status code: {response.status_code}")
        return False

# Check if the PDF is text-based or image-based
def is_pdf_text_based(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return bool(text.strip())

# Process PDF with OCR if it's image-based
def process_pdf(pdf_path, ocr_save_path):
    # Always process OCR, skipping the check for text-based PDFs
    logging.info(f"Applying OCR to PDF: {pdf_path}")
    return process_pdf_with_ocr(pdf_path, ocr_save_path)

def process_pdf_with_ocr(pdf_path, ocr_save_path):
    try:
        images = convert_from_path(pdf_path)
        text = ""
        for image in images:
            text += pytesseract.image_to_string(image)
        with open(ocr_save_path, 'w') as f:
            f.write(text)
        logging.info(f"OCR processed file saved at {ocr_save_path}")
        return True
    except Exception as e:
        logging.error(f"OCR processing failed for {pdf_path}: {e}")
        return False

# Save metadata to JSON
def save_metadata(metadata_file, announcements, delta_file):
    existing_data = {"announcements": []}
    try:
        with open(metadata_file, 'r') as f:
            existing_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    
    existing_data["announcements"].extend(announcements)
    if announcements:
        existing_data["last_download_time"] = max(a["date_time"] for a in announcements)
    
    with open(metadata_file, 'w') as f:
        json.dump(existing_data, f, indent=2)
    logging.info(f"Metadata saved successfully.")

    # Delta file
    with open(delta_file, 'w') as f:
        json.dump({"announcements": announcements}, f, indent=2)
    logging.info(f"Delta data saved successfully.")

# Load metadata from JSON
def load_metadata(metadata_file):
    if os.path.exists(metadata_file) and os.path.getsize(metadata_file) > 0:
        try:
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
            logging.info(f"Metadata loaded from {metadata_file}")
            return metadata
        except json.JSONDecodeError:
            logging.error(f"Error: {metadata_file} is corrupted or invalid JSON.")
            return {"announcements": []}
    else:
        logging.info(f"Metadata file not found or empty. Initializing with defaults.")
        return {"announcements": []}

# Modify the scraping logic for pagination
def scrape_psx_announcements(driver, metadata_file, symbols=None, download_dir='psx_announcements', delta_file='delta_data.json'):
    url = "https://dps.psx.com.pk/announcements/companies"
    driver.get(url)
    if not wait_for_table(driver):
        return

    # Load initial metadata and get the first last_download_time
    scraper_state = load_metadata(metadata_file)
    last_download_time = max((
        datetime.fromisoformat(a["date_time"]) for a in scraper_state["announcements"]
    ), default=None)

    if not os.path.exists(download_dir):
        os.makedirs(download_dir)

    announcements = []

    # Start scraping
    while True:
        rows = driver.find_elements(By.CSS_SELECTOR, 'table#announcementsTable tbody tr')
        
        for row in rows:
            cols = row.find_elements(By.TAG_NAME, 'td')
            if len(cols) < 5:
                continue

            # Extract date and time
            date_time = datetime.strptime(f"{cols[0].text} {cols[1].text}", "%b %d, %Y %I:%M %p")
            symbol = cols[2].text.strip()
            title = cols[3].text.strip()  # Assuming the 4th column contains the title

            logging.info(f"Checking symbol: {symbol}, title: {title}")

            # Stop scraping if the current row's date_time is <= the initial last_download_time
            if last_download_time and date_time <= last_download_time:
                logging.info(f"Reached last_download_time: {last_download_time}, stopping scraper.")
                save_metadata(metadata_file, announcements, delta_file)  # Save metadata before stopping
                return  # Stop scraping

            if symbols and symbol not in symbols:
                continue

            # Extract PDF link
            pdf_link_element = row.find_elements(By.CSS_SELECTOR, 'td a[href*="download/document"]')
            if not pdf_link_element:
                continue

            link = pdf_link_element[0].get_attribute('href')
            file_name = f"{date_time.strftime('%Y%m%d_%H%M')}_{symbol}.pdf"
            pdf_save_path = os.path.join(download_dir, file_name)

            # Download the PDF and process it
            if download_file(link, pdf_save_path):
                ocr_file_path = pdf_save_path.replace('.pdf', '_OCR.txt') if process_pdf(pdf_save_path, pdf_save_path.replace('.pdf', '_OCR.txt')) else None
                announcements.append({
                    'date_time': date_time.isoformat(),
                    'symbol': symbol,
                    'title': title,  # Add title to metadata
                    'file_path': pdf_save_path,
                    'ocr_file_path': ocr_file_path
                })

        # Save metadata after processing the rows
        save_metadata(metadata_file, announcements, delta_file)

        # Handle pagination with an updated approach
        try:
            # Wait for the next button to be present (using an explicit wait)
            next_button = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, 'button.form__button.next'))
            )

            # Check if the button is disabled (indicating the last page)
            if "disabled" in next_button.get_attribute("class"):
                logging.info("No more pages to load. Stopping pagination.")
                break

            # Click the next button
            next_button.click()
            time.sleep(5)

        except Exception as e:
            logging.error(f"Pagination error: {e}")
            break

# Continuous scraping
def continuous_scraping(metadata_file, symbols=None, download_dir='psx_announcements', delta_file='delta_data.json', interval=300):
    driver = set_up_driver()
    try:
        while True:
            print("Starting scraping cycle...")
            scrape_psx_announcements(driver, metadata_file, symbols, download_dir, delta_file)
            print(f"Scraping done. Sleeping for {interval} seconds...")
            time.sleep(interval)  # Sleep for 5 minutes (300 seconds)
            print("Restarting scraping cycle...")
    except KeyboardInterrupt:
        print("Stopping continuous scraping...")
    finally:
        driver.quit()

# Example usage
metadata_file = "psx_scraper_metadata.json"
delta_file = "delta_data.json"
continuous_scraping(metadata_file=metadata_file, symbols=[ "MARI", "ASC",  "DAWH",  "PAKD", "PSO", "POL" , "IBFL"])
