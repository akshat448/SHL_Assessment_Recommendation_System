"""
SHL Catalog Scraper - A tool to extract assessment data from SHL's product catalog

This script extracts information about pre-packaged job solutions and individual 
test assessments from the SHL product catalog website, including metadata and PDF downloads.
"""

import os
import json
import time
import re
from urllib.parse import urljoin
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup

# Configuration
OUTPUT_DIR = "/Users/akshat/Developer/Tasks/SHL/data"
BASE_URL = "https://www.shl.com/solutions/products/product-catalog/"
PREPACKAGED_TYPE = "2"
INDIVIDUAL_TYPE = "1"

# Output file paths
PREPACKAGED_FILE = os.path.join(OUTPUT_DIR, "shl_prepackaged_solutions_clean.json")
INDIVIDUAL_FILE = os.path.join(OUTPUT_DIR, "shl_individual_tests_clean.json")
ALL_FILE = os.path.join(OUTPUT_DIR, "shl_all_assessments_clean.json")

def setup_driver(headless=True):
    """Configure and initialize a Chrome WebDriver instance"""
    options = Options()
    if headless:
        options.add_argument("--headless=new")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    
    # Avoid detection as automation
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    
    driver = webdriver.Chrome(options=options)
    driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
    driver.set_page_load_timeout(45)
    
    return driver

def wait_for_element(driver, by, selector, timeout=15):
    """Wait for an element to be present on the page"""
    try:
        element = WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((by, selector))
        )
        return element
    except Exception as e:
        print(f"Error waiting for element {selector}: {e}")
        return None

def generate_catalog_urls(solution_type):
    """Generate URLs for all catalog pages of a given solution type"""
    # Items per page in catalog
    items_per_page = 12
    
    # Expected total number of items
    max_items = 141 if solution_type == PREPACKAGED_TYPE else 377
    
    urls = []
    # Standard pagination format
    for start in range(0, max_items, items_per_page):
        urls.append(f"{BASE_URL}?start={start}&type={solution_type}")
    
    return urls

def extract_basic_data_from_table(driver, category_name):
    """Extract basic assessment data from the catalog table"""
    assessments = []
    
    try:
        # Wait for table to load
        table_wrapper = wait_for_element(driver, By.CSS_SELECTOR, ".custom__table-wrapper")
        if not table_wrapper:
            print("Table not found")
            return []
        
        # Extract data from rows
        rows = table_wrapper.find_elements(By.CSS_SELECTOR, "tr:not(:first-child)")
        print(f"Found {len(rows)} rows in {category_name} table")
        
        for row in rows:
            cols = row.find_elements(By.TAG_NAME, "td")
            if len(cols) < 4:
                continue
            
            # Extract assessment name and URL
            link_element = cols[0].find_element(By.TAG_NAME, "a")
            assessment_name = link_element.text.strip()
            assessment_url = link_element.get_attribute("href")
            
            # Extract remote testing and adaptive circle indicators
            remote_testing = 'Yes' if cols[1].find_elements(By.CSS_SELECTOR, ".catalogue__circle.-yes") else 'No'
            adaptive_support = 'Yes' if cols[2].find_elements(By.CSS_SELECTOR, ".catalogue__circle.-yes") else 'No'
            
            # Extract test type keys
            test_type_keys = []
            for key in cols[3].find_elements(By.CSS_SELECTOR, ".product-catalogue__key"):
                key_text = key.get_attribute("innerHTML").strip()
                if key_text:
                    test_type_keys.append(key_text)
            
            assessments.append({
                "Assessment Name": assessment_name,
                "URL": assessment_url,
                "Remote Testing": remote_testing,
                "Adaptive Support": adaptive_support,
                "Test Type Keys": test_type_keys
            })
            
    except Exception as e:
        print(f"Error extracting table data: {e}")
    
    return assessments

def extract_pdf_links(driver, soup, url):
    """Extract PDF download links from the assessment detail page"""
    pdf_links = []
    
    # 1. Find PDF links in the dedicated download section
    download_section = soup.find("div", class_="product-catalogue-training-calendar__row", 
                              string=lambda text: text and "Downloads" in text if text else False)
    
    if download_section:
        download_list = download_section.find_next("ul", class_="product-catalogue__downloads")
        if download_list:
            for item in download_list.find_all("li", class_="product-catalogue__download"):
                link = item.find("a")
                if link and link.get("href", "").lower().endswith(".pdf"):
                    pdf_links.append(link.get("href").strip())
    
    # 2. Find all PDF links on the page
    for link in soup.find_all("a", href=True):
        if link.get("href", "").lower().endswith(".pdf"):
            pdf_links.append(link.get("href").strip())
    
    # 3. Look for service.shl.com PDF links
    shl_pdf_links = re.findall(r'https?://service\.shl\.com/[^"\'>\s]+\.pdf', str(soup))
    pdf_links.extend(shl_pdf_links)
    
    # 4. Execute JavaScript to find PDF links
    try:
        js_links = driver.execute_script("""
            return Array.from(document.querySelectorAll('a'))
                .filter(link => link.href && link.href.toLowerCase().endsWith('.pdf'))
                .map(link => link.href);
        """)
        pdf_links.extend(js_links)
    except Exception:
        pass
    
    # Process links to make them absolute and unique
    unique_links = set()
    for link in pdf_links:
        # Make relative URLs absolute
        if not link.startswith(('http:', 'https:')):
            link = urljoin(url, link)
        
        unique_links.add(link)
    
    return list(unique_links)

def scrape_detail_page(driver, url):
    """Scrape detailed information from an assessment page"""
    print(f"Visiting detail page: {url}")
    
    try:
        driver.get(url)
        wait_for_element(driver, By.TAG_NAME, "h1")
        
        # Wait for page to load completely
        time.sleep(3)
        
        # Scroll through page to ensure all content loads
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight/2);")
        time.sleep(0.5)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(0.5)
        
        # Parse the page content
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        
        # Find content container
        content_div = soup.find('div', class_='product-catalogue')
        if not content_div:
            print("No content div found")
            return {}
        
        # Extract section data
        sections = {
            "Description": "",
            "Job Levels": "",
            "Languages": "",
            "Assessment Length": ""
        }
        
        # Look for sections with h4 headers
        section_rows = content_div.find_all('div', class_='product-catalogue-training-calendar__row')
        for row in section_rows:
            header = row.find('h4')
            if not header:
                continue
            
            header_text = header.text.strip()
            p_tag = row.find('p')
            if not p_tag:
                continue
                
            content = p_tag.text.strip()
            
            # Map headers to our section names
            if "Description" in header_text:
                sections["Description"] = content
            elif "Job levels" in header_text or "Job Levels" in header_text:
                sections["Job Levels"] = content
            elif "Languages" in header_text:
                sections["Languages"] = content
            elif "Assessment length" in header_text or "Assessment Length" in header_text:
                sections["Assessment Length"] = content
        
        # Extract PDF download links
        pdf_links = extract_pdf_links(driver, soup, url)
        
        # Combine all data
        detail_data = {
            "Description": sections["Description"],
            "Job Levels": sections["Job Levels"],
            "Languages": sections["Languages"],
            "Assessment Length": sections["Assessment Length"],
            "PDF Downloads": pdf_links
        }
        
        return detail_data
        
    except Exception as e:
        print(f"Error scraping detail page {url}: {e}")
        return {}

def scrape_catalog(solution_type):
    """Scrape all assessments of a specific type (prepackaged or individual)"""
    category_name = "Pre-packaged Job Solutions" if solution_type == PREPACKAGED_TYPE else "Individual Test Solutions"
    print(f"\n=== SCRAPING {category_name.upper()} ===")
    
    driver = setup_driver()
    assessments = []
    
    try:
        # Get all catalog URLs for this solution type
        catalog_urls = generate_catalog_urls(solution_type)
        
        # Store URLs of assessments already processed to avoid duplicates
        processed_urls = set()
        
        # Process each catalog page
        for url_index, url in enumerate(catalog_urls):
            try:
                print(f"Processing catalog page {url_index+1}/{len(catalog_urls)}: {url}")
                driver.get(url)
                
                # Extract basic assessment data from table
                page_assessments = extract_basic_data_from_table(driver, category_name)
                
                # Store only assessments we haven't seen before
                for assessment in page_assessments:
                    assessment_url = assessment["URL"]
                    if assessment_url not in processed_urls:
                        processed_urls.add(assessment_url)
                        
                        # Fetch detailed information for this assessment
                        print(f"Scraping details for: {assessment['Assessment Name']}")
                        detail_data = scrape_detail_page(driver, assessment_url)
                        
                        # Combine basic and detailed data
                        full_assessment = {**assessment, **detail_data}
                        assessments.append(full_assessment)
                
                # Small delay between catalog pages
                time.sleep(1)
                
            except Exception as e:
                print(f"Error processing catalog page {url}: {e}")
        
        print(f"Successfully scraped {len(assessments)} {category_name}")
        
    finally:
        driver.quit()
    
    return assessments

def save_data_to_file(data, filename):
    """Save data to a JSON file"""
    try:
        with open(filename, "w", encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"Data saved to {filename}")
    except Exception as e:
        print(f"Error saving data to {filename}: {e}")

def main():
    """Main execution function"""
    print("Starting SHL catalog extraction...")
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    try:
        # Step 1: Scrape pre-packaged solutions
        prepackaged_solutions = scrape_catalog(PREPACKAGED_TYPE)
        
        # Step 2: Scrape individual test solutions
        individual_solutions = scrape_catalog(INDIVIDUAL_TYPE)
        
        # Save results to files
        save_data_to_file(prepackaged_solutions, PREPACKAGED_FILE)
        save_data_to_file(individual_solutions, INDIVIDUAL_FILE)
        
        # Create and save combined file
        combined_data = {
            "Pre-packaged Job Solutions": prepackaged_solutions,
            "Individual Test Solutions": individual_solutions
        }
        save_data_to_file(combined_data, ALL_FILE)
        
        # Report statistics
        total_items = len(prepackaged_solutions) + len(individual_solutions)
        prepackaged_with_pdf = sum(1 for item in prepackaged_solutions if item.get("PDF Downloads"))
        individual_with_pdf = sum(1 for item in individual_solutions if item.get("PDF Downloads"))
        
        print("\n=== SUMMARY ===")
        print(f"Pre-packaged Job Solutions: {len(prepackaged_solutions)}")
        print(f"Individual Test Solutions: {len(individual_solutions)}")
        print(f"Total assessments: {total_items}")
        print(f"Pre-packaged solutions with PDFs: {prepackaged_with_pdf}/{len(prepackaged_solutions)}")
        print(f"Individual tests with PDFs: {individual_with_pdf}/{len(individual_solutions)}")
        
    except Exception as e:
        print(f"Error during extraction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()