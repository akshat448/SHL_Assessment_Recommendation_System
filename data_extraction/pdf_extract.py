import os
import json
import re
import requests
from PyPDF2 import PdfReader
from langdetect import detect, DetectorFactory

# Ensure consistent language detection results
DetectorFactory.seed = 0

# File paths
INPUT_JSON = "/Users/akshat/Developer/Tasks/SHL/data/shl_all_assessments_final.json"
OUTPUT_JSON = "/Users/akshat/Developer/Tasks/SHL/data/pdf_extracted_data.json"
TEMP_PDF_DIR = "/Users/akshat/Developer/Tasks/SHL/temp_pdfs"  # Temporary directory for downloaded PDFs

# Ensure the temporary directory exists
os.makedirs(TEMP_PDF_DIR, exist_ok=True)

def download_pdf(pdf_url, save_path):
    """Download a PDF from a URL and save it locally."""
    try:
        response = requests.get(pdf_url, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"Downloaded: {pdf_url}")
        return True
    except Exception as e:
        print(f"Error downloading {pdf_url}: {e}")
        return False

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text()
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
        return None
    return text

def is_text_english(text):
    """Check if the given text is in English."""
    try:
        language = detect(text)
        return language == "en"
    except Exception as e:
        print(f"Error detecting language: {e}")
        return False

def clean_text(text):
    """Clean the extracted text."""
    # Remove headers, footers, and page numbers
    text = re.sub(r"©\s*\d{4}\s*SHL.*www\.shl\.com", "", text)
    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)
    # Remove special characters
    text = re.sub(r"[^\w\s]", "", text)
    # Remove duplicate lines
    lines = text.splitlines()
    unique_lines = list(dict.fromkeys(lines))  # Preserve order while removing duplicates
    return " ".join(unique_lines).strip()

def process_solutions(input_json, output_json, temp_pdf_dir, limit=5):
    """Process the first `limit` solutions, extract PDF data, and save to a JSON file."""
    try:
        # Load the input JSON
        with open(input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract the solutions
        solutions = data.get("Pre-packaged Job Solutions", []) + data.get("Individual Test Solutions", [])
        solutions = solutions[:limit]  # Limit to the first `limit` solutions
        
        extracted_data = []
        
        for solution in solutions:
            solution_name = solution.get("Assessment Name", "Unknown Solution")
            pdf_links = solution.get("PDF Downloads", [])
            extracted_pdfs = []
            
            print(f"Processing solution: {solution_name}")
            
            for pdf_link in pdf_links:
                # Extract the filename from the URL
                pdf_filename = os.path.basename(pdf_link)
                pdf_path = os.path.join(temp_pdf_dir, pdf_filename)
                
                # Download the PDF
                if download_pdf(pdf_link, pdf_path):
                    # Extract text from the PDF
                    pdf_text = extract_text_from_pdf(pdf_path)
                    pdf_clean_text = clean_text(pdf_text)
                    if pdf_text:
                        # Check if the text is in English
                        if is_text_english(pdf_clean_text):
                            extracted_pdfs.append({
                                "PDF Link": pdf_link,
                                "Extracted Data": pdf_clean_text.strip()
                            })
                        else:
                            print(f"  Skipping non-English PDF: {pdf_link}")
                    
                    # Delete the PDF after processing
                    os.remove(pdf_path)
                else:
                    print(f"  Failed to download: {pdf_link}")
            
            # Add the extracted data for this solution
            extracted_data.append({
                "Assessment Name": solution_name,
                "URL": solution.get("URL", ""),
                "PDF Data": extracted_pdfs
            })
        
        # Save the extracted data to the output JSON
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(extracted_data, f, indent=4, ensure_ascii=False)
        
        print(f"Extracted data saved to {output_json}")
    
    except Exception as e:
        print(f"Error processing solutions: {e}")

if __name__ == "__main__":
    # Process the first 5 solutions
    process_solutions(INPUT_JSON, OUTPUT_JSON, TEMP_PDF_DIR, limit=5)
