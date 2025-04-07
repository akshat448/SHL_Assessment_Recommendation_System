import os
import json
from pdf_extract import download_pdf, extract_text_from_pdf, clean_text, is_text_english

# File paths
INPUT_JSON = "/Users/akshat/Developer/Tasks/SHL/data/shl_all_assessments_final.json"
OUTPUT_JSON = "/Users/akshat/Developer/Tasks/SHL/data/processed_assessments.json"
TEMP_PDF_DIR = "/Users/akshat/Developer/Tasks/SHL/temp_pdfs"  # Temporary directory for downloaded PDFs

# Ensure the temporary directory exists
os.makedirs(TEMP_PDF_DIR, exist_ok=True)

def process_and_format_data(input_json, output_json, temp_pdf_dir, limit=None):
    """Process the data, extract and clean PDF text, and merge it with the original fields."""
    try:
        # Load the input JSON
        with open(input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Combine all solutions into a single list
        solutions = data.get("Pre-packaged Job Solutions", []) + data.get("Individual Test Solutions", [])
        
        # Apply the limit if specified
        if limit is not None:
            solutions = solutions[:limit]
        
        processed_data = []

        # Iterate through each assessment
        for assessment in solutions:
            assessment_name = assessment.get("Assessment Name", "Unknown Assessment")
            pdf_links = assessment.get("PDF Downloads", [])
            cleaned_pdfs = []

            print(f"Processing assessment: {assessment_name}")

            for pdf_link in pdf_links:
                # Extract the filename from the URL
                pdf_filename = os.path.basename(pdf_link)
                pdf_path = os.path.join(temp_pdf_dir, pdf_filename)

                # Download the PDF
                if download_pdf(pdf_link, pdf_path):
                    # Extract text from the PDF
                    pdf_text = extract_text_from_pdf(pdf_path)
                    if pdf_text:
                        # Clean the extracted text
                        cleaned_text = clean_text(pdf_text)

                        # Check if the text is in English
                        if is_text_english(cleaned_text):
                            cleaned_pdfs.append({
                                "PDF Link": pdf_link,
                                "Cleaned Text": cleaned_text
                            })
                        else:
                            print(f"  Skipping non-English PDF: {pdf_link}")
                    
                    # Delete the PDF after processing
                    os.remove(pdf_path)
                else:
                    print(f"  Failed to download: {pdf_link}")

            # Merge the cleaned PDF data with the original assessment fields
            processed_assessment = {
                **assessment,  # Include all original fields
                "Cleaned PDF Data": cleaned_pdfs  # Add the cleaned PDF data
            }
            processed_data.append(processed_assessment)

        # Save the processed data to the output JSON
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(processed_data, f, indent=4, ensure_ascii=False)
        
        print(f"Processed data saved to {output_json}")
    
    except Exception as e:
        print(f"Error processing and formatting data: {e}")

if __name__ == "__main__":
    # Set the limit (e.g., process only the first 5 solutions)
    process_and_format_data(INPUT_JSON, OUTPUT_JSON, TEMP_PDF_DIR, limit=None)