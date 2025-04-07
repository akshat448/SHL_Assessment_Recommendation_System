import json

# File paths
INPUT_JSON = "/Users/akshat/Developer/Tasks/SHL/data/processed_assessments.json"
OUTPUT_JSON = "/Users/akshat/Developer/Tasks/SHL/data/processed_assessments_no_duplicates.json"

def remove_duplicates(input_json, output_json):
    """Remove duplicate PDFs and save the updated JSON."""
    try:
        # Load the input JSON
        with open(input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        seen_pdf_links = set()
        updated_data = []

        for assessment in data:
            cleaned_pdfs = []
            for pdf in assessment.get("Cleaned PDF Data", []):
                pdf_link = pdf.get("PDF Link", "")
                if pdf_link not in seen_pdf_links:
                    cleaned_pdfs.append(pdf)
                    seen_pdf_links.add(pdf_link)

            # Update the assessment with unique PDFs
            assessment["Cleaned PDF Data"] = cleaned_pdfs
            updated_data.append(assessment)

        # Save the updated data to the output JSON
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(updated_data, f, indent=4, ensure_ascii=False)
        
        print(f"Duplicates removed. Updated data saved to {output_json}")
    
    except Exception as e:
        print(f"Error removing duplicates: {e}")

if __name__ == "__main__":
    remove_duplicates(INPUT_JSON, OUTPUT_JSON)