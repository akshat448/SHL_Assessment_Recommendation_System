"""
SHL Data Validation Script

This script analyzes the extracted SHL assessment data to validate:
1. Data structure and integrity
2. Absence of duplicate assessments
3. Presence of required fields
4. PDF link format validation
5. Summary statistics
"""

import json
import re
from collections import Counter
import urllib.parse

# Configuration
DATA_FILE = "/Users/akshat/Developer/Tasks/SHL/data/shl_all_assessments_final.json"

# Required fields that should be present in each assessment
REQUIRED_FIELDS = [
    "Assessment Name", 
    "URL", 
    "Remote Testing", 
    "Adaptive Support",
    "Test Type Keys"
]

# Optional fields that might not be present in all assessments
OPTIONAL_FIELDS = [
    "Description",
    "Job Levels",
    "Languages",
    "Assessment Length",
    "PDF Downloads"
]

def load_json_data(file_path):
    """Load and parse JSON data from a file"""
    print(f"Loading data from: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"Error loading JSON file: {e}")
        return None

def check_duplicates(category_data, category_name):
    """Check for duplicate assessment names within a category"""
    print(f"\n===== {category_name} ({len(category_data)} total items) =====")
    
    # Extract assessment names
    assessment_names = [item.get("Assessment Name", "NO_NAME") for item in category_data 
                       if "Assessment Name" in item]
    name_counter = Counter(assessment_names)
    
    # Check for duplicates
    duplicates = {name: count for name, count in name_counter.items() if count > 1}
    
    print(f"Unique assessment names: {len(name_counter)}")
    if duplicates:
        print(f"Found {len(duplicates)} duplicate names:")
        for name, count in sorted(duplicates.items(), key=lambda x: x[1], reverse=True):
            print(f"  - '{name}': {count} occurrences")
    else:
        print("No duplicates found.")
        
    return duplicates

def check_cross_category_duplicates(data):
    """Check for assessment names that appear in both categories"""
    if "Pre-packaged Job Solutions" in data and "Individual Test Solutions" in data:
        prepackaged_names = {item.get("Assessment Name") for item in data["Pre-packaged Job Solutions"] 
                           if "Assessment Name" in item}
        individual_names = {item.get("Assessment Name") for item in data["Individual Test Solutions"] 
                          if "Assessment Name" in item}
        
        common_names = prepackaged_names.intersection(individual_names)
        
        print(f"\n===== Cross-Category Analysis =====")
        if common_names:
            print(f"Found {len(common_names)} assessment names that appear in both categories:")
            for name in sorted(common_names):
                print(f"  - '{name}'")
        else:
            print("No common assessment names between categories.")
        
        return common_names
    return set()

def check_required_fields(category_data, category_name):
    """Check for presence of required fields in each assessment"""
    print(f"\n===== Field Validation: {category_name} =====")
    
    missing_fields = {field: 0 for field in REQUIRED_FIELDS}
    total_items = len(category_data)
    
    for item in category_data:
        for field in REQUIRED_FIELDS:
            if field not in item or not item[field]:
                missing_fields[field] += 1
    
    if any(missing_fields.values()):
        print("Missing required fields:")
        for field, count in missing_fields.items():
            if count > 0:
                print(f"  - {field}: {count}/{total_items} items ({count/total_items*100:.1f}%)")
    else:
        print("All required fields present in all items.")
    
    return missing_fields

def validate_pdf_links(category_data, category_name):
    """Validate PDF links format and count statistics"""
    print(f"\n===== PDF Validation: {category_name} =====")
    
    total_items = len(category_data)
    items_with_pdfs = sum(1 for item in category_data if item.get("PDF Downloads"))
    pdf_percentage = items_with_pdfs / total_items * 100 if total_items > 0 else 0
    
    print(f"Items with PDF Downloads: {items_with_pdfs}/{total_items} ({pdf_percentage:.1f}%)")
    
    # Count and validate PDF URLs
    valid_pdf_links = 0
    invalid_pdf_links = 0
    total_pdfs = 0
    service_shl_links = 0
    
    # PDF validation regex
    pdf_pattern = re.compile(r'^https?://[^/]+/.*\.pdf$', re.IGNORECASE)
    
    for item in category_data:
        pdf_links = item.get("PDF Downloads", [])
        total_pdfs += len(pdf_links)
        
        for link in pdf_links:
            if pdf_pattern.match(link):
                valid_pdf_links += 1
                if "service.shl.com" in link:
                    service_shl_links += 1
            else:
                invalid_pdf_links += 1
                print(f"  Invalid PDF link found: {link}")
    
    print(f"Total PDF links: {total_pdfs}")
    print(f"Valid PDF links: {valid_pdf_links}")
    if invalid_pdf_links > 0:
        print(f"Invalid PDF links: {invalid_pdf_links}")
    print(f"Links from service.shl.com: {service_shl_links} ({service_shl_links/total_pdfs*100:.1f}% of total)")
    
    # PDF distribution analysis
    pdf_counts = [len(item.get("PDF Downloads", [])) for item in category_data]
    pdf_counter = Counter(pdf_counts)
    
    print("\nPDF count distribution:")
    for count in sorted(pdf_counter.keys()):
        items = pdf_counter[count]
        print(f"  {count} PDFs: {items} items ({items/total_items*100:.1f}%)")
    
    return {
        "items_with_pdfs": items_with_pdfs,
        "total_pdfs": total_pdfs,
        "valid_pdfs": valid_pdf_links,
        "invalid_pdfs": invalid_pdf_links,
        "service_shl_links": service_shl_links
    }

def validate_urls(category_data, category_name):
    """Validate assessment URLs format and structure"""
    print(f"\n===== URL Validation: {category_name} =====")
    
    base_url = "https://www.shl.com/solutions/products/product-catalog/view/"
    total_items = len(category_data)
    valid_urls = 0
    invalid_urls = 0
    
    for item in category_data:
        url = item.get("URL", "")
        
        # Check if URL starts with expected base
        if url.startswith(base_url):
            valid_urls += 1
        else:
            invalid_urls += 1
            print(f"  Invalid URL format: {url}")
    
    print(f"Valid URLs: {valid_urls}/{total_items} ({valid_urls/total_items*100:.1f}%)")
    if invalid_urls > 0:
        print(f"Invalid URLs: {invalid_urls}/{total_items} ({invalid_urls/total_items*100:.1f}%)")
    
    return {
        "valid_urls": valid_urls,
        "invalid_urls": invalid_urls
    }

def analyze_test_types(category_data, category_name):
    """Analyze test type key distribution"""
    print(f"\n===== Test Type Analysis: {category_name} =====")
    
    test_type_counts = Counter()
    total_items = len(category_data)
    
    for item in category_data:
        test_types = item.get("Test Type Keys", [])
        for test_type in test_types:
            test_type_counts[test_type] += 1
    
    print("Test type distribution:")
    for test_type, count in sorted(test_type_counts.items(), key=lambda x: x[1], reverse=True):
        print(f"  {test_type}: {count} items ({count/total_items*100:.1f}%)")
    
    return test_type_counts

def main():
    """Main execution function"""
    # Load data
    data = load_json_data(DATA_FILE)
    if not data:
        return
    
    # Ensure expected categories exist
    if "Pre-packaged Job Solutions" not in data or "Individual Test Solutions" not in data:
        print("Error: JSON file doesn't have expected categories")
        return
    
    # Extract category data
    prepackaged_solutions = data["Pre-packaged Job Solutions"]
    individual_solutions = data["Individual Test Solutions"]
    
    # 1. Check for duplicates within each category
    prepackaged_duplicates = check_duplicates(prepackaged_solutions, "Pre-packaged Job Solutions")
    individual_duplicates = check_duplicates(individual_solutions, "Individual Test Solutions")
    
    # 2. Check for cross-category duplicates
    cross_category_duplicates = check_cross_category_duplicates(data)
    
    # 3. Check required fields
    prepackaged_missing = check_required_fields(prepackaged_solutions, "Pre-packaged Job Solutions")
    individual_missing = check_required_fields(individual_solutions, "Individual Test Solutions")
    
    # 4. Validate PDF links
    prepackaged_pdfs = validate_pdf_links(prepackaged_solutions, "Pre-packaged Job Solutions")
    individual_pdfs = validate_pdf_links(individual_solutions, "Individual Test Solutions")
    
    # 5. Validate URLs
    prepackaged_urls = validate_urls(prepackaged_solutions, "Pre-packaged Job Solutions")
    individual_urls = validate_urls(individual_solutions, "Individual Test Solutions")
    
    # 6. Analyze test types
    prepackaged_test_types = analyze_test_types(prepackaged_solutions, "Pre-packaged Job Solutions")
    individual_test_types = analyze_test_types(individual_solutions, "Individual Test Solutions")
    
    # Final Summary
    print("\n===== OVERALL SUMMARY =====")
    print(f"Total assessments analyzed: {len(prepackaged_solutions) + len(individual_solutions)}")
    print(f"Pre-packaged Job Solutions: {len(prepackaged_solutions)}")
    print(f"Individual Test Solutions: {len(individual_solutions)}")
    
    has_issues = (
        bool(prepackaged_duplicates) or 
        bool(individual_duplicates) or 
        bool(cross_category_duplicates) or
        any(prepackaged_missing.values()) or 
        any(individual_missing.values()) or
        prepackaged_pdfs["invalid_pdfs"] > 0 or 
        individual_pdfs["invalid_pdfs"] > 0 or
        prepackaged_urls["invalid_urls"] > 0 or 
        individual_urls["invalid_urls"] > 0
    )
    
    if has_issues:
        print("\n Some issues were detected in the data. See details above.")
    else:
        print("\n All validation checks passed successfully!")

if __name__ == "__main__":
    main()