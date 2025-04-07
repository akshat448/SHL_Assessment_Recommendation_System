import streamlit as st
import requests
import pandas as pd
import os
import json

# Backend API URL
API_URL = os.environ.get("API_URL", "http://localhost:8080/api/recommendations/")
QUERY_API_URL = os.environ.get("QUERY_API_URL", "http://localhost:8080/api/recommendations/query")

st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Query Recommendations"])

if page == "Home":
    st.title("🔍 SHL Assessment Recommendation System")
    st.markdown("Enter a job description or hiring query to get the most relevant SHL assessments.")

    # Input form
    with st.form("query_form"):
        query = st.text_area("Enter your query or URL to a Job Description", height=150, 
                             placeholder="e.g. I am hiring for a Java developer who can collaborate with business teams. Need a test under 40 minutes.")
        top_k = st.number_input("Number of candidates to retrieve (top_k)", min_value=5, max_value=100, value=20)
        top_n = st.number_input("Number of final recommendations to show (top_n)", min_value=1, max_value=10, value=10)
        use_reranker = st.checkbox("Use reranker", value=False)
        submitted = st.form_submit_button("Get Recommendations")
        use_llm_explanations = True

    # Add note about reranker timing
    if use_reranker:
        st.info("💡 Note: Using the reranker provides more accurate results but may take a little longer to process. Please be patient.")
    
    if submitted and query:
        with st.spinner("Processing your query..."):
            try:
                # Make API request
                response = requests.post(
                    API_URL, 
                    json={
                        "query": query,
                        "top_k": int(top_k),
                        "top_n": int(top_n),
                        "use_reranker": use_reranker,
                        "use_llm_explanations": use_llm_explanations
                    }
                )
                response.raise_for_status()
                data = response.json()

                if not data.get("recommendations"):
                    st.warning("No recommendations found for this query.")
                else:
                    st.success(f"Found {len(data['recommendations'])} recommendations in {data.get('processing_time', 0):.2f} seconds.")
                    
                    # After creating the DataFrame from recommendations
                    df = pd.DataFrame(data["recommendations"])
                    
                    # Add test_types from metadata if test_type_categories are empty
                    if "test_type_categories" in df.columns and "metadata" in data and "test_types" in data["metadata"]:
                        # Replace test_type_categories with test_types from metadata
                        if all(not cats for cats in df["test_type_categories"]):
                            df["test_types"] = [data["metadata"]["test_types"]] * len(df)
                            # Remove the empty test_type_categories column
                            df = df.drop("test_type_categories", axis=1)
                        
                    # Process all list columns (test_types, job_levels, languages)
                    for col in df.columns:
                        if any(isinstance(item, list) for item in df[col] if item is not None):
                            df[col] = df[col].apply(lambda x: ", ".join(x) if isinstance(x, list) and x else "")
                    
                    # Create a clickable hyperlink for assessment name
                    if "assessment_url" in df.columns and "assessment_name" in df.columns:
                        df["Assessment"] = df.apply(
                            lambda row: f'<a href="{row["assessment_url"]}" target="_blank">{row["assessment_name"]}</a>', 
                            axis=1
                        )
                    else:
                        df["Assessment"] = df["assessment_name"]
                    
                    # Rename other columns for display
                    column_mapping = {
                        "duration_minutes": "Duration (min)",
                        "remote_testing": "Remote",
                        "adaptive_support": "Adaptive",
                        "test_types": "Test Types",
                        "job_levels": "Job Levels",
                        "languages": "Languages",
                        "explanation": "Why Recommended"  # Updated name
                    }

                    # Only rename columns that exist in the DataFrame
                    rename_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
                    df = df.rename(columns=rename_cols) 
                    
                    # Remove "**Explanation:**" prefix if present
                    if "Why Recommended" in df.columns:
                        df["Why Recommended"] = df["Why Recommended"].apply(
                            lambda x: x.replace("**Explanation:** ", "") if isinstance(x, str) and x.startswith("**Explanation:**") else x
                        )
                    
                    # Define display columns (prioritize required columns)
                    display_cols = ["Assessment", "Duration (min)", "Remote", "Adaptive", "Test Types"]
                    
                    # Add optional columns if they exist
                    optional_cols = ["Job Levels", "Languages", "Why Recommended"]
                    display_cols.extend([col for col in optional_cols if col in df.columns])
                    
                    # Filter to only include columns that exist in the DataFrame
                    display_cols = [col for col in display_cols if col in df.columns]
                    
                    # Display the table
                    st.markdown("### Recommended Assessments")
                    st.write(
                        df[display_cols].to_html(escape=False, index=False),
                        unsafe_allow_html=True
                    )
                    
                    # Add CSV download option
                    csv = df[display_cols].to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download as CSV",
                        csv,
                        "shl_recommendations.csv",
                        "text/csv",
                        key='download-csv'
                    )
                    
            except requests.exceptions.RequestException as e:
                st.error(f"API request failed: {e}")

elif page == "Query Recommendations":
    st.title("🔍 Query Recommendations API")
    st.markdown("Use the GET endpoint to fetch recommendations directly as JSON.")

    # Input form for GET request
    with st.form("get_query_form"):
        query = st.text_input("Enter your query", placeholder="e.g. I am hiring for a Java developer...")
        top_k = st.number_input("Number of candidates to retrieve (top_k)", min_value=5, max_value=100, value=20)
        top_n = st.number_input("Number of final recommendations to show (top_n)", min_value=1, max_value=10, value=10)
        use_reranker = st.checkbox("Use reranker", value=False)
        use_llm_explanations = st.checkbox("Use LLM explanations", value=True)
        submitted = st.form_submit_button("Fetch JSON")
    
    # Add note about reranker timing
    if use_reranker:
        st.info("💡 Note: Using the reranker provides more accurate results but may take a little longer to process. Please be patient.")

    if submitted and query:
        with st.spinner("Fetching recommendations..."):
            try:
                # Make GET request
                response = requests.get(
                    QUERY_API_URL,
                    params={
                        "query": query,
                        "top_k": int(top_k),
                        "top_n": int(top_n),
                        "use_reranker": use_reranker,
                        "use_llm_explanations": use_llm_explanations
                    },
                    allow_redirects=True
                )
                   # Print debugging info
                st.write(f"Status Code: {response.status_code}")
                st.write(f"Request URL: {response.request.url}")
                
                response.raise_for_status()
                data = response.json()

                if not data.get("recommendations"):
                    st.warning("No recommendations found for this query.")
                else:
                    # Process the JSON to add test_types to each recommendation
                    if "metadata" in data and "test_types" in data["metadata"]:
                        for rec in data["recommendations"]:
                            # Replace empty test_type_categories with test_types from metadata
                            if "test_type_categories" in rec and not rec["test_type_categories"]:
                                # Update both fields for consistency
                                rec["test_types"] = data["metadata"]["test_types"]
                                rec["test_type_categories"] = data["metadata"]["test_types"]
                            
                            # Remove the score field if it exists
                            if "score" in rec:
                                del rec["score"]
                            
                            # Clean up explanation field - remove "**Explanation:**" prefix
                            if "explanation" in rec and isinstance(rec["explanation"], str):
                                rec["explanation"] = rec["explanation"].replace("**Explanation:** ", "")
                                rec["explanation"] = rec["explanation"].replace("**Explanation:**", "")
                        
                        # After all recommendations are processed, update the overall data object
                        data["recommendations"] = data["recommendations"]
                        
                            
                    st.success(f"Found {len(data['recommendations'])} recommendations in {data.get('processing_time', 0):.2f} seconds.")
                    
                    # Display raw JSON output
                    st.markdown("### JSON Response")
                    st.json(data)
                    
                    # Add JSON download option
                    json_str = json.dumps(data, indent=2)
                    st.download_button(
                        "Download JSON",
                        json_str,
                        "shl_recommendations.json",
                        "application/json",
                        key='download-json'
                    )
                    
            except requests.exceptions.RequestException as e:
                st.error(f"API request failed: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    st.error(f"Response status: {e.response.status_code}")
                    st.error(f"Response text: {e.response.text}")