import streamlit as st
import requests
import pandas as pd
import os
import json

# Backend API URL
API_URL = os.environ.get("API_URL", "http://localhost:8080/api/recommendations/")

st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")

st.title("🔍 SHL Assessment Recommendation System")
st.markdown("Enter a job description or hiring query to get the most relevant SHL assessments.")

# Input form
with st.form("query_form"):
    query = st.text_area("Enter your query or URL to a Job Description", height=150, 
                         placeholder="e.g. I am hiring for a Java developer who can collaborate with business teams. Need a test under 40 minutes.")
    top_k = st.number_input("Number of candidates to retrieve (top_k)", min_value=5, max_value=100, value=20)
    top_n = st.number_input("Number of final recommendations to show (top_n)", min_value=1, max_value=10, value=5)
    use_reranker = st.checkbox("Use reranker", value=False)
    submitted = st.form_submit_button("Get Recommendations")
    use_llm_explanations = True

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
                    "explanation": "Why Recommended (Test Type)"  # Updated name
                }

                # Only rename columns that exist in the DataFrame
                rename_cols = {k: v for k, v in column_mapping.items() if k in df.columns}
                df = df.rename(columns=rename_cols) 
                
                # Remove "**Explanation:**" prefix if present
                if "Why Recommended (Test Type)" in df.columns:
                    df["Why Recommended (Test Type)"] = df["Why Recommended (Test Type)"].apply(
                        lambda x: x.replace("**Explanation:** ", "") if isinstance(x, str) and x.startswith("**Explanation:**") else x
                    )
                
                # Define display columns (prioritize required columns)
                display_cols = ["Assessment", "Duration (min)", "Remote", "Adaptive", "Test Types"]
                
                # Add optional columns if they exist
                optional_cols = ["Job Levels", "Languages", "Why Recommended (Test Type)"]
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