import streamlit as st
import requests
import pandas as pd
import os
import json

# Backend API URLs
BASE_URL = os.environ.get("BASE_URL", "http://localhost:8080")
HEALTH_URL = f"{BASE_URL}/health"
RECOMMEND_URL = f"{BASE_URL}/recommend"

st.set_page_config(page_title="SHL Assessment Recommender", layout="wide")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Assessment Recommendations (POST)", "Query API (GET)", "Health Check"])

# Health Check Page
if page == "Health Check":
    st.title("🏥 Health Check")
    st.markdown("Check the health status of the SHL Assessment Recommendation API.")

    if st.button("Check Health Status"):
        with st.spinner("Checking API health..."):
            try:
                response = requests.get(HEALTH_URL)
                response.raise_for_status()
                data = response.json()
                
                if data.get("status") == "healthy":
                    st.success("✅ API is healthy!")
                else:
                    st.warning("⚠️ API returned an unexpected response.")
                
                st.json(data)
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Health check failed: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    st.error(f"Response status: {e.response.status_code}")
                    st.error(f"Response text: {e.response.text}")

# Assessment Recommendations Page (POST /recommend)
elif page == "Assessment Recommendations (POST)":
    st.title("🔍 SHL Assessment Recommendations")
    st.markdown("Enter a job description or hiring query to get the most relevant SHL assessments.")

    # Input form
    with st.form("recommend_form"):
        query = st.text_area("Enter your query or URL to a Job Description", height=150, 
                            placeholder="e.g. I am hiring for a Java developer who can collaborate with business teams. Need a test under 40 minutes.")
        
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.number_input("Number of candidates to retrieve (top_k)", min_value=5, max_value=100, value=25)
        
        with col2:
            top_n = st.number_input("Number of final recommendations to show (top_n)", min_value=1, max_value=10, value=5)
        
        col1, col2 = st.columns(2)
        with col1:
            use_reranker = st.checkbox("Use reranker", value=True)  # Default to true as per requirement
            if use_reranker:
                st.info("⚠️ Using the reranker will provide better results but may increase processing time.")
        
        with col2:
            use_llm_explanations = st.checkbox("Use LLM explanations", value=False)  # Default to false as per requirement
            if use_llm_explanations:
                st.info("ℹ️ Explanations will be provided for why each assessment was recommended.")
                
        submitted = st.form_submit_button("Get Recommendations")

    if submitted and query:
        with st.spinner("Processing your query..."):
            try:
                # Make POST request to /recommend endpoint 
                response = requests.post(
                    RECOMMEND_URL, 
                    json={
                        "query": query
                    }
                )
                response.raise_for_status()
                data = response.json()

                if not data.get("recommended_assessments") or len(data["recommended_assessments"]) == 0:
                    st.warning("No recommendations found for this query.")
                else:
                    st.success(f"Found {len(data['recommended_assessments'])} recommendations.")
                    
                    # Display format selection
                    display_format = st.radio("Display format:", ["Table", "JSON", "Both"])
                    
                    if display_format in ["Table", "Both"]:
                        # Display the table
                        st.markdown("### Recommended Assessments")
                        
                        # Convert recommended assessments to DataFrame
                        df = pd.DataFrame(data["recommended_assessments"])
                        
                        # Add clickable URL column
                        df["Assessment URL"] = df["url"].apply(
                            lambda url: f'<a href="{url}" target="_blank">{url}</a>' if url else ""
                        )
                        
                        # Rename columns for better display
                        column_mapping = {
                            "duration": "Duration (min)",
                            "remote_support": "Remote Testing",
                            "adaptive_support": "Adaptive Support",
                            "test_type": "Test Types",
                            "description": "Description"
                        }
                        df = df.rename(columns=column_mapping)
                        
                        # Format list columns
                        if "Test Types" in df.columns:
                            df["Test Types"] = df["Test Types"].apply(
                                lambda types: ", ".join(types) if isinstance(types, list) else types
                            )
                        
                        # Define display columns
                        display_cols = ["Assessment URL", "Duration (min)", "Remote Testing", 
                                       "Adaptive Support", "Test Types", "Description"]
                        
                        # Filter to only include columns that exist in the DataFrame
                        display_cols = [col for col in display_cols if col in df.columns]
                        
                        # Display the table
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
                    
                    if display_format in ["JSON", "Both"]:
                        # Display the JSON
                        st.markdown("### JSON Response")
                        st.json(data)
                        
                        # Add JSON download option
                        json_str = json.dumps(data, indent=2)
                        st.download_button(
                            "Download JSON",
                            json_str,
                            "shl_recommendations.json",
                            "application/json",
                            key='download-json-2'
                        )
                    
                    # Add curl command example for reference
                    st.markdown("### Example CURL Command")
                    curl_cmd = f"""curl -X POST "{RECOMMEND_URL}" -H "Content-Type: application/json" -d '{{"query": "{query}"}}'"""
                    st.code(curl_cmd, language="bash")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"API request failed: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    st.error(f"Response status: {e.response.status_code}")
                    st.error(f"Response text: {e.response.text}")

# Query API Page (GET /recommend)
elif page == "Query API (GET)":
    st.title("🔍 Query Recommendations API")
    st.markdown("Use the GET endpoint to fetch recommendations directly as JSON.")

    # Input form for GET request
    with st.form("query_form"):
        query = st.text_input("Enter your query", placeholder="e.g. I am hiring for a Java developer...")
        top_k = st.number_input("Number of candidates to retrieve (top_k)", min_value=5, max_value=100, value=25)
        top_n = st.number_input("Number of final recommendations to show (top_n)", min_value=1, max_value=10, value=5)
        submitted = st.form_submit_button("Fetch JSON")
    
    if submitted and query:
        with st.spinner("Fetching recommendations..."):
            try:
                # Make GET request to /recommend endpoint
                response = requests.get(
                    RECOMMEND_URL,
                    params={
                        "query": query,
                        "top_k": int(top_k),
                        "top_n": int(top_n),
                    }
                )
                
                # Print debugging info
                st.write(f"Status Code: {response.status_code}")
                st.write(f"Request URL: {response.request.url}")
                
                response.raise_for_status()
                data = response.json()

                if not data.get("recommended_assessments"):
                    st.warning("No recommendations found for this query.")
                else:
                    st.success(f"Found {len(data['recommended_assessments'])} recommendations.")
                    
                    # Display JSON only
                    st.markdown("### JSON Response")
                    st.json(data)
                    
                    # Add JSON download option
                    json_str = json.dumps(data, indent=2)
                    st.download_button(
                        "Download JSON",
                        json_str,
                        "shl_query_recommendations.json",
                        "application/json",
                        key='download-json'
                    )
                    
                    # Add curl command example for reference
                    st.markdown("### Example CURL Command")
                    curl_cmd = f"""curl "{RECOMMEND_URL}?query={query.replace(' ', '%20')}&top_k={top_k}&top_n={top_n}" """
                    st.code(curl_cmd, language="bash")
                    
            except requests.exceptions.RequestException as e:
                st.error(f"API request failed: {e}")
                if hasattr(e, 'response') and e.response is not None:
                    st.error(f"Response status: {e.response.status_code}")
                    st.error(f"Response text: {e.response.text}")