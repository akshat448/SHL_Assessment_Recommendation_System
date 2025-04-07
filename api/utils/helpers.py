import requests
from bs4 import BeautifulSoup
import logging
from typing import Dict, Any
import json
import os
from google.genai import Client, types

logger = logging.getLogger(__name__)

# Setup Google Gemini API
try:
    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        client = Client(api_key=api_key)
    else:
        logger.warning("GEMINI_API_KEY not set in environment variables")
except Exception as e:
    logger.error(f"Error configuring Gemini AI: {e}")

def scrape_job_description(url: str) -> str:
    """
    Scrape the job description from a given URL.

    Args:
        url: The URL of the job posting.

    Returns:
        The extracted job description as a string.
    """
    try:
        logger.info(f"Scraping job description from URL: {url}")
        # Send a GET request to the URL
        response = requests.get(url, timeout=10)
        response.raise_for_status()

        # Parse the HTML content
        soup = BeautifulSoup(response.text, "html.parser")

        # LinkedIn-specific job description container
        jd_container = soup.find("div", {"class": "jobs-description__content"})
        if jd_container:
            # Extract the text content from the container
            return jd_container.get_text(strip=True)

        # General fallback for other job description containers
        jd_container = soup.find("div", {"class": "description__text"})  # LinkedIn-specific fallback
        if not jd_container:
            jd_container = soup.find("div", {"class": "job-description"})  # General fallback

        # Extract and clean the text
        if jd_container:
            return jd_container.get_text(strip=True)
        else:
            raise ValueError("Job description not found on the page.")
    except Exception as e:
        logger.error(f"Failed to scrape job description: {e}")
        raise ValueError(f"Failed to scrape job description: {e}")

def summarize_job_description(job_description: str) -> str:
    """
    Create a concise summary of a job description while preserving key information.
    
    Args:
        job_description: The full job description text
        
    Returns:
        A summarized version of the job description
    """
    logger.info(f"Creating summary of job description ({len(job_description)} chars)")
    
    try:
        # Use Gemini to create a summary
        prompt = """
        Create a concise summary of this job description. Include all key information such as:
        - Role title and seniority level
        - Required technical skills and experience
        - Education requirements
        - Soft skills and behavioral traits needed
        - Job responsibilities
        - Team structure and reporting relationships
        - Industry information
        
        Keep the summary focused and concise (under 300 words) while preserving all important details.
        Do not add any information not present in the original text.
        
        JOB DESCRIPTION:
        """
        
        if len(job_description) > 15000:
            # For very long JDs, truncate but keep the most important parts (usually at beginning)
            job_description = job_description[:15000] + "..."
        
        response = client.models.generate_content(
            prompt + "\n\n" + job_description,
            generation_config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=350,
            )
        )
        
        summary = response.candidates[0].content.parts[0].text
        logger.info(f"Created summary: {len(summary)} chars")
        return summary
    
    except Exception as e:
        logger.error(f"Error generating JD summary: {e}")
        # If summary generation fails, return a truncated version of the original
        truncated = job_description[:1000] + "..." if len(job_description) > 1000 else job_description
        logger.info(f"Falling back to truncated JD: {len(truncated)} chars")
        return truncated