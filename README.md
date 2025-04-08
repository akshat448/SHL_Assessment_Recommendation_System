# SHL Assessment Recommendation System

A comprehensive system for recommending SHL assessments based on natural language queries using AI agents, vector search, and NLP techniques.

## Table of Contents
- [Overview](#overview)
- [Project Architecture](#project-architecture)
- [Directory Structure](#directory-structure)
- [Agent Components](#agent-components)
- [Technology Stack](#technology-stack)
- [Features](#features)
- [Setup and Installation](#setup-and-installation)
- [Usage](#usage)
- [Configuration](#configuration)

## Overview

The SHL Assessment Recommendation System helps talent acquisition professionals and HR teams find the most suitable assessment products from SHL's extensive catalog based on their specific hiring needs. By analyzing natural language queries (including job descriptions), the system recommends appropriate assessment tools, saving time and improving assessment selection accuracy.

## Project Architecture

The system follows a multi-agent architecture with a pipeline approach:

User Query → Query Enhancer → Retriever → Reranker → Output Formatter → Response


- **Query Enhancer**: Extracts structured metadata from natural language queries
- **Retriever**: Searches for relevant assessments using vector similarity and metadata filtering
- **Reranker**: Re-ranks search results to improve relevance (optional)
- **Output Formatter**: Formats results with explanations for recommendations

The service is deployed as a containerized application with:
- Streamlit frontend for user interaction
- FastAPI backend for processing requests
- Docker containers for easy deployment

## Agent Components

### 1. Query Enhancer Agent
- **Purpose**: Extracts structured metadata from natural language queries
- **Implementation**: Uses Google's Gemini Flash 2.0 lite model with function calling
- **Key features**:
  - Identifies test types, job levels, skills, and constraints
  - Handles both simple queries and full job descriptions
  - Converts unstructured text to structured JSON metadata
  - Outputs test_types, job_levels, languages, duration_minutes, etc.

### 2. Retriever Agent
- **Purpose**: Finds relevant assessments based on query and metadata
- **Implementation**: Uses vector search and metadata filtering
- **Key features**:
  - Combined search strategy (filtered and semantic)
  - Supports exact metadata filtering (test types, job levels, etc.)
  - Uses semantic similarity for content matching

### 3. Reranker Agent
- **Purpose**: Re-ranks retrieved assessments for better relevance
- **Implementation**: Uses BAAI/bge-reranker-v2-m3 model
- **Key features**:
  - Cross-encoder architecture for precise ranking
  - Query-document pair scoring
  - Can be disabled to improve performance when not needed

### 4. Output Formatter Agent
- **Purpose**: Formats results with explanations
- **Implementation**: Uses rule-based and LLM-based explanation generation
- **Key features**:
  - Formats recommendations into consistent structure
  - Generates natural language explanations for recommendations
  - Handles test type mapping and standardization
  - Falls back to rule-based explanations if LLM fails

### 5. Agent Manager
- **Purpose**: Coordinates the agent workflow
- **Implementation**: Orchestrates the entire pipeline
- **Key features**:
  - Manages flow from query to formatted results
  - Handles errors and fallbacks
  - Provides timing and debugging information

## Technology Stack

- **Frontend**: Streamlit
- **Backend**: FastAPI
- **Models**:
  - Google's Gemini Flash 2.0 lite model (for query enhancement and explanations)
  - Vector embedding models for semantic search
  - BAAI/bge-large-en-v1.5 for embedding generation
  - BAAI/bge-reranker-v2-m3 for result re-ranking
- **Vector Database**: Qdrant
- **Containerization**: Docker
- **Programming Language**: Python 3.13+

## Features

The system can process both natural language queries and URLs to job descriptions:

1. **Natural Language Queries**:
   - Example: "I am hiring for a Python developer with SQL skills. Need a test under 30 minutes."
   - The Query Enhancer extracts structured metadata such as:
     - Skills: Python, SQL
     - Job Levels: Mid-Professional
     - Duration: 30 minutes
     - Test Types: Knowledge and Skills

2. **Job Description URLs**:
   - Example: Provide a URL to a job description (e.g., `https://example.com/job-description`).
   - The system fetches the content of the URL and extracts relevant context using NLP techniques.
   - This context is used to enhance the query and find the best matching assessments.

Both approaches ensure that the recommendations are tailored to the user's specific hiring needs.

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd SHL
2. Create and activate a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
3. Install the requirements:
    ```bash
    pip install -r requirements.txt
4. Setup environment variables:

    Ensure you have configured the necessary environment variables as described in the [Configuration](#configuration) section. Create a `.env` file in the root directory and populate it with the required keys.

5. Launch the application:

    Start the application using Docker Compose:
    ```bash
    docker-compose up --build
    ```
    This will start both the Streamlit frontend and the FastAPI backend.


---

### Usage

### Streamlit Interface

1. Navigate to `http://localhost:8501` in your browser.
2. Enter your query in the text area. You can:
   - Provide a **natural language query** (e.g., "I am hiring for a Java developer who can collaborate with business teams. Need a test under 40 minutes.")
   - Provide a **URL to a job description**. The system will extract relevant context from the URL.
3. Adjust parameters such as:
   - `Number of candidates to retrieve (top_k)`
   - `Number of final recommendations to show (top_n)`
   - Enable or disable the reranker
4. Click "Get Recommendations" to see results.

## API Endpoints

The backend provides several API endpoints for programmatic access:

### **POST `/recommend`**
- **Description**: Get basic assessment recommendations without explanations
- **Parameters**: 
  - `query` (string): Natural language query describing the assessment needs
- **Response Format**:
  ```json
  {
    "recommended_assessments": [
      {
        "url": "Valid URL in string",
        "adaptive_support": "Yes/No",
        "description": "Description in string",
        "duration": 60,
        "remote_support": "Yes/No",
        "test_type": ["List of string"]
      }
    ]
  }
    ```
### **GET `/recommend`**
- **Description**: Get basic assessment recommendations without explanations
- **Parameters**: 
  - `query` (string): Natural language query describing the assessment needs
- **Response Format**:
  ```json
  {
    "recommended_assessments": [
      {
        "url": "Valid URL in string",
        "adaptive_support": "Yes/No",
        "description": "Description in string",
        "duration": 60,
        "remote_support": "Yes/No",
        "test_type": ["List of string"]
      }
    ]
  }
  ```
### **POST `/dashboard`**

- **Description**: Enhanced endpoint specifically designed for the dashboard interface.
- **Parameters**:
  - `query` (string): Natural language query or URL to a job description.
  - `top_k` (integer, default: 25): Maximum number of candidates to retrieve.
  - `top_n` (integer, default: 10): Maximum number of final recommendations to return.
  - `use_reranker` (boolean, default: true): Whether to use the reranker for improved results.
  - `use_llm_explanations` (boolean, default: false): Whether to include explanations for recommendations.
- **Response Format**:
```json
{
  "recommended_assessments": [
    {
      "url": "Valid URL in string",
      "adaptive_support": "Yes/No",
      "description": "Description in string",
      "duration": 60,
      "remote_support": "Yes/No",
      "test_type": ["List of string"],
      "explanation": "Optional explanation when use_llm_explanations=true"
    }
  ]
}
```

### **GET `/health`**

- **Description**: Simple health check endpoint to verify if the service is running properly.
- **Response Format**:
```json
{
  "status": "healthy"
}
```



## Streamlit Frontend

The frontend provides a user-friendly interface to interact with the API:

### Assessment Recommendations Page
- Enter a job description or hiring query or URL to JD
- Configure parameters:
  - Number of candidates to retrieve (`top_k`)
  - Number of final recommendations (`top_n`) 
  - Toggle reranker usage for better results (may increase processing time)
  - Toggle LLM explanations to get reasoning for each recommendation
- View results in a formatted table with:
  - Clickable assessment names linked to their respective URLs
  - Duration, test types, and other assessment metadata
  - Explanation column (when LLM explanations are enabled)
- Download results as CSV or JSON

### Health Check Page
- Verify the API's operational status
- View complete health check response

## Configuration

### Environment Variables

The application uses a `.env` file for configuration. Below are the key variables:

- `API_URL`: The base URL for the backend API (default: `http://localhost:8080/api/recommendations/`)
- `GEMINI_API_KEY`: API key for Google's Gemini Flash 2.0 lite model
-  `Together_API_KEY`: API key for Together AI's bge-large-en-v1.5
- `QDRANT_API_KEY`: API key for Qdrant vector database
- `QDRANT_URL`: URL for the Qdrant instance

### Agent Settings

You can configure agent behavior in `agents/config.py`:
- Enable or disable the reranker
- Adjust the number of candidates retrieved (`top_k`) and final recommendations (`top_n`)
- Modify LLM prompt templates in `agents/prompts/`

### Docker Configuration

Modify `docker-compose.yml` to adjust container settings, such as ports and resource limits.
