from typing import Dict, List, Any
import os
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

class OutputFormatterAgent:
    """
    Agent that formats search results into user-friendly output with explanations.
    """
    
    def __init__(self, use_llm: bool = True, prompt_path: str = "agents/prompts/output_formatter.txt"):
        """Initialize the Output Formatter agent."""
        self.use_llm = use_llm
        self.prompt_path = prompt_path
        
        # Load the prompt from the file
        self.prompt_template = self._load_prompt()

        # Initialize LLM for explanations if enabled
        if self.use_llm:
            try:
                api_key = os.getenv("GEMINI_API_KEY")
                if not api_key:
                    print("Warning: GEMINI_API_KEY not found in environment. Explanations will be basic.")
                    self.llm_available = False
                else:
                    # Initialize client properly
                    self.client = genai.Client(api_key=api_key)
                    self.llm_available = True
                    print("LLM initialized for generating explanations")
            except Exception as e:
                print(f"Error initializing LLM: {e}")
                self.llm_available = False
        else:
            self.llm_available = False

    def _load_prompt(self) -> str:
        """
        Load the prompt template from the specified file.
        
        Returns:
            The prompt template as a string.
        """
        try:
            with open(self.prompt_path, "r") as f:
                return f.read()
        except FileNotFoundError:
            print(f"Warning: Prompt file not found at {self.prompt_path}. Using default instructions.")
            return self._get_default_prompt()

    def _get_default_prompt(self) -> str:
        """Return default prompt instructions if file is not found."""
        return "Write a short, professional explanation for why an assessment is recommended."

    def _ensure_none_values(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert any JSON null values or string 'null' to Python None.
        
        Args:
            data: Dictionary to process
            
        Returns:
            Processed dictionary with null values converted to None
        """
        if not isinstance(data, dict):
            return data
            
        result = {}
        for key, value in data.items():
            if value == "null" or value == "undefined" or value is None:
                result[key] = None
            elif isinstance(value, dict):
                result[key] = self._ensure_none_values(value)
            elif isinstance(value, list):
                result[key] = [
                    self._ensure_none_values(item) if isinstance(item, dict) else
                    None if item == "null" or item == "undefined" else item
                    for item in value
                ]
            else:
                result[key] = value
        return result
    
    # def format_results(self, reranked_results: List[Dict[str, Any]], query: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
    #     """
    #     Format reranked search results into a user-friendly structure.
        
    #     Args:
    #         reranked_results: List of assessment dictionaries from RerankerAgent
    #         query: Original user query
    #         metadata: Dictionary with extracted metadata from QueryEnhancer
            
    #     Returns:
    #         Dictionary with formatted results in the new structure
    #     """
    #     # Ensure metadata fields are properly converted from null to None
    #     metadata = self._ensure_none_values(metadata)
        
    #     # Create the main output structure
    #     formatted_output = {
    #         "query": query,
    #         "metadata": metadata,  # Include original metadata for reference
    #         "recommendations": []
    #     }
        
    #     # Process each assessment result
    #     for i, result in enumerate(reranked_results):
    #         # Convert results from null to None
    #         result = self._ensure_none_values(result)
    #         payload = result.get("metadata", {})
    #         payload = self._ensure_none_values(payload)
            
    #         # Generate explanation for this recommendation
    #         explanation = self.generate_explanation(
    #             query, metadata, payload, i+1
    #         )
            
    #         # Format the assessment data in the requested structure
    #         assessment = {
    #             "rank": i + 1,
    #             "assessment_name": payload.get("assessment_name", "Unknown Assessment"),
    #             "assessment_url": payload.get("url", ""),
    #             "remote_testing": payload.get("remote_testing", None),
    #             "adaptive_support": payload.get("adaptive_support", None),
    #             "duration_minutes": payload.get("duration_minutes"),
    #             "job_levels": payload.get("job_levels", []),
    #             "test_types": payload.get("test_type_categories", []),
    #             "languages": payload.get("languages", []),
    #             "explanation": explanation,
    #             "description": self._truncate_description(payload.get("description", "")),
    #             "source": result.get("source", "unknown")
    #         }
            
    #         # Add PDF links if available
    #         if "pdf_links" in payload and payload["pdf_links"]:
    #             assessment["pdf_links"] = payload["pdf_links"]
                
    #         # Add extracted text summary if available (truncated)
    #         if "extracted_text" in payload and payload["extracted_text"]:
    #             assessment["text_summary"] = self._truncate_description(payload["extracted_text"], max_length=200)
                
    #         formatted_output["recommendations"].append(assessment)
        
    #     return formatted_output
    
    
    def format_results(self, results: List[Dict[str, Any]], query: str, metadata: Dict[str, Any]) -> Any:
        """
        Format the results for output.
        """
        formatted_results = []
        for i, result in enumerate(results):
            # Create a new result object with standardized fields
            formatted_result = {
                "rank": result.get("rank"),
                "assessment_name": result["metadata"].get("assessment_name"),
                "assessment_url": result["metadata"].get("url"),
                "remote_testing": result["metadata"].get("remote_testing"),
                "adaptive_support": result["metadata"].get("adaptive_support"),
                "duration_minutes": result["metadata"].get("duration_minutes"),
                "job_levels": result["metadata"].get("job_levels"),
                # Standardize on test_types (use test_type_categories if available)
                "test_types": result["metadata"].get("test_type_categories", []),
                "languages": result["metadata"].get("languages"),
                "description": result["metadata"].get("description"),
                "source": result.get("source"),
                "pdf_links": result["metadata"].get("pdf_links"),
            }
            
            # Add explanation if we're using LLM - FIX: Call with correct parameters
            if self.use_llm:
                formatted_result["explanation"] = self.generate_explanation(
                    query, 
                    metadata, 
                    result["metadata"],  # Pass the source metadata, not the formatted result
                    i+1  # Pass the rank
                )
                    
            formatted_results.append(formatted_result)
            
        return {
            "query": query,
            "metadata": metadata,
            "recommendations": formatted_results
        }
    
    def _truncate_description(self, description: str, max_length: int = 300) -> str:
        """Helper method to truncate long descriptions."""
        if not description:
            return ""
        if len(description) > max_length:
            return description[:max_length - 3] + "..."
        return description
    
    def generate_explanation(self, query: str, metadata: Dict[str, Any], assessment: Dict[str, Any], rank: int) -> str:
        """
        Generate an explanation for why this assessment was recommended.
        
        Args:
            query: Original query string
            metadata: Query metadata
            assessment: Assessment details
            rank: The ranking position
            
        Returns:
            Explanation string
        """
        if not self.use_llm or not self.llm_available:
            return self._rule_based_explanation(query, metadata, assessment, rank)
        else:
            try:
                # Create a prompt for structured explanation generation
                explanation = self._llm_explanation(query, metadata, assessment, rank)
                if explanation:
                    return explanation
                # Fall back to rule-based if LLM fails or returns empty
                return self._rule_based_explanation(query, metadata, assessment, rank)
            except Exception as e:
                print(f"Error generating LLM explanation: {e}")
                # Fall back to rule-based explanation
                return self._rule_based_explanation(query, metadata, assessment, rank)
    
    def _rule_based_explanation(self, query: str, metadata: Dict[str, Any], assessment: Dict[str, Any], rank: int) -> str:
        """Generate explanation using rule-based approach."""
        reasons = []
        
        # Check job level match
        if metadata.get("job_levels") and assessment.get("job_levels"):
            if any(level in assessment["job_levels"] for level in metadata["job_levels"]):
                reasons.append("matching job level requirements")
                
        # Check test type match
        # if metadata.get("test_types") and assessment.get("test_type_categories"):
        #     matching_types = [t for t in metadata["test_types"] if t in assessment.get("test_type_categories", [])]
        #     if matching_types:
        if metadata.get("test_types"):
            test_types_in_assessment = assessment.get("test_types", assessment.get("test_type_categories", []))
            matching_types = [t for t in metadata["test_types"] if t in test_types_in_assessment]
            if matching_types:
                if len(matching_types) == 1:
                    reasons.append(f"includes {matching_types[0]} assessment")
                else:
                    reasons.append(f"includes {' and '.join(matching_types)} assessments")
        
        # Check skills (looking in description)
        if metadata.get("skills") and assessment.get("description"):
            matched_skills = []
            for skill in metadata["skills"]:
                if skill.lower() in assessment["description"].lower():
                    matched_skills.append(skill)
            if matched_skills:
                reasons.append(f"focuses on {', '.join(matched_skills)}")
        
        # Check remote testing if specified
        if metadata.get("remote_testing") is not None:
            remote_value = "Yes" if metadata["remote_testing"] else "No"
            if assessment.get("remote_testing") == remote_value:
                remote_text = "remote testing support" if metadata["remote_testing"] else "in-person testing option"
                reasons.append(remote_text)
        
        # Check duration if specified
        if metadata.get("duration_minutes") and assessment.get("duration_minutes"):
            if assessment.get("duration_minutes") <= metadata["duration_minutes"]:
                reasons.append(f"completion time within {metadata['duration_minutes']} minutes")
        
        if not reasons:
            if rank <= 3:
                return "This assessment closely aligns with your search query based on semantic similarity and content relevance."
            else:
                return "This assessment covers aspects relevant to your search requirements."
        else:
            return "Recommended for " + ", ".join(reasons) + "."
    
    def _llm_explanation(self, query: str, metadata: Dict[str, Any], assessment: Dict[str, Any], rank: int) -> str:
        """Generate explanation using LLM."""
        try:
            # Use the loaded prompt template
            prompt_message = self.prompt_template.format(
                query=query,
                assessment_name=assessment.get("assessment_name", "Unknown Assessment"),
                job_levels=", ".join(assessment.get("job_levels", ["Not specified"])),
                test_types=", ".join(assessment.get("test_type_categories", ["Not specified"])),
                duration_minutes=assessment.get("duration_minutes", "Not specified"),
                remote_testing=assessment.get("remote_testing", "Not specified"),
                adaptive_support=assessment.get("adaptive_support", "Not specified"),
                languages=", ".join(assessment.get("languages", ["Not specified"])),
                description=assessment.get("description", "Not available")
            )

            # Create content from prompt
            contents = [
                types.Content(
                    role="user", 
                    parts=[types.Part(text=prompt_message)]
                )
            ]
            
            # Create Tool and config objects
            config = types.GenerateContentConfig(
                temperature=0.5,
                max_output_tokens=300
            )
            
            # Call the model
            response = self.client.models.generate_content(
                model="gemini-2.0-flash-lite",
                contents=contents,
                config=config
            )
            
            explanation = ""  # Initialize explanation with empty string
            
            if hasattr(response, 'candidates') and len(response.candidates) > 0:
                candidate = response.candidates[0]
                if hasattr(candidate, 'content') and candidate.content:
                    explanation = candidate.content.parts[0].text
            
            return explanation.strip()
            
        except Exception as e:
            print(f"Error in LLM explanation generation: {e}")
            print(f"Assessment data: {assessment}")  # Add this to see the actual assessment data
            return ""  # Return empty string to trigger fallback