import os
import json
from google import genai
from google.genai import types
from typing import Dict, Any
from dotenv import load_dotenv

load_dotenv()

class QueryEnhancerAgent:
    """
    Agent that enhances user queries by extracting structured metadata
    using Google's Gemini Flash 2.0 lite model with function calling.
    """
    
    def __init__(self, prompt_path: str = "agents/promts/query_enhancer_promt.txt"):
        """
        Initialize the Query Enhancer agent with the Gemini API key and prompt.
        
        Args:
            prompt_path: Path to the prompt template file
        """
        # Set up Gemini API
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set")
        
        self.client = genai.Client(api_key=api_key)
        self.model_name = "gemini-2.0-flash-lite"
        
        # Use default prompt path if none provided
        if prompt_path is None:
            # Use a path relative to the current file
            current_dir = os.path.dirname(os.path.abspath(__file__))
            prompt_path = os.path.join(current_dir, "prompts", "query_enhancer_prompt.txt")
            
        # Load prompt for context
        try:
            with open(prompt_path, "r") as f:
                self.prompt_template = f.read()
        except FileNotFoundError:
            print(f"Warning: Prompt file not found at {prompt_path}. Using default instructions.")
            self.prompt_template = self._get_default_prompt()
        
        # Add job description specific prompt
        self._job_description_prompt = """
        Extract key assessment-relevant information from this job description:

        1. skills: List the technical and soft skills mentioned in the job description
        2. job_levels: Identify the job level (Entry-Level, Mid-Professional, Professional Individual Contributor, Manager, Director) 
        3. duration_minutes: If specified, maximum assessment time in minutes (or null)
        4. test_types: Relevant assessment types from: Ability and Aptitude, Personality and Behavior, Knowledge and Skills
        5. languages: Any languages mentioned as requirements
        6. remote_testing: Whether remote testing is requested (true/false/null)
        7. adaptive_support: Whether adaptive testing is needed (true/false/null)

        Format your response as a valid JSON object with these fields.
        """
        
        # Standard query prompt
        self._query_prompt = self.prompt_template

    def _get_default_prompt(self) -> str:
        """Return default prompt instructions if file is not found."""
        return """
        Convert natural language hiring queries into structured format with skills, job levels,
        duration, test types, languages, remote testing and adaptive support information.
        """

    def enhance_query(self, query: str) -> Dict[str, Any]:
        """
        Enhance a natural language query by extracting structured fields.
        
        Args:
            query: User's natural language query (or job description summary)
            
        Returns:
            Dictionary with structured metadata
        """
        print(f"Enhancing query: '{query[:100]}...' ({len(query)} chars)")
        
        # Detect if this is a job description query
        is_job_description = query.startswith("Job Description:") or "job description" in query.lower()
        
        # Select the appropriate prompt
        prompt = self._job_description_prompt if is_job_description else self._query_prompt
        
        try:
            # Try using function calling with Gemini
            instructions = prompt
            
            # Define the function schema that will structure our output
            query_enhancer_schema = {
                "name": "extract_query_metadata",
                "description": "Extract structured metadata from a hiring query or job description",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "skills": {
                            "type": "array",
                            "description": "List of technical or behavioral skills mentioned or implied",
                            "items": {"type": "string"}
                        },
                        "job_levels": {
                            "type": "array",
                            "description": "Job levels from: Entry-Level, Mid-Professional, Professional Individual Contributor, Manager, Director",
                            "items": {"type": "string"}
                        },
                        "duration_minutes": {
                            "type": "integer",
                            "description": "Maximum assessment time in minutes (infer if not explicitly stated)"
                        },
                        "test_types": {
                            "type": "array",
                            "description": "Test types from: Ability and Aptitude, Personality and Behavior, Knowledge and Skills",
                            "items": {"type": "string"}
                        },
                        "languages": {
                            "type": "array",
                            "description": "Languages required for the test, default to English (USA) if not specified",
                            "items": {"type": "string"}
                        },
                        "remote_testing": {
                            "type": "boolean",
                            "description": "Whether remote testing is requested (true) or not (false)"
                        },
                        "adaptive_support": {
                            "type": "boolean",
                            "description": "Whether adaptive testing is needed (true) or not (false)"
                        }
                    },
                    "required": ["skills", "job_levels", "test_types", "languages"]
                }
            }
            
            # Add clarifying instructions along with the query
            prompt_message = f"""
            {instructions}
            
            Please analyze the following {("job description" if is_job_description else "query")} and extract structured metadata:
            
            Input: {query}
            """
            
            # Create content from prompt
            contents = [
                types.Content(
                    role="user", 
                    parts=[types.Part(text=prompt_message)]
                )
            ]
            
            # Create Tool and config objects as per the documentation
            tools = types.Tool(function_declarations=[query_enhancer_schema])
            config = types.GenerateContentConfig(
                tools=[tools],
                tool_config=types.ToolConfig(function_calling_config={"mode": "AUTO"}),
                temperature=0.2,
                max_output_tokens=300
            )
            
            # Call the model with function calling using the config parameter
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=config
            )
            
            # Extract the function call results
            function_call = response.candidates[0].content.parts[0].function_call
            
            if function_call and function_call.name == "extract_query_metadata":
                # Handle different function_call.args types (could be dict or str)
                if isinstance(function_call.args, dict):
                    result = function_call.args
                else:
                    result = json.loads(function_call.args)
                    
                # Add original query for reference (truncated if JD)
                if is_job_description:
                    # Get first 200 chars of the query to avoid huge metadata
                    result["original_query"] = query[:200] + "..." if len(query) > 200 else query
                else:
                    result["original_query"] = query
                    
                return result
            
            # Fall back to traditional method if function call fails
            print("Function calling failed, falling back to text parsing")
            return self._fallback_with_specific_prompt(query, prompt, is_job_description)
                
        except Exception as e:
            print(f"Error enhancing query with function calling: {e}")
            # Try fallback method
            return self._fallback_with_specific_prompt(query, prompt, is_job_description)
    
    def _fallback_with_specific_prompt(self, query: str, prompt: str, is_job_description: bool) -> Dict[str, Any]:
        """Fallback extraction using the specific prompt type."""
        try:
            # Generate response from Gemini without function calling
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=[types.Content(role="user", parts=[
                    types.Part(text=f"{prompt}\n\nPlease provide a valid JSON output for this input:\n\n{query}")
                ])],
                generation_config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=300
                )
            )
            
            # Extract text from response
            response_text = response.candidates[0].content.parts[0].text
            
            # Find JSON data in the response (between curly braces)
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response_text[json_start:json_end]
                # Parse the JSON
                try:
                    result = json.loads(json_str)
                    # Add original query for reference (truncated if JD)
                    if is_job_description:
                        result["original_query"] = query[:200] + "..." if len(query) > 200 else query
                    else:
                        result["original_query"] = query
                        
                    # Ensure all required fields exist
                    result.setdefault("skills", [])
                    result.setdefault("job_levels", [])
                    result.setdefault("duration_minutes", None)
                    result.setdefault("test_types", [])
                    result.setdefault("languages", ["English (USA)"])
                    result.setdefault("remote_testing", None)
                    result.setdefault("adaptive_support", None)
                    
                    return result
                except json.JSONDecodeError:
                    print(f"Error parsing JSON: {json_str}")
                    return self._fallback_parsing(response_text, query, is_job_description)
            else:
                # Fallback to manual parsing
                return self._fallback_parsing(response_text, query, is_job_description)
        except Exception as e:
            print(f"Fallback extraction failed: {e}")
            return self._default_result(query, is_job_description)
    
    def _fallback_parsing(self, text: str, query: str, is_job_description: bool = False) -> Dict[str, Any]:
        """Manual parsing as a fallback when JSON parsing fails."""
        result = self._default_result(query, is_job_description)
        
        # Simple extraction based on keywords
        lines = text.split('\n')
        for line in lines:
            line = line.strip()
            if '"skills":' in line:
                try:
                    skills_text = line.split('[')[1].split(']')[0]
                    result["skills"] = [s.strip(' "\'') for s in skills_text.split(',')]
                except (IndexError, KeyError):
                    pass
            elif '"job_levels":' in line:
                try:
                    levels_text = line.split('[')[1].split(']')[0]
                    result["job_levels"] = [l.strip(' "\'') for l in levels_text.split(',')]
                except (IndexError, KeyError):
                    pass
            elif '"duration_minutes":' in line:
                try:
                    result["duration_minutes"] = int(line.split(':')[1].strip(' ,'))
                except (ValueError, IndexError):
                    pass
            elif '"test_types":' in line:
                try:
                    types_text = line.split('[')[1].split(']')[0]
                    result["test_types"] = [t.strip(' "\'') for t in types_text.split(',')]
                except (IndexError, KeyError):
                    pass
            elif '"languages":' in line:
                try:
                    languages_text = line.split('[')[1].split(']')[0]
                    result["languages"] = [l.strip(' "\'') for l in languages_text.split(',')]
                except (IndexError, KeyError):
                    pass
            elif '"remote_testing":' in line:
                try:
                    value = line.split(':')[1].strip(' ,').lower()
                    result["remote_testing"] = True if value == "true" else (False if value == "false" else None)
                except (IndexError, ValueError):
                    pass
            elif '"adaptive_support":' in line:
                try:
                    value = line.split(':')[1].strip(' ,').lower()
                    result["adaptive_support"] = True if value == "true" else (False if value == "false" else None)
                except (IndexError, ValueError):
                    pass
                
        return result
    
    def enhance_job_description(self, job_description: str) -> Dict[str, Any]:
        """
        Process a job description to extract structured metadata.
        This is a specialized version of enhance_query for handling longer job descriptions.
        
        Args:
            job_description: The scraped job description text
            
        Returns:
            Dictionary with structured metadata
        """
        # Just use enhance_query with a job description prefix
        return self.enhance_query(f"Job Description: {job_description}")
    
    def _create_default_metadata(self, job_description: str) -> Dict[str, Any]:
        """Create a default metadata structure with the job description."""
        return {
            "skills": [],
            "job_levels": ["Mid-Professional"],  # Default to mid-professional
            "test_types": ["Ability and Aptitude", "Knowledge and Skills"],
            "languages": ["English (USA)"],
            "duration_minutes": None,
            "remote_testing": None,
            "adaptive_support": None,
            "original_query": job_description[:200] + "..." if len(job_description) > 200 else job_description
        }
    
    def _default_result(self, query: str, is_job_description: bool = False) -> Dict[str, Any]:
        """Return a default result structure."""
        if is_job_description:
            return self._create_default_metadata(query)
            
        return {
            "skills": [],
            "job_levels": [],
            "duration_minutes": None,
            "test_types": [],
            "languages": ["English (USA)"],
            "remote_testing": None,
            "adaptive_support": None,
            "original_query": query
        }

# Test the agent if run directly
if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Instantiate the agent
    enhancer = QueryEnhancerAgent()
    
    # Test query
    test_query = "Looking to hire mid-level professionals who are proficient in Python, SQL and Java Script. Need an assessment package that can test all skills with max duration of 60 minutes."
    
    # Process the query
    result = enhancer.enhance_query(test_query)
    
    # Print the result
    print(json.dumps(result, indent=2))
    
    # Test with a job description
    test_jd = "Job Description: We are looking for a Senior Software Engineer with 5+ years of experience in Python, JavaScript, and React. The candidate should have strong communication skills and be able to work in an Agile environment."
    
    # Process the job description
    jd_result = enhancer.enhance_query(test_jd)
    
    # Print the result
    print("\nJob Description Result:")
    print(json.dumps(jd_result, indent=2))