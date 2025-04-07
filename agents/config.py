"""
Configuration settings for the multi-agent system.
"""

# Vector database configuration
QDRANT_COLLECTION_NAME = "shl_assessments_with_metadata"
MODEL_CACHE_DIR = "/Users/akshat/Developer/Tasks/SHL/.model_cache"

# Agent configuration
DEFAULT_RESULTS_COUNT = 5
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Job level mappings (for query enhancer)
JOB_LEVEL_MAPPINGS = {
    "entry": "Entry-Level",
    "junior": "Entry-Level",
    "mid": "Mid-Professional",
    "senior": "Professional Individual Contributor",
    "lead": "Manager",
    "manager": "Manager",
    "director": "Director",
    "executive": "Director"
}

# Test type mappings (for query enhancer)
TEST_TYPE_MAPPING = {
    "A": "Ability and Aptitude",
    "B": "Biodata and Situational Judgement",
    "C": "Competencies",
    "D": "Development and 360",
    "E": "Assessment Exercises",
    "K": "Knowledge and Skills",
    "P": "Personality and Behavior",
    "S": "Simulations"
}

# Path configurations
PROMPTS_DIR = "agents/prompts"
QUERY_ENHANCER_PROMPT_PATH = f"{PROMPTS_DIR}/query_enhancer_prompt.txt"
OUTPUT_FORMATTER_PROMPT_PATH = f"{PROMPTS_DIR}/output_formatter_prompt.txt"