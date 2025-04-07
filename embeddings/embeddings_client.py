from together import Together
import os
from typing import List

class TogetherEmbeddingsClient:
    """A wrapper for Together.ai embeddings API."""
    
    def __init__(self, model_name="BAAI/bge-large-en-v1.5"):
        self.model_name = model_name
        
        # Get API key - MAKE SURE THIS ENVIRONMENT VARIABLE IS CORRECTLY SET
        together_api_key = os.getenv("TOGETHER_API_KEY")
        if not together_api_key:
            raise ValueError("TOGETHER_API_KEY not found in environment. Make sure it's correctly set.")
            
        # Initialize the client with the API key
        self.client = Together(api_key=together_api_key)
        print(f"Initialized Together AI embeddings client with model: {model_name}")
    
    def embed_query(self, text: str) -> List[float]:
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=text,
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"Error calling Together API for embeddings: {e}")
            raise
        
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        try:
            response = self.client.embeddings.create(
                model=self.model_name,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except Exception as e:
            print(f"Error calling Together API for batch embeddings: {e}")
            raise