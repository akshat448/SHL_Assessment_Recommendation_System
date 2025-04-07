import os
import time
import uuid
import numpy as np
from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.exceptions import UnexpectedResponse
from dotenv import load_dotenv

load_dotenv()

class QdrantManager:
    def __init__(self, collection_name, vector_size):
        """Initialize QdrantManager with collection name and vector size."""
        self.qdrant_url = os.getenv("QDRANT_URL", None)
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY", None)
        if not self.qdrant_api_key:
            raise ValueError("QDRANT_API_KEY environment variable is not set.")
        if not self.qdrant_url:
            raise ValueError("QDRANT_URL environment variable is not set.")
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.client = self._connect_to_qdrant()
        self._ensure_collection()

    def _connect_to_qdrant(self):
        """Connect to the Qdrant server."""
        print(f"Connecting to Qdrant at {self.qdrant_url} with API key: {self.qdrant_api_key}")
        try:
            client = QdrantClient(
                url=self.qdrant_url,
                api_key=self.qdrant_api_key,
                timeout=10.0
            )
            # Test connection by getting collection list
            client.get_collections()
            print(f"Connected to Qdrant successfully at {self.qdrant_url}")
            return client
        except Exception as e:
            raise Exception(f"Failed to connect to Qdrant: {e}")

    def _ensure_collection(self):
        """Ensure the collection exists in Qdrant."""
        try:
            collections = self.client.get_collections().collections
            collection_names = [collection.name for collection in collections]

            if self.collection_name not in collection_names:
                print(f"Creating collection: {self.collection_name} with vector size: {self.vector_size}")
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=self.vector_size,
                        distance=models.Distance.COSINE
                    )
                )
                print(f"Collection '{self.collection_name}' created successfully.")
            else:
                print(f"Collection '{self.collection_name}' already exists.")
        except Exception as e:
            raise Exception(f"Error ensuring collection: {e}")

    def add_embedding(self, vector, payload, id=None):
        """Add embedding with metadata to Qdrant."""
        try:
            # Always use UUID for Qdrant point ID - this is required by Qdrant
            point_id = str(uuid.uuid4())
            
            # Store the original ID (if provided) in the payload to preserve it
            if id:
                payload["original_id"] = id
            
            # Convert numpy array to list if needed
            if isinstance(vector, np.ndarray):
                vector = vector.tolist()
            
            # Add embedding to Qdrant
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload
                    )
                ]
            )
            return point_id
        except Exception as e:
            print(f"Error adding embedding: {e}")
            import traceback
            traceback.print_exc()
            return None

    def search(self, query_vector, filter_conditions=None, limit=5):
        """Search for similar vectors in Qdrant with optional filtering."""
        try:
            # Convert numpy array to list if needed
            if isinstance(query_vector, np.ndarray):
                query_vector = query_vector.tolist()
            
            # Prepare search filter if provided
            search_filter = None
            if filter_conditions:
                search_filter = models.Filter(**filter_conditions)
            
            # Execute search
            results = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                query_filter=search_filter,
                limit=limit
            )
            
            return results
        except Exception as e:
            print(f"Error searching: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def store_embedding(self, ml_id: str, vector: np.ndarray) -> None:
        """Update embedding for existing ML ID"""
        try:
            # Check if ml_id exists
            search_results = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="ml_id",
                            match=models.MatchValue(value=ml_id)
                        )
                    ]
                ),
                limit=1
            )
            
            points, _ = search_results
            
            if not points:
                print(f"ML ID {ml_id} not found, creating a new entry")
                self.add_new_embedding(ml_id, vector)
            else:
                # Update the vector for the existing ml_id
                print(f"Updating embedding for ML ID {ml_id}")
                point_id = points[0].id
                self.client.update_vectors(
                    collection_name=self.collection_name,
                    points=[
                        models.PointVectors(
                            id=point_id,
                            vector=vector.tolist()
                        )
                    ]
                )
                print(f"Updated embedding for ML ID {ml_id}")
        except Exception as e:
            print(f"Error storing embedding: {e}")
            import traceback
            traceback.print_exc()
    
    def add_new_embedding(self, ml_id: str, vector: np.ndarray) -> Optional[str]:
        """Add a new embedding with a given ML ID"""
        try:
            # Generate a unique ID (Qdrant requires a unique point ID)
            point_id = str(uuid.uuid4())
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    models.PointStruct(
                        id=point_id,
                        vector=vector.tolist(),
                        payload={"ml_id": ml_id}
                    )
                ]
            )
            print(f"Added new embedding for ML ID {ml_id}")
            return ml_id
        except Exception as e:
            print(f"Error adding new embedding: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def generate_ml_id(self) -> str:
        """Generate a new unique ML ID"""
        return str(uuid.uuid4())