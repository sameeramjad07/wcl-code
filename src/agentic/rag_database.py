"""
RAG Database for Physics Rules
===============================

Uses FAISS for efficient similarity search of EM physics principles.

Author: Research Team
Date: January 2026
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple, Optional
import json


class RAGDatabase:
    """Retrieval-Augmented Generation database for physics rules."""
    
    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        index_path: Optional[str] = None
    ):
        """
        Initialize RAG database.
        
        Args:
            embedding_model: Sentence transformer model
            index_path: Path to save/load FAISS index
        """
        self.encoder = SentenceTransformer(embedding_model)
        self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        self.index_path = index_path
        
        self.rules = []  # List of physics rules
        self.embeddings = []  # Corresponding embeddings
        
    def add_rule(self, rule: Dict):
        """
        Add physics rule to database.
        
        Args:
            rule: Dictionary with keys: 'title', 'condition', 'action', 'priority'
        """
        # Create text representation
        text = f"{rule['title']}: {rule['condition']} -> {rule['action']}"
        
        # Encode
        embedding = self.encoder.encode(text, convert_to_numpy=True)
        
        # Add to index
        self.index.add(np.array([embedding], dtype=np.float32))
        
        # Store rule
        self.rules.append(rule)
        self.embeddings.append(embedding)
    
    def add_rules_from_json(self, json_path: str):
        """Load rules from JSON file."""
        with open(json_path, 'r') as f:
            rules = json.load(f)
        
        for rule in rules:
            self.add_rule(rule)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 3
    ) -> List[Tuple[Dict, float]]:
        """
        Retrieve most relevant physics rules.
        
        Args:
            query: Query string (user situation description)
            top_k: Number of rules to retrieve
            
        Returns:
            List of (rule, score) tuples
        """
        # Encode query
        query_embedding = self.encoder.encode(query, convert_to_numpy=True)
        
        # Search
        distances, indices = self.index.search(
            np.array([query_embedding], dtype=np.float32),
            top_k
        )
        
        # Return rules with scores
        results = []
        for i, idx in enumerate(indices[0]):
            if idx < len(self.rules):
                results.append((self.rules[idx], float(distances[0][i])))
        
        return results
    
    def save(self, path: str):
        """Save database."""
        # Save FAISS index
        faiss.write_index(self.index, f"{path}.index")
        
        # Save rules
        with open(f"{path}.json", 'w') as f:
            json.dump(self.rules, f, indent=2)
    
    def load(self, path: str):
        """Load database."""
        # Load FAISS index
        self.index = faiss.read_index(f"{path}.index")
        
        # Load rules
        with open(f"{path}.json", 'r') as f:
            self.rules = json.load(f)