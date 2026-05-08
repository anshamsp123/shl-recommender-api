import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Optional

class CatalogIndex:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(CatalogIndex, cls).__new__(cls)
        return cls._instance

    def __init__(self, catalog_path: str, embed_model: str = "all-MiniLM-L6-v2"):
        if hasattr(self, 'initialized') and self.initialized:
            return
            
        self.catalog_path = catalog_path
        self.embed_model_name = embed_model
        
        try:
            with open(catalog_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load catalog from {catalog_path}: {e}")
            
        if not self.data:
            raise RuntimeError(f"Catalog at {catalog_path} is empty.")
            
        # Create a dictionary for O(1) lookups by name
        self.name_to_doc = {item["name"]: item for item in self.data}
        
        # Prepare text for embedding
        self.texts = [f"{item['name']}. {item.get('description', '')}" for item in self.data]
        
        # Initialize model and embed
        try:
            self.model = SentenceTransformer(embed_model)
            embeddings = self.model.encode(self.texts, convert_to_numpy=True)
            
            # Normalize embeddings for cosine similarity
            faiss.normalize_L2(embeddings)
            
            # Build index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatIP(dimension)
            self.index.add(embeddings)
            self.use_fallback = False
        except Exception as e:
            print(f"Warning: Failed to build FAISS index: {e}. Falling back to keyword search.")
            self.use_fallback = True
            
        self.initialized = True

    def _fallback_search(self, query: str, top_k: int) -> List[Dict]:
        query_words = set(query.lower().split())
        scored_items = []
        for item in self.data:
            text = f"{item['name']} {item.get('description', '')}".lower()
            score = sum(1 for w in query_words if w in text)
            scored_items.append((score, item))
            
        scored_items.sort(key=lambda x: x[0], reverse=True)
        return [item for score, item in scored_items[:top_k]]

    def search(self, query: str, top_k: int = 15) -> List[Dict]:
        if getattr(self, 'use_fallback', False):
            return self._fallback_search(query, top_k)
            
        query_emb = self.model.encode([query], convert_to_numpy=True)
        faiss.normalize_L2(query_emb)
        
        D, I = self.index.search(query_emb, top_k)
        
        results = []
        for idx in I[0]:
            if idx >= 0 and idx < len(self.data):
                results.append(self.data[idx])
        return results

    def get_by_name(self, name: str) -> Optional[Dict]:
        return self.name_to_doc.get(name)

    def get_by_names(self, names: List[str]) -> List[Dict]:
        results = []
        for name in names:
            if name in self.name_to_doc:
                results.append(self.name_to_doc[name])
        return results

# Singleton instance getter
def get_catalog() -> CatalogIndex:
    return CatalogIndex(catalog_path="data/catalog.json", embed_model="all-MiniLM-L6-v2")
