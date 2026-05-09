import json
import re
import math
from collections import Counter
from typing import List, Dict, Optional

class CatalogIndex:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(CatalogIndex, cls).__new__(cls)
        return cls._instance

    def __init__(self, catalog_path: str):
        if hasattr(self, 'initialized') and self.initialized:
            return
            
        self.catalog_path = catalog_path
        
        try:
            with open(catalog_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"Failed to load catalog from {catalog_path}: {e}")
            
        if not self.data:
            raise RuntimeError(f"Catalog at {catalog_path} is empty.")
            
        self.name_to_doc = {item["name"]: item for item in self.data}
        
        # Precompute BM25 statistics
        self.docs_tokens = []
        for item in self.data:
            text = f"{item['name']} {item.get('description', '')} {item.get('test_type', '')} {item.get('job_levels', '')}"
            self.docs_tokens.append(self._tokenize(text))
            
        self.k1 = 1.5
        self.b = 0.75
        self.doc_len = [len(tokens) for tokens in self.docs_tokens]
        self.avgdl = sum(self.doc_len) / len(self.docs_tokens) if self.docs_tokens else 1
        
        self.df = Counter()
        for tokens in self.docs_tokens:
            self.df.update(set(tokens))
            
        N = len(self.docs_tokens)
        self.idf = {word: math.log((N - freq + 0.5) / (freq + 0.5) + 1) for word, freq in self.df.items()}
            
        self.initialized = True

    def _tokenize(self, text: str) -> List[str]:
        return re.findall(r'\w+', text.lower())

    def search(self, query: str, top_k: int = 15) -> List[Dict]:
        query_tokens = self._tokenize(query)
        scores = []
        for i, tokens in enumerate(self.docs_tokens):
            score = 0
            doc_counts = Counter(tokens)
            for qt in query_tokens:
                if qt in doc_counts:
                    tf = doc_counts[qt]
                    numerator = tf * (self.k1 + 1)
                    denominator = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avgdl)
                    score += self.idf.get(qt, 0) * (numerator / denominator)
            scores.append((score, self.data[i]))
            
        scores.sort(key=lambda x: x[0], reverse=True)
        return [item for score, item in scores[:top_k]]

    def get_by_name(self, name: str) -> Optional[Dict]:
        return self.name_to_doc.get(name)

    def get_by_names(self, names: List[str]) -> List[Dict]:
        return [self.name_to_doc[n] for n in names if n in self.name_to_doc]

def get_catalog() -> CatalogIndex:
    return CatalogIndex(catalog_path="data/catalog.json")
