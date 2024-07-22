from dataclasses import dataclass
from typing import List, Dict
import numpy as np

@dataclass
class EvaluationCase:
    model: str
    method: str
    w1: float
    w2: float
    K: int
    N: int

    def __eq__(self, other):
        if not isinstance(other, EvaluationCase):
            return False
        return (self.model, self.method, self.w1, self.w2, self.K, self.N) == \
               (other.model, other.method, other.w1, other.w2, other.K, other.N)

    def __hash__(self):
        return hash((self.model, self.method, self.w1, self.w2, self.K, self.N))
    
@dataclass
class RecommendedItem:
    item_id: str
    score: float
    origin: str

    def __repr__(self):
        return self.item_id
    
@dataclass
class Recommendation:
    item_id: str
    user_id: str
    items_map: Dict[str, RecommendedItem]
    items: List[RecommendedItem]
    item_ids: List[str]

    def softmax_normalize_scores(self):
        # Extract scores from RecommendedItem objects
        scores = [item.score for item in self.items]
        
        # Apply softmax normalization
        exp_scores = np.exp(scores)
        softmax_scores = exp_scores / np.sum(exp_scores)
        
        # Update scores of RecommendedItem objects with normalized scores
        for item, softmax_score in zip(self.items, softmax_scores):
            item.score = softmax_score
