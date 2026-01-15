"""
Shape matching and retrieval module.

Provides similarity metrics, efficient retrieval, and ranking algorithms.
"""

import numpy as np
import trimesh
from typing import Tuple, List, Dict, Optional
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler


class SimilarityMetrics:
    """Compute similarity between shape features."""
    
    @staticmethod
    def euclidean_distance(feature1: np.ndarray, feature2: np.ndarray) -> float:
        """Euclidean distance between features."""
        return float(np.linalg.norm(feature1 - feature2))
    
    @staticmethod
    def cosine_similarity(feature1: np.ndarray, feature2: np.ndarray) -> float:
        """Cosine similarity between features."""
        f1_norm = feature1 / (np.linalg.norm(feature1) + 1e-8)
        f2_norm = feature2 / (np.linalg.norm(feature2) + 1e-8)
        return float(np.dot(f1_norm, f2_norm))
    
    @staticmethod
    def chi_square_distance(feature1: np.ndarray, feature2: np.ndarray) -> float:
        """Chi-square distance (for histogram features)."""
        denominator = feature1 + feature2 + 1e-8
        numerator = (feature1 - feature2) ** 2
        return float(np.sum(numerator / denominator))
    
    @staticmethod
    def wasserstein_distance(feature1: np.ndarray, feature2: np.ndarray) -> float:
        """Wasserstein distance for histogram features."""
        # Normalize to probability distributions
        f1 = feature1 / (feature1.sum() + 1e-8)
        f2 = feature2 / (feature2.sum() + 1e-8)
        
        # Cumulative sums
        cdf1 = np.cumsum(f1)
        cdf2 = np.cumsum(f2)
        
        return float(np.sum(np.abs(cdf1 - cdf2)))
    
    @staticmethod
    def hamming_distance(feature1: np.ndarray, feature2: np.ndarray) -> float:
        """Hamming distance for binary features."""
        binary1 = (feature1 > feature1.mean()).astype(int)
        binary2 = (feature2 > feature2.mean()).astype(int)
        return float(np.sum(binary1 != binary2))


class ShapeRetriever:
    """Efficient shape retrieval and ranking."""
    
    def __init__(self, metric: str = 'euclidean', normalize_features: bool = True):
        """
        Initialize retriever.
        
        Args:
            metric: Similarity metric ('euclidean', 'cosine', 'chi_square')
            normalize_features: Whether to normalize features
        """
        self.metric = metric
        self.normalize_features = normalize_features
        self.scaler = StandardScaler() if normalize_features else None
        self.database_features = None
        self.database_names = None
    
    def add_to_database(self, features: np.ndarray,
                       names: Optional[List[str]] = None) -> None:
        """
        Add features to retrieval database.
        
        Args:
            features: Feature matrix (N, D)
            names: Optional names for each shape
        """
        if self.normalize_features:
            features = self.scaler.fit_transform(features)
        
        self.database_features = features
        self.database_names = names if names else [f"shape_{i}" for i in range(len(features))]
    
    def retrieve(self, query_features: np.ndarray,
                k: int = 5,
                return_scores: bool = True) -> Tuple[List[int], np.ndarray]:
        """
        Retrieve top-k most similar shapes.
        
        Args:
            query_features: Query feature vector (D,)
            k: Number of results to return
            return_scores: Whether to return similarity scores
            
        Returns:
            (indices, scores) or indices if not return_scores
        """
        if self.database_features is None:
            raise ValueError("Database is empty")
        
        # Normalize query
        query = query_features.reshape(1, -1)
        if self.normalize_features:
            query = self.scaler.transform(query)[0]
        else:
            query = query_features
        
        # Compute distances
        if self.metric == 'euclidean':
            distances = cdist([query], self.database_features, metric='euclidean')[0]
        elif self.metric == 'cosine':
            # Convert cosine similarity to distance
            similarities = cdist([query], self.database_features, metric='cosine')[0]
            distances = 1.0 - similarities
        elif self.metric == 'chi_square':
            distances = cdist([query], self.database_features, metric='chebyshev')[0]
        else:
            distances = cdist([query], self.database_features, metric='euclidean')[0]
        
        # Get top-k indices
        k = min(k, len(distances))
        indices = np.argsort(distances)[:k]
        
        if return_scores:
            scores = 1.0 / (1.0 + distances[indices])  # Convert to similarity
            return indices.tolist(), scores
        
        return indices.tolist()
    
    def retrieve_from_meshes(self, query_mesh: trimesh.Trimesh,
                            database_meshes: List[trimesh.Trimesh],
                            feature_extractor,
                            k: int = 5) -> Tuple[List[int], np.ndarray]:
        """
        Retrieve similar shapes from mesh objects.
        
        Args:
            query_mesh: Query mesh
            database_meshes: List of database meshes
            feature_extractor: Feature extractor function/object
            k: Number of results
            
        Returns:
            (indices, scores)
        """
        # Extract features
        if hasattr(feature_extractor, 'extract'):
            query_feat = feature_extractor.extract(query_mesh)
            db_features = feature_extractor.extract_batch(database_meshes)
        else:
            query_feat = feature_extractor(query_mesh)
            db_features = np.array([feature_extractor(mesh) for mesh in database_meshes])
        
        # Add to database
        self.add_to_database(db_features)
        
        # Retrieve
        return self.retrieve(query_feat, k=k)
    
    def evaluate_retrieval(self, query_indices: List[int],
                          retrieved_indices: List[List[int]],
                          ground_truth: Optional[Dict[int, List[int]]] = None) -> Dict[str, float]:
        """
        Evaluate retrieval results.
        
        Args:
            query_indices: Indices of query shapes
            retrieved_indices: List of retrieved index lists
            ground_truth: Ground truth similar shapes per query
            
        Returns:
            Evaluation metrics
        """
        metrics = {
            'mean_average_precision': 0.0,
            'mean_reciprocal_rank': 0.0,
            'recall_at_5': 0.0,
            'recall_at_10': 0.0,
        }
        
        if not query_indices:
            return metrics
        
        ap_scores = []
        mrr_scores = []
        recall5 = []
        recall10 = []
        
        for q_idx, retrieved in zip(query_indices, retrieved_indices):
            if ground_truth and q_idx in ground_truth:
                gt = set(ground_truth[q_idx])
                
                # Average Precision
                retrieved_set = set(retrieved)
                correct = len(retrieved_set & gt)
                ap = correct / (len(retrieved) + 1e-8) if retrieved else 0.0
                ap_scores.append(ap)
                
                # Reciprocal Rank
                if gt:
                    for rank, idx in enumerate(retrieved, 1):
                        if idx in gt:
                            mrr_scores.append(1.0 / rank)
                            break
                
                # Recall at k
                recall5.append(len({r for r in retrieved[:5]} & gt) / len(gt))
                recall10.append(len({r for r in retrieved[:10]} & gt) / len(gt))
        
        if ap_scores:
            metrics['mean_average_precision'] = float(np.mean(ap_scores))
        if mrr_scores:
            metrics['mean_reciprocal_rank'] = float(np.mean(mrr_scores))
        if recall5:
            metrics['recall_at_5'] = float(np.mean(recall5))
        if recall10:
            metrics['recall_at_10'] = float(np.mean(recall10))
        
        return metrics


class HierarchicalRetrieval:
    """Hierarchical approach to shape retrieval."""
    
    def __init__(self, num_levels: int = 3):
        """
        Initialize hierarchical retriever.
        
        Args:
            num_levels: Number of hierarchy levels
        """
        self.num_levels = num_levels
        self.retrievers = [ShapeRetriever(metric='cosine') for _ in range(num_levels)]
    
    def retrieve_hierarchical(self, query_features: np.ndarray,
                             database_features: np.ndarray,
                             k: int = 5) -> Tuple[List[int], np.ndarray]:
        """
        Hierarchical retrieval with coarse-to-fine refinement.
        
        Args:
            query_features: Query feature vector
            database_features: Database features (N, D)
            k: Final number of results
            
        Returns:
            (indices, scores)
        """
        candidates = set(range(len(database_features)))
        
        for level in range(self.num_levels - 1):
            # Progressive filtering
            k_level = max(k * (2 ** level), k)
            
            self.retrievers[level].add_to_database(database_features)
            indices, _ = self.retrievers[level].retrieve(query_features, k=k_level)
            
            candidates = {idx for idx in indices if idx in candidates}
        
        # Final retrieval
        final_features = database_features[list(candidates)]
        self.retrievers[-1].add_to_database(final_features)
        final_indices, scores = self.retrievers[-1].retrieve(query_features, k=k)
        
        # Map back to original indices
        candidate_list = sorted(list(candidates))
        mapped_indices = [candidate_list[idx] for idx in final_indices]
        
        return mapped_indices, scores


def compute_similarity_matrix(features: np.ndarray,
                             metric: str = 'euclidean') -> np.ndarray:
    """
    Compute pairwise similarity matrix.
    
    Args:
        features: Feature matrix (N, D)
        metric: Distance metric
        
    Returns:
        Similarity matrix (N, N)
    """
    distances = cdist(features, features, metric=metric)
    
    # Convert distances to similarities
    if metric == 'euclidean':
        similarities = 1.0 / (1.0 + distances)
    elif metric == 'cosine':
        similarities = 1.0 - distances
    else:
        similarities = 1.0 / (1.0 + distances)
    
    return similarities
