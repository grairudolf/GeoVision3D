"""
Machine learning pipeline with scikit-learn integration.

Provides classification, clustering, dimensionality reduction, and evaluation.
"""

import numpy as np
from typing import Tuple, Dict, Optional, List
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, silhouette_score, davies_bouldin_score
)


class FeatureNormalizer:
    """Normalize feature vectors for ML pipeline."""
    
    def __init__(self, method: str = 'standard'):
        """
        Initialize normalizer.
        
        Args:
            method: Normalization method ('standard', 'minmax', 'robust')
        """
        self.method = method
        if method == 'standard':
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def fit(self, features: np.ndarray) -> None:
        """Fit normalizer on training data."""
        self.scaler.fit(features)
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transform features."""
        return self.scaler.transform(features)
    
    def fit_transform(self, features: np.ndarray) -> np.ndarray:
        """Fit and transform features."""
        return self.scaler.fit_transform(features)


class DimensionalityReducer:
    """Reduce feature dimensionality using PCA."""
    
    def __init__(self, n_components: Optional[int] = None,
                variance_ratio: float = 0.95):
        """
        Initialize reducer.
        
        Args:
            n_components: Number of components (uses variance_ratio if None)
            variance_ratio: Variance explained ratio target
        """
        self.n_components = n_components
        self.variance_ratio = variance_ratio
        self.pca = None
    
    def fit(self, features: np.ndarray) -> None:
        """Fit PCA."""
        if self.n_components is None:
            n_comp = len(features[0])
        else:
            n_comp = self.n_components
        
        self.pca = PCA(n_components=min(n_comp, len(features)))
        self.pca.fit(features)
        
        # Find n_components for variance ratio
        if self.n_components is None:
            cumsum = np.cumsum(self.pca.explained_variance_ratio_)
            self.n_components = np.argmax(cumsum >= self.variance_ratio) + 1
    
    def transform(self, features: np.ndarray) -> np.ndarray:
        """Transform to reduced space."""
        if self.pca is None:
            raise ValueError("Reducer not fitted")
        return self.pca.transform(features)[:, :self.n_components]
    
    def get_explained_variance(self) -> float:
        """Get total explained variance ratio."""
        if self.pca is None:
            return 0.0
        return float(self.pca.explained_variance_ratio_[:self.n_components].sum())


class ShapeClassifier:
    """Supervised shape classification."""
    
    def __init__(self, classifier_type: str = 'svm',
                normalize: bool = True):
        """
        Initialize classifier.
        
        Args:
            classifier_type: 'svm' or 'random_forest'
            normalize: Whether to normalize features
        """
        if classifier_type == 'svm':
            self.classifier = SVC(kernel='rbf', gamma='scale', probability=True)
        elif classifier_type == 'random_forest':
            self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        else:
            raise ValueError(f"Unknown classifier: {classifier_type}")
        
        self.normalizer = FeatureNormalizer() if normalize else None
        self.classes_ = None
    
    def fit(self, features: np.ndarray, labels: np.ndarray) -> None:
        """
        Train classifier.
        
        Args:
            features: Feature matrix (N, D)
            labels: Label vector (N,)
        """
        if self.normalizer is not None:
            features = self.normalizer.fit_transform(features)
        
        self.classifier.fit(features, labels)
        self.classes_ = self.classifier.classes_
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict labels.
        
        Args:
            features: Feature matrix
            
        Returns:
            Predicted labels
        """
        if self.normalizer is not None:
            features = self.normalizer.transform(features)
        
        return self.classifier.predict(features)
    
    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            features: Feature matrix
            
        Returns:
            Probability matrix
        """
        if self.normalizer is not None:
            features = self.normalizer.transform(features)
        
        return self.classifier.predict_proba(features)
    
    def evaluate(self, features: np.ndarray,
                labels: np.ndarray) -> Dict[str, float]:
        """
        Evaluate classifier performance.
        
        Args:
            features: Feature matrix
            labels: True labels
            
        Returns:
            Evaluation metrics
        """
        predictions = self.predict(features)
        
        metrics = {
            'accuracy': float(accuracy_score(labels, predictions)),
            'precision': float(precision_score(labels, predictions, average='weighted',
                                              zero_division=0)),
            'recall': float(recall_score(labels, predictions, average='weighted',
                                        zero_division=0)),
            'f1': float(f1_score(labels, predictions, average='weighted', zero_division=0)),
        }
        
        return metrics


class ShapeClusterer:
    """Unsupervised shape clustering."""
    
    def __init__(self, n_clusters: int = 5,
                normalize: bool = True):
        """
        Initialize clusterer.
        
        Args:
            n_clusters: Number of clusters
            normalize: Whether to normalize features
        """
        self.n_clusters = n_clusters
        self.clustering = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        self.normalizer = FeatureNormalizer() if normalize else None
        self.labels_ = None
    
    def fit(self, features: np.ndarray) -> None:
        """
        Fit clustering model.
        
        Args:
            features: Feature matrix (N, D)
        """
        if self.normalizer is not None:
            features = self.normalizer.fit_transform(features)
        
        self.clustering.fit(features)
        self.labels_ = self.clustering.labels_
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels.
        
        Args:
            features: Feature matrix
            
        Returns:
            Cluster labels
        """
        if self.normalizer is not None:
            features = self.normalizer.transform(features)
        
        return self.clustering.predict(features)
    
    def evaluate(self, features: np.ndarray) -> Dict[str, float]:
        """
        Evaluate clustering quality.
        
        Args:
            features: Feature matrix
            
        Returns:
            Evaluation metrics
        """
        if self.normalizer is not None:
            features = self.normalizer.transform(features)
        
        labels = self.clustering.labels_
        
        metrics = {
            'silhouette_score': float(silhouette_score(features, labels)),
            'davies_bouldin_index': float(davies_bouldin_score(features, labels)),
            'inertia': float(self.clustering.inertia_),
        }
        
        return metrics


class MLPipeline:
    """Complete ML pipeline for shape analysis."""
    
    def __init__(self, normalize: bool = True,
                reduce_dims: bool = True,
                n_components: Optional[int] = None):
        """
        Initialize pipeline.
        
        Args:
            normalize: Whether to normalize features
            reduce_dims: Whether to reduce dimensionality
            n_components: Number of PCA components
        """
        self.normalizer = FeatureNormalizer() if normalize else None
        self.reducer = DimensionalityReducer(n_components=n_components) if reduce_dims else None
        self.classifier = None
        self.clusterer = None
    
    def preprocess(self, features: np.ndarray,
                  fit: bool = True) -> np.ndarray:
        """
        Preprocess features (normalize and reduce).
        
        Args:
            features: Raw feature matrix
            fit: Whether to fit normalizer and reducer
            
        Returns:
            Preprocessed features
        """
        if self.normalizer is not None:
            if fit:
                features = self.normalizer.fit_transform(features)
            else:
                features = self.normalizer.transform(features)
        
        if self.reducer is not None:
            if fit:
                self.reducer.fit(features)
            features = self.reducer.transform(features)
        
        return features
    
    def train_classifier(self, features: np.ndarray,
                        labels: np.ndarray,
                        classifier_type: str = 'svm') -> None:
        """Train classifier."""
        processed = self.preprocess(features, fit=True)
        
        self.classifier = ShapeClassifier(classifier_type=classifier_type,
                                         normalize=False)
        self.classifier.fit(processed, labels)
    
    def train_clusterer(self, features: np.ndarray,
                       n_clusters: int = 5) -> None:
        """Train clusterer."""
        processed = self.preprocess(features, fit=True)
        
        self.clusterer = ShapeClusterer(n_clusters=n_clusters, normalize=False)
        self.clusterer.fit(processed)
    
    def classify(self, features: np.ndarray) -> np.ndarray:
        """Classify shapes."""
        if self.classifier is None:
            raise ValueError("Classifier not trained")
        
        processed = self.preprocess(features, fit=False)
        return self.classifier.predict(processed)
    
    def cluster(self, features: np.ndarray) -> np.ndarray:
        """Cluster shapes."""
        if self.clusterer is None:
            raise ValueError("Clusterer not trained")
        
        processed = self.preprocess(features, fit=False)
        return self.clusterer.predict(processed)
    
    def evaluate_classification(self, features: np.ndarray,
                               labels: np.ndarray) -> Dict[str, float]:
        """Evaluate classification."""
        if self.classifier is None:
            raise ValueError("Classifier not trained")
        
        processed = self.preprocess(features, fit=False)
        return self.classifier.evaluate(processed, labels)
    
    def evaluate_clustering(self, features: np.ndarray) -> Dict[str, float]:
        """Evaluate clustering."""
        if self.clusterer is None:
            raise ValueError("Clusterer not trained")
        
        processed = self.preprocess(features, fit=False)
        return self.clusterer.evaluate(processed)


def create_confusion_matrix(y_true: np.ndarray,
                           y_pred: np.ndarray) -> np.ndarray:
    """Create confusion matrix."""
    return confusion_matrix(y_true, y_pred)
