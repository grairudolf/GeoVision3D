"""
Learned shape descriptors and feature learning module.

Provides shallow neural networks and feature extraction for shape analysis.
"""

import numpy as np
import trimesh
from typing import Tuple, Dict, Optional
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class ShapeDescriptor:
    """Extract learned shape descriptors."""
    
    @staticmethod
    def compute_histogram_features(mesh: trimesh.Trimesh,
                                  n_bins: int = 20) -> np.ndarray:
        """
        Compute histogram-based features from shape properties.
        
        Args:
            mesh: Input mesh
            n_bins: Number of histogram bins
            
        Returns:
            Feature vector
        """
        vertices = mesh.vertices
        faces = mesh.faces
        edges = mesh.edges_unique
        
        # Edge length distribution
        edge_lengths = np.linalg.norm(
            vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1
        )
        edge_hist, _ = np.histogram(edge_lengths, bins=n_bins)
        edge_hist = edge_hist / (len(edges) + 1e-8)
        
        # Face area distribution
        face_areas = mesh.area_faces
        area_hist, _ = np.histogram(face_areas, bins=n_bins)
        area_hist = area_hist / (len(faces) + 1e-8)
        
        # Vertex distance distribution
        centroid = mesh.centroid
        vertex_dist = np.linalg.norm(vertices - centroid, axis=1)
        dist_hist, _ = np.histogram(vertex_dist, bins=n_bins)
        dist_hist = dist_hist / (len(vertices) + 1e-8)
        
        # Combine histograms
        descriptor = np.concatenate([edge_hist, area_hist, dist_hist])
        
        return descriptor.astype(np.float32)
    
    @staticmethod
    def compute_pca_descriptors(mesh: trimesh.Trimesh,
                               n_components: int = 10) -> np.ndarray:
        """
        Compute PCA-based shape descriptors.
        
        Args:
            mesh: Input mesh
            n_components: Number of PCA components
            
        Returns:
            Feature vector
        """
        vertices = mesh.vertices
        
        # Normalize vertices
        centered = vertices - vertices.mean(axis=0)
        
        # Apply PCA
        pca = PCA(n_components=min(n_components, len(vertices), 3))
        pca.fit(centered)
        
        # Use eigenvalues as descriptors
        descriptors = np.concatenate([
            pca.explained_variance_,
            pca.explained_variance_ratio_,
            [np.sum(pca.explained_variance_)],
        ])
        
        # Pad to desired length
        if len(descriptors) < n_components:
            descriptors = np.pad(descriptors, (0, n_components - len(descriptors)))
        
        return descriptors[:n_components].astype(np.float32)
    
    @staticmethod
    def compute_multiscale_descriptors(mesh: trimesh.Trimesh,
                                      scales: Tuple[int, ...] = (5, 10, 15)) -> np.ndarray:
        """
        Compute multi-scale shape descriptors.
        
        Args:
            mesh: Input mesh
            scales: Scales for histogram bins
            
        Returns:
            Combined multi-scale feature vector
        """
        descriptors = []
        
        for scale in scales:
            hist_feat = ShapeDescriptor.compute_histogram_features(mesh, n_bins=scale)
            descriptors.append(hist_feat)
        
        return np.concatenate(descriptors).astype(np.float32)
    
    @staticmethod
    def compute_spectral_features(mesh: trimesh.Trimesh,
                                 n_features: int = 10) -> np.ndarray:
        """
        Compute spectral features from Laplacian eigenvalues.
        
        Args:
            mesh: Input mesh
            n_features: Number of spectral features
            
        Returns:
            Feature vector
        """
        try:
            # Get Laplacian matrix
            laplacian = mesh.laplacian_sparse
            
            # Compute eigenvalues (smallest ones are most informative)
            from scipy.sparse.linalg import eigsh
            
            n_eigs = min(n_features, laplacian.shape[0] - 1)
            eigenvalues = eigsh(laplacian, k=n_eigs, which='SM',
                              return_eigenvectors=False)
            
            # Normalize
            eigenvalues = np.abs(eigenvalues)
            if eigenvalues.max() > 0:
                eigenvalues = eigenvalues / eigenvalues.max()
            
            # Pad if necessary
            if len(eigenvalues) < n_features:
                eigenvalues = np.pad(eigenvalues, (0, n_features - len(eigenvalues)))
            
            return eigenvalues[:n_features].astype(np.float32)
        except Exception:
            # Fallback to histogram features
            return ShapeDescriptor.compute_histogram_features(mesh, n_bins=n_features)


class LearnedFeatureExtractor:
    """Extract and learn shape features using shallow networks."""
    
    def __init__(self, feature_dim: int = 128):
        """
        Initialize extractor.
        
        Args:
            feature_dim: Output feature dimension
        """
        self.feature_dim = feature_dim
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=feature_dim)
        self.fitted = False
    
    def extract_raw_features(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """
        Extract raw multi-descriptor features.
        
        Args:
            mesh: Input mesh
            
        Returns:
            Raw feature vector
        """
        features = {
            'histogram': ShapeDescriptor.compute_histogram_features(mesh, n_bins=20),
            'pca': ShapeDescriptor.compute_pca_descriptors(mesh, n_components=10),
            'spectral': ShapeDescriptor.compute_spectral_features(mesh, n_features=10),
            'multiscale': ShapeDescriptor.compute_multiscale_descriptors(mesh),
        }
        
        # Concatenate all features
        raw = np.concatenate([
            features['histogram'],
            features['pca'],
            features['spectral'],
            features['multiscale'],
        ])
        
        return raw.astype(np.float32)
    
    def fit(self, meshes: list) -> None:
        """
        Fit the feature extractor on a set of meshes.
        
        Args:
            meshes: List of trimesh.Trimesh objects
        """
        # Extract raw features
        raw_features = np.array([self.extract_raw_features(mesh) for mesh in meshes])
        
        # Normalize
        raw_features = self.scaler.fit_transform(raw_features)
        
        # Learn PCA projection
        self.pca.fit(raw_features)
        self.fitted = True
    
    def extract(self, mesh: trimesh.Trimesh) -> np.ndarray:
        """
        Extract learned features.
        
        Args:
            mesh: Input mesh
            
        Returns:
            Learned feature vector
        """
        raw = self.extract_raw_features(mesh)
        
        if self.fitted:
            raw = self.scaler.transform(raw.reshape(1, -1))[0]
            features = self.pca.transform(raw.reshape(1, -1))[0]
        else:
            # Return normalized raw features
            features = self.scaler.fit_transform(raw.reshape(1, -1))[0]
        
        return features.astype(np.float32)
    
    def extract_batch(self, meshes: list) -> np.ndarray:
        """
        Extract features for multiple meshes.
        
        Args:
            meshes: List of meshes
            
        Returns:
            Feature matrix (N_meshes, feature_dim)
        """
        features = []
        for mesh in meshes:
            feat = self.extract(mesh)
            features.append(feat)
        
        return np.array(features)


def extract_learned_features(mesh: trimesh.Trimesh,
                            extractor: Optional[LearnedFeatureExtractor] = None) -> np.ndarray:
    """
    Extract learned features from a mesh.
    
    Args:
        mesh: Input mesh
        extractor: Feature extractor (creates new if None)
        
    Returns:
        Feature vector
    """
    if extractor is None:
        extractor = LearnedFeatureExtractor()
    
    return extractor.extract(mesh)


def extract_all_descriptors(mesh: trimesh.Trimesh) -> Dict[str, np.ndarray]:
    """Extract all available descriptors."""
    return {
        'histogram': ShapeDescriptor.compute_histogram_features(mesh),
        'pca': ShapeDescriptor.compute_pca_descriptors(mesh),
        'spectral': ShapeDescriptor.compute_spectral_features(mesh),
        'multiscale': ShapeDescriptor.compute_multiscale_descriptors(mesh),
    }
