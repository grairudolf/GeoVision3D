"""
Geometric feature extraction from 3D meshes.

Includes curvature estimation, shape moments, and local geometric descriptors.
"""

import numpy as np
import trimesh
from typing import Dict, Tuple, Optional
from scipy.spatial import cKDTree
from scipy.linalg import svd as scipy_svd


class GeometricFeatures:
    """Extract geometric features from meshes."""
    
    @staticmethod
    def compute_vertex_normals(mesh: trimesh.Trimesh) -> np.ndarray:
        """
        Compute per-vertex normals (area-weighted).
        
        Args:
            mesh: Input mesh
            
        Returns:
            Vertex normals (N_vertices, 3)
        """
        return mesh.vertex_normals
    
    @staticmethod
    def compute_face_normals(mesh: trimesh.Trimesh) -> np.ndarray:
        """
        Get per-face normals.
        
        Args:
            mesh: Input mesh
            
        Returns:
            Face normals (N_faces, 3)
        """
        return mesh.face_normals
    
    @staticmethod
    def compute_mean_curvature(mesh: trimesh.Trimesh, 
                              use_laplacian: bool = True) -> np.ndarray:
        """
        Estimate mean curvature at each vertex.
        
        Uses discrete Laplacian approximation: H ≈ ||Δ|| / 2
        
        Args:
            mesh: Input mesh
            use_laplacian: Use Laplacian-based method if True
            
        Returns:
            Mean curvature per vertex (N_vertices,)
        """
        vertices = mesh.vertices
        faces = mesh.faces
        
        if use_laplacian:
            # Compute Laplacian
            laplacian = mesh.laplacian_sparse.toarray()
            laplacian_coords = laplacian @ vertices
            
            # Curvature approximation
            curvature = np.linalg.norm(laplacian_coords, axis=1) / 2.0
        else:
            # Alternative: use angular defect
            vertex_angles = np.zeros(vertices.shape[0])
            vertex_areas = np.zeros(vertices.shape[0])
            
            for face_idx, face in enumerate(faces):
                v0, v1, v2 = vertices[face]
                
                # Compute angles at each vertex
                angles = [
                    np.arccos(np.clip(np.dot(v1 - v0, v2 - v0) / 
                            (np.linalg.norm(v1 - v0) * np.linalg.norm(v2 - v0) + 1e-10),
                            -1, 1)),
                    np.arccos(np.clip(np.dot(v0 - v1, v2 - v1) / 
                            (np.linalg.norm(v0 - v1) * np.linalg.norm(v2 - v1) + 1e-10),
                            -1, 1)),
                    np.arccos(np.clip(np.dot(v0 - v2, v1 - v2) / 
                            (np.linalg.norm(v0 - v2) * np.linalg.norm(v1 - v2) + 1e-10),
                            -1, 1))
                ]
                
                face_area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
                
                for vi, angle in enumerate(angles):
                    vertex_angles[face[vi]] += angle
                    vertex_areas[face[vi]] += face_area / 3.0
            
            # Gaussian curvature from angle defect
            gaussian_curvature = (2 * np.pi - vertex_angles) / (vertex_areas + 1e-10)
            
            # Mean curvature (simplified)
            curvature = np.abs(gaussian_curvature)
        
        return np.clip(curvature, -10, 10)  # Clip outliers
    
    @staticmethod
    def compute_gaussian_curvature(mesh: trimesh.Trimesh) -> np.ndarray:
        """
        Estimate Gaussian curvature (angle defect method).
        
        Args:
            mesh: Input mesh
            
        Returns:
            Gaussian curvature per vertex
        """
        vertices = mesh.vertices
        faces = mesh.faces
        
        vertex_angles = np.zeros(vertices.shape[0])
        vertex_areas = np.zeros(vertices.shape[0])
        
        for face in faces:
            v0, v1, v2 = vertices[face]
            
            # Compute angles
            angles = []
            edges = [(v1 - v0, v2 - v0), (v0 - v1, v2 - v1), (v0 - v2, v1 - v2)]
            for e1, e2 in edges:
                cos_angle = np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2) + 1e-10)
                angle = np.arccos(np.clip(cos_angle, -1, 1))
                angles.append(angle)
            
            # Face area
            face_area = 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
            
            for vi, angle in enumerate(angles):
                vertex_angles[face[vi]] += angle
                vertex_areas[face[vi]] += face_area / 3.0
        
        # Gaussian curvature: (2π - Σ angles) / area
        gaussian_curvature = (2 * np.pi - vertex_angles) / (vertex_areas + 1e-10)
        
        return np.clip(gaussian_curvature, -10, 10)
    
    @staticmethod
    def compute_principal_curvatures(mesh: trimesh.Trimesh,
                                    neighborhood_size: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate principal curvatures using local PCA.
        
        Args:
            mesh: Input mesh
            neighborhood_size: Number of neighbors for local fitting
            
        Returns:
            (k1, k2) principal curvatures per vertex
        """
        vertices = mesh.vertices
        normals = mesh.vertex_normals
        
        # Build KDTree for neighbor search
        kdtree = cKDTree(vertices)
        
        k1 = np.zeros(len(vertices))
        k2 = np.zeros(len(vertices))
        
        for i in range(len(vertices)):
            # Find neighbors
            _, indices = kdtree.query(vertices[i], k=neighborhood_size + 1)
            neighbors = vertices[indices[1:]]  # Exclude self
            
            # Center on vertex
            centered = neighbors - vertices[i]
            
            # Project onto tangent plane
            n = normals[i]
            tangent = centered - np.outer(np.dot(centered, n), n)
            
            # PCA on tangent projection
            if np.linalg.norm(tangent) > 1e-6:
                _, _, Vt = scipy_svd(tangent)
                # Principal curvatures from curvature tensor
                k1[i] = np.linalg.norm(np.dot(tangent, Vt[0]))
                k2[i] = np.linalg.norm(np.dot(tangent, Vt[1]))
            
        return k1, k2
    
    @staticmethod
    def compute_shape_moments(mesh: trimesh.Trimesh) -> Dict[str, float]:
        """
        Compute shape moments (up to second order).
        
        Args:
            mesh: Input mesh
            
        Returns:
            Dictionary of moment features
        """
        vertices = mesh.vertices
        
        # Center of mass
        com = vertices.mean(axis=0)
        centered = vertices - com
        
        # Moments
        m0 = vertices.shape[0]
        m1 = centered.mean(axis=0)
        m2 = np.sum(centered ** 2, axis=0)
        
        # Covariance matrix
        cov = np.cov(centered.T)
        
        # Eigenvalues
        eigenvalues = np.linalg.eigvals(cov)
        eigenvalues = np.sort(np.abs(eigenvalues))[::-1]
        
        moments = {
            'moment_0': float(m0),
            'moment_1_x': float(m1[0]),
            'moment_1_y': float(m1[1]),
            'moment_1_z': float(m1[2]),
            'moment_2_x': float(m2[0]),
            'moment_2_y': float(m2[1]),
            'moment_2_z': float(m2[2]),
            'moment_2_total': float(np.sum(m2)),
            'eigenvalue_0': float(eigenvalues[0]),
            'eigenvalue_1': float(eigenvalues[1]),
            'eigenvalue_2': float(eigenvalues[2]),
        }
        
        return moments
    
    @staticmethod
    def compute_local_descriptors(mesh: trimesh.Trimesh,
                                 neighborhood_size: int = 20) -> np.ndarray:
        """
        Compute local geometric descriptors for each vertex.
        
        Features: local curvature, normal, and shape statistics
        
        Args:
            mesh: Input mesh
            neighborhood_size: Number of neighbors
            
        Returns:
            Local descriptor matrix (N_vertices, features)
        """
        vertices = mesh.vertices
        normals = mesh.vertex_normals
        
        kdtree = cKDTree(vertices)
        mean_curv = GeometricFeatures.compute_mean_curvature(mesh)
        
        descriptors = []
        
        for i in range(len(vertices)):
            _, indices = kdtree.query(vertices[i], k=min(neighborhood_size + 1, len(vertices)))
            indices = indices[1:]  # Exclude self
            
            neighbors = vertices[indices]
            n = normals[i]
            
            # Local statistics
            local_distances = np.linalg.norm(neighbors - vertices[i], axis=1)
            local_curvatures = mean_curv[indices]
            
            descriptor = np.array([
                mean_curv[i],
                local_distances.mean(),
                local_distances.std() + 1e-8,
                local_curvatures.mean(),
                local_curvatures.std() + 1e-8,
                n[0], n[1], n[2],  # Normal vector
            ], dtype=np.float32)
            
            descriptors.append(descriptor)
        
        return np.array(descriptors)


def extract_curvature(mesh: trimesh.Trimesh) -> Dict[str, np.ndarray]:
    """Extract all curvature measures."""
    return {
        'mean_curvature': GeometricFeatures.compute_mean_curvature(mesh),
        'gaussian_curvature': GeometricFeatures.compute_gaussian_curvature(mesh),
    }


def extract_shape_moments(mesh: trimesh.Trimesh) -> Dict[str, float]:
    """Extract shape moments."""
    return GeometricFeatures.compute_shape_moments(mesh)


def extract_local_descriptors(mesh: trimesh.Trimesh,
                             neighborhood_size: int = 20) -> np.ndarray:
    """Extract local geometric descriptors."""
    return GeometricFeatures.compute_local_descriptors(mesh, neighborhood_size)


def extract_all_features(mesh: trimesh.Trimesh) -> Dict:
    """
    Extract comprehensive geometric features from mesh.
    
    Returns:
        Dictionary containing all geometric features
    """
    return {
        'mean_curvature': GeometricFeatures.compute_mean_curvature(mesh),
        'gaussian_curvature': GeometricFeatures.compute_gaussian_curvature(mesh),
        'moments': GeometricFeatures.compute_shape_moments(mesh),
        'local_descriptors': GeometricFeatures.compute_local_descriptors(mesh),
        'normals': GeometricFeatures.compute_vertex_normals(mesh),
    }
