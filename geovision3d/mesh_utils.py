"""
Mesh utilities for loading, processing, and analyzing 3D mesh files.

Supports OBJ and STL formats with validation, normalization, and statistical analysis.
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional, Union
import trimesh


class MeshProcessor:
    """Handle mesh loading, validation, and preprocessing."""
    
    SUPPORTED_FORMATS = {'.obj', '.stl', '.ply', '.gltf'}
    
    @staticmethod
    def load_mesh(filepath: Union[str, Path]) -> trimesh.Trimesh:
        """
        Load a 3D mesh from file.
        
        Args:
            filepath: Path to mesh file (OBJ, STL, PLY, GLTF)
            
        Returns:
            trimesh.Trimesh: Loaded mesh object
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If format is not supported
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Mesh file not found: {filepath}")
        
        if filepath.suffix.lower() not in MeshProcessor.SUPPORTED_FORMATS:
            raise ValueError(f"Unsupported format: {filepath.suffix}")
        
        try:
            mesh = trimesh.load(str(filepath), process=False)
            if isinstance(mesh, trimesh.Trimesh):
                return mesh
            elif hasattr(mesh, 'geometry'):
                # Handle multi-mesh formats
                return mesh.geometry[0] if mesh.geometry else mesh
            else:
                return mesh
        except Exception as e:
            raise RuntimeError(f"Failed to load mesh: {str(e)}")
    
    @staticmethod
    def export_mesh(mesh: trimesh.Trimesh, filepath: Union[str, Path],
                    file_type: Optional[str] = None) -> None:
        """
        Export mesh to file.
        
        Args:
            mesh: Mesh to export
            filepath: Output file path
            file_type: Format (auto-detected if None)
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if file_type is None:
            file_type = filepath.suffix[1:].lower()
        
        mesh.export(str(filepath), file_type=file_type)
    
    @staticmethod
    def normalize_mesh(mesh: trimesh.Trimesh, 
                      target_scale: float = 1.0) -> trimesh.Trimesh:
        """
        Normalize mesh: center at origin and scale to unit size.
        
        Args:
            mesh: Input mesh
            target_scale: Target scale (default 1.0 for unit scale)
            
        Returns:
            Normalized mesh copy
        """
        normalized = mesh.copy()
        
        # Center at origin
        normalized.apply_translation(-normalized.centroid)
        
        # Scale to target size
        current_size = np.linalg.norm(normalized.bounds[1] - normalized.bounds[0])
        if current_size > 0:
            normalized.apply_scale(target_scale / current_size)
        
        return normalized
    
    @staticmethod
    def validate_mesh(mesh: trimesh.Trimesh) -> Tuple[bool, str]:
        """
        Validate mesh integrity.
        
        Args:
            mesh: Mesh to validate
            
        Returns:
            (is_valid, message): Validation result and details
        """
        issues = []
        
        if mesh.vertices.shape[0] < 3:
            issues.append("Mesh has fewer than 3 vertices")
        
        if mesh.faces.shape[0] < 1:
            issues.append("Mesh has no faces")
        
        if not mesh.is_watertight:
            issues.append("Mesh is not watertight (has holes)")
        
        if mesh.has_degenerate_faces:
            issues.append(f"Mesh has {np.sum(mesh.degenerate_faces)} degenerate faces")
        
        if mesh.volume < 1e-6:
            issues.append("Mesh has near-zero volume")
        
        is_valid = len(issues) == 0
        message = " | ".join(issues) if issues else "Mesh is valid"
        
        return is_valid, message


class MeshStatistics:
    """Compute statistical properties of meshes."""
    
    @staticmethod
    def get_basic_stats(mesh: trimesh.Trimesh) -> Dict[str, float]:
        """
        Compute basic mesh statistics.
        
        Args:
            mesh: Input mesh
            
        Returns:
            Dictionary with statistics
        """
        vertices = mesh.vertices
        faces = mesh.faces
        
        # Vertex statistics
        vertex_distances = np.linalg.norm(vertices, axis=1)
        
        # Face statistics
        face_areas = mesh.area_faces
        
        # Edge length statistics
        edges = mesh.edges_unique
        edge_lengths = np.linalg.norm(
            vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1
        )
        
        stats = {
            'num_vertices': float(vertices.shape[0]),
            'num_faces': float(faces.shape[0]),
            'num_edges': float(len(edges)),
            'surface_area': float(mesh.area),
            'volume': float(mesh.volume),
            'centroid_x': float(mesh.centroid[0]),
            'centroid_y': float(mesh.centroid[1]),
            'centroid_z': float(mesh.centroid[2]),
            'min_vertex_distance': float(vertex_distances.min()),
            'max_vertex_distance': float(vertex_distances.max()),
            'mean_vertex_distance': float(vertex_distances.mean()),
            'min_edge_length': float(edge_lengths.min()),
            'max_edge_length': float(edge_lengths.max()),
            'mean_edge_length': float(edge_lengths.mean()),
            'std_edge_length': float(edge_lengths.std()),
            'min_face_area': float(face_areas.min()),
            'max_face_area': float(face_areas.max()),
            'mean_face_area': float(face_areas.mean()),
        }
        
        return stats
    
    @staticmethod
    def get_shape_descriptors(mesh: trimesh.Trimesh) -> Dict[str, float]:
        """
        Compute shape descriptors and invariants.
        
        Args:
            mesh: Input mesh
            
        Returns:
            Dictionary with descriptors
        """
        volume = mesh.volume
        surface_area = mesh.area
        
        # Compute bounding box
        bounds = mesh.bounds
        bbox_dims = bounds[1] - bounds[0]
        bbox_volume = np.prod(bbox_dims)
        
        # Aspect ratio and elongation
        bbox_dims_sorted = np.sort(bbox_dims)[::-1]
        aspect_ratio_1 = bbox_dims_sorted[0] / (bbox_dims_sorted[1] + 1e-8)
        aspect_ratio_2 = bbox_dims_sorted[1] / (bbox_dims_sorted[2] + 1e-8)
        
        # Sphericity (volume / volume_bounding_sphere)
        bounding_sphere = mesh.bounding_box_oriented.extents.max() / 2
        bounding_sphere_volume = (4/3) * np.pi * (bounding_sphere ** 3)
        sphericity = volume / (bounding_sphere_volume + 1e-8)
        
        # Compactness
        compactness = (36 * np.pi * volume**2) / (surface_area**3 + 1e-8)
        
        # Convexity
        convex_hull = mesh.convex_hull
        convexity = volume / (convex_hull.volume + 1e-8)
        
        descriptors = {
            'aspect_ratio_1': float(aspect_ratio_1),
            'aspect_ratio_2': float(aspect_ratio_2),
            'sphericity': float(np.clip(sphericity, 0, 1)),
            'compactness': float(compactness),
            'convexity': float(np.clip(convexity, 0, 1)),
            'bbox_volume': float(bbox_volume),
            'volume_ratio': float(volume / bbox_volume) if bbox_volume > 0 else 0.0,
            'surface_area_to_volume': float(surface_area / (volume + 1e-8)),
        }
        
        return descriptors
    
    @staticmethod
    def get_all_stats(mesh: trimesh.Trimesh) -> Dict[str, Dict]:
        """
        Get comprehensive mesh statistics.
        
        Args:
            mesh: Input mesh
            
        Returns:
            Dictionary with all statistics categories
        """
        return {
            'basic': MeshStatistics.get_basic_stats(mesh),
            'descriptors': MeshStatistics.get_shape_descriptors(mesh),
        }


def load_mesh(filepath: Union[str, Path]) -> trimesh.Trimesh:
    """Convenience function to load a mesh."""
    return MeshProcessor.load_mesh(filepath)


def export_mesh(mesh: trimesh.Trimesh, filepath: Union[str, Path]) -> None:
    """Convenience function to export a mesh."""
    MeshProcessor.export_mesh(mesh, filepath)


def normalize_mesh(mesh: trimesh.Trimesh, 
                  target_scale: float = 1.0) -> trimesh.Trimesh:
    """Convenience function to normalize a mesh."""
    return MeshProcessor.normalize_mesh(mesh, target_scale)


def get_mesh_stats(mesh: trimesh.Trimesh) -> Dict[str, Dict]:
    """Convenience function to get mesh statistics."""
    return MeshStatistics.get_all_stats(mesh)
