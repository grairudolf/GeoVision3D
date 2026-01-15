"""
Example 1: Comprehensive Mesh Analysis
Demonstrates mesh loading, feature extraction, and statistical analysis.
"""

import numpy as np
from geovision3d import mesh_utils, geometric_features
from pathlib import Path


def example_basic_mesh_loading():
    """Load and validate mesh files."""
    print("=" * 60)
    print("EXAMPLE 1: Basic Mesh Loading and Validation")
    print("=" * 60)
    
    # Create a sample mesh (unit sphere)
    import trimesh
    mesh = trimesh.creation.icosphere(subdivisions=3)
    
    # Export for demonstration
    sample_path = Path("sample_mesh.obj")
    mesh_utils.export_mesh(mesh, sample_path)
    print(f"Created sample mesh: {sample_path}")
    
    # Load mesh
    loaded_mesh = mesh_utils.load_mesh(sample_path)
    print(f"Loaded mesh with {loaded_mesh.vertices.shape[0]} vertices "
          f"and {loaded_mesh.faces.shape[0]} faces")
    
    # Validate mesh
    is_valid, message = mesh_utils.MeshProcessor.validate_mesh(loaded_mesh)
    print(f"Validation: {message}")
    
    # Normalize mesh
    normalized_mesh = mesh_utils.normalize_mesh(loaded_mesh)
    print(f"Normalized mesh bounds: {normalized_mesh.bounds}")
    
    # Get statistics
    stats = mesh_utils.get_mesh_stats(loaded_mesh)
    print("\nMesh Statistics:")
    print(f"  Volume: {stats['basic']['volume']:.4f}")
    print(f"  Surface Area: {stats['basic']['surface_area']:.4f}")
    print(f"  Number of Vertices: {int(stats['basic']['num_vertices'])}")
    print(f"  Sphericity: {stats['descriptors']['sphericity']:.4f}")
    print(f"  Compactness: {stats['descriptors']['compactness']:.4f}")
    
    # Clean up
    sample_path.unlink()


def example_geometric_features():
    """Extract geometric features from mesh."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Geometric Feature Extraction")
    print("=" * 60)
    
    # Create test mesh
    import trimesh
    mesh = trimesh.creation.icosphere(subdivisions=2)
    mesh_utils.export_mesh(mesh, "test_mesh.obj")
    mesh = mesh_utils.load_mesh("test_mesh.obj")
    
    # Extract curvature
    mean_curv = geometric_features.GeometricFeatures.compute_mean_curvature(mesh)
    gaussian_curv = geometric_features.GeometricFeatures.compute_gaussian_curvature(mesh)
    
    print(f"Mean Curvature - Min: {mean_curv.min():.4f}, "
          f"Max: {mean_curv.max():.4f}, Mean: {mean_curv.mean():.4f}")
    print(f"Gaussian Curvature - Min: {gaussian_curv.min():.4f}, "
          f"Max: {gaussian_curv.max():.4f}, Mean: {gaussian_curv.mean():.4f}")
    
    # Extract shape moments
    moments = geometric_features.extract_shape_moments(mesh)
    print("\nShape Moments:")
    print(f"  Eigenvalue 0: {moments['eigenvalue_0']:.4f}")
    print(f"  Eigenvalue 1: {moments['eigenvalue_1']:.4f}")
    print(f"  Eigenvalue 2: {moments['eigenvalue_2']:.4f}")
    
    # Extract local descriptors
    descriptors = geometric_features.extract_local_descriptors(mesh)
    print(f"\nLocal Descriptors shape: {descriptors.shape}")
    print(f"  Descriptor 0 mean: {descriptors[0].mean():.4f}")
    print(f"  Descriptor 0 std: {descriptors[0].std():.4f}")
    
    # Clean up
    Path("test_mesh.obj").unlink()


def example_feature_comparison():
    """Compare features between different shapes."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Feature Comparison Between Shapes")
    print("=" * 60)
    
    import trimesh
    
    # Create different shapes
    sphere = trimesh.creation.icosphere(subdivisions=2)
    cube = trimesh.creation.box(extents=[1, 1, 1])
    cylinder = trimesh.creation.cylinder(radius=0.5, height=1.0)
    
    shapes = {'sphere': sphere, 'cube': cube, 'cylinder': cylinder}
    
    # Compare statistics
    print("Shape Comparison:")
    print(f"{'Shape':<12} {'Volume':<12} {'Surface Area':<15} {'Sphericity':<12}")
    print("-" * 50)
    
    for name, mesh in shapes.items():
        stats = mesh_utils.get_mesh_stats(mesh)
        vol = stats['basic']['volume']
        area = stats['basic']['surface_area']
        sphericity = stats['descriptors']['sphericity']
        print(f"{name:<12} {vol:<12.4f} {area:<15.4f} {sphericity:<12.4f}")
    
    # Compare curvature
    print("\nMean Curvature Statistics:")
    print(f"{'Shape':<12} {'Min':<12} {'Max':<12} {'Mean':<12}")
    print("-" * 50)
    
    for name, mesh in shapes.items():
        mean_curv = geometric_features.GeometricFeatures.compute_mean_curvature(mesh)
        print(f"{name:<12} {mean_curv.min():<12.4f} {mean_curv.max():<12.4f} "
              f"{mean_curv.mean():<12.4f}")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("GeoVision3D - Mesh Analysis Examples")
    print("=" * 60)
    
    example_basic_mesh_loading()
    example_geometric_features()
    example_feature_comparison()
    
    print("\n" + "=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)
