"""
Example 2: Camera Calibration and 3D Reconstruction
Demonstrates camera calibration and depth-based 3D reconstruction.
"""

import numpy as np
import cv2
from pathlib import Path
from geovision3d import camera_calibration, reconstruction


def example_camera_calibration():
    """Demonstrate camera calibration concepts."""
    print("=" * 60)
    print("EXAMPLE 1: Camera Calibration")
    print("=" * 60)
    
    # Create a calibrator
    calibrator = camera_calibration.CameraCalibrator(
        checkerboard_size=(9, 6),
        square_size=0.025  # 25mm squares
    )
    
    print("Camera Calibrator initialized:")
    print(f"  Checkerboard size: {calibrator.checkerboard_size}")
    print(f"  Square size: {calibrator.square_size}m")
    
    # Create synthetic calibration images for demonstration
    print("\nGenerating synthetic calibration images...")
    
    # We'll create a simple example with known camera matrix
    K_true = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ], dtype=np.float32)
    
    dist_coeffs_true = np.array([0.1, -0.05, 0.001, 0.002], dtype=np.float32)
    
    print(f"Synthetic camera matrix:")
    print(K_true)
    print(f"Distortion coefficients: {dist_coeffs_true}")
    
    # Create intrinsics object
    intrinsics = camera_calibration.CameraIntrinsics.from_camera_matrix(
        K_true, dist_coeffs_true
    )
    
    print("\nExtracted Intrinsics:")
    print(f"  fx: {intrinsics.fx:.2f}, fy: {intrinsics.fy:.2f}")
    print(f"  cx: {intrinsics.cx:.2f}, cy: {intrinsics.cy:.2f}")
    print(f"  k1: {intrinsics.k1:.4f}, k2: {intrinsics.k2:.4f}")


def example_depth_estimation():
    """Demonstrate depth map processing."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Depth Estimation and Processing")
    print("=" * 60)
    
    # Create synthetic stereo pair with known depth
    h, w = 480, 640
    
    # Create depth map (varying depth from 1 to 5 meters)
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    xx, yy = np.meshgrid(x, y)
    
    depth_map = 1.0 + 4.0 * np.sqrt(xx**2 + (yy - 0.5)**2)  # Cone-shaped depth
    depth_map = depth_map.astype(np.float32)
    
    print(f"Created synthetic depth map:")
    print(f"  Shape: {depth_map.shape}")
    print(f"  Depth range: {depth_map.min():.2f}m - {depth_map.max():.2f}m")
    
    # Refine depth map
    estimator = reconstruction.DepthEstimator()
    refined_depth = estimator.refine_depth_map(depth_map)
    
    print(f"\nRefined depth map:")
    print(f"  Min: {refined_depth.min():.2f}m")
    print(f"  Max: {refined_depth.max():.2f}m")
    print(f"  Mean: {refined_depth.mean():.2f}m")


def example_point_cloud_generation():
    """Generate and process point clouds."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Point Cloud Generation and Processing")
    print("=" * 60)
    
    # Create synthetic depth map
    h, w = 480, 640
    x = np.linspace(0, 1, w)
    y = np.linspace(0, 1, h)
    xx, yy = np.meshgrid(x, y)
    
    depth_map = 2.0 + np.sin(xx * 4 * np.pi) * 0.5 + np.cos(yy * 4 * np.pi) * 0.5
    depth_map = depth_map.astype(np.float32)
    
    # Camera matrix
    K = np.array([
        [500, 0, 320],
        [0, 500, 240],
        [0, 0, 1]
    ], dtype=np.float32)
    
    # Generate point cloud
    processor = reconstruction.PointCloudProcessor()
    points_3d, colors = processor.depth_to_point_cloud(depth_map, K)
    
    print(f"Generated point cloud:")
    print(f"  Number of points: {points_3d.shape[0]}")
    print(f"  Bounds X: [{points_3d[:, 0].min():.3f}, {points_3d[:, 0].max():.3f}]")
    print(f"  Bounds Y: [{points_3d[:, 1].min():.3f}, {points_3d[:, 1].max():.3f}]")
    print(f"  Bounds Z: [{points_3d[:, 2].min():.3f}, {points_3d[:, 2].max():.3f}]")
    
    # Filter outliers
    filtered_points = processor.filter_outliers(points_3d)
    print(f"\nAfter outlier filtering:")
    print(f"  Remaining points: {filtered_points.shape[0]}")
    print(f"  Removed: {points_3d.shape[0] - filtered_points.shape[0]} points")
    
    # Downsample
    downsampled = processor.downsample_point_cloud(filtered_points, voxel_size=0.05)
    print(f"\nAfter downsampling (voxel_size=0.05):")
    print(f"  Points: {downsampled.shape[0]}")
    
    # Estimate normals
    if downsampled.shape[0] > 10:
        normals = processor.estimate_normals(downsampled, k_neighbors=5)
        print(f"\nEstimated normals:")
        print(f"  Normal 0: {normals[0]}")
        print(f"  Normal norm: {np.linalg.norm(normals[0]):.4f}")


def example_multi_view_reconstruction():
    """Demonstrate multi-view reconstruction concept."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Multi-View Reconstruction Setup")
    print("=" * 60)
    
    # Camera parameters
    K = np.array([
        [800, 0, 320],
        [0, 800, 240],
        [0, 0, 1]
    ], dtype=np.float32)
    
    dist_coeffs = np.zeros(4)
    
    # Create reconstructor
    reconstructor = reconstruction.MultiViewReconstructor(K, dist_coeffs)
    
    print("Multi-View Reconstructor initialized:")
    print(f"  Camera matrix:")
    print(K)
    print(f"  Focal length: fx={K[0,0]:.1f}, fy={K[1,1]:.1f}")
    print(f"  Principal point: cx={K[0,2]:.1f}, cy={K[1,2]:.1f}")
    
    print("\nReconstructor ready for:")
    print("  - Stereo reconstruction from image pairs")
    print("  - Multi-view geometry processing")
    print("  - Depth map fusion")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("GeoVision3D - Camera and Reconstruction Examples")
    print("=" * 60)
    
    example_camera_calibration()
    example_depth_estimation()
    example_point_cloud_generation()
    example_multi_view_reconstruction()
    
    print("\n" + "=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)
