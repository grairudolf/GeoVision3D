"""
3D reconstruction module for multi-view geometry and depth estimation.

Supports depth map processing, point cloud generation, and multi-view fusion.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional, List, Union, Dict
import trimesh


class DepthEstimator:
    """Estimate depth from stereo or monocular images."""
    
    @staticmethod
    def compute_disparity_map(left_image: np.ndarray,
                             right_image: np.ndarray,
                             block_size: int = 15,
                             num_disparities: int = 64) -> np.ndarray:
        """
        Compute disparity map using stereo matching.
        
        Args:
            left_image: Left stereo image (grayscale)
            right_image: Right stereo image (grayscale)
            block_size: Matching block size (odd number)
            num_disparities: Number of disparity levels
            
        Returns:
            Disparity map
        """
        # Ensure grayscale
        if len(left_image.shape) == 3:
            left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        if len(right_image.shape) == 3:
            right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        
        # Create stereo matcher
        stereo = cv2.StereoBM_create(numDisparities=num_disparities,
                                     blockSize=block_size)
        
        disparity = stereo.compute(left_image, right_image)
        
        # Convert to float and normalize
        disparity = disparity.astype(np.float32) / 16.0
        
        return disparity
    
    @staticmethod
    def disparity_to_depth(disparity: np.ndarray,
                          baseline: float,
                          focal_length: float) -> np.ndarray:
        """
        Convert disparity map to depth map.
        
        Args:
            disparity: Disparity map
            baseline: Stereo baseline distance
            focal_length: Camera focal length (in pixels)
            
        Returns:
            Depth map
        """
        # Avoid division by zero
        depth = np.zeros_like(disparity)
        valid_mask = disparity > 0
        
        depth[valid_mask] = (baseline * focal_length) / (disparity[valid_mask] + 1e-8)
        
        return depth
    
    @staticmethod
    def refine_depth_map(depth_map: np.ndarray,
                        kernel_size: int = 5,
                        iterations: int = 2) -> np.ndarray:
        """
        Refine depth map using bilateral filtering and morphology.
        
        Args:
            depth_map: Raw depth map
            kernel_size: Filter kernel size
            iterations: Number of refinement iterations
            
        Returns:
            Refined depth map
        """
        refined = depth_map.copy()
        
        for _ in range(iterations):
            # Bilateral filter to preserve edges
            refined = cv2.bilateralFilter(refined.astype(np.float32),
                                         kernel_size, 50, 50)
            
            # Morphological closing to fill small holes
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                              (kernel_size, kernel_size))
            refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)
        
        return refined


class PointCloudProcessor:
    """Process and analyze point clouds from depth maps."""
    
    @staticmethod
    def depth_to_point_cloud(depth_map: np.ndarray,
                            K: np.ndarray,
                            max_depth: float = 100.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert depth map to 3D point cloud.
        
        Args:
            depth_map: Depth map (H, W)
            K: Camera intrinsic matrix (3, 3)
            max_depth: Maximum valid depth threshold
            
        Returns:
            (points_3d, colors) where colors are from depth intensity
        """
        h, w = depth_map.shape
        
        # Create pixel coordinates
        x = np.arange(w)
        y = np.arange(h)
        xx, yy = np.meshgrid(x, y)
        
        # Backproject to 3D
        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        
        z = depth_map
        x_3d = (xx - cx) * z / fx
        y_3d = (yy - cy) * z / fy
        
        # Stack to point cloud
        points = np.stack([x_3d, y_3d, z], axis=-1).reshape(-1, 3)
        
        # Filter invalid points
        valid_mask = (z.flatten() > 0) & (z.flatten() < max_depth)
        points = points[valid_mask]
        
        # Normalize depth for color visualization
        colors = (z.flatten()[valid_mask] / max_depth * 255).astype(np.uint8)
        colors = np.stack([colors, colors, colors], axis=-1)
        
        return points, colors
    
    @staticmethod
    def filter_outliers(points: np.ndarray,
                       statistical_threshold: float = 2.0) -> np.ndarray:
        """
        Remove outlier points using statistical distance.
        
        Args:
            points: Point cloud (N, 3)
            statistical_threshold: Distance threshold in standard deviations
            
        Returns:
            Filtered point cloud
        """
        # Compute mean and covariance
        mean = points.mean(axis=0)
        cov = np.cov(points.T)
        
        # Compute Mahalanobis distance
        diff = points - mean
        inv_cov = np.linalg.inv(cov + np.eye(3) * 1e-6)
        distances = np.sqrt(np.sum(diff @ inv_cov * diff, axis=1))
        
        # Filter
        threshold = distances.mean() + statistical_threshold * distances.std()
        valid_mask = distances < threshold
        
        return points[valid_mask]
    
    @staticmethod
    def downsample_point_cloud(points: np.ndarray,
                              voxel_size: float = 0.01) -> np.ndarray:
        """
        Downsample point cloud using voxel grid.
        
        Args:
            points: Point cloud (N, 3)
            voxel_size: Size of voxel grid cells
            
        Returns:
            Downsampled point cloud
        """
        if len(points) == 0:
            return points
        
        # Quantize to voxel grid
        quantized = np.floor(points / voxel_size).astype(np.int32)
        
        # Find unique voxels
        unique_voxels, inverse_indices = np.unique(quantized, axis=0,
                                                   return_inverse=True)
        
        # Compute voxel centers
        downsampled = np.zeros_like(unique_voxels, dtype=np.float32)
        for i in range(len(unique_voxels)):
            mask = inverse_indices == i
            downsampled[i] = points[mask].mean(axis=0)
        
        return downsampled
    
    @staticmethod
    def estimate_normals(points: np.ndarray,
                        k_neighbors: int = 10) -> np.ndarray:
        """
        Estimate surface normals for point cloud.
        
        Args:
            points: Point cloud (N, 3)
            k_neighbors: Number of neighbors for local PCA
            
        Returns:
            Normal vectors (N, 3)
        """
        from scipy.spatial import cKDTree
        
        normals = np.zeros_like(points)
        kdtree = cKDTree(points)
        
        for i in range(len(points)):
            # Find neighbors
            _, indices = kdtree.query(points[i], k=min(k_neighbors + 1, len(points)))
            neighbors = points[indices[1:]]
            
            # Local PCA
            centered = neighbors - neighbors.mean(axis=0)
            _, _, Vt = np.linalg.svd(centered)
            
            # Normal is smallest principal component
            normals[i] = Vt[-1]
        
        return normals


class MultiViewReconstructor:
    """Reconstruct 3D from multiple views."""
    
    def __init__(self, K: np.ndarray, dist_coeffs: np.ndarray):
        """
        Initialize reconstructor.
        
        Args:
            K: Camera intrinsic matrix
            dist_coeffs: Distortion coefficients
        """
        self.K = K
        self.dist_coeffs = dist_coeffs
    
    def reconstruct_from_stereo(self, left_image: np.ndarray,
                               right_image: np.ndarray,
                               baseline: float,
                               num_disparities: int = 64) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct from stereo pair.
        
        Args:
            left_image: Left image
            right_image: Right image
            baseline: Stereo baseline
            num_disparities: Disparity levels
            
        Returns:
            (points_3d, normals)
        """
        # Undistort images
        left_undist = cv2.undistort(left_image, self.K, self.dist_coeffs)
        right_undist = cv2.undistort(right_image, self.K, self.dist_coeffs)
        
        # Estimate disparity
        estimator = DepthEstimator()
        disparity = estimator.compute_disparity_map(left_undist, right_undist,
                                                    num_disparities=num_disparities)
        
        # Convert to depth
        focal_length = self.K[0, 0]
        depth = estimator.disparity_to_depth(disparity, baseline, focal_length)
        
        # Refine depth
        depth = estimator.refine_depth_map(depth)
        
        # Convert to point cloud
        processor = PointCloudProcessor()
        points_3d, _ = processor.depth_to_point_cloud(depth, self.K)
        points_3d = processor.filter_outliers(points_3d)
        
        # Estimate normals
        normals = processor.estimate_normals(points_3d)
        
        return points_3d, normals
    
    def reconstruct_from_sequence(self, image_dir: Union[str, Path],
                                 pattern: str = "*.jpg",
                                 use_stereo: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Reconstruct from image sequence.
        
        Args:
            image_dir: Directory with images
            pattern: Image pattern
            use_stereo: Use stereo reconstruction if True
            
        Returns:
            (points_3d, normals)
        """
        image_dir = Path(image_dir)
        image_paths = sorted(image_dir.glob(pattern))
        
        if not image_paths:
            raise FileNotFoundError(f"No images matching '{pattern}'")
        
        # For simple reconstruction, use first two images as stereo pair
        if len(image_paths) >= 2:
            left_image = cv2.imread(str(image_paths[0]))
            right_image = cv2.imread(str(image_paths[1]))
            
            if left_image is not None and right_image is not None:
                return self.reconstruct_from_stereo(left_image, right_image,
                                                   baseline=0.1)
        
        raise ValueError("Need at least 2 images for reconstruction")


def load_point_cloud(filepath: Union[str, Path]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load point cloud from file (OBJ or PLY format)."""
    mesh = trimesh.load(str(filepath))
    if isinstance(mesh, trimesh.Trimesh):
        return mesh.vertices, mesh.vertex_normals
    return np.array(mesh.vertices), None
