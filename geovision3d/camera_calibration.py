"""
Camera calibration module for intrinsic/extrinsic parameter estimation.

Supports checkerboard-based calibration, lens distortion correction, and pose estimation.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Union
from dataclasses import dataclass


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""
    fx: float  # Focal length in x
    fy: float  # Focal length in y
    cx: float  # Principal point x
    cy: float  # Principal point y
    k1: float = 0.0  # Radial distortion 1
    k2: float = 0.0  # Radial distortion 2
    p1: float = 0.0  # Tangential distortion 1
    p2: float = 0.0  # Tangential distortion 2
    
    def to_camera_matrix(self) -> np.ndarray:
        """Get 3x3 camera intrinsic matrix."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ], dtype=np.float64)
    
    def to_dist_coeffs(self) -> np.ndarray:
        """Get distortion coefficients vector."""
        return np.array([self.k1, self.k2, self.p1, self.p2], dtype=np.float64)
    
    @staticmethod
    def from_camera_matrix(K: np.ndarray, 
                          dist_coeffs: Optional[np.ndarray] = None) -> 'CameraIntrinsics':
        """Create from camera matrix and distortion coefficients."""
        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])
        
        k1, k2, p1, p2 = 0.0, 0.0, 0.0, 0.0
        if dist_coeffs is not None and len(dist_coeffs) >= 4:
            k1, k2, p1, p2 = float(dist_coeffs[0]), float(dist_coeffs[1]), \
                              float(dist_coeffs[2]), float(dist_coeffs[3])
        
        return CameraIntrinsics(fx=fx, fy=fy, cx=cx, cy=cy, 
                               k1=k1, k2=k2, p1=p1, p2=p2)


class CameraCalibrator:
    """Perform camera calibration using checkerboard patterns."""
    
    def __init__(self, checkerboard_size: Tuple[int, int] = (9, 6),
                 square_size: float = 1.0):
        """
        Initialize calibrator.
        
        Args:
            checkerboard_size: (width, height) of checkerboard (number of corners)
            square_size: Size of each square in real-world units
        """
        self.checkerboard_size = checkerboard_size
        self.square_size = square_size
        self.K = None
        self.dist_coeffs = None
        self.calibration_error = None
    
    def detect_corners(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Detect checkerboard corners in image.
        
        Args:
            image: Input image (BGR or grayscale)
            
        Returns:
            Array of corner coordinates (N, 1, 2) or None if not found
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
        
        if ret:
            # Refine corner positions
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            return corners
        
        return None
    
    def calibrate_from_images(self, image_paths: List[Union[str, Path]],
                             min_corners: int = 3) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calibrate camera from multiple images.
        
        Args:
            image_paths: List of image file paths
            min_corners: Minimum number of images with detected corners
            
        Returns:
            (camera_matrix, distortion_coefficients)
        """
        object_points = []  # 3D points
        image_points = []   # 2D image points
        image_shape = None
        
        # Prepare 3D object points (0,0,0), (1,0,0), ..., (8,5,0)
        objp = np.zeros((self.checkerboard_size[0] * self.checkerboard_size[1], 3),
                       dtype=np.float32)
        objp[:, :2] = np.mgrid[0:self.checkerboard_size[0],
                               0:self.checkerboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        
        successful_images = 0
        
        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            
            if image_shape is None:
                image_shape = image.shape[:2]
            
            corners = self.detect_corners(image)
            
            if corners is not None:
                object_points.append(objp)
                image_points.append(corners)
                successful_images += 1
        
        if successful_images < min_corners:
            raise ValueError(f"Found only {successful_images} images with corners, "
                           f"need at least {min_corners}")
        
        # Calibrate
        ret, K, dist_coeffs, _, _ = cv2.calibrateCamera(
            object_points, image_points, image_shape,
            None, None
        )
        
        if not ret:
            raise RuntimeError("Camera calibration failed")
        
        self.K = K
        self.dist_coeffs = dist_coeffs
        
        # Compute calibration error
        self.calibration_error = self._compute_reprojection_error(
            object_points, image_points, K, dist_coeffs
        )
        
        return K, dist_coeffs
    
    def calibrate_from_directory(self, image_dir: Union[str, Path],
                                pattern: str = "*.jpg") -> Tuple[np.ndarray, np.ndarray]:
        """
        Calibrate from images in a directory.
        
        Args:
            image_dir: Directory containing calibration images
            pattern: File pattern to match
            
        Returns:
            (camera_matrix, distortion_coefficients)
        """
        image_dir = Path(image_dir)
        image_paths = sorted(image_dir.glob(pattern))
        
        if not image_paths:
            raise FileNotFoundError(f"No images matching '{pattern}' in {image_dir}")
        
        return self.calibrate_from_images(image_paths)
    
    def undistort_image(self, image: np.ndarray,
                       K: Optional[np.ndarray] = None,
                       dist_coeffs: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Remove lens distortion from image.
        
        Args:
            image: Input image
            K: Camera matrix (uses calibrated if not provided)
            dist_coeffs: Distortion coefficients (uses calibrated if not provided)
            
        Returns:
            Undistorted image
        """
        if K is None:
            K = self.K
        if dist_coeffs is None:
            dist_coeffs = self.dist_coeffs
        
        if K is None or dist_coeffs is None:
            raise ValueError("Calibration parameters not available")
        
        return cv2.undistort(image, K, dist_coeffs)
    
    def estimate_pose(self, image: np.ndarray,
                     K: Optional[np.ndarray] = None,
                     dist_coeffs: Optional[np.ndarray] = None
                     ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Estimate camera pose (rotation and translation) from checkerboard.
        
        Args:
            image: Input image
            K: Camera matrix
            dist_coeffs: Distortion coefficients
            
        Returns:
            (rotation_vector, translation_vector) or None if no checkerboard found
        """
        if K is None:
            K = self.K
        if dist_coeffs is None:
            dist_coeffs = self.dist_coeffs
        
        corners = self.detect_corners(image)
        if corners is None:
            return None
        
        # 3D object points
        objp = np.zeros((self.checkerboard_size[0] * self.checkerboard_size[1], 3),
                       dtype=np.float32)
        objp[:, :2] = np.mgrid[0:self.checkerboard_size[0],
                               0:self.checkerboard_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        
        # Solve PnP
        success, rvec, tvec = cv2.solvePnP(objp, corners, K, dist_coeffs)
        
        if success:
            return rvec, tvec
        return None
    
    def project_3d_to_2d(self, points_3d: np.ndarray,
                        K: Optional[np.ndarray] = None,
                        rvec: Optional[np.ndarray] = None,
                        tvec: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Project 3D points to 2D image coordinates.
        
        Args:
            points_3d: 3D points (N, 3)
            K: Camera matrix
            rvec: Rotation vector
            tvec: Translation vector
            
        Returns:
            2D image points (N, 2)
        """
        if K is None:
            K = self.K
        
        if K is None:
            raise ValueError("Camera matrix not available")
        
        # Default: identity rotation and zero translation
        if rvec is None:
            rvec = np.array([0, 0, 0], dtype=np.float64)
        if tvec is None:
            tvec = np.array([0, 0, 0], dtype=np.float64)
        
        points_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, K, None)
        return points_2d.reshape(-1, 2)
    
    @staticmethod
    def _compute_reprojection_error(object_points: List[np.ndarray],
                                   image_points: List[np.ndarray],
                                   K: np.ndarray,
                                   dist_coeffs: np.ndarray) -> float:
        """Compute mean reprojection error."""
        total_error = 0.0
        total_points = 0
        
        for objp, imgp in zip(object_points, image_points):
            rvec, tvec, _ = cv2.solvePnPRansac(objp, imgp, K, dist_coeffs)
            projected_points, _ = cv2.projectPoints(objp, rvec, tvec, K, dist_coeffs)
            error = cv2.norm(imgp, projected_points, cv2.NORM_L2) / len(objp)
            total_error += error
            total_points += 1
        
        return total_error / (total_points + 1e-8)
    
    def get_intrinsics(self) -> Optional[CameraIntrinsics]:
        """Get calibrated intrinsic parameters."""
        if self.K is None:
            return None
        return CameraIntrinsics.from_camera_matrix(self.K, self.dist_coeffs)
    
    def save_calibration(self, filepath: Union[str, Path]) -> None:
        """Save calibration parameters to file."""
        if self.K is None:
            raise ValueError("No calibration data to save")
        
        np.savez(filepath, K=self.K, dist_coeffs=self.dist_coeffs,
                error=self.calibration_error)
    
    def load_calibration(self, filepath: Union[str, Path]) -> None:
        """Load calibration parameters from file."""
        data = np.load(filepath)
        self.K = data['K']
        self.dist_coeffs = data['dist_coeffs']
        if 'error' in data:
            self.calibration_error = float(data['error'])
