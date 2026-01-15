# GeoVision3D - Module Reference Card

## Quick Module Lookup

### 1. mesh_utils - Mesh Processing
```
Classes:
  - MeshProcessor (static methods)
    * load_mesh(filepath)
    * export_mesh(mesh, filepath)
    * normalize_mesh(mesh, target_scale=1.0)
    * validate_mesh(mesh) → (bool, str)
  
  - MeshStatistics (static methods)
    * get_basic_stats(mesh) → Dict
    * get_shape_descriptors(mesh) → Dict
    * get_all_stats(mesh) → Dict

Convenience functions:
  - load_mesh(filepath)
  - export_mesh(mesh, filepath)
  - normalize_mesh(mesh, target_scale)
  - get_mesh_stats(mesh)
```

### 2. camera_calibration - Camera Calibration
```
Classes:
  - CameraIntrinsics (data class)
    * fx, fy (focal lengths)
    * cx, cy (principal point)
    * k1, k2, p1, p2 (distortion coefficients)
    * to_camera_matrix() → np.ndarray
    * to_dist_coeffs() → np.ndarray
    * from_camera_matrix(K, dist_coeffs) → CameraIntrinsics
  
  - CameraCalibrator
    * __init__(checkerboard_size, square_size)
    * detect_corners(image) → np.ndarray
    * calibrate_from_images(paths, min_corners) → (K, dist)
    * calibrate_from_directory(image_dir, pattern) → (K, dist)
    * undistort_image(image, K, dist) → np.ndarray
    * estimate_pose(image, K, dist) → (rvec, tvec)
    * project_3d_to_2d(points_3d, K, rvec, tvec) → np.ndarray
    * save_calibration(filepath)
    * load_calibration(filepath)
```

### 3. geometric_features - Geometric Analysis
```
Classes:
  - GeometricFeatures (static methods)
    * compute_vertex_normals(mesh) → np.ndarray
    * compute_face_normals(mesh) → np.ndarray
    * compute_mean_curvature(mesh, use_laplacian) → np.ndarray
    * compute_gaussian_curvature(mesh) → np.ndarray
    * compute_principal_curvatures(mesh, neighborhood_size) → (k1, k2)
    * compute_shape_moments(mesh) → Dict
    * compute_local_descriptors(mesh, neighborhood_size) → np.ndarray

Convenience functions:
  - extract_curvature(mesh) → Dict
  - extract_shape_moments(mesh) → Dict
  - extract_local_descriptors(mesh, neighborhood_size) → np.ndarray
  - extract_all_features(mesh) → Dict
```

### 4. reconstruction - 3D Reconstruction
```
Classes:
  - DepthEstimator (static methods)
    * compute_disparity_map(left, right, block_size, num_disparities) → np.ndarray
    * disparity_to_depth(disparity, baseline, focal_length) → np.ndarray
    * refine_depth_map(depth_map, kernel_size, iterations) → np.ndarray
  
  - PointCloudProcessor (static methods)
    * depth_to_point_cloud(depth_map, K, max_depth) → (points_3d, colors)
    * filter_outliers(points, statistical_threshold) → np.ndarray
    * downsample_point_cloud(points, voxel_size) → np.ndarray
    * estimate_normals(points, k_neighbors) → np.ndarray
  
  - MultiViewReconstructor
    * __init__(K, dist_coeffs)
    * reconstruct_from_stereo(left, right, baseline, num_disparities) → (points, normals)
    * reconstruct_from_sequence(image_dir, pattern, use_stereo) → (points, normals)

Functions:
  - load_point_cloud(filepath) → (points, normals)
```

### 5. learned_features - Feature Learning
```
Classes:
  - ShapeDescriptor (static methods)
    * compute_histogram_features(mesh, n_bins) → np.ndarray
    * compute_pca_descriptors(mesh, n_components) → np.ndarray
    * compute_multiscale_descriptors(mesh, scales) → np.ndarray
    * compute_spectral_features(mesh, n_features) → np.ndarray
  
  - LearnedFeatureExtractor
    * __init__(feature_dim=128)
    * extract_raw_features(mesh) → np.ndarray
    * fit(meshes)
    * extract(mesh) → np.ndarray
    * extract_batch(meshes) → np.ndarray

Functions:
  - extract_learned_features(mesh, extractor) → np.ndarray
  - extract_all_descriptors(mesh) → Dict
```

### 6. matching - Shape Retrieval
```
Classes:
  - SimilarityMetrics (static methods)
    * euclidean_distance(f1, f2) → float
    * cosine_similarity(f1, f2) → float
    * chi_square_distance(f1, f2) → float
    * wasserstein_distance(f1, f2) → float
    * hamming_distance(f1, f2) → float
  
  - ShapeRetriever
    * __init__(metric='euclidean', normalize_features=True)
    * add_to_database(features, names) 
    * retrieve(query_features, k, return_scores) → (indices, scores)
    * retrieve_from_meshes(query_mesh, db_meshes, extractor, k) → (indices, scores)
    * evaluate_retrieval(query_idx, retrieved_idx, ground_truth) → Dict
  
  - HierarchicalRetrieval
    * __init__(num_levels=3)
    * retrieve_hierarchical(query_feat, db_features, k) → (indices, scores)

Functions:
  - compute_similarity_matrix(features, metric) → np.ndarray
```

### 7. ml_pipeline - Machine Learning
```
Classes:
  - FeatureNormalizer
    * __init__(method='standard')
    * fit(features)
    * transform(features) → np.ndarray
    * fit_transform(features) → np.ndarray
  
  - DimensionalityReducer
    * __init__(n_components=None, variance_ratio=0.95)
    * fit(features)
    * transform(features) → np.ndarray
    * get_explained_variance() → float
  
  - ShapeClassifier
    * __init__(classifier_type='svm', normalize=True)
    * fit(features, labels)
    * predict(features) → np.ndarray
    * predict_proba(features) → np.ndarray
    * evaluate(features, labels) → Dict
  
  - ShapeClusterer
    * __init__(n_clusters=5, normalize=True)
    * fit(features)
    * predict(features) → np.ndarray
    * evaluate(features) → Dict
  
  - MLPipeline
    * __init__(normalize=True, reduce_dims=True, n_components=None)
    * preprocess(features, fit=True) → np.ndarray
    * train_classifier(features, labels, classifier_type)
    * train_clusterer(features, n_clusters)
    * classify(features) → np.ndarray
    * cluster(features) → np.ndarray
    * evaluate_classification(features, labels) → Dict
    * evaluate_clustering(features) → Dict

Functions:
  - create_confusion_matrix(y_true, y_pred) → np.ndarray
```

---

## Common Imports

```python
# Basic imports
from geovision3d import mesh_utils, geometric_features, learned_features
from geovision3d import camera_calibration, reconstruction, matching, ml_pipeline

# Working with meshes
from geovision3d.mesh_utils import MeshProcessor, MeshStatistics

# Feature extraction
from geovision3d.geometric_features import GeometricFeatures
from geovision3d.learned_features import ShapeDescriptor, LearnedFeatureExtractor

# Camera and 3D
from geovision3d.camera_calibration import CameraCalibrator, CameraIntrinsics
from geovision3d.reconstruction import PointCloudProcessor, MultiViewReconstructor

# ML and retrieval
from geovision3d.matching import ShapeRetriever, SimilarityMetrics
from geovision3d.ml_pipeline import MLPipeline, ShapeClassifier, ShapeClusterer
```

---

## Common Workflows

### Workflow 1: Mesh Feature Extraction
```python
from geovision3d import mesh_utils, geometric_features

mesh = mesh_utils.load_mesh("model.obj")
mesh = mesh_utils.normalize_mesh(mesh)

features = geometric_features.extract_all_features(mesh)
stats = mesh_utils.get_mesh_stats(mesh)

# Features available:
# - features['mean_curvature']
# - features['gaussian_curvature']
# - features['moments']
# - features['local_descriptors']
# - features['normals']
```

### Workflow 2: Feature-Based Classification
```python
from geovision3d import learned_features, ml_pipeline
import numpy as np

# Prepare data
extractor = learned_features.LearnedFeatureExtractor()
extractor.fit(training_meshes)
features = extractor.extract_batch(training_meshes)

# Train classifier
pipeline = ml_pipeline.MLPipeline(normalize=True, reduce_dims=True)
pipeline.train_classifier(features, labels, classifier_type='svm')

# Classify new shape
query_features = extractor.extract(test_mesh)
pred_class = pipeline.classify(query_features.reshape(1, -1))[0]
```

### Workflow 3: Shape Retrieval
```python
from geovision3d import learned_features, matching

# Extract and index
extractor = learned_features.LearnedFeatureExtractor()
extractor.fit(database_meshes)
db_features = extractor.extract_batch(database_meshes)

# Create retriever
retriever = matching.ShapeRetriever(metric='cosine')
retriever.add_to_database(db_features, shape_names)

# Query
query_features = extractor.extract(query_mesh)
indices, scores = retriever.retrieve(query_features, k=10)
```

### Workflow 4: 3D Reconstruction
```python
from geovision3d import camera_calibration, reconstruction
import cv2

# Calibrate camera
calibrator = camera_calibration.CameraCalibrator()
K, dist = calibrator.calibrate_from_directory("calibration_images/")

# Reconstruct from stereo
reconstructor = reconstruction.MultiViewReconstructor(K, dist)
points_3d, normals = reconstructor.reconstruct_from_stereo(
    left_image, right_image, baseline=0.1
)

# Process point cloud
processor = reconstruction.PointCloudProcessor()
filtered = processor.filter_outliers(points_3d)
downsampled = processor.downsample_point_cloud(filtered, voxel_size=0.02)
```

### Workflow 5: Clustering
```python
from geovision3d import learned_features, ml_pipeline

extractor = learned_features.LearnedFeatureExtractor()
extractor.fit(shapes)
features = extractor.extract_batch(shapes)

# Train clusterer
pipeline = ml_pipeline.MLPipeline(normalize=True)
pipeline.train_clusterer(features, n_clusters=5)

# Cluster new data
clusters = pipeline.cluster(new_features)
metrics = pipeline.evaluate_clustering(features)
```

---

## Data Structures

### Key NumPy Arrays

```python
# Vertex properties
vertices: np.ndarray         # Shape (N_vertices, 3)
normals: np.ndarray          # Shape (N_vertices, 3)
curvature: np.ndarray        # Shape (N_vertices,)

# Face properties
faces: np.ndarray            # Shape (N_faces, 3) - vertex indices
face_normals: np.ndarray     # Shape (N_faces, 3)
face_areas: np.ndarray       # Shape (N_faces,)

# Features
features: np.ndarray         # Shape (N_samples, N_features)
labels: np.ndarray          # Shape (N_samples,)

# Point clouds
points_3d: np.ndarray       # Shape (N_points, 3)
colors: np.ndarray          # Shape (N_points, 3) - RGB

# Camera matrices
K: np.ndarray               # Shape (3, 3) - Camera matrix
dist_coeffs: np.ndarray     # Shape (4,) - Distortion
```

### Dictionary Returns

```python
# Statistics dictionary
stats = {
    'basic': {
        'volume': float,
        'surface_area': float,
        'num_vertices': float,
        ...
    },
    'descriptors': {
        'sphericity': float,
        'compactness': float,
        ...
    }
}

# Moments dictionary
moments = {
    'moment_0': float,
    'moment_1_x': float,
    'moment_2_total': float,
    'eigenvalue_0': float,
    ...
}

# Evaluation metrics dictionary
metrics = {
    'accuracy': float,
    'precision': float,
    'recall': float,
    'f1': float,
}
```

---

## Parameters Summary

### Frequently Used Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `n_components` (PCA) | None | 1-D | Dimensionality of output |
| `n_bins` (histogram) | 20 | 5-100 | Histogram resolution |
| `neighborhood_size` | 10 | 3-50 | Local descriptor size |
| `n_clusters` | 5 | 2-1000 | Number of clusters |
| `voxel_size` | 0.01 | 0.001-1.0 | Point cloud downsampling |
| `statistical_threshold` | 2.0 | 1-5 | Outlier filter sensitivity |
| `metric` | 'euclidean' | See SimilarityMetrics | Distance metric |

---

## Error Handling

### Common Exceptions

```python
# FileNotFoundError
load_mesh("nonexistent.obj")  # Raises FileNotFoundError

# ValueError
normalize_mesh(None)           # Raises ValueError
compute_disparity_map(img, img)  # Bad size raises ValueError

# RuntimeError
calibrate_camera(bad_images)   # Calibration failure
```

### Validation Examples

```python
# Validate mesh before processing
is_valid, message = MeshProcessor.validate_mesh(mesh)
if not is_valid:
    print(f"Mesh issues: {message}")

# Check feature vector size
if features.shape[0] < min_samples:
    raise ValueError(f"Need {min_samples} samples")
```

---

## Performance Tips

1. **Pre-process features once**: Normalize before multiple operations
2. **Use batch operations**: `extract_batch()` faster than loop
3. **Cache computed features**: Save to disk, reload when needed
4. **Downsample point clouds**: Use voxel grid for >10M points
5. **Use appropriate metric**: Cosine for normalized, Euclidean for raw
6. **Parallelize when possible**: NumPy operations are naturally parallel

---

**Last Updated**: January 15, 2026
