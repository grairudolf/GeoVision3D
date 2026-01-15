"""
API Reference and Detailed Documentation for GeoVision3D
"""

# GeoVision3D API Reference

## Core Modules

### 1. mesh_utils - Mesh Loading and Processing

**Classes:**
- `MeshProcessor`: Static methods for mesh operations
- `MeshStatistics`: Compute mesh properties and descriptors

**Key Functions:**

#### MeshProcessor.load_mesh(filepath: str | Path) -> trimesh.Trimesh
Load mesh from OBJ, STL, PLY, or GLTF file.
```python
from geovision3d import mesh_utils
mesh = mesh_utils.load_mesh("model.obj")
```

#### MeshProcessor.export_mesh(mesh: Trimesh, filepath: str | Path) -> None
Export mesh to file.
```python
mesh_utils.export_mesh(mesh, "output.stl")
```

#### MeshProcessor.normalize_mesh(mesh: Trimesh, target_scale: float = 1.0) -> Trimesh
Normalize mesh to unit scale and center at origin.
```python
normalized = mesh_utils.normalize_mesh(mesh, target_scale=1.0)
```

#### MeshProcessor.validate_mesh(mesh: Trimesh) -> (bool, str)
Validate mesh integrity.
```python
is_valid, message = mesh_utils.MeshProcessor.validate_mesh(mesh)
print(message)  # Details on any issues
```

#### MeshStatistics.get_basic_stats(mesh: Trimesh) -> Dict
Get basic mesh statistics:
- Volume, surface area, number of vertices/faces/edges
- Vertex distance statistics
- Edge length distribution
- Face area distribution

```python
stats = mesh_utils.MeshStatistics.get_basic_stats(mesh)
print(stats['volume'])
```

#### MeshStatistics.get_shape_descriptors(mesh: Trimesh) -> Dict
Get shape descriptors:
- Aspect ratios (elongation)
- Sphericity
- Compactness
- Convexity
- Volume ratio

---

### 2. camera_calibration - Camera Parameter Estimation

**Classes:**
- `CameraIntrinsics`: Data class for camera parameters
- `CameraCalibrator`: Calibration interface

**Key Classes:**

#### CameraIntrinsics
```python
intrinsics = camera_calibration.CameraIntrinsics(
    fx=500, fy=500, cx=320, cy=240,
    k1=0.1, k2=-0.05, p1=0.001, p2=0.002
)

K = intrinsics.to_camera_matrix()  # 3x3 matrix
dist = intrinsics.to_dist_coeffs()  # Distortion vector
```

#### CameraCalibrator.calibrate_from_directory(image_dir, pattern="*.jpg")
Calibrate from checkerboard images:
```python
calibrator = camera_calibration.CameraCalibrator(
    checkerboard_size=(9, 6),
    square_size=0.025  # 25mm squares
)

K, dist_coeffs = calibrator.calibrate_from_directory("calibration_images/")
print(f"Calibration error: {calibrator.calibration_error:.4f}")
```

#### CameraCalibrator.undistort_image(image, K, dist_coeffs)
Remove lens distortion:
```python
undistorted = calibrator.undistort_image(image, K, dist_coeffs)
```

#### CameraCalibrator.estimate_pose(image, K, dist_coeffs)
Estimate camera pose from checkerboard:
```python
rvec, tvec = calibrator.estimate_pose(image, K, dist_coeffs)
# rvec: rotation vector, tvec: translation vector
```

---

### 3. geometric_features - Geometric Feature Extraction

**Classes:**
- `GeometricFeatures`: Static methods for feature computation

**Key Functions:**

#### extract_curvature(mesh) -> Dict
Compute mean and Gaussian curvature:
```python
from geovision3d import geometric_features

curvature = geometric_features.extract_curvature(mesh)
mean_curv = curvature['mean_curvature']  # Per-vertex values
gaussian_curv = curvature['gaussian_curvature']
```

#### extract_shape_moments(mesh) -> Dict
Shape moment invariants:
```python
moments = geometric_features.extract_shape_moments(mesh)
# eigenvalues, moment_0, moment_1_x, moment_1_y, etc.
```

#### extract_local_descriptors(mesh, neighborhood_size=20) -> ndarray
Local geometric descriptors per vertex:
- Local curvature statistics
- Neighborhood distances
- Normal vector information
```python
descriptors = geometric_features.extract_local_descriptors(mesh, neighborhood_size=15)
# Shape: (num_vertices, num_features)
```

#### extract_all_features(mesh) -> Dict
Comprehensive feature extraction:
```python
all_features = geometric_features.extract_all_features(mesh)
# Contains: mean_curvature, gaussian_curvature, moments, 
#           local_descriptors, normals
```

---

### 4. reconstruction - 3D Reconstruction from Images

**Classes:**
- `DepthEstimator`: Depth map computation
- `PointCloudProcessor`: Point cloud operations
- `MultiViewReconstructor`: Multi-view reconstruction

**Key Functions:**

#### DepthEstimator.compute_disparity_map(left, right, block_size=15, num_disparities=64)
Stereo matching:
```python
from geovision3d import reconstruction

estimator = reconstruction.DepthEstimator()
disparity = estimator.compute_disparity_map(left_image, right_image)
```

#### DepthEstimator.disparity_to_depth(disparity, baseline, focal_length)
Convert disparity to depth:
```python
depth = estimator.disparity_to_depth(disparity, baseline=0.1, focal_length=500)
```

#### PointCloudProcessor.depth_to_point_cloud(depth_map, K, max_depth=100)
Generate 3D point cloud:
```python
processor = reconstruction.PointCloudProcessor()
points_3d, colors = processor.depth_to_point_cloud(depth_map, K, max_depth=100)
```

#### PointCloudProcessor.filter_outliers(points, statistical_threshold=2.0)
Remove outlier points:
```python
filtered = processor.filter_outliers(points_3d, statistical_threshold=2.0)
```

#### PointCloudProcessor.downsample_point_cloud(points, voxel_size=0.01)
Downsample using voxel grid:
```python
downsampled = processor.downsample_point_cloud(points, voxel_size=0.05)
```

#### PointCloudProcessor.estimate_normals(points, k_neighbors=10)
Estimate surface normals:
```python
normals = processor.estimate_normals(points, k_neighbors=10)
```

#### MultiViewReconstructor.reconstruct_from_stereo(left, right, baseline, num_disparities)
Complete stereo reconstruction:
```python
reconstructor = reconstruction.MultiViewReconstructor(K, dist_coeffs)
points_3d, normals = reconstructor.reconstruct_from_stereo(
    left_image, right_image, baseline=0.1
)
```

---

### 5. learned_features - Learned Shape Descriptors

**Classes:**
- `ShapeDescriptor`: Static descriptor computation methods
- `LearnedFeatureExtractor`: Learnable feature extraction

**Key Functions:**

#### ShapeDescriptor.compute_histogram_features(mesh, n_bins=20)
Histogram-based features:
```python
from geovision3d import learned_features

hist = learned_features.ShapeDescriptor.compute_histogram_features(mesh, n_bins=20)
```

#### ShapeDescriptor.compute_pca_descriptors(mesh, n_components=10)
PCA-based descriptors:
```python
pca_feat = learned_features.ShapeDescriptor.compute_pca_descriptors(mesh)
```

#### ShapeDescriptor.compute_spectral_features(mesh, n_features=10)
Laplacian eigenvalue features:
```python
spectral = learned_features.ShapeDescriptor.compute_spectral_features(mesh)
```

#### LearnedFeatureExtractor
Trainable feature extractor with dimensionality reduction:
```python
extractor = learned_features.LearnedFeatureExtractor(feature_dim=128)

# Fit on training shapes
extractor.fit(training_meshes)

# Extract features
features = extractor.extract(query_mesh)

# Batch extraction
batch_features = extractor.extract_batch(meshes_list)
```

---

### 6. matching - Shape Retrieval and Similarity

**Classes:**
- `SimilarityMetrics`: Distance and similarity metrics
- `ShapeRetriever`: Efficient retrieval interface
- `HierarchicalRetrieval`: Coarse-to-fine retrieval

**Key Functions:**

#### SimilarityMetrics (Static methods)
```python
from geovision3d.matching import SimilarityMetrics

# Available metrics
euclidean_dist = SimilarityMetrics.euclidean_distance(f1, f2)
cosine_sim = SimilarityMetrics.cosine_similarity(f1, f2)
chi_sq = SimilarityMetrics.chi_square_distance(f1, f2)
wasserstein = SimilarityMetrics.wasserstein_distance(f1, f2)
```

#### ShapeRetriever
```python
from geovision3d import matching

retriever = matching.ShapeRetriever(metric='euclidean', normalize_features=True)

# Add database
retriever.add_to_database(database_features, shape_names)

# Retrieve
indices, scores = retriever.retrieve(query_features, k=5, return_scores=True)

# Retrieve from meshes
indices, scores = retriever.retrieve_from_meshes(
    query_mesh, database_meshes, feature_extractor, k=5
)

# Evaluate
metrics = retriever.evaluate_retrieval(query_indices, retrieved_indices, ground_truth)
```

#### compute_similarity_matrix(features, metric='euclidean')
Full similarity matrix:
```python
sim_matrix = matching.compute_similarity_matrix(features, metric='cosine')
```

---

### 7. ml_pipeline - Machine Learning Pipeline

**Classes:**
- `FeatureNormalizer`: Feature standardization
- `DimensionalityReducer`: PCA wrapper
- `ShapeClassifier`: Supervised classification
- `ShapeClusterer`: Unsupervised clustering
- `MLPipeline`: Complete integrated pipeline

**Key Components:**

#### FeatureNormalizer
```python
from geovision3d import ml_pipeline

normalizer = ml_pipeline.FeatureNormalizer(method='standard')
normalizer.fit(training_features)
normalized = normalizer.transform(test_features)
```

#### DimensionalityReducer
```python
reducer = ml_pipeline.DimensionalityReducer(
    n_components=32,
    variance_ratio=0.95
)
reducer.fit(features)
reduced = reducer.transform(features)
print(f"Explained variance: {reducer.get_explained_variance():.2%}")
```

#### ShapeClassifier
```python
classifier = ml_pipeline.ShapeClassifier(
    classifier_type='svm',  # or 'random_forest'
    normalize=True
)

# Train
classifier.fit(training_features, training_labels)

# Predict
predictions = classifier.predict(test_features)
probabilities = classifier.predict_proba(test_features)

# Evaluate
metrics = classifier.evaluate(test_features, test_labels)
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

#### ShapeClusterer
```python
clusterer = ml_pipeline.ShapeClusterer(n_clusters=5, normalize=True)

# Fit
clusterer.fit(features)

# Predict
cluster_labels = clusterer.predict(new_features)

# Evaluate
metrics = clusterer.evaluate(features)
print(f"Silhouette score: {metrics['silhouette_score']:.4f}")
```

#### MLPipeline - Complete Workflow
```python
# Initialize with all components
pipeline = ml_pipeline.MLPipeline(
    normalize=True,
    reduce_dims=True,
    n_components=64
)

# Classification
pipeline.train_classifier(training_features, training_labels, classifier_type='svm')
predictions = pipeline.classify(test_features)
metrics = pipeline.evaluate_classification(test_features, test_labels)

# Clustering
pipeline.train_clusterer(features, n_clusters=5)
clusters = pipeline.cluster(new_features)
metrics = pipeline.evaluate_clustering(features)
```

---

## Common Workflows

### Workflow 1: Mesh Analysis and Classification

```python
from geovision3d import mesh_utils, geometric_features, ml_pipeline

# Load mesh
mesh = mesh_utils.load_mesh("model.obj")

# Normalize
mesh = mesh_utils.normalize_mesh(mesh)

# Extract features
features = geometric_features.extract_all_features(mesh)

# Flatten for ML
feat_vector = np.concatenate([
    features['moments'].values(),
    features['mean_curvature'].mean(axis=0),
])

# Classify
pipeline = ml_pipeline.MLPipeline()
pipeline.train_classifier(training_features, labels)
predicted_class = pipeline.classify(feat_vector)
```

### Workflow 2: Shape Retrieval

```python
from geovision3d import learned_features, matching

# Train feature extractor
extractor = learned_features.LearnedFeatureExtractor()
extractor.fit(database_meshes)

# Extract features for all shapes
db_features = extractor.extract_batch(database_meshes)
query_features = extractor.extract(query_mesh)

# Retrieve similar shapes
retriever = matching.ShapeRetriever()
retriever.add_to_database(db_features, shape_names)
indices, scores = retriever.retrieve(query_features, k=10)

# Display results
for idx, score in zip(indices, scores):
    print(f"{shape_names[idx]}: {score:.4f}")
```

### Workflow 3: Camera Calibration and Reconstruction

```python
from geovision3d import camera_calibration, reconstruction

# Calibrate camera
calibrator = camera_calibration.CameraCalibrator()
K, dist = calibrator.calibrate_from_directory("calibration_images/")

# Reconstruct from stereo pair
reconstructor = reconstruction.MultiViewReconstructor(K, dist)
points_3d, normals = reconstructor.reconstruct_from_stereo(
    left_image, right_image, baseline=0.1
)

# Process point cloud
processor = reconstruction.PointCloudProcessor()
filtered = processor.filter_outliers(points_3d)
downsampled = processor.downsample_point_cloud(filtered, voxel_size=0.02)
```

---

## Best Practices

1. **Feature Normalization**: Always normalize features before ML operations
2. **Mesh Validation**: Validate meshes before processing
3. **Outlier Handling**: Use statistical filtering for robust results
4. **Cross-Validation**: Evaluate on held-out test sets
5. **Parameter Tuning**: Use grid search for classifier/clustering parameters

---

## Performance Considerations

- **Large Meshes**: Use downsampling for meshes with >100K vertices
- **Feature Extraction**: Pre-compute and cache features when possible
- **Retrieval**: Use hierarchical retrieval for large databases (>1000 shapes)
- **Memory**: Batch process point clouds to reduce memory usage

---

## Troubleshooting

**Issue: Low calibration error**
- Ensure checkerboard is fully visible in all images
- Use images from different angles and distances
- Ensure good lighting conditions

**Issue: Poor mesh features**
- Validate mesh is watertight and well-formed
- Check for degenerate faces
- Normalize mesh before feature extraction

**Issue: Low classification accuracy**
- Ensure sufficient training data
- Check feature normalization
- Increase training set size or complexity
