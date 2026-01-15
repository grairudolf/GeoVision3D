# GeoVision3D: High-Accuracy 3D Shape Analysis, Classification, and Retrieval

A comprehensive Python system for analyzing, classifying, and retrieving 3D shapes from mesh files and camera feeds using advanced computer vision, geometric processing, and machine learning techniques.

## Features

### 1. **Multi-Source Input Processing**
   - Load and process 3D mesh files (OBJ, STL formats)
   - Reconstruct 3D geometry from calibrated camera feeds
   - Support for multi-view geometry and depth estimation

### 2. **Robust Feature Extraction**
   - **Geometric Features**: Surface curvature, normal estimation, shape moments
   - **Mesh Statistics**: Volume, surface area, compactness, aspect ratio
   - **Learned Descriptors**: Deep-learned shape embeddings and neural descriptors

### 3. **Camera Vision Pipeline**
   - Intrinsic and extrinsic camera calibration
   - Lens distortion correction
   - Multi-view stereo reconstruction
   - Depth estimation and 3D point cloud generation

### 4. **Shape Matching & Retrieval**
   - Feature-based similarity metrics
   - Efficient indexing and ranking
   - Nearest-neighbor retrieval with quantitative evaluation

### 5. **Machine Learning Integration**
   - Feature normalization and dimensionality reduction (PCA)
   - Supervised classification (SVM, RandomForest)
   - Unsupervised clustering (K-means, hierarchical)
   - Comprehensive evaluation metrics

## Project Structure

```
GeoVision3D/
├── geovision3d/                    # Main package
│   ├── __init__.py
│   ├── mesh_utils.py              # Mesh I/O and processing
│   ├── camera_calibration.py       # Camera calibration module
│   ├── reconstruction.py           # 3D reconstruction from camera
│   ├── geometric_features.py       # Geometric feature extraction
│   ├── learned_features.py         # Learned shape descriptors
│   ├── matching.py                 # Shape matching and retrieval
│   └── ml_pipeline.py              # ML pipeline with scikit-learn
├── examples/                       # Example scripts
│   ├── example_mesh_analysis.py
│   ├── example_camera_reconstruction.py
│   └── example_shape_retrieval.py
├── tests/                          # Unit tests
│   └── test_*.py
├── data/                           # Sample data directory
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
cd c:\Users\Rudolf\Desktop\projects\GeoVision3D
```

2. Create virtual environment (optional):
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Mesh Analysis
```python
from geovision3d import mesh_utils, geometric_features

# Load mesh
mesh = mesh_utils.load_mesh("path/to/model.obj")

# Extract geometric features
features = geometric_features.extract_all_features(mesh)
print(f"Shape volume: {features['volume']:.4f}")
print(f"Mean curvature: {features['mean_curvature']:.4f}")
```

### Camera Calibration and Reconstruction
```python
from geovision3d import camera_calibration, reconstruction

# Calibrate camera
calib = camera_calibration.CameraCalibrator()
K, dist_coeffs = calib.calibrate_from_checkerboard(images_dir="calibration_images/")

# Reconstruct 3D from camera feed
reconstructor = reconstruction.MultiViewReconstructor(K, dist_coeffs)
points_3d, normals = reconstructor.reconstruct_from_sequence("video_frames/")
```

### Shape Matching and Retrieval
```python
from geovision3d import mesh_utils, geometric_features, matching

# Load query and database shapes
query_mesh = mesh_utils.load_mesh("query.obj")
database_meshes = [mesh_utils.load_mesh(f"db_{i}.obj") for i in range(10)]

# Extract and match features
query_features = geometric_features.extract_all_features(query_mesh)
retriever = matching.ShapeRetriever()
ranks, scores = retriever.retrieve(query_features, database_meshes)
```

## Key Algorithms

### Geometric Feature Extraction
- **Mean Curvature**: Discrete mesh curvature estimation
- **Gaussian Curvature**: Per-vertex curvature analysis
- **Shape Descriptors**: Moment invariants, PCA-based shape characterization
- **Mesh Statistics**: Aspect ratio, sphericity, compactness

### Computer Vision
- **Camera Calibration**: Zhang's method for checkerboard calibration
- **Multi-View Geometry**: Structure from Motion (SfM) concepts
- **Depth Estimation**: Stereo reconstruction and depth refinement
- **Point Cloud Processing**: Filtering, downsampling, normal estimation

### Machine Learning
- **Feature Normalization**: StandardScaler for zero-mean unit-variance
- **Dimensionality Reduction**: Principal Component Analysis (PCA)
- **Classification**: Support Vector Machines, Random Forest
- **Clustering**: K-means, Hierarchical clustering with linkage analysis
- **Evaluation**: Precision, Recall, F1-score, Silhouette coefficient

## API Reference

### mesh_utils
- `load_mesh(path)` - Load OBJ or STL file
- `export_mesh(mesh, path)` - Export mesh to file
- `normalize_mesh(mesh)` - Center and scale mesh
- `get_mesh_stats(mesh)` - Compute basic statistics

### camera_calibration
- `CameraCalibrator.calibrate_from_checkerboard()` - Calibrate from images
- `CameraCalibrator.undistort_image()` - Remove lens distortion
- `CameraCalibrator.project_3d_to_2d()` - Project points

### reconstruction
- `MultiViewReconstructor` - Reconstruct from image sequence
- `DepthEstimator` - Estimate depth from stereo/monocular
- `PointCloudProcessor` - Process and filter point clouds

### geometric_features
- `extract_curvature()` - Compute surface curvature
- `extract_shape_moments()` - Shape moment invariants
- `extract_all_features()` - All geometric features at once

### matching
- `ShapeRetriever` - Efficient shape retrieval and ranking
- `compute_similarity()` - Feature similarity metrics
- `evaluate_retrieval()` - Quantitative evaluation

### ml_pipeline
- `FeatureNormalizer` - Normalize feature vectors
- `DimensionalityReducer` - PCA wrapper
- `ShapeClassifier` - Train/test classification
- `ShapeClusterer` - Unsupervised clustering

## Performance & Accuracy

The system emphasizes accuracy through:
- **Camera Calibration**: Proper intrinsic/extrinsic parameter estimation
- **Noise Handling**: Outlier removal, smoothing, robust statistics
- **Feature Validation**: Cross-validation, sensitivity analysis
- **Quantitative Evaluation**: Precision-recall curves, confusion matrices

## Requirements

- Python 3.8+
- NumPy >= 1.24
- SciPy >= 1.11
- scikit-learn >= 1.3
- OpenCV >= 4.8
- trimesh >= 3.20
- Pillow >= 10.0

## Future Enhancements

- Integration with deep learning (CNN-based feature extraction)
- Real-time streaming 3D reconstruction
- GPU acceleration for large-scale retrieval
- Web API for remote shape analysis
- Interactive 3D visualization module

## License

MIT License

## Contact

For questions and feedback, please open an issue on the project repository.
