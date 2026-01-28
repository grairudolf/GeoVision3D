# GeoVision3D - 3D Shape Analysis & Retrieval System

A comprehensive Python system for analyzing, classifying, and retrieving 3D shapes from mesh files using computer vision and machine learning techniques.

---

## What is GeoVision3D?

GeoVision3D is a complete 3D shape analysis toolkit that enables you to:
- Load and analyze 3D mesh files (OBJ, STL, PLY, GLTF formats)
- Extract geometric features like surface curvature, shape moments, and descriptors
- Calibrate cameras and reconstruct 3D geometry from camera feeds
- Classify and retrieve shapes using machine learning
- Match shapes efficiently using multiple similarity metrics

It combines geometric processing, computer vision, and machine learning in an easy-to-use package.

---

## Quick Setup

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Installation

1. Navigate to the project directory:
```bash
cd GeoVision3D
```

2. Create and activate virtual environment (recommended):
```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## Quick Usage Examples

### Load and Analyze a Mesh
```python
from geovision3d import mesh_utils, geometric_features

mesh = mesh_utils.load_mesh("model.obj")
stats = mesh_utils.get_mesh_stats(mesh)

print(f"Volume: {stats['basic']['volume']:.4f}")
print(f"Surface Area: {stats['basic']['surface_area']:.4f}")
print(f"Sphericity: {stats['descriptors']['sphericity']:.4f}")
```

### Train a Shape Classifier
```python
import numpy as np
import trimesh
from geovision3d import learned_features, ml_pipeline

# Prepare training data
shapes = [
    trimesh.creation.icosphere(subdivisions=2),
    trimesh.creation.box(),
    trimesh.creation.cylinder()
]
labels = np.array([0, 1, 2])

# Train classifier
extractor = learned_features.LearnedFeatureExtractor()
extractor.fit(shapes)
features = extractor.extract_batch(shapes)

pipeline = ml_pipeline.MLPipeline(normalize=True, reduce_dims=True)
pipeline.train_classifier(features, labels, classifier_type='svm')

# Classify new shape
test_mesh = trimesh.creation.icosphere()
test_features = extractor.extract(test_mesh)
prediction = pipeline.classify(test_features.reshape(1, -1))[0]
```

### Retrieve Similar Shapes
```python
from geovision3d import learned_features, matching

# Build database
extractor = learned_features.LearnedFeatureExtractor()
extractor.fit(shapes)
features = extractor.extract_batch(shapes)

retriever = matching.ShapeRetriever(metric='euclidean')
retriever.add_to_database(features, ['sphere', 'cube', 'cylinder'])

# Query
query_features = extractor.extract(test_mesh)
results = retriever.retrieve(query_features, top_k=2)
```

---

## Project Structure

```
GeoVision3D/
├── geovision3d/
│   ├── mesh_utils.py              # Load, process, analyze meshes
│   ├── camera_calibration.py       # Camera calibration & distortion
│   ├── reconstruction.py           # 3D reconstruction from images
│   ├── geometric_features.py       # Extract geometric features
│   ├── learned_features.py         # Learn shape descriptors
│   ├── matching.py                 # Shape retrieval & similarity
│   └── ml_pipeline.py              # ML classification & clustering
├── examples/                       # Demo scripts
├── tests/                          # Unit tests
└── requirements.txt
```

---

## Key Concepts & Skills Learned

### Computer Vision & 3D Geometry
- **Mesh processing**: Loading, validating, and analyzing 3D mesh files
- **Geometric features**: Computing curvature, surface normals, shape moments
- **Camera calibration**: Intrinsic/extrinsic calibration, lens distortion correction
- **3D reconstruction**: Stereo depth estimation, point cloud processing, multi-view reconstruction

### Feature Engineering
- **Geometric descriptors**: Statistical shape analysis, local neighborhood features
- **Learned features**: PCA-based descriptors, spectral features, histogram-based representations
- **Feature learning**: Training on shape datasets, dimensionality reduction, normalization

### Machine Learning
- **Classification**: Training supervised models (SVM, Random Forest) on shape features
- **Clustering**: Unsupervised grouping (K-means, hierarchical clustering)
- **Retrieval**: Efficient shape matching using multiple similarity metrics
- **Evaluation**: Cross-validation, performance metrics, feature importance

### Software Engineering
- **Object-oriented design**: Clean module architecture with single responsibilities
- **Type hints & documentation**: Type annotations and comprehensive docstrings
- **Testing**: Unit tests for robust functionality
- **Data processing pipelines**: Standardized workflows for feature extraction and ML

---

## Core Modules

| Module | Purpose |
|--------|---------|
| `mesh_utils.py` | Load/export meshes, compute statistics, normalize geometry |
| `geometric_features.py` | Extract curvature, moments, normals, local descriptors |
| `learned_features.py` | Compute histogram, PCA, spectral, and multi-scale features |
| `camera_calibration.py` | Calibrate cameras, undistort images, estimate pose |
| `reconstruction.py` | Stereo depth estimation, point cloud generation |
| `matching.py` | Shape retrieval, similarity metrics, ranking |
| `ml_pipeline.py` | Classification, clustering, dimensionality reduction |

---

## Camera Calibration & Reconstruction

For camera-based 3D reconstruction:

```python
from geovision3d import camera_calibration, reconstruction

calib = camera_calibration.CameraCalibrator()
K, dist_coeffs = calib.calibrate_from_directory("calibration_images/")

reconstructor = reconstruction.MultiViewReconstructor(K, dist_coeffs)
points_3d, normals = reconstructor.reconstruct_from_sequence("video_frames/")
```

---

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
