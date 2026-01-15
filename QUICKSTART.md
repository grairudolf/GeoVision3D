# GeoVision3D - Quick Start Guide

## Installation

### 1. Prerequisites
- Python 3.8 or higher
- pip package manager

### 2. Setup

```bash
# Navigate to project directory
cd c:\Users\Rudolf\Desktop\projects\GeoVision3D

# Create virtual environment (optional but recommended)
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 5-Minute Quick Start

### Example 1: Load and Analyze a Mesh

```python
from geovision3d import mesh_utils, geometric_features
import trimesh

# Create a sample mesh
mesh = trimesh.creation.icosphere(subdivisions=2)

# Get mesh statistics
stats = mesh_utils.get_mesh_stats(mesh)
print(f"Volume: {stats['basic']['volume']:.4f}")
print(f"Surface Area: {stats['basic']['surface_area']:.4f}")
print(f"Sphericity: {stats['descriptors']['sphericity']:.4f}")

# Extract geometric features
mean_curv = geometric_features.GeometricFeatures.compute_mean_curvature(mesh)
print(f"Mean Curvature - Min: {mean_curv.min():.4f}, Max: {mean_curv.max():.4f}")
```

### Example 2: Shape Classification

```python
from geovision3d import learned_features, ml_pipeline
import trimesh
import numpy as np

# Create training shapes
sphere = trimesh.creation.icosphere(subdivisions=2)
cube = trimesh.creation.box()
cylinder = trimesh.creation.cylinder()

training_meshes = [sphere, cube, cylinder]
training_labels = np.array([0, 1, 2])

# Extract features
extractor = learned_features.LearnedFeatureExtractor()
extractor.fit(training_meshes)

features = extractor.extract_batch(training_meshes)

# Train classifier
pipeline = ml_pipeline.MLPipeline(normalize=True, reduce_dims=True)
pipeline.train_classifier(features, training_labels, classifier_type='svm')

# Classify new shape
test_sphere = trimesh.creation.icosphere()
test_features = extractor.extract(test_sphere)
predicted_class = pipeline.classify(test_features.reshape(1, -1))[0]
print(f"Predicted class: {predicted_class}")
```

### Example 3: Shape Retrieval

```python
from geovision3d import learned_features, matching
import trimesh

# Create shape database
shapes = {
    'sphere': trimesh.creation.icosphere(),
    'cube': trimesh.creation.box(),
    'cylinder': trimesh.creation.cylinder(),
}

# Extract features
extractor = learned_features.LearnedFeatureExtractor()
extractor.fit(list(shapes.values()))

features = extractor.extract_batch(list(shapes.values()))
shape_names = list(shapes.keys())

# Create retriever
retriever = matching.ShapeRetriever(metric='euclidean')
retriever.add_to_database(features, shape_names)

# Query
query_shape = trimesh.creation.icosphere()
query_features = extractor.extract(query_shape)

indices, scores = retriever.retrieve(query_features, k=2)
print("Similar shapes:")
for idx, score in zip(indices, scores):
    print(f"  {shape_names[idx]}: {score:.4f}")
```

### Example 4: Camera Calibration

```python
from geovision3d import camera_calibration
import cv2
import numpy as np

# Initialize calibrator
calibrator = camera_calibration.CameraCalibrator(
    checkerboard_size=(9, 6),
    square_size=0.025
)

# Calibrate from image directory (requires checkerboard images)
# K, dist_coeffs = calibrator.calibrate_from_directory("calibration_images/")

# Or use pre-computed parameters
K = np.array([
    [500, 0, 320],
    [0, 500, 240],
    [0, 0, 1]
], dtype=np.float32)

dist_coeffs = np.zeros(4)

# Undistort an image
# undistorted = calibrator.undistort_image(image, K, dist_coeffs)

print("Camera matrix:")
print(K)
```

### Example 5: 3D Reconstruction

```python
from geovision3d import reconstruction
import numpy as np

# Create synthetic depth map
h, w = 480, 640
depth_map = np.ones((h, w), dtype=np.float32) * 2.0

# Camera matrix
K = np.array([
    [500, 0, 320],
    [0, 500, 240],
    [0, 0, 1]
], dtype=np.float32)

# Convert to point cloud
processor = reconstruction.PointCloudProcessor()
points_3d, colors = processor.depth_to_point_cloud(depth_map, K)

# Filter and downsample
filtered = processor.filter_outliers(points_3d)
downsampled = processor.downsample_point_cloud(filtered, voxel_size=0.05)

print(f"Generated {points_3d.shape[0]} points")
print(f"After filtering: {filtered.shape[0]} points")
print(f"After downsampling: {downsampled.shape[0]} points")
```

## Running Examples

The package includes comprehensive examples:

```bash
# Mesh analysis examples
python examples/example_mesh_analysis.py

# Camera and reconstruction examples
python examples/example_camera_reconstruction.py

# Shape retrieval and ML pipeline examples
python examples/example_shape_retrieval.py
```

## Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_geovision3d.py -v

# Run with coverage report
python -m pytest tests/ --cov=geovision3d --cov-report=html
```

## Project Structure

```
GeoVision3D/
├── geovision3d/                 # Main package
│   ├── __init__.py
│   ├── mesh_utils.py            # Mesh loading and processing
│   ├── camera_calibration.py     # Camera calibration
│   ├── reconstruction.py         # 3D reconstruction
│   ├── geometric_features.py     # Geometric feature extraction
│   ├── learned_features.py       # Learned shape descriptors
│   ├── matching.py               # Shape retrieval and matching
│   └── ml_pipeline.py            # ML pipeline components
├── examples/                     # Example scripts
│   ├── example_mesh_analysis.py
│   ├── example_camera_reconstruction.py
│   └── example_shape_retrieval.py
├── tests/                        # Unit tests
│   └── test_geovision3d.py
├── data/                         # Data directory
├── README.md                     # Project overview
├── API_REFERENCE.md              # Detailed API documentation
├── CONTRIBUTING.md               # Contributing guidelines
├── requirements.txt              # Python dependencies
└── QUICKSTART.md                # This file
```

## Key Features

### Mesh Processing
- Load OBJ, STL, PLY, GLTF files
- Mesh validation and normalization
- Statistical analysis (volume, surface area, etc.)
- Shape descriptors (sphericity, compactness, etc.)

### Geometric Features
- Mean and Gaussian curvature estimation
- Shape moment invariants
- Local geometric descriptors
- Vertex normal computation

### Computer Vision
- Camera calibration from checkerboard images
- Lens distortion correction
- Stereo depth estimation
- Point cloud processing and filtering
- Multi-view reconstruction

### Learned Features
- Histogram-based descriptors
- PCA-based shape characterization
- Spectral features (Laplacian eigenvalues)
- Multi-scale shape analysis

### Shape Matching & Retrieval
- Multiple similarity metrics (Euclidean, cosine, etc.)
- Efficient nearest-neighbor search
- Hierarchical retrieval for large databases
- Quantitative evaluation metrics

### Machine Learning
- Feature normalization and standardization
- Dimensionality reduction (PCA)
- Classification (SVM, Random Forest)
- Unsupervised clustering (K-means)
- Comprehensive evaluation metrics

## Common Workflows

### 1. Mesh Analysis Pipeline
```python
mesh = load_mesh("model.obj")
mesh = normalize_mesh(mesh)
features = extract_all_features(mesh)
stats = get_mesh_stats(mesh)
```

### 2. Shape Classification
```python
features = extract_features(training_meshes)
pipeline.train_classifier(features, labels)
predicted_class = pipeline.classify(test_features)
```

### 3. Shape Retrieval
```python
db_features = extractor.extract_batch(database_meshes)
retriever.add_to_database(db_features, names)
results = retriever.retrieve(query_features, k=10)
```

### 4. Clustering
```python
pipeline.train_clusterer(features, n_clusters=5)
clusters = pipeline.cluster(new_features)
metrics = pipeline.evaluate_clustering(features)
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'geovision3d'"
**Solution**: Install package in development mode:
```bash
pip install -e .
```

### Issue: "Cannot load mesh" error
**Solution**: Check file path and supported format (OBJ, STL, PLY, GLTF)

### Issue: Low classification accuracy
**Solution**: 
- Ensure sufficient training data
- Check feature normalization
- Increase model complexity or training iterations

### Issue: Memory error with large point clouds
**Solution**: Use point cloud downsampling:
```python
downsampled = processor.downsample_point_cloud(points, voxel_size=0.05)
```

## Next Steps

1. **Read the documentation**: See [API_REFERENCE.md](API_REFERENCE.md) for detailed API
2. **Explore examples**: Run the example scripts in `examples/`
3. **Run tests**: Execute `pytest tests/` to verify installation
4. **Read contributing guide**: See [CONTRIBUTING.md](CONTRIBUTING.md) for development

## Support

- Check [README.md](README.md) for project overview
- Review [API_REFERENCE.md](API_REFERENCE.md) for detailed documentation
- Look at [examples/](examples/) for usage patterns
- Run tests for validation: `pytest tests/ -v`

## License

MIT License - See project repository for details

---

**Happy analyzing! 🚀**
