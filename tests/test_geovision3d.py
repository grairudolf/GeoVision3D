"""
Comprehensive unit tests for GeoVision3D package.
"""

import unittest
import numpy as np
import tempfile
from pathlib import Path
import trimesh

from geovision3d import (
    mesh_utils, geometric_features, learned_features,
    camera_calibration, reconstruction, matching, ml_pipeline
)


class TestMeshUtils(unittest.TestCase):
    """Test mesh loading and processing."""
    
    def setUp(self):
        """Create test mesh."""
        self.mesh = trimesh.creation.icosphere(subdivisions=2)
        self.temp_dir = tempfile.TemporaryDirectory()
    
    def tearDown(self):
        """Clean up temporary files."""
        self.temp_dir.cleanup()
    
    def test_mesh_export_import(self):
        """Test mesh export and import."""
        filepath = Path(self.temp_dir.name) / "test.obj"
        mesh_utils.export_mesh(self.mesh, filepath)
        loaded = mesh_utils.load_mesh(filepath)
        
        self.assertEqual(loaded.vertices.shape[0], self.mesh.vertices.shape[0])
        self.assertEqual(loaded.faces.shape[0], self.mesh.faces.shape[0])
    
    def test_mesh_normalization(self):
        """Test mesh normalization."""
        normalized = mesh_utils.normalize_mesh(self.mesh, target_scale=1.0)
        
        # Check normalization
        size = np.linalg.norm(normalized.bounds[1] - normalized.bounds[0])
        self.assertAlmostEqual(size, 1.0, places=1)
    
    def test_mesh_validation(self):
        """Test mesh validation."""
        is_valid, message = mesh_utils.MeshProcessor.validate_mesh(self.mesh)
        self.assertTrue(is_valid)
    
    def test_get_mesh_stats(self):
        """Test mesh statistics extraction."""
        stats = mesh_utils.get_mesh_stats(self.mesh)
        
        self.assertIn('basic', stats)
        self.assertIn('descriptors', stats)
        self.assertGreater(stats['basic']['volume'], 0)
        self.assertGreater(stats['basic']['surface_area'], 0)


class TestGeometricFeatures(unittest.TestCase):
    """Test geometric feature extraction."""
    
    def setUp(self):
        """Create test mesh."""
        self.mesh = trimesh.creation.icosphere(subdivisions=2)
    
    def test_curvature_computation(self):
        """Test curvature estimation."""
        curvature = geometric_features.GeometricFeatures.compute_mean_curvature(self.mesh)
        
        self.assertEqual(curvature.shape[0], self.mesh.vertices.shape[0])
        self.assertTrue(np.all(np.isfinite(curvature)))
    
    def test_normals(self):
        """Test normal computation."""
        normals = geometric_features.GeometricFeatures.compute_vertex_normals(self.mesh)
        
        self.assertEqual(normals.shape, self.mesh.vertices.shape)
        # Check normals are unit vectors
        norms = np.linalg.norm(normals, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(len(norms)), decimal=5)
    
    def test_shape_moments(self):
        """Test shape moment extraction."""
        moments = geometric_features.extract_shape_moments(self.mesh)
        
        self.assertIn('eigenvalue_0', moments)
        self.assertIn('moment_0', moments)
        self.assertTrue(all(np.isfinite(v) for v in moments.values()))


class TestLearnedFeatures(unittest.TestCase):
    """Test learned feature extraction."""
    
    def setUp(self):
        """Create test shapes."""
        self.shapes = [
            trimesh.creation.icosphere(subdivisions=1),
            trimesh.creation.box(),
            trimesh.creation.cylinder(),
        ]
    
    def test_histogram_features(self):
        """Test histogram features."""
        feat = learned_features.ShapeDescriptor.compute_histogram_features(self.shapes[0])
        
        self.assertEqual(feat.shape[0], 60)  # 3 histograms of 20 bins each
        self.assertTrue(np.all(feat >= 0))
        self.assertTrue(np.allclose(feat.sum(), 3.0, atol=0.1))
    
    def test_pca_descriptors(self):
        """Test PCA descriptors."""
        feat = learned_features.ShapeDescriptor.compute_pca_descriptors(self.shapes[0])
        
        self.assertEqual(feat.shape[0], 10)  # n_components default
        self.assertTrue(np.all(np.isfinite(feat)))
    
    def test_feature_extractor_fit(self):
        """Test feature extractor fitting."""
        extractor = learned_features.LearnedFeatureExtractor(feature_dim=16)
        extractor.fit(self.shapes)
        
        self.assertTrue(extractor.fitted)
        
        # Test extraction
        feat = extractor.extract(self.shapes[0])
        self.assertEqual(feat.shape[0], 16)
    
    def test_batch_extraction(self):
        """Test batch feature extraction."""
        extractor = learned_features.LearnedFeatureExtractor()
        extractor.fit(self.shapes)
        
        batch_feat = extractor.extract_batch(self.shapes)
        self.assertEqual(batch_feat.shape[0], len(self.shapes))


class TestCameraCalibration(unittest.TestCase):
    """Test camera calibration."""
    
    def test_camera_intrinsics(self):
        """Test intrinsics data structure."""
        intrinsics = camera_calibration.CameraIntrinsics(
            fx=500, fy=500, cx=320, cy=240,
            k1=0.1, k2=-0.05, p1=0.001, p2=0.002
        )
        
        K = intrinsics.to_camera_matrix()
        self.assertEqual(K.shape, (3, 3))
        self.assertEqual(K[0, 0], 500)
        self.assertEqual(K[1, 1], 500)
    
    def test_intrinsics_from_matrix(self):
        """Test intrinsics creation from matrix."""
        K = np.array([
            [500, 0, 320],
            [0, 500, 240],
            [0, 0, 1]
        ], dtype=np.float64)
        
        dist_coeffs = np.array([0.1, -0.05, 0.001, 0.002])
        
        intrinsics = camera_calibration.CameraIntrinsics.from_camera_matrix(K, dist_coeffs)
        
        self.assertEqual(intrinsics.fx, 500)
        self.assertEqual(intrinsics.k1, 0.1)


class TestReconstruction(unittest.TestCase):
    """Test 3D reconstruction."""
    
    def setUp(self):
        """Set up synthetic data."""
        self.K = np.array([
            [500, 0, 320],
            [0, 500, 240],
            [0, 0, 1]
        ], dtype=np.float32)
        
        # Create synthetic depth map
        h, w = 480, 640
        self.depth_map = np.ones((h, w), dtype=np.float32) * 2.0
    
    def test_depth_to_point_cloud(self):
        """Test depth to point cloud conversion."""
        processor = reconstruction.PointCloudProcessor()
        points, colors = processor.depth_to_point_cloud(self.depth_map, self.K)
        
        self.assertGreater(points.shape[0], 0)
        self.assertEqual(colors.shape[1], 3)
    
    def test_point_cloud_downsampling(self):
        """Test point cloud downsampling."""
        processor = reconstruction.PointCloudProcessor()
        points = np.random.randn(1000, 3)
        
        downsampled = processor.downsample_point_cloud(points, voxel_size=0.1)
        
        self.assertLess(downsampled.shape[0], points.shape[0])


class TestMatching(unittest.TestCase):
    """Test shape matching and retrieval."""
    
    def setUp(self):
        """Create test features."""
        np.random.seed(42)
        self.database_features = np.random.randn(10, 32)
        self.query_features = np.random.randn(32)
    
    def test_euclidean_distance(self):
        """Test Euclidean distance."""
        f1 = np.array([0, 0, 0], dtype=np.float32)
        f2 = np.array([3, 4, 0], dtype=np.float32)
        
        dist = SimilarityMetrics.euclidean_distance(f1, f2)
        self.assertAlmostEqual(dist, 5.0, places=5)
    
    def test_retriever_basic(self):
        """Test shape retriever."""
        retriever = matching.ShapeRetriever(metric='euclidean')
        retriever.add_to_database(self.database_features)
        
        indices, scores = retriever.retrieve(self.query_features, k=3)
        
        self.assertEqual(len(indices), 3)
        self.assertEqual(len(scores), 3)
        self.assertTrue(all(0 <= idx < 10 for idx in indices))
    
    def test_similarity_matrix(self):
        """Test similarity matrix computation."""
        sim_matrix = matching.compute_similarity_matrix(self.database_features)
        
        self.assertEqual(sim_matrix.shape, (10, 10))
        # Diagonal should be 1 (maximum similarity)
        np.testing.assert_array_almost_equal(np.diag(sim_matrix), np.ones(10), decimal=5)


class TestMLPipeline(unittest.TestCase):
    """Test ML pipeline components."""
    
    def setUp(self):
        """Create test data."""
        np.random.seed(42)
        self.features = np.random.randn(30, 16)
        self.labels = np.repeat([0, 1, 2], 10)
    
    def test_normalizer(self):
        """Test feature normalization."""
        normalizer = ml_pipeline.FeatureNormalizer()
        normalized = normalizer.fit_transform(self.features)
        
        # Check zero mean and unit variance
        np.testing.assert_array_almost_equal(normalized.mean(axis=0), np.zeros(16), decimal=5)
        np.testing.assert_array_almost_equal(normalized.std(axis=0), np.ones(16), decimal=1)
    
    def test_dimensionality_reducer(self):
        """Test PCA dimensionality reduction."""
        reducer = ml_pipeline.DimensionalityReducer(n_components=8)
        reducer.fit(self.features)
        
        reduced = reducer.transform(self.features)
        self.assertEqual(reduced.shape[1], 8)
        self.assertGreater(reducer.get_explained_variance(), 0.5)
    
    def test_classifier(self):
        """Test shape classifier."""
        classifier = ml_pipeline.ShapeClassifier(classifier_type='svm')
        classifier.fit(self.features, self.labels)
        
        predictions = classifier.predict(self.features)
        self.assertEqual(predictions.shape[0], self.features.shape[0])
        
        metrics = classifier.evaluate(self.features, self.labels)
        self.assertIn('accuracy', metrics)
        self.assertGreater(metrics['accuracy'], 0)
    
    def test_clusterer(self):
        """Test shape clusterer."""
        clusterer = ml_pipeline.ShapeClusterer(n_clusters=3)
        clusterer.fit(self.features)
        
        predictions = clusterer.predict(self.features)
        self.assertEqual(predictions.shape[0], self.features.shape[0])
        
        metrics = clusterer.evaluate(self.features)
        self.assertIn('silhouette_score', metrics)


class SimilarityMetrics:
    """Helper for similarity metric tests."""
    
    @staticmethod
    def euclidean_distance(f1, f2):
        return float(np.linalg.norm(f1 - f2))


if __name__ == '__main__':
    unittest.main()
