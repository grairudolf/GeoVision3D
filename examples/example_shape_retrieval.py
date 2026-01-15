"""
Example 3: Shape Retrieval, Matching, and ML Pipeline
Demonstrates shape similarity search and machine learning classification/clustering.
"""

import numpy as np
from geovision3d import (
    mesh_utils, geometric_features, learned_features, 
    matching, ml_pipeline
)
import trimesh


def example_shape_retrieval():
    """Demonstrate shape retrieval and ranking."""
    print("=" * 60)
    print("EXAMPLE 1: Shape Retrieval and Similarity Ranking")
    print("=" * 60)
    
    # Create synthetic shape database
    print("Creating synthetic shape database...")
    
    # Generate test shapes
    shapes = {
        'sphere_1': trimesh.creation.icosphere(subdivisions=2),
        'sphere_2': trimesh.creation.icosphere(subdivisions=2),
        'sphere_small': trimesh.creation.icosphere(subdivisions=1),
        'cube_1': trimesh.creation.box(extents=[1, 1, 1]),
        'cube_2': trimesh.creation.box(extents=[1, 1, 1]),
        'cylinder': trimesh.creation.cylinder(radius=0.5, height=1.0),
    }
    
    # Extract features for all shapes
    print("Extracting geometric features...")
    extractor = learned_features.LearnedFeatureExtractor()
    
    database_features = []
    database_names = []
    
    for name, mesh in shapes.items():
        feat = extractor.extract(mesh)
        database_features.append(feat)
        database_names.append(name)
    
    database_features = np.array(database_features)
    print(f"Extracted features: shape {database_features.shape}")
    
    # Create retriever
    retriever = matching.ShapeRetriever(metric='euclidean')
    retriever.add_to_database(database_features, database_names)
    
    # Query: retrieve similar to first sphere
    query_features = database_features[0]  # Query is first sphere
    
    print("\nRetrieving similar shapes to 'sphere_1':")
    print("-" * 50)
    
    indices, scores = retriever.retrieve(query_features, k=5)
    
    for rank, (idx, score) in enumerate(zip(indices, scores), 1):
        name = database_names[idx]
        print(f"{rank}. {name:<20} (similarity: {score:.4f})")
    
    # Demonstrate similarity matrix
    print("\nComputing similarity matrix...")
    similarity_matrix = matching.compute_similarity_matrix(database_features, metric='euclidean')
    
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    print("\nSimilarity between shapes:")
    print("-" * 50)
    for i in range(min(3, len(database_names))):
        for j in range(min(3, len(database_names))):
            sim = similarity_matrix[i, j]
            print(f"{database_names[i]:<15} - {database_names[j]:<15}: {sim:.4f}")


def example_learned_features():
    """Demonstrate learned feature extraction."""
    print("\n" + "=" * 60)
    print("EXAMPLE 2: Learned Feature Extraction")
    print("=" * 60)
    
    # Create shapes
    shapes = [
        trimesh.creation.icosphere(subdivisions=2),
        trimesh.creation.box(extents=[1, 1, 1]),
        trimesh.creation.cylinder(radius=0.5, height=1.0),
    ]
    
    shape_names = ['sphere', 'cube', 'cylinder']
    
    # Create and fit extractor
    extractor = learned_features.LearnedFeatureExtractor(feature_dim=32)
    
    print("Training feature extractor on shape database...")
    extractor.fit(shapes)
    
    # Extract features
    print("\nExtracted learned features:")
    print("-" * 50)
    
    for name, mesh in zip(shape_names, shapes):
        feat = extractor.extract(mesh)
        print(f"{name:<15}: feature dimension {feat.shape[0]}, "
              f"mean={feat.mean():.4f}, std={feat.std():.4f}")
    
    # Extract all descriptor types
    print("\nAvailable descriptor types:")
    print("-" * 50)
    
    test_mesh = shapes[0]
    descriptors = learned_features.extract_all_descriptors(test_mesh)
    
    for desc_type, desc_vec in descriptors.items():
        print(f"{desc_type:<15}: {desc_vec.shape[0]} features")


def example_classification():
    """Demonstrate shape classification."""
    print("\n" + "=" * 60)
    print("EXAMPLE 3: Shape Classification with ML Pipeline")
    print("=" * 60)
    
    # Create training data
    print("Creating training dataset...")
    
    np.random.seed(42)
    n_samples_per_class = 10
    
    training_features = []
    training_labels = []
    
    for class_idx in range(3):
        # Generate random features for each class
        class_features = np.random.randn(n_samples_per_class, 32) + class_idx * 2
        training_features.append(class_features)
        training_labels.extend([class_idx] * n_samples_per_class)
    
    training_features = np.vstack(training_features)
    training_labels = np.array(training_labels)
    
    print(f"Training set: {training_features.shape[0]} samples, "
          f"{training_features.shape[1]} features, "
          f"{len(np.unique(training_labels))} classes")
    
    # Create and train pipeline
    print("\nTraining classification pipeline...")
    
    pipeline = ml_pipeline.MLPipeline(normalize=True, reduce_dims=True, n_components=16)
    pipeline.train_classifier(training_features, training_labels, classifier_type='svm')
    
    print("Pipeline trained successfully")
    
    # Evaluate on training set
    print("\nEvaluation on training set:")
    print("-" * 50)
    
    metrics = pipeline.evaluate_classification(training_features, training_labels)
    
    for metric_name, value in metrics.items():
        print(f"{metric_name:<15}: {value:.4f}")
    
    # Predict on new samples
    print("\nPredicting on new samples...")
    
    test_features = np.random.randn(5, 32)
    predictions = pipeline.classify(test_features)
    
    print(f"Predictions: {predictions}")


def example_clustering():
    """Demonstrate unsupervised clustering."""
    print("\n" + "=" * 60)
    print("EXAMPLE 4: Shape Clustering")
    print("=" * 60)
    
    # Create data with natural clusters
    print("Creating clusterable dataset...")
    
    np.random.seed(42)
    
    # 3 clusters with distinct centers
    cluster_centers = np.array([
        [0, 0],
        [5, 5],
        [-5, 5]
    ])
    
    features = []
    true_labels = []
    
    for cluster_idx, center in enumerate(cluster_centers):
        cluster_features = np.random.randn(15, 2) * 0.5 + center
        features.append(cluster_features)
        true_labels.extend([cluster_idx] * 15)
    
    features = np.vstack(features)
    true_labels = np.array(true_labels)
    
    print(f"Dataset: {features.shape[0]} samples, {features.shape[1]} features, "
          f"{len(cluster_centers)} true clusters")
    
    # Train clusterer
    print("\nTraining clustering model...")
    
    pipeline = ml_pipeline.MLPipeline(normalize=True, reduce_dims=False)
    pipeline.train_clusterer(features, n_clusters=3)
    
    # Evaluate clustering
    print("\nClustering evaluation:")
    print("-" * 50)
    
    metrics = pipeline.evaluate_clustering(features)
    
    for metric_name, value in metrics.items():
        print(f"{metric_name:<20}: {value:.4f}")
    
    # Predict clusters
    print("\nPredicted cluster assignments (first 10 samples):")
    print("-" * 50)
    
    clusters = pipeline.cluster(features)
    for i in range(min(10, len(clusters))):
        true_cluster = true_labels[i]
        pred_cluster = clusters[i]
        match = "✓" if true_cluster == pred_cluster else "✗"
        print(f"Sample {i}: predicted={pred_cluster}, true={true_cluster} {match}")


def example_complete_pipeline():
    """Complete pipeline: load, extract, classify, retrieve."""
    print("\n" + "=" * 60)
    print("EXAMPLE 5: Complete End-to-End Pipeline")
    print("=" * 60)
    
    # Create synthetic shape database
    print("Step 1: Creating shape database...")
    
    shapes = {
        'sphere': trimesh.creation.icosphere(subdivisions=2),
        'cube': trimesh.creation.box(),
        'cylinder': trimesh.creation.cylinder(),
        'torus': trimesh.creation.torus(),
        'cone': trimesh.creation.cone(),
    }
    
    # Extract features
    print("\nStep 2: Extracting features...")
    
    extractor = learned_features.LearnedFeatureExtractor()
    extractor.fit(list(shapes.values()))
    
    features = extractor.extract_batch(list(shapes.values()))
    shape_names = list(shapes.keys())
    
    # Train classifier
    print("\nStep 3: Training classifier...")
    
    labels = np.arange(len(shapes))  # Each shape is a class
    pipeline = ml_pipeline.MLPipeline(normalize=True, reduce_dims=True)
    pipeline.train_classifier(features, labels, classifier_type='random_forest')
    
    # Classify new shape
    print("\nStep 4: Classifying new shape...")
    
    new_mesh = trimesh.creation.icosphere(subdivisions=2)
    new_features = extractor.extract(new_mesh)
    predicted_class = pipeline.classify(new_features.reshape(1, -1))[0]
    
    print(f"Classified as: {shape_names[predicted_class]}")
    
    # Retrieve similar shapes
    print("\nStep 5: Retrieving similar shapes...")
    
    retriever = matching.ShapeRetriever(metric='cosine')
    retriever.add_to_database(features, shape_names)
    
    indices, scores = retriever.retrieve(new_features, k=3)
    
    print("Top similar shapes:")
    for rank, (idx, score) in enumerate(zip(indices, scores), 1):
        print(f"  {rank}. {shape_names[idx]:<15} (similarity: {score:.4f})")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("GeoVision3D - Shape Retrieval and ML Pipeline Examples")
    print("=" * 60)
    
    example_shape_retrieval()
    example_learned_features()
    example_classification()
    example_clustering()
    example_complete_pipeline()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
