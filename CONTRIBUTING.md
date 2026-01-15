# Contributing to GeoVision3D

Thank you for interest in contributing to GeoVision3D! This document provides guidelines and instructions for contributing.

## Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please be respectful and professional in all interactions.

## Getting Started

### Setting Up Development Environment

1. Fork and clone the repository
2. Create virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install in development mode:
   ```bash
   pip install -e .
   pip install -r requirements.txt
   ```

4. Run tests to verify setup:
   ```bash
   python -m pytest tests/
   ```

## Contributing Code

### Style Guide

- **Python Version**: Python 3.8+
- **Code Style**: PEP 8 with 100-character line limit
- **Type Hints**: Use type hints for all functions
- **Docstrings**: Use Google-style docstrings

Example:
```python
def compute_feature(mesh: trimesh.Trimesh, n_bins: int = 20) -> np.ndarray:
    """
    Compute histogram features from mesh.
    
    Args:
        mesh: Input mesh
        n_bins: Number of histogram bins
        
    Returns:
        Feature vector of shape (3*n_bins,)
        
    Raises:
        ValueError: If mesh is empty
    """
```

### Code Organization

- **Core modules**: `geovision3d/`
- **Tests**: `tests/`
- **Examples**: `examples/`
- **Documentation**: Root directory and module docstrings

### Making Changes

1. Create feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes with clear commits:
   ```bash
   git commit -m "Add feature: clear description"
   ```

3. Write/update tests:
   ```bash
   python -m pytest tests/test_your_module.py -v
   ```

4. Run all tests:
   ```bash
   python -m pytest tests/ --cov=geovision3d
   ```

5. Update documentation if needed

6. Submit pull request with:
   - Clear description of changes
   - Link to related issues
   - Test results
   - Any breaking changes noted

## Adding Features

### 1. Geometric/CV Features

If adding new geometric features:

1. Add method to `GeometricFeatures` class
2. Include both mathematical description and implementation
3. Test on multiple mesh types (sphere, cube, cylinder, torus)
4. Document parameters and return values
5. Add to `extract_all_features()` if general purpose

Example PR checklist:
- [ ] Implementation with tests
- [ ] Example usage in documentation
- [ ] Comparison with alternative methods
- [ ] Performance benchmark for typical meshes
- [ ] Type hints and docstrings

### 2. ML/Classification Features

If adding classification or clustering:

1. Implement in `ml_pipeline` module
2. Inherit from appropriate base class
3. Support normalization/preprocessing
4. Include evaluation metrics
5. Test on synthetic and real data

Requirements:
- [ ] Training/prediction interface
- [ ] Parameter documentation
- [ ] Evaluation metrics
- [ ] Integration with MLPipeline
- [ ] Example usage script

### 3. Camera/Reconstruction Features

If adding camera or 3D reconstruction features:

1. Implement in `camera_calibration` or `reconstruction`
2. Follow opencv/scipy conventions
3. Handle edge cases and invalid inputs
4. Provide numerical stability checks
5. Test with synthetic and real data

Checklist:
- [ ] Input validation
- [ ] Error handling
- [ ] Numerical tests
- [ ] Performance for typical inputs
- [ ] Integration with existing modules

## Testing

### Test Requirements

All new code must include tests:

```python
class TestNewFeature(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.mesh = trimesh.creation.icosphere()
    
    def test_basic_functionality(self):
        """Test core functionality."""
        result = compute_feature(self.mesh)
        self.assertIsNotNone(result)
        self.assertEqual(result.shape[0], expected_dim)
    
    def test_edge_cases(self):
        """Test edge cases and error handling."""
        with self.assertRaises(ValueError):
            compute_feature(invalid_input)
    
    def test_numerical_properties(self):
        """Test mathematical properties."""
        np.testing.assert_array_almost_equal(
            result.sum(), expected_sum, decimal=5
        )
```

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_geometric_features.py -v

# Run with coverage
python -m pytest tests/ --cov=geovision3d --cov-report=html

# Run specific test class
python -m pytest tests/test_geometric_features.py::TestCurvature -v
```

### Test Coverage Goals

- Aim for >80% code coverage
- 100% coverage for critical paths
- Include both success and failure cases

## Documentation

### Module Documentation

Each module should include:

1. **Docstring** at module level:
   ```python
   """
   Brief description of module.
   
   Detailed description of what this module does, key classes/functions,
   and typical usage patterns.
   """
   ```

2. **Class docstrings**:
   ```python
   class FeatureExtractor:
       """
       Extract features from shapes.
       
       This class provides methods for extracting both geometric and
       learned features from 3D meshes.
       """
   ```

3. **Function docstrings** (Google style):
   ```python
   def compute_curvature(mesh: trimesh.Trimesh) -> np.ndarray:
       """
       Compute mean curvature at mesh vertices.
       
       Uses discrete Laplacian approximation for efficient computation.
       
       Args:
           mesh: Input mesh
           
       Returns:
           Mean curvature values per vertex (N_vertices,)
           
       Raises:
           ValueError: If mesh is empty
           
       Note:
           Curvature values are clipped to [-10, 10] to handle outliers.
       """
   ```

### Documentation Files

- **README.md**: Project overview, features, installation
- **API_REFERENCE.md**: Complete API documentation
- **CONTRIBUTING.md**: This file
- **Examples**: Executable examples in `examples/`

## Performance and Optimization

### Guidelines

1. **Profile before optimizing**:
   ```python
   import cProfile
   cProfile.run('compute_feature(mesh)')
   ```

2. **Vectorize operations** using NumPy:
   ```python
   # Good - vectorized
   distances = np.linalg.norm(points - center, axis=1)
   
   # Avoid - loops
   distances = [np.linalg.norm(p - center) for p in points]
   ```

3. **Use efficient algorithms**:
   - KDTree for spatial queries
   - Sparse matrices for large graphs
   - Batch operations when possible

4. **Document complexity**:
   ```python
   def compute_features(mesh: trimesh.Trimesh) -> np.ndarray:
       """
       Time complexity: O(V + E) where V=vertices, E=edges
       Space complexity: O(V)
       """
   ```

## Reporting Issues

When reporting bugs:

1. **Use issue template** if available
2. **Include**:
   - Python version and OS
   - Package versions
   - Minimal reproducible example
   - Expected vs actual behavior
   - Error traceback (if applicable)

3. **Example**:
   ```
   **Describe the bug**
   Curvature computation returns NaN for certain meshes.
   
   **To Reproduce**
   ```python
   mesh = trimesh.load("problematic_mesh.obj")
   curv = geometric_features.compute_mean_curvature(mesh)
   # Result contains NaN
   ```
   
   **Environment**
   - Python 3.9.2
   - NumPy 1.20.0
   - trimesh 3.9.1
   ```

## Feature Requests

When proposing features:

1. **Check existing issues** first
2. **Describe use case** and motivation
3. **Provide examples** of how it would be used
4. **Discuss implementation** approach
5. **Consider performance** implications

## Pull Request Process

1. **Update documentation** and docstrings
2. **Add/update tests** for changes
3. **Run full test suite**: `pytest tests/ --cov=geovision3d`
4. **Update CHANGELOG** (if exists)
5. **Write clear PR description**:
   - What problem does this solve?
   - How does it work?
   - Any breaking changes?
   - Performance impact?

6. **Address review feedback** promptly

## Code Review Checklist

Reviewers will check:

- [ ] Tests included and passing
- [ ] Docstrings complete and accurate
- [ ] Type hints present
- [ ] PEP 8 compliant
- [ ] No duplicate functionality
- [ ] Performance acceptable
- [ ] Documentation updated
- [ ] No breaking changes (or clearly noted)

## Performance Benchmarking

When adding computationally intensive features:

```python
import time
import numpy as np

def benchmark_feature_extraction():
    mesh = trimesh.creation.icosphere(subdivisions=3)
    
    times = []
    for _ in range(10):
        start = time.perf_counter()
        feature = compute_feature(mesh)
        times.append(time.perf_counter() - start)
    
    print(f"Mean: {np.mean(times):.4f}s")
    print(f"Std:  {np.std(times):.4f}s")
```

## Additional Resources

- **NumPy Guide**: https://numpy.org/devdocs/
- **Trimesh Docs**: https://trimesh.org/
- **scikit-learn Guide**: https://scikit-learn.org/
- **OpenCV Docs**: https://docs.opencv.org/
- **Google Python Style**: https://google.github.io/styleguide/pyguide.html

## Questions?

- Open an issue for discussions
- Check existing documentation
- Review example scripts
- Look at similar implementations

Thank you for contributing!
