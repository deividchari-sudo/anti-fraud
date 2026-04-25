import pytest
import sys
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def setup_test_environment():
    """Setup test environment."""
    # Create necessary directories
    Path("models").mkdir(exist_ok=True)
    Path("tests").mkdir(exist_ok=True)
    
    yield
    
    # Cleanup if needed
    pass


def pytest_configure(config):
    """Configure pytest."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
