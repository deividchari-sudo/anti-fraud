"""
Tests for JSONUserProfileRepository.
"""

import pytest
from datetime import datetime
from pathlib import Path
import tempfile
import shutil
import json

from src.repositories import JSONUserProfileRepository
from src.user_profile.models import (
    UserProfile,
    Statistics,
    ValueStatistics,
    HourStatistics,
    FrequencyStatistics,
    Destinations,
    Canais,
    Produtos,
    CanalInfo,
    ProdutoInfo
)


@pytest.fixture
def temp_repo():
    """Create a temporary repository for testing."""
    temp_dir = tempfile.mkdtemp()
    profiles_file = Path(temp_dir) / "user_profiles.json"
    global_profile_file = Path(temp_dir) / "global_profile.json"
    
    # Create repository
    repo = JSONUserProfileRepository(str(profiles_file), str(global_profile_file))
    
    yield repo
    
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_profile():
    """Create a sample user profile for testing."""
    return UserProfile(
        cpf="12345678901",
        created_at=datetime.utcnow(),
        last_updated=datetime.utcnow(),
        transaction_count=45,
        is_cold_start=False,
        statistics=Statistics(
            valor=ValueStatistics(
                mean=500.0,
                std=150.0,
                median=450.0,
                p25=350.0,
                p75=650.0,
                p95=850.0,
                min=100.0,
                max=1000.0
            ),
            hora=HourStatistics(
                mean=12.0,
                std=2.0,
                median=12.0,
                p25=10.0,
                p75=14.0
            ),
            frequencia=FrequencyStatistics(
                transactions_per_day_mean=2.5,
                transactions_per_day_std=1.0,
                days_active=30
            )
        ),
        destinations=Destinations(),
        canais=Canais(
            app=CanalInfo(count=36, percentage=0.8),
            web=CanalInfo(count=7, percentage=0.15),
            api=CanalInfo(count=2, percentage=0.05)
        ),
        produtos=Produtos(
            pix=ProdutoInfo(count=30, percentage=0.67),
            ted=ProdutoInfo(count=10, percentage=0.22),
            boleto=ProdutoInfo(count=5, percentage=0.11)
        )
    )


class TestJSONUserProfileRepository:
    """Test JSONUserProfileRepository."""
    
    def test_initialization_creates_files(self, temp_repo):
        """Test that repository initialization creates JSON files."""
        assert temp_repo.profiles_file.exists()
        assert temp_repo.global_profile_file.exists()
    
    def test_save_and_load_profile(self, temp_repo, sample_profile):
        """Test saving and loading a profile."""
        # Save profile
        temp_repo.save_profile("12345678901", sample_profile)
        
        # Load profile
        loaded_profile = temp_repo.load_profile("12345678901")
        
        assert loaded_profile is not None
        assert loaded_profile.cpf == "12345678901"
        assert loaded_profile.transaction_count == 45
        assert loaded_profile.statistics.valor.mean == 500.0
    
    def test_load_nonexistent_profile(self, temp_repo):
        """Test loading a profile that doesn't exist."""
        profile = temp_repo.load_profile("99999999999")
        assert profile is None
    
    def test_update_existing_profile(self, temp_repo, sample_profile):
        """Test updating an existing profile."""
        # Save initial profile
        temp_repo.save_profile("12345678901", sample_profile)
        
        # Update profile
        sample_profile.transaction_count = 50
        sample_profile.last_updated = datetime.utcnow()
        temp_repo.save_profile("12345678901", sample_profile.model_copy())
        
        # Load updated profile
        loaded_profile = temp_repo.load_profile("12345678901")
        
        assert loaded_profile.transaction_count == 50
    
    def test_delete_profile(self, temp_repo, sample_profile):
        """Test deleting a profile."""
        # Save profile
        temp_repo.save_profile("12345678901", sample_profile)
        
        # Delete profile
        result = temp_repo.delete_profile("12345678901")
        
        assert result == True
        
        # Verify deletion
        loaded_profile = temp_repo.load_profile("12345678901")
        assert loaded_profile is None
    
    def test_delete_nonexistent_profile(self, temp_repo):
        """Test deleting a profile that doesn't exist."""
        result = temp_repo.delete_profile("99999999999")
        assert result == False
    
    def test_list_all_profiles(self, temp_repo, sample_profile):
        """Test listing all profiles."""
        # Save multiple profiles
        temp_repo.save_profile("12345678901", sample_profile)
        
        profile2 = sample_profile.model_copy()
        profile2.cpf = "98765432100"
        temp_repo.save_profile("98765432100", profile2)
        
        # List all profiles
        profiles = temp_repo.list_all_profiles()
        
        assert len(profiles) == 2
        assert "12345678901" in profiles
        assert "98765432100" in profiles
    
    def test_metadata_updated_on_save(self, temp_repo, sample_profile):
        """Test that metadata is updated when saving profiles."""
        # Save profile
        temp_repo.save_profile("12345678901", sample_profile)
        
        # Load JSON and check metadata
        data = temp_repo._load_json(temp_repo.profiles_file)
        
        assert data["metadata"]["total_profiles"] == 1
        assert "last_sync" in data["metadata"]
    
    def test_save_global_profile(self, temp_repo):
        """Test saving and loading global profile."""
        global_profile_data = {
            "global_profile": {
                "statistics": {
                    "valor": {"mean": 450.0, "std": 250.0, "median": 350.0, "p25": 150.0, "p75": 600.0, "p95": 1200.0, "min": 0.0, "max": 5000.0},
                    "hora": {"mean": 13.0, "std": 4.5, "median": 13.0, "p25": 10.0, "p75": 16.0},
                    "frequencia": {"transactions_per_day_mean": 1.8, "transactions_per_day_std": 1.2, "days_active": 100}
                },
                "canais": {
                    "app": {"count": 65000, "percentage": 0.65},
                    "web": {"count": 25000, "percentage": 0.25},
                    "api": {"count": 10000, "percentage": 0.10}
                },
                "produtos": {
                    "pix": {"count": 55000, "percentage": 0.55},
                    "ted": {"count": 25000, "percentage": 0.25},
                    "boleto": {"count": 15000, "percentage": 0.15}
                },
                "destinations": {
                    "common_cpfs": {},
                    "common_bancos": {}
                }
            },
            "metadata": {
                "based_on_transactions": 100000,
                "last_updated": datetime.utcnow().isoformat(),
                "version": "1.0.0"
            }
        }
        
        # Save global profile manually
        temp_repo._save_json(temp_repo.global_profile_file, global_profile_data)
        
        # Load global profile
        loaded_profile = temp_repo.load_global_profile()
        
        assert loaded_profile is not None
        assert loaded_profile.cpf == "GLOBAL"
        assert loaded_profile.statistics.valor.mean == 450.0
    
    def test_load_global_profile_none(self, temp_repo):
        """Test loading global profile when none exists."""
        # Global profile file exists but is null
        loaded_profile = temp_repo.load_global_profile()
        assert loaded_profile is None
    
    def test_json_file_structure(self, temp_repo, sample_profile):
        """Test that JSON file has correct structure."""
        # Save profile
        temp_repo.save_profile("12345678901", sample_profile)
        
        # Load and verify structure
        data = temp_repo._load_json(temp_repo.profiles_file)
        
        assert "profiles" in data
        assert "metadata" in data
        assert isinstance(data["profiles"], dict)
        assert isinstance(data["metadata"], dict)
        assert "total_profiles" in data["metadata"]
