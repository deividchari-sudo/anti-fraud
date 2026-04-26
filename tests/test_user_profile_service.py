"""
Tests for UserProfileService.
"""

import pytest
from datetime import datetime
from pathlib import Path
import tempfile
import shutil

from src.user_profile.service import UserProfileService
from src.repositories import JSONUserProfileRepository


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
def sample_transactions():
    """Sample transactions for testing."""
    return [
        {
            "id": "1",
            "timestamp": "2026-04-25T10:00:00Z",
            "canal": "app",
            "produto": "pix",
            "valor": 100.0,
            "sender": {"cpfSender": "12345678901"},
            "receiver": {"cpfReceiver": "98765432100", "banco": 152}
        },
        {
            "id": "2",
            "timestamp": "2026-04-25T11:00:00Z",
            "canal": "app",
            "produto": "pix",
            "valor": 150.0,
            "sender": {"cpfSender": "12345678901"},
            "receiver": {"cpfReceiver": "98765432100", "banco": 152}
        },
        {
            "id": "3",
            "timestamp": "2026-04-25T12:00:00Z",
            "canal": "web",
            "produto": "ted",
            "valor": 200.0,
            "sender": {"cpfSender": "12345678901"},
            "receiver": {"cpfReceiver": "45678901234", "banco": 1}
        }
    ]


class TestUserProfileService:
    """Test UserProfileService."""
    
    def test_calculate_profile_basic(self, temp_repo, sample_transactions):
        """Test basic profile calculation."""
        service = UserProfileService(temp_repo)
        
        profile = service.calculate_profile(sample_transactions, "12345678901")
        
        assert profile.cpf == "12345678901"
        assert profile.transaction_count == 3
        assert profile.statistics.valor.mean == 150.0
        assert profile.statistics.valor.min == 100.0
        assert profile.statistics.valor.max == 200.0
        # With only 3 transactions (< 30), it should be cold start
        assert profile.is_cold_start
    
    def test_calculate_profile_with_insufficient_transactions(self, temp_repo):
        """Test profile calculation with insufficient transactions (cold start)."""
        service = UserProfileService(temp_repo)
        
        transactions = [
            {
                "id": "1",
                "timestamp": "2026-04-25T10:00:00Z",
                "canal": "app",
                "produto": "pix",
                "valor": 100.0,
                "sender": {"cpfSender": "12345678901"},
                "receiver": {"cpfReceiver": "98765432100", "banco": 152}
            }
        ]
        
        profile = service.calculate_profile(transactions, "12345678901")
        
        assert profile.transaction_count == 1
        assert profile.is_cold_start  # Should be cold start with < 30 transactions
    
    def test_get_or_create_profile_existing(self, temp_repo, sample_transactions):
        """Test getting existing profile."""
        service = UserProfileService(temp_repo)
        
        # Create and save profile
        profile = service.calculate_profile(sample_transactions, "12345678901")
        temp_repo.save_profile("12345678901", profile)
        
        # Get profile
        retrieved_profile = service.get_or_create_profile("12345678901")
        
        assert retrieved_profile.cpf == "12345678901"
        assert retrieved_profile.transaction_count == 3
    
    def test_get_or_create_profile_cold_start(self, temp_repo):
        """Test creating profile for cold start (global profile fallback)."""
        service = UserProfileService(temp_repo)
        
        # No profile exists, should use global profile or minimal profile
        profile = service.get_or_create_profile("99999999999")
        
        assert profile.cpf == "99999999999"
        assert profile.is_cold_start
        assert profile.transaction_count == 0
    
    def test_update_profile(self, temp_repo, sample_transactions):
        """Test updating profile with new transaction."""
        service = UserProfileService(temp_repo)
        
        # Create initial profile
        profile = service.calculate_profile(sample_transactions, "12345678901")
        temp_repo.save_profile("12345678901", profile)
        
        # Update with new transaction
        new_transaction = {
            "id": "4",
            "timestamp": "2026-04-25T13:00:00Z",
            "canal": "app",
            "produto": "pix",
            "valor": 300.0,
            "sender": {"cpfSender": "12345678901"},
            "receiver": {"cpfReceiver": "98765432100", "banco": 152}
        }
        
        service.update_profile("12345678901", new_transaction)
        
        # Verify update
        updated_profile = temp_repo.load_profile("12345678901")
        assert updated_profile.transaction_count == 4
    
    def test_analyze_transaction(self, temp_repo, sample_transactions):
        """Test analyzing transaction with behavioral profiling."""
        service = UserProfileService(temp_repo)
        
        # Create profile first
        profile = service.calculate_profile(sample_transactions, "12345678901")
        temp_repo.save_profile("12345678901", profile)
        
        # Analyze normal transaction
        normal_transaction = sample_transactions[0]
        analysis = service.analyze_transaction(normal_transaction)
        
        # With only 3 transactions (< 30), it should be cold start
        assert analysis.has_profile == False
        assert analysis.transaction_count == 3
        assert analysis.is_cold_start == True
    
    def test_analyze_transaction_cold_start(self, temp_repo):
        """Test analyzing transaction for cold start user."""
        service = UserProfileService(temp_repo)
        
        transaction = {
            "id": "1",
            "timestamp": "2026-04-25T10:00:00Z",
            "canal": "app",
            "produto": "pix",
            "valor": 100.0,
            "sender": {"cpfSender": "99999999999"},
            "receiver": {"cpfReceiver": "98765432100", "banco": 152}
        }
        
        analysis = service.analyze_transaction(transaction)
        
        assert analysis.has_profile == False
        assert analysis.is_cold_start == True
