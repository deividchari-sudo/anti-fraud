"""
Repository Pattern for data access.
"""

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional
from datetime import datetime
import sqlite3

import pandas as pd


class TransactionRepository(ABC):
    """Abstract repository for transaction data access."""

    @abstractmethod
    def load_dataset(self) -> pd.DataFrame:
        """Load the transaction dataset."""
        pass

    @abstractmethod
    def save_dataset(self, df: pd.DataFrame) -> None:
        """Save the transaction dataset."""
        pass


class CSVTransactionRepository(TransactionRepository):
    """CSV-based implementation of transaction repository."""

    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path

    def load_dataset(self) -> pd.DataFrame:
        """Load the transaction dataset from CSV."""
        return pd.read_csv(self.dataset_path)

    def save_dataset(self, df: pd.DataFrame) -> None:
        """Save the transaction dataset to CSV."""
        df.to_csv(self.dataset_path, index=False)


class ModelRepository(ABC):
    """Abstract repository for model persistence."""

    @abstractmethod
    def load_model(self):
        """Load the trained model."""
        pass

    @abstractmethod
    def save_model(self, model) -> None:
        """Save the trained model."""
        pass

    @abstractmethod
    def load_feature_names(self) -> list:
        """Load feature names."""
        pass

    @abstractmethod
    def save_feature_names(self, feature_names: list) -> None:
        """Save feature names."""
        pass


class JoblibModelRepository(ModelRepository):
    """Joblib-based implementation of model repository."""

    def __init__(self, model_path: str, feature_names_path: str):
        self.model_path = model_path
        self.feature_names_path = feature_names_path

    def load_model(self):
        """Load the trained model from disk."""
        import joblib

        return joblib.load(self.model_path)

    def save_model(self, model) -> None:
        """Save the trained model to disk."""
        import joblib

        Path(self.model_path).parent.mkdir(exist_ok=True)
        joblib.dump(model, self.model_path)

    def load_feature_names(self) -> list:
        """Load feature names from JSON."""
        if Path(self.feature_names_path).exists():
            with open(self.feature_names_path, "r") as f:
                return json.load(f)
        return []

    def save_feature_names(self, feature_names: list) -> None:
        """Save feature names to JSON."""
        Path(self.feature_names_path).parent.mkdir(exist_ok=True)
        with open(self.feature_names_path, "w") as f:
            json.dump(feature_names, f)


class UserProfileRepository(ABC):
    """Abstract repository for user profile data access."""

    @abstractmethod
    def save_profile(self, cpf: str, profile) -> None:
        """Save user profile."""
        pass

    @abstractmethod
    def load_profile(self, cpf: str) -> Optional:
        """Load user profile."""
        pass

    @abstractmethod
    def load_global_profile(self) -> Optional:
        """Load global profile."""
        pass

    @abstractmethod
    def delete_profile(self, cpf: str) -> bool:
        """Delete user profile."""
        pass


class JSONUserProfileRepository(UserProfileRepository):
    """JSON-based implementation of user profile repository with cache."""

    def __init__(self, profiles_file: str = "data/user_profiles.json",
                 global_profile_file: str = "data/global_profile.json"):
        self.profiles_file = Path(profiles_file)
        self.global_profile_file = Path(global_profile_file)
        self._ensure_files_exist()
        # Simple in-memory cache
        self._profiles_cache = {}
        self._global_profile_cache = None
        self._load_cache()

    def _ensure_files_exist(self):
        """Ensure JSON files exist with proper structure."""
        self.profiles_file.parent.mkdir(exist_ok=True)
        
        if not self.profiles_file.exists():
            initial_data = {
                "profiles": {},
                "metadata": {
                    "total_profiles": 0,
                    "last_sync": datetime.utcnow().isoformat(),
                    "version": "1.0.0"
                }
            }
            with open(self.profiles_file, 'w', encoding='utf-8') as f:
                json.dump(initial_data, f, indent=2)

        if not self.global_profile_file.exists():
            initial_data = {
                "global_profile": None,
                "metadata": {
                    "last_updated": datetime.utcnow().isoformat(),
                    "version": "1.0.0"
                }
            }
            with open(self.global_profile_file, 'w', encoding='utf-8') as f:
                json.dump(initial_data, f, indent=2)

    def _load_cache(self):
        """Load profiles into memory cache (lazy loading - don't load all at once)."""
        # Don't load all profiles at startup - use lazy loading instead
        # Cache will be populated on demand
        self._profiles_cache = {}
        
        # Load global profile cache
        try:
            self._global_profile_cache = self.load_global_profile()
        except Exception:
            self._global_profile_cache = None

    def _load_json(self, file_path: Path) -> dict:
        """Load JSON file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_json(self, file_path: Path, data: dict):
        """Save JSON file."""
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)

    def save_profile(self, cpf: str, profile) -> None:
        """Save user profile to JSON."""
        data = self._load_json(self.profiles_file)
        
        # Convert profile to dict if it has dict() method
        if hasattr(profile, 'dict'):
            profile_dict = profile.dict()
        else:
            profile_dict = profile
        
        data["profiles"][cpf] = profile_dict
        data["metadata"]["total_profiles"] = len(data["profiles"])
        data["metadata"]["last_sync"] = datetime.utcnow().isoformat()
        
        self._save_json(self.profiles_file, data)
        
        # Update cache
        self._profiles_cache[cpf] = profile_dict

    def load_profile(self, cpf: str) -> Optional:
        """Load user profile from JSON (uses cache)."""
        # Try cache first
        profile_data = self._profiles_cache.get(cpf)
        
        if profile_data is None:
            return None
        
        # Import here to avoid circular dependency
        try:
            from src.user_profile.models import UserProfile
        except ImportError:
            from user_profile.models import UserProfile
        return UserProfile(**profile_data)

    def load_global_profile(self) -> Optional:
        """Load global profile from JSON (uses cache)."""
        # Return cached version if available
        if self._global_profile_cache is not None:
            return self._global_profile_cache
        
        data = self._load_json(self.global_profile_file)
        global_profile_data = data.get("global_profile")
        
        if global_profile_data is None:
            return None
        
        # Import here to avoid circular dependency
        try:
            from src.user_profile.models import UserProfile, Statistics, ValueStatistics, HourStatistics, FrequencyStatistics, Destinations, Canais, Produtos
        except ImportError:
            from user_profile.models import UserProfile, Statistics, ValueStatistics, HourStatistics, FrequencyStatistics, Destinations, Canais, Produtos
        
        # Reconstruct UserProfile from global profile data
        stats_data = global_profile_data.get("statistics", {})
        canais_data = global_profile_data.get("canais", {})
        produtos_data = global_profile_data.get("produtos", {})
        
        profile = UserProfile(
            cpf="GLOBAL",
            created_at=datetime.utcnow(),
            last_updated=datetime.utcnow(),
            transaction_count=0,
            is_cold_start=False,
            statistics=Statistics(**stats_data),
            destinations=Destinations(),
            canais=Canais(**canais_data),
            produtos=Produtos(**produtos_data)
        )
        
        # Cache the global profile
        self._global_profile_cache = profile
        return profile

    def delete_profile(self, cpf: str) -> bool:
        """Delete user profile from JSON."""
        data = self._load_json(self.profiles_file)
        
        if cpf in data["profiles"]:
            del data["profiles"][cpf]
            data["metadata"]["total_profiles"] = len(data["profiles"])
            data["metadata"]["last_sync"] = datetime.utcnow().isoformat()
            self._save_json(self.profiles_file, data)
            
            # Update cache
            if cpf in self._profiles_cache:
                del self._profiles_cache[cpf]
            
            return True
        
        return False

    def list_all_profiles(self) -> dict:
        """List all profiles."""
        data = self._load_json(self.profiles_file)
        return data["profiles"]


class SQLiteUserProfileRepository(UserProfileRepository):
    """SQLite-based implementation of user profile repository for better performance."""
    
    def __init__(self, db_file: str = "data/user_profiles.db", global_profile_file: str = "data/global_profile.json"):
        self.db_file = Path(db_file)
        self.global_profile_file = Path(global_profile_file)
        self.global_profile_cache = None
        self._ensure_db_exists()
        self._load_global_profile_cache()
    
    def _ensure_db_exists(self):
        """Ensure SQLite database exists with proper schema."""
        self.db_file.parent.mkdir(exist_ok=True)
        
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS profiles (
                cpf TEXT PRIMARY KEY,
                profile_json TEXT NOT NULL,
                created_at TEXT,
                last_updated TEXT,
                transaction_count INTEGER
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_cpf ON profiles(cpf)
        ''')
        
        conn.commit()
        conn.close()
    
    def _load_global_profile_cache(self):
        """Load global profile into cache."""
        try:
            self.global_profile_cache = self.load_global_profile()
        except Exception:
            self.global_profile_cache = None
    
    def save_profile(self, cpf: str, profile) -> None:
        """Save user profile to SQLite."""
        # Convert profile to dict if it has dict() method
        if hasattr(profile, 'dict'):
            profile_dict = profile.dict()
        else:
            profile_dict = profile
        
        import json
        profile_json = json.dumps(profile_dict, default=str)
        
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT OR REPLACE INTO profiles (cpf, profile_json, created_at, last_updated, transaction_count)
            VALUES (?, ?, ?, ?, ?)
        ''', (
            cpf,
            profile_json,
            profile_dict.get('created_at', datetime.utcnow().isoformat()),
            profile_dict.get('last_updated', datetime.utcnow().isoformat()),
            profile_dict.get('transaction_count', 0)
        ))
        
        conn.commit()
        conn.close()
    
    def load_profile(self, cpf: str) -> Optional:
        """Load user profile from SQLite."""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('SELECT profile_json FROM profiles WHERE cpf = ?', (cpf,))
        row = cursor.fetchone()
        
        conn.close()
        
        if row is None:
            return None
        
        # Import here to avoid circular dependency
        try:
            from src.user_profile.models import UserProfile
        except ImportError:
            from user_profile.models import UserProfile
        
        profile_dict = json.loads(row[0])
        return UserProfile(**profile_dict)
    
    def load_global_profile(self) -> Optional:
        """Load global profile from JSON."""
        if self.global_profile_cache is not None:
            return self.global_profile_cache
        
        if not self.global_profile_file.exists():
            return None
        
        with open(self.global_profile_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        global_profile_data = data.get("global_profile")
        
        if global_profile_data is None:
            return None
        
        # Import here to avoid circular dependency
        try:
            from src.user_profile.models import UserProfile, Statistics, ValueStatistics, HourStatistics, FrequencyStatistics, Destinations, Canais, Produtos
        except ImportError:
            from user_profile.models import UserProfile, Statistics, ValueStatistics, HourStatistics, FrequencyStatistics, Destinations, Canais, Produtos
        
        stats_data = global_profile_data.get("statistics", {})
        canais_data = global_profile_data.get("canais", {})
        produtos_data = global_profile_data.get("produtos", {})
        
        profile = UserProfile(
            cpf="GLOBAL",
            created_at=datetime.utcnow(),
            last_updated=datetime.utcnow(),
            transaction_count=0,
            is_cold_start=False,
            statistics=Statistics(**stats_data),
            destinations=Destinations(),
            canais=Canais(**canais_data),
            produtos=Produtos(**produtos_data)
        )
        
        self.global_profile_cache = profile
        return profile
    
    def delete_profile(self, cpf: str) -> bool:
        """Delete user profile from SQLite."""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM profiles WHERE cpf = ?', (cpf,))
        affected = cursor.rowcount
        
        conn.commit()
        conn.close()
        
        return affected > 0
    
    def list_all_profiles(self) -> dict:
        """List all profiles (returns dict of CPF -> profile)."""
        conn = sqlite3.connect(self.db_file)
        cursor = conn.cursor()
        
        cursor.execute('SELECT cpf, profile_json FROM profiles')
        rows = cursor.fetchall()
        
        conn.close()
        
        return {row[0]: json.loads(row[1]) for row in rows}
