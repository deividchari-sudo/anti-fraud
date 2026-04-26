"""
Backfill script to generate user profiles from existing dataset.
This script reads the transaction dataset and creates behavioral profiles for each user.
"""

import sys
import pandas as pd
from pathlib import Path
from collections import defaultdict
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from user_profile.service import UserProfileService


class MockUserProfileRepository:
    """Mock repository for backfill (will be replaced by Backend's JSON repository)."""
    
    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.profiles = {}
    
    def save_profile(self, cpf: str, profile):
        """Save profile to memory (for backfill)."""
        self.profiles[cpf] = profile
    
    def load_profile(self, cpf: str):
        """Load profile from memory."""
        return self.profiles.get(cpf)
    
    def load_global_profile(self):
        """No global profile during backfill."""
        return None
    
    def export_to_json(self, output_file: str = "data/user_profiles.json"):
        """Export all profiles to JSON file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(exist_ok=True)
        
        profiles_dict = {
            "profiles": {
                cpf: profile.dict()
                for cpf, profile in self.profiles.items()
            },
            "metadata": {
                "total_profiles": len(self.profiles),
                "last_sync": pd.Timestamp.now().isoformat(),
                "version": "1.0.0"
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(profiles_dict, f, indent=2, default=str)
        
        print(f"✅ Exported {len(self.profiles)} profiles to {output_path}")


def load_dataset(dataset_path: str = "dataset_transacoes_expanded.csv") -> pd.DataFrame:
    """Load transaction dataset."""
    print(f"📂 Loading dataset from {dataset_path}...")
    
    path = Path(dataset_path)
    if not path.exists():
        print(f"⚠️  Dataset not found at {dataset_path}")
        print(f"   Trying dataset_transacoes.csv...")
        path = Path("dataset_transacoes.csv")
    
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path} or dataset_transacoes.csv")
    
    df = pd.read_csv(path)
    print(f"✅ Loaded {len(df)} transactions")
    return df


def group_transactions_by_cpf(df: pd.DataFrame) -> dict:
    """Group transactions by sender CPF."""
    print("📊 Grouping transactions by CPF...")
    
    # Normalize column names
    df.columns = [col.lower().replace('.', '_') for col in df.columns]
    
    # Extract CPF from nested structure if needed
    if 'sender' in df.columns:
        # Parse JSON sender column if it's a string
        if df['sender'].dtype == 'object':
            df['sender'] = df['sender'].apply(lambda x: eval(x) if isinstance(x, str) else x)
        
        df['cpf_sender'] = df['sender'].apply(lambda x: x.get('cpfSender') if isinstance(x, dict) else None)
    elif 'cpf_sender' in df.columns:
        df['cpf_sender'] = df['cpf_sender']
    else:
        raise ValueError("Could not find CPF column in dataset")
    
    # Filter out rows without CPF
    df = df[df['cpf_sender'].notna()]
    df['cpf_sender'] = df['cpf_sender'].astype(str)
    
    # Group by CPF
    cpf_groups = defaultdict(list)
    for _, row in df.iterrows():
        cpf = row['cpf_sender']
        # Convert row to dict
        transaction = row.to_dict()
        cpf_groups[cpf].append(transaction)
    
    print(f"✅ Found {len(cpf_groups)} unique CPFs")
    return dict(cpf_groups)


def backfill_profiles(dataset_path: str = "dataset_transacoes_expanded.csv", 
                      min_transactions: int = 5,
                      output_file: str = "data/user_profiles.json"):
    """
    Backfill user profiles from dataset.
    
    Args:
        dataset_path: Path to transaction dataset
        min_transactions: Minimum transactions required to create a profile
        output_file: Output JSON file for profiles
    """
    print("=" * 60)
    print("BEHAVIORAL PROFILING - BACKFILL SCRIPT")
    print("=" * 60)
    
    # Load dataset
    df = load_dataset(dataset_path)
    
    # Group by CPF
    cpf_groups = group_transactions_by_cpf(df)
    
    # Create service with mock repository
    repository = MockUserProfileRepository()
    service = UserProfileService(repository)
    
    # Generate profiles
    print(f"\n🔄 Generating profiles (min {min_transactions} transactions per CPF)...")
    profiles_created = 0
    profiles_skipped = 0
    
    for cpf, transactions in cpf_groups.items():
        if len(transactions) >= min_transactions:
            try:
                profile = service.calculate_profile(transactions, cpf)
                repository.save_profile(cpf, profile)
                profiles_created += 1
                
                if profiles_created % 100 == 0:
                    print(f"   Progress: {profiles_created} profiles created...")
            except Exception as e:
                print(f"   ⚠️  Error creating profile for CPF {cpf}: {e}")
                profiles_skipped += 1
        else:
            profiles_skipped += 1
    
    print(f"\n✅ Profile generation complete:")
    print(f"   - Profiles created: {profiles_created}")
    print(f"   - Profiles skipped (< {min_transactions} transactions): {profiles_skipped}")
    
    # Export to JSON
    repository.export_to_json(output_file)
    
    print("\n" + "=" * 60)
    print("BACKFILL COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Backfill user profiles from dataset")
    parser.add_argument(
        "--dataset",
        default="dataset_transacoes_expanded.csv",
        help="Path to transaction dataset"
    )
    parser.add_argument(
        "--min-transactions",
        type=int,
        default=5,
        help="Minimum transactions required to create a profile"
    )
    parser.add_argument(
        "--output",
        default="data/user_profiles.json",
        help="Output JSON file for profiles"
    )
    
    args = parser.parse_args()
    
    backfill_profiles(
        dataset_path=args.dataset,
        min_transactions=args.min_transactions,
        output_file=args.output
    )
