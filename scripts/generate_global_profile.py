"""
Generate global profile from dataset.
Global profile is used as fallback for cold start users (users with insufficient transaction history).
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

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


def calculate_global_statistics(df: pd.DataFrame) -> Statistics:
    """Calculate global statistics from all transactions."""
    print("📊 Calculating global statistics...")
    
    # Parse JSON payload column
    df['parsed_payload'] = df['payload'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    
    # Extract values
    valores = df['parsed_payload'].apply(lambda x: float(x.get("valor", 0)) if isinstance(x, dict) and x.get("valor", 0) > 0 else None).dropna()
    
    # Extract hours
    horas = []
    for payload in df['parsed_payload']:
        if isinstance(payload, dict):
            timestamp = payload.get("timestamp", "")
            if timestamp:
                try:
                    dt = pd.to_datetime(timestamp)
                    horas.append(dt.hour)
                except:
                    pass
    
    # Calculate value statistics
    valor_stats = ValueStatistics(
        mean=float(np.mean(valores)),
        std=float(np.std(valores)),
        median=float(np.median(valores)),
        p25=float(np.percentile(valores, 25)),
        p75=float(np.percentile(valores, 75)),
        p95=float(np.percentile(valores, 95)),
        min=float(np.min(valores)),
        max=float(np.max(valores))
    )
    
    # Calculate hour statistics
    if horas:
        horas_array = np.array(horas)
        hora_stats = HourStatistics(
            mean=float(np.mean(horas_array)),
            std=float(np.std(horas_array)),
            median=float(np.median(horas_array)),
            p25=float(np.percentile(horas_array, 25)),
            p75=float(np.percentile(horas_array, 75))
        )
    else:
        hora_stats = HourStatistics(
            mean=13.0, std=4.5, median=13.0,
            p25=10.0, p75=16.0
        )
    
    # Calculate frequency statistics (simplified)
    unique_days = df['parsed_payload'].apply(lambda x: pd.to_datetime(x.get("timestamp", "")).date() if isinstance(x, dict) and x.get("timestamp") else None).nunique()
    total_transactions = len(df)
    
    if unique_days > 0:
        transactions_per_day = total_transactions / unique_days
    else:
        transactions_per_day = 1.8
    
    freq_stats = FrequencyStatistics(
        transactions_per_day_mean=round(transactions_per_day, 2),
        transactions_per_day_std=round(transactions_per_day * 0.3, 2),
        days_active=int(unique_days)
    )
    
    return Statistics(
        valor=valor_stats,
        hora=hora_stats,
        frequencia=freq_stats
    )


def calculate_global_channels(df: pd.DataFrame) -> Canais:
    """Calculate global channel distribution."""
    print("📊 Calculating global channel distribution...")
    
    # Parse JSON payload column
    df['parsed_payload'] = df['payload'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    
    canal_counts = df['parsed_payload'].apply(lambda x: x.get("canal", "") if isinstance(x, dict) else "").value_counts()
    total = len(df)
    
    app_count = canal_counts.get('app', 0)
    web_count = canal_counts.get('web', 0)
    api_count = canal_counts.get('api', 0)
    
    return Canais(
        app=CanalInfo(count=int(app_count), percentage=round(app_count / total, 2)),
        web=CanalInfo(count=int(web_count), percentage=round(web_count / total, 2)),
        api=CanalInfo(count=int(api_count), percentage=round(api_count / total, 2))
    )


def calculate_global_products(df: pd.DataFrame) -> Produtos:
    """Calculate global product distribution."""
    print("📊 Calculating global product distribution...")
    
    # Parse JSON payload column
    df['parsed_payload'] = df['payload'].apply(lambda x: json.loads(x) if isinstance(x, str) else x)
    
    produto_counts = df['parsed_payload'].apply(lambda x: x.get("produto", "") if isinstance(x, dict) else "").value_counts()
    total = len(df)
    
    pix_count = produto_counts.get('pix', 0)
    ted_count = produto_counts.get('ted', 0)
    boleto_count = produto_counts.get('boleto', 0)
    autenticacao_count = produto_counts.get('autenticacao', 0)
    
    produtos = Produtos(
        pix=ProdutoInfo(count=int(pix_count), percentage=round(pix_count / total, 2)),
        ted=ProdutoInfo(count=int(ted_count), percentage=round(ted_count / total, 2)),
        boleto=ProdutoInfo(count=int(boleto_count), percentage=round(boleto_count / total, 2))
    )
    
    if autenticacao_count > 0:
        produtos.autenticacao = ProdutoInfo(
            count=int(autenticacao_count),
            percentage=round(autenticacao_count / total, 2)
        )
    
    return produtos


def generate_global_profile(dataset_path: str = "dataset_transacoes_expanded.csv",
                           output_file: str = "data/global_profile.json"):
    """
    Generate global profile from dataset.
    
    Args:
        dataset_path: Path to transaction dataset
        output_file: Output JSON file for global profile
    """
    print("=" * 60)
    print("BEHAVIORAL PROFILING - GLOBAL PROFILE GENERATION")
    print("=" * 60)
    
    # Load dataset
    df = load_dataset(dataset_path)
    
    # Calculate statistics
    statistics = calculate_global_statistics(df)
    canais = calculate_global_channels(df)
    produtos = calculate_global_products(df)
    
    # Create global profile
    now = datetime.utcnow()
    
    global_profile_dict = {
        "global_profile": {
            "statistics": statistics.dict(),
            "canais": canais.dict(),
            "produtos": produtos.dict(),
            "destinations": {
                "common_cpfs": {},
                "common_bancos": {}
            }
        },
        "metadata": {
            "based_on_transactions": len(df),
            "last_updated": now.isoformat(),
            "version": "1.0.0"
        }
    }
    
    # Export to JSON
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(global_profile_dict, f, indent=2, default=str)
    
    print(f"\n✅ Global profile exported to {output_file}")
    print(f"   Based on {len(df)} transactions")
    
    # Print summary
    print("\n📊 Global Profile Summary:")
    print(f"   - Mean value: R$ {statistics.valor.mean:.2f}")
    print(f"   - Mean hour: {statistics.hora.mean:.0f}:00")
    print(f"   - Transactions/day: {statistics.frequencia.transactions_per_day_mean:.2f}")
    print(f"   - Channel distribution: app {canais.app.percentage*100:.0f}%, web {canais.web.percentage*100:.0f}%, api {canais.api.percentage*100:.0f}%")
    print(f"   - Product distribution: pix {produtos.pix.percentage*100:.0f}%, ted {produtos.ted.percentage*100:.0f}%, boleto {produtos.boleto.percentage*100:.0f}%")
    
    print("\n" + "=" * 60)
    print("GLOBAL PROFILE GENERATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate global profile from dataset")
    parser.add_argument(
        "--dataset",
        default="dataset_transacoes_expanded.csv",
        help="Path to transaction dataset"
    )
    parser.add_argument(
        "--output",
        default="data/global_profile.json",
        help="Output JSON file for global profile"
    )
    
    args = parser.parse_args()
    
    generate_global_profile(
        dataset_path=args.dataset,
        output_file=args.output
    )
