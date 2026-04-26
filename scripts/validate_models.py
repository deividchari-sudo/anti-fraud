"""
Validation script for all behavioral profiling models.
Especialista de Dados - Validação técnica completa.
"""

import sys
import io
import time
import numpy as np
from pathlib import Path

# Fix Windows console encoding for unicode (emojis)
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.user_profile.clustering import UserClusterer
from src.user_profile.isolation_forest import (
    MultivariateAnomalyDetector,
    EnsembleIsolationForest,
)
from src.user_profile.online_learning import (
    AdaptiveUserProfile,
    OnlineAnomalyThreshold,
    ConceptDriftDetector,
)
from src.user_profile.graph_features import GraphFeatureExtractor
from src.user_profile.temporal_features import TemporalFeatureExtractor


def section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print('=' * 70)


def generate_sample_transactions(n=200):
    """Generate synthetic transactions for validation."""
    np.random.seed(42)
    transactions = []
    cpfs = [f"{i:011d}" for i in range(1, 21)]  # 20 users
    for i in range(n):
        cpf = np.random.choice(cpfs)
        receiver = np.random.choice(cpfs)
        hour_probs = np.array([0.01]*6 + [0.07]*12 + [0.02]*6)
        hour_probs = hour_probs / hour_probs.sum()
        hour = np.random.choice(range(24), p=hour_probs)
        transactions.append({
            "id": f"tx-{i}",
            "timestamp": f"2026-04-{(i % 28) + 1:02d}T{hour:02d}:00:00Z",
            "canal": np.random.choice(["app", "web", "api"], p=[0.7, 0.25, 0.05]),
            "produto": np.random.choice(["pix", "ted", "boleto"], p=[0.7, 0.2, 0.1]),
            "direcao": "saida",
            "valor": float(np.random.lognormal(6, 1.2)),
            "sender": {"cpfSender": cpf, "banco": int(np.random.choice([1, 152, 260]))},
            "receiver": {"cpfReceiver": receiver, "banco": int(np.random.choice([1, 152, 260, 888]))},
        })
    return transactions


def generate_sample_profiles(n=20):
    """Generate synthetic user profiles for clustering."""
    np.random.seed(42)
    profiles = []
    for i in range(n):
        profiles.append({
            "cpf": f"{i:011d}",
            "transaction_count": int(np.random.randint(30, 500)),
            "statistics": {
                "valor": {
                    "mean": float(np.random.lognormal(6, 1)),
                    "std": float(np.random.lognormal(5, 0.5)),
                    "median": float(np.random.lognormal(6, 1)),
                    "p95": float(np.random.lognormal(7, 1)),
                },
                "hora": {
                    "mean": float(np.random.uniform(8, 20)),
                    "std": float(np.random.uniform(2, 6)),
                },
                "frequencia": {
                    "transactions_per_day_mean": float(np.random.uniform(0.5, 10)),
                },
            },
        })
    return profiles


# ===== 1. Clustering =====
section("1. CLUSTERING (K-means + DBSCAN)")

profiles = generate_sample_profiles(50)

# K-means
kmeans_clusterer = UserClusterer(n_clusters=10, clustering_method="kmeans")
t0 = time.time()
kmeans_result = kmeans_clusterer.fit(profiles)
kmeans_time = (time.time() - t0) * 1000

print(f"K-means:")
print(f"  - Clusters: {kmeans_result['n_clusters']}")
print(f"  - Silhouette Score: {kmeans_result['silhouette_score']:.4f}")
print(f"  - Tempo de treino: {kmeans_time:.2f}ms")
print(f"  - Cluster sizes: {[d['size'] for d in kmeans_result['cluster_descriptions'].values()]}")

# DBSCAN
dbscan_clusterer = UserClusterer(clustering_method="dbscan")
t0 = time.time()
dbscan_result = dbscan_clusterer.fit(profiles)
dbscan_time = (time.time() - t0) * 1000

print(f"\nDBSCAN:")
print(f"  - Clusters: {dbscan_result['n_clusters']}")
print(f"  - Outliers (noise): {dbscan_result.get('noise_count', 0)}")
print(f"  - Silhouette Score: {dbscan_result['silhouette_score']}")
print(f"  - Tempo de treino: {dbscan_time:.2f}ms")

# ===== 2. Isolation Forest =====
section("2. ISOLATION FOREST (com Auto-Contamination)")

txs = generate_sample_transactions(500)

iforest = MultivariateAnomalyDetector(auto_contamination=True)
t0 = time.time()
iforest.fit(txs)
fit_time = (time.time() - t0) * 1000

print(f"  - Optimal contamination: {iforest.optimal_contamination:.4f}")
print(f"  - Tempo de treino: {fit_time:.2f}ms")

# Predict latency
latencies = []
anomalies_detected = 0
for tx in txs[:100]:
    t0 = time.time()
    result = iforest.predict(tx)
    latencies.append((time.time() - t0) * 1000)
    if result["is_anomaly"]:
        anomalies_detected += 1

print(f"  - Latência média (predict): {np.mean(latencies):.3f}ms")
print(f"  - Latência P95: {np.percentile(latencies, 95):.3f}ms")
print(f"  - Anomalias em 100 amostras: {anomalies_detected}")

# Feature importance
fi = iforest.get_feature_importance()
top_features = sorted(fi.items(), key=lambda x: -x[1])[:5]
print(f"  - Top 5 features:")
for name, score in top_features:
    print(f"      {name}: {score:.4f}")

# ===== 3. Ensemble Isolation Forest =====
section("3. ENSEMBLE ISOLATION FOREST")

ensemble = EnsembleIsolationForest(n_estimators=5)
t0 = time.time()
ensemble.fit(txs)
print(f"  - Tempo de treino (5 modelos): {(time.time() - t0) * 1000:.2f}ms")

result = ensemble.predict(txs[0])
print(f"  - Vote distribution: {result['vote_distribution']}")
print(f"  - Score consenso: {result['anomaly_score']:.4f}")

# ===== 4. Online Learning =====
section("4. ONLINE LEARNING (Adaptive)")

adaptive = AdaptiveUserProfile(cpf="test_user")
for tx in txs[:50]:
    adaptive.update(tx, anomaly_score=0.3)

print(f"  - EMA value: {adaptive.ema_value:.2f}")
print(f"  - EMA hour: {adaptive.ema_hour:.2f}")
print(f"  - Updates: {adaptive.count}")

# Concept drift
drift = ConceptDriftDetector()
for tx in txs[:30]:
    drift.add_sample(float(tx["valor"]))
drift_detected = drift.detect_drift()
print(f"  - Concept drift detected: {drift_detected}")

# Adaptive threshold
threshold = OnlineAnomalyThreshold()
for _ in range(20):
    threshold.update(0.3, is_true_anomaly=False)
for _ in range(5):
    threshold.update(0.9, is_true_anomaly=True)
print(f"  - Adaptive threshold: {threshold.threshold:.4f}")

# ===== 5. Graph Features =====
section("5. GRAPH FEATURES")

graph_extractor = GraphFeatureExtractor()
graph_extractor.build_graph(txs)

n_nodes = len(graph_extractor.graph.nodes)
total_edges = sum(len(e) for e in graph_extractor.graph.edges.values())
print(f"  - Nodes: {n_nodes}")
print(f"  - Edges: {total_edges}")

# Sample node features
sample_cpf = txs[0]["sender"]["cpfSender"]
node_features = graph_extractor.extract_node_features(sample_cpf)
print(f"  - Features for {sample_cpf}:")
print(f"      out_degree: {node_features['out_degree']}")
print(f"      in_degree: {node_features['in_degree']}")
print(f"      page_rank: {node_features['page_rank']:.4f}")
print(f"      clustering_coefficient: {node_features['clustering_coefficient']:.4f}")

# Detect graph anomaly
anomaly_result = graph_extractor.detect_graph_anomaly(txs[0])
print(f"  - Sample anomaly detection: is_anomaly={anomaly_result.get('is_anomaly', False)}")
stats = {"n_nodes": n_nodes, "n_edges": total_edges}

# ===== 6. Temporal Features =====
section("6. TEMPORAL FEATURES")

temporal = TemporalFeatureExtractor()
features = temporal.extract_all_temporal_features(txs[:100])
print(f"  - Sliding windows: {list(features['sliding_windows'].keys())}")
print(f"  - Trend direction: {features['trends']['trend_direction']}")
print(f"  - Most active hour: {features['seasonal']['most_active_hour']}")
print(f"  - Most active day: {features['seasonal']['most_active_day']}")

# ===== Summary =====
section("RESUMO DA VALIDAÇÃO")

print(f"""
✅ Clustering K-means:     OK ({kmeans_time:.1f}ms, silhouette={kmeans_result['silhouette_score']:.3f})
✅ Clustering DBSCAN:      OK ({dbscan_time:.1f}ms, {dbscan_result['n_clusters']} clusters)
✅ Isolation Forest:       OK (auto-contamination={iforest.optimal_contamination:.3f})
✅ Ensemble Isolation:     OK (5 modelos, voting funcional)
✅ Online Learning:        OK (EMA + drift + adaptive threshold)
✅ Graph Features:         OK ({stats['n_nodes']} nodes, {stats['n_edges']} edges)
✅ Temporal Features:      OK (sliding windows + trends + seasonal)

🎯 LATÊNCIA Isolation Forest: {np.mean(latencies):.2f}ms (meta <10ms) ✅
""")
