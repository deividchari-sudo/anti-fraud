"""
Simple GraphSAGE-inspired Node Embedding (Sprint 3).

Implementation in pure NumPy (no PyTorch Geometric dependency).
Generates node embeddings via mean-aggregation of neighbor features over K hops,
mimicking the inductive learning approach of GraphSAGE (Hamilton et al., 2017).

Used by Nubank and Bradesco P&D for money mule and circular fraud detection.

NOTE: This is a simplified inductive embedding - not a trained GNN. It computes
deterministic neighborhood aggregation features that downstream supervised models
can use as inputs (replacing the simpler NetworkX graph features).
"""

from collections import defaultdict
from typing import Dict, List, Optional

import numpy as np


class SimpleGraphSAGE:
    """Mean-aggregator GraphSAGE-style node embedding (untrained, deterministic).

    Steps:
    1. Build directed transaction graph (sender -> receiver) with edge values
    2. For each node, build a base feature vector from local statistics
    3. Aggregate neighbor features over K hops with mean pooling
    4. Concatenate own + aggregated features -> embedding
    """

    def __init__(self, num_hops: int = 2, embedding_dim: int = 8):
        """
        Args:
            num_hops: Number of GraphSAGE-style aggregation hops (1 or 2)
            embedding_dim: Dimension of base node feature vector
        """
        if num_hops not in (1, 2):
            raise ValueError("num_hops must be 1 or 2")
        if embedding_dim < 4:
            raise ValueError("embedding_dim must be at least 4")

        self.num_hops = num_hops
        self.embedding_dim = embedding_dim

        # Graph storage
        self.out_neighbors: Dict[str, List[str]] = defaultdict(list)
        self.in_neighbors: Dict[str, List[str]] = defaultdict(list)
        self.edge_weights: Dict[str, float] = {}  # "sender->receiver" -> total amount
        self.nodes: set = set()
        self._base_features: Dict[str, np.ndarray] = {}

    # ----------------------------------------------------------- graph build

    def build_graph(self, transactions: List[dict]) -> None:
        """Build transaction graph from a list of transactions."""
        self.out_neighbors.clear()
        self.in_neighbors.clear()
        self.edge_weights.clear()
        self.nodes = set()

        for tx in transactions:
            sender = tx.get("sender", {}).get("cpfSender") or ""
            receiver = tx.get("receiver", {}).get("cpfReceiver") or ""
            amount = float(tx.get("valor", 0) or 0)

            if not sender or not receiver:
                continue

            self.nodes.add(sender)
            self.nodes.add(receiver)
            self.out_neighbors[sender].append(receiver)
            self.in_neighbors[receiver].append(sender)
            edge_key = f"{sender}->{receiver}"
            self.edge_weights[edge_key] = self.edge_weights.get(edge_key, 0.0) + amount

        # Pre-compute base features for all nodes
        self._base_features = {
            node: self._compute_base_features(node) for node in self.nodes
        }

    # -------------------------------------------------------- node features

    def _compute_base_features(self, node: str) -> np.ndarray:
        """Compute base feature vector for a node (degree, value stats, etc.)."""
        out_n = self.out_neighbors.get(node, [])
        in_n = self.in_neighbors.get(node, [])

        out_degree = len(set(out_n))
        in_degree = len(set(in_n))
        total_sent = sum(
            self.edge_weights.get(f"{node}->{r}", 0.0) for r in set(out_n)
        )
        total_received = sum(
            self.edge_weights.get(f"{s}->{node}", 0.0) for s in set(in_n)
        )

        # Padding to embedding_dim with normalized derived features
        balance = total_received - total_sent
        send_recv_ratio = total_sent / (total_received + 1.0)
        unique_out_ratio = out_degree / (len(out_n) + 1.0)
        unique_in_ratio = in_degree / (len(in_n) + 1.0)

        features = np.array(
            [
                np.log1p(out_degree),
                np.log1p(in_degree),
                np.log1p(total_sent),
                np.log1p(total_received),
                np.tanh(balance / 1000.0),
                np.tanh(send_recv_ratio),
                unique_out_ratio,
                unique_in_ratio,
            ],
            dtype=float,
        )

        # Pad / truncate to embedding_dim
        if len(features) < self.embedding_dim:
            features = np.concatenate(
                [features, np.zeros(self.embedding_dim - len(features))]
            )
        else:
            features = features[: self.embedding_dim]
        return features

    # ----------------------------------------------------- mean aggregation

    def _aggregate_neighbors(self, node: str) -> np.ndarray:
        """Mean-aggregate features of neighbors (1-hop)."""
        # Combine in + out neighbors as "1-hop neighborhood"
        neighbors = list(set(self.out_neighbors.get(node, []))) + list(
            set(self.in_neighbors.get(node, []))
        )
        if not neighbors:
            return np.zeros(self.embedding_dim)
        feats = [self._base_features.get(n, np.zeros(self.embedding_dim)) for n in neighbors]
        return np.mean(feats, axis=0)

    def get_embedding(self, node: str) -> np.ndarray:
        """Get K-hop GraphSAGE-style embedding for a node.

        Args:
            node: Node identifier (typically CPF)

        Returns:
            Embedding vector of size embedding_dim * (1 + num_hops)
        """
        if node not in self.nodes:
            return np.zeros(self.embedding_dim * (1 + self.num_hops))

        own = self._base_features[node]
        hop1 = self._aggregate_neighbors(node)

        if self.num_hops == 1:
            return np.concatenate([own, hop1])

        # 2-hop: aggregate aggregated features of neighbors
        neighbors = list(set(self.out_neighbors.get(node, []))) + list(
            set(self.in_neighbors.get(node, []))
        )
        if not neighbors:
            hop2 = np.zeros(self.embedding_dim)
        else:
            hop2 = np.mean(
                [self._aggregate_neighbors(n) for n in neighbors], axis=0
            )

        return np.concatenate([own, hop1, hop2])

    # ---------------------------------------------------------- convenience

    def extract_features_dict(
        self, cpf: Optional[str], prefix: str = "gsage"
    ) -> Dict[str, float]:
        """Return embedding as a flat feature dict suitable for ML pipelines."""
        if not cpf:
            zeros = np.zeros(self.embedding_dim * (1 + self.num_hops))
            return {f"{prefix}_{i}": float(v) for i, v in enumerate(zeros)}
        emb = self.get_embedding(cpf)
        return {f"{prefix}_{i}": float(v) for i, v in enumerate(emb)}

    @property
    def output_dim(self) -> int:
        """Total embedding dimension."""
        return self.embedding_dim * (1 + self.num_hops)
