"""
Graph-based features for transaction network analysis.
Implements graph metrics to detect anomalies in connection patterns.
"""

from typing import List, Dict, Optional, Set
from collections import defaultdict, deque
import numpy as np


class TransactionGraph:
    """Represents the transaction network as a graph."""
    
    def __init__(self):
        """Initialize transaction graph."""
        self.nodes = set()  # CPFs
        self.edges = defaultdict(set)  # sender -> set of receivers
        self.reverse_edges = defaultdict(set)  # receiver -> set of senders
        self.edge_weights = defaultdict(float)  # (sender, receiver) -> total amount
        self.edge_counts = defaultdict(int)  # (sender, receiver) -> count
        self.node_degree = defaultdict(int)  # node -> degree
        self.node_total_sent = defaultdict(float)  # node -> total sent
        self.node_total_received = defaultdict(float)  # node -> total received
    
    def add_transaction(self, sender_cpf: str, receiver_cpf: str, amount: float):
        """
        Add a transaction to the graph.
        
        Args:
            sender_cpf: Sender CPF
            receiver_cpf: Receiver CPF
            amount: Transaction amount
        """
        # Add nodes
        self.nodes.add(sender_cpf)
        self.nodes.add(receiver_cpf)
        
        # Add edges
        self.edges[sender_cpf].add(receiver_cpf)
        self.reverse_edges[receiver_cpf].add(sender_cpf)
        
        # Update edge weights and counts
        edge_key = (sender_cpf, receiver_cpf)
        self.edge_weights[edge_key] += amount
        self.edge_counts[edge_key] += 1
        
        # Update node degrees
        self.node_degree[sender_cpf] += 1
        self.node_degree[receiver_cpf] += 1
        
        # Update node totals
        self.node_total_sent[sender_cpf] += amount
        self.node_total_received[receiver_cpf] += amount
    
    def get_node_degree(self, cpf: str) -> int:
        """Get degree of a node (number of connections)."""
        return self.node_degree.get(cpf, 0)
    
    def get_out_degree(self, cpf: str) -> int:
        """Get out-degree of a node (number of unique receivers)."""
        return len(self.edges.get(cpf, set()))
    
    def get_in_degree(self, cpf: str) -> int:
        """Get in-degree of a node (number of unique senders)."""
        return len(self.reverse_edges.get(cpf, set()))
    
    def get_clustering_coefficient(self, cpf: str) -> float:
        """
        Calculate local clustering coefficient for a node.
        Measures how connected neighbors are to each other.
        
        Args:
            cpf: Node CPF
            
        Returns:
            Clustering coefficient (0.0 to 1.0)
        """
        neighbors = self.edges.get(cpf, set()) | self.reverse_edges.get(cpf, set())
        
        if len(neighbors) < 2:
            return 0.0
        
        # Count edges between neighbors
        neighbor_edges = 0
        for n1 in neighbors:
            for n2 in neighbors:
                if n1 != n2:
                    if n2 in self.edges.get(n1, set()) or n1 in self.edges.get(n2, set()):
                        neighbor_edges += 1
        
        # Clustering coefficient = actual edges / possible edges
        possible_edges = len(neighbors) * (len(neighbors) - 1)
        return neighbor_edges / possible_edges if possible_edges > 0 else 0.0
    
    def get_shortest_path_length(self, source: str, target: str) -> Optional[int]:
        """
        Calculate shortest path length between two nodes using BFS.
        
        Args:
            source: Source CPF
            target: Target CPF
            
        Returns:
            Shortest path length or None if no path exists
        """
        if source == target:
            return 0
        
        if source not in self.nodes or target not in self.nodes:
            return None
        
        # BFS
        visited = set()
        queue = deque([(source, 0)])
        visited.add(source)
        
        while queue:
            node, distance = queue.popleft()
            
            # Get neighbors
            neighbors = self.edges.get(node, set()) | self.reverse_edges.get(node, set())
            
            for neighbor in neighbors:
                if neighbor == target:
                    return distance + 1
                
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, distance + 1))
        
        return None
    
    def get_common_neighbors(self, cpf1: str, cpf2: str) -> Set[str]:
        """
        Get common neighbors between two nodes.
        
        Args:
            cpf1: First CPF
            cpf2: Second CPF
            
        Returns:
            Set of common neighbors
        """
        neighbors1 = self.edges.get(cpf1, set()) | self.reverse_edges.get(cpf1, set())
        neighbors2 = self.edges.get(cpf2, set()) | self.reverse_edges.get(cpf2, set())
        
        return neighbors1 & neighbors2
    
    def get_page_rank(self, cpf: str, damping_factor: float = 0.85, 
                     iterations: int = 20) -> float:
        """
        Calculate PageRank for a node (simplified).
        
        Args:
            cpf: Node CPF
            damping_factor: Damping factor for PageRank
            iterations: Number of iterations
            
        Returns:
            PageRank score
        """
        if cpf not in self.nodes:
            return 0.0
        
        # Initialize PageRank
        n_nodes = len(self.nodes)
        page_rank = {node: 1.0 / n_nodes for node in self.nodes}
        
        # Iterate
        for _ in range(iterations):
            new_page_rank = {}
            
            for node in self.nodes:
                # Get incoming neighbors
                incoming = self.reverse_edges.get(node, set())
                
                if not incoming:
                    new_page_rank[node] = (1 - damping_factor) / n_nodes
                else:
                    rank_sum = 0.0
                    for neighbor in incoming:
                        neighbor_out_degree = len(self.edges.get(neighbor, set()))
                        if neighbor_out_degree > 0:
                            rank_sum += page_rank[neighbor] / neighbor_out_degree
                    
                    new_page_rank[node] = (1 - damping_factor) / n_nodes + damping_factor * rank_sum
            
            page_rank = new_page_rank
        
        return page_rank.get(cpf, 0.0)


class GraphFeatureExtractor:
    """Extracts graph-based features from transaction network."""
    
    def __init__(self):
        """Initialize graph feature extractor."""
        self.graph = TransactionGraph()
    
    def build_graph(self, transactions: List[dict]):
        """
        Build transaction graph from transaction history.
        
        Args:
            transactions: List of transaction dictionaries
        """
        for tx in transactions:
            sender = tx.get("sender", {}).get("cpfSender", "")
            receiver = tx.get("receiver", {}).get("cpfReceiver", "")
            amount = float(tx.get("valor", 0))
            
            if sender and receiver:
                self.graph.add_transaction(sender, receiver, amount)
    
    def extract_node_features(self, cpf: str) -> Dict:
        """
        Extract graph-based features for a specific CPF.
        
        Args:
            cpf: CPF to extract features for
            
        Returns:
            Dictionary with graph features
        """
        if cpf not in self.graph.nodes:
            return self._empty_features()
        
        return {
            "node_degree": self.graph.get_node_degree(cpf),
            "out_degree": self.graph.get_out_degree(cpf),
            "in_degree": self.graph.get_in_degree(cpf),
            "clustering_coefficient": self.graph.get_clustering_coefficient(cpf),
            "total_sent": self.graph.node_total_sent.get(cpf, 0.0),
            "total_received": self.graph.node_total_received.get(cpf, 0.0),
            "net_flow": self.graph.node_total_sent.get(cpf, 0.0) - self.graph.node_total_received.get(cpf, 0.0),
            "page_rank": self.graph.get_page_rank(cpf)
        }
    
    def extract_edge_features(self, sender_cpf: str, receiver_cpf: str) -> Dict:
        """
        Extract features for a specific edge.
        
        Args:
            sender_cpf: Sender CPF
            receiver_cpf: Receiver CPF
            
        Returns:
            Dictionary with edge features
        """
        edge_key = (sender_cpf, receiver_cpf)
        
        return {
            "total_amount": self.graph.edge_weights.get(edge_key, 0.0),
            "transaction_count": self.graph.edge_counts.get(edge_key, 0),
            "avg_amount": self.graph.edge_weights.get(edge_key, 0.0) / max(self.graph.edge_counts.get(edge_key, 1), 1),
            "common_neighbors": len(self.graph.get_common_neighbors(sender_cpf, receiver_cpf)),
            "shortest_path_length": self.graph.get_shortest_path_length(sender_cpf, receiver_cpf) or 0
        }
    
    def detect_graph_anomaly(self, transaction: dict, threshold_degree: int = 50) -> Dict:
        """
        Detect graph-based anomalies in a transaction.
        
        Args:
            transaction: Transaction dictionary
            threshold_degree: Degree threshold for anomaly detection
            
        Returns:
            Dictionary with anomaly detection result
        """
        sender = transaction.get("sender", {}).get("cpfSender", "")
        receiver = transaction.get("receiver", {}).get("cpfReceiver", "")
        
        if not sender or not receiver:
            return {
                "is_anomaly": False,
                "anomaly_score": 0.0,
                "reason": "invalid_cpf"
            }
        
        # Check if nodes exist
        sender_exists = sender in self.graph.nodes
        receiver_exists = receiver in self.graph.nodes
        
        if not sender_exists and not receiver_exists:
            return {
                "is_anomaly": False,
                "anomaly_score": 0.0,
                "reason": "new_users"
            }
        
        # Extract features
        sender_features = self.extract_node_features(sender) if sender_exists else self._empty_features()
        receiver_features = self.extract_node_features(receiver) if receiver_exists else self._empty_features()
        edge_features = self.extract_edge_features(sender, receiver)
        
        # Detect anomalies
        anomalies = []
        anomaly_score = 0.0
        
        # High degree sender (potential money mule)
        if sender_features["node_degree"] > threshold_degree:
            anomalies.append("high_degree_sender")
            anomaly_score += 0.3
        
        # High degree receiver (potential money mule)
        if receiver_features["node_degree"] > threshold_degree:
            anomalies.append("high_degree_receiver")
            anomaly_score += 0.3
        
        # New connection between high-degree nodes
        if edge_features["transaction_count"] == 0 and sender_features["node_degree"] > 10 and receiver_features["node_degree"] > 10:
            anomalies.append("new_high_degree_connection")
            anomaly_score += 0.4
        
        # Shortest path length (circular transactions)
        if edge_features["shortest_path_length"] == 2:
            anomalies.append("circular_transaction")
            anomaly_score += 0.2
        
        # Cap anomaly score
        anomaly_score = min(anomaly_score, 1.0)
        
        return {
            "is_anomaly": anomaly_score > 0.5,
            "anomaly_score": anomaly_score,
            "anomalies": anomalies,
            "sender_features": sender_features,
            "receiver_features": receiver_features,
            "edge_features": edge_features
        }
    
    def _empty_features(self) -> Dict:
        """Return empty feature dictionary."""
        return {
            "node_degree": 0,
            "out_degree": 0,
            "in_degree": 0,
            "clustering_coefficient": 0.0,
            "total_sent": 0.0,
            "total_received": 0.0,
            "net_flow": 0.0,
            "page_rank": 0.0
        }
