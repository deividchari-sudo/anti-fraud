"""
User clustering for behavioral profiling.
Implements K-means and DBSCAN clustering to segment users into behavioral groups.
"""

from typing import List, Dict, Optional
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score


class UserClusterer:
    """Clusters users into behavioral groups for anomaly detection."""
    
    def __init__(self, n_clusters: int = 5, clustering_method: str = "kmeans"):
        """
        Initialize user clusterer.
        
        Args:
            n_clusters: Number of behavioral clusters (for K-means)
            clustering_method: "kmeans" or "dbscan"
        """
        self.n_clusters = n_clusters
        self.clustering_method = clustering_method
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.dbscan = DBSCAN(eps=1.5, min_samples=3, metric='euclidean')
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.silhouette_score = None
    
    def _extract_features(self, user_profiles: List[Dict]) -> np.ndarray:
        """
        Extract features from user profiles for clustering.
        
        Args:
            user_profiles: List of user profile dictionaries
            
        Returns:
            Feature matrix (n_users, n_features)
        """
        features = []
        
        for profile in user_profiles:
            stats = profile.get("statistics", {})
            valor_stats = stats.get("valor", {})
            hora_stats = stats.get("hora", {})
            freq_stats = stats.get("frequencia", {})
            
            # Extract key features
            feature_vector = [
                valor_stats.get("mean", 0),
                valor_stats.get("std", 0),
                valor_stats.get("median", 0),
                valor_stats.get("p95", 0),
                hora_stats.get("mean", 0),
                hora_stats.get("std", 0),
                freq_stats.get("transactions_per_day_mean", 0),
                profile.get("transaction_count", 0)
            ]
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def fit(self, user_profiles: List[Dict]) -> Dict:
        """
        Fit clustering model on user profiles.
        
        Args:
            user_profiles: List of user profile dictionaries
            
        Returns:
            Dictionary with cluster information
        """
        if len(user_profiles) < 2:
            # Not enough profiles for clustering
            return {
                "n_clusters": 1,
                "cluster_labels": [0] * len(user_profiles),
                "cluster_centers": None,
                "cluster_descriptions": {"0": "insufficient_data"},
                "silhouette_score": None,
                "method": self.clustering_method
            }
        
        # Extract features
        features = self._extract_features(user_profiles)
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Fit clustering model
        if self.clustering_method == "dbscan":
            # Use DBSCAN for density-based clustering
            self.dbscan.fit(features_scaled)
            self.is_fitted = True
            
            # DBSCAN labels: -1 = noise/outlier, >=0 = cluster
            labels = self.dbscan.labels_
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            # Calculate silhouette score (ignore noise points)
            if n_clusters > 1 and n_clusters < len(labels):
                mask = labels != -1
                if np.sum(mask) > 1:
                    self.silhouette_score = silhouette_score(features_scaled[mask], labels[mask])
                else:
                    self.silhouette_score = None
            else:
                self.silhouette_score = None
            
            # Generate cluster descriptions
            cluster_descriptions = self._generate_dbscan_descriptions(
                features_scaled, 
                labels
            )
            
            return {
                "n_clusters": n_clusters,
                "cluster_labels": labels.tolist(),
                "cluster_centers": None,  # DBSCAN doesn't have centers
                "cluster_descriptions": cluster_descriptions,
                "silhouette_score": self.silhouette_score,
                "method": "dbscan",
                "noise_count": int(np.sum(labels == -1))
            }
        else:
            # Use K-means (default)
            if len(user_profiles) < self.n_clusters:
                return {
                    "n_clusters": 1,
                    "cluster_labels": [0] * len(user_profiles),
                    "cluster_centers": None,
                    "cluster_descriptions": {"0": "all_users"},
                    "silhouette_score": None,
                    "method": "kmeans"
                }
            
            self.kmeans.fit(features_scaled)
            self.is_fitted = True
            
            # Calculate silhouette score
            if self.n_clusters > 1:
                self.silhouette_score = silhouette_score(features_scaled, self.kmeans.labels_)
            else:
                self.silhouette_score = None
            
            # Generate cluster descriptions
            cluster_descriptions = self._generate_cluster_descriptions(
                features_scaled, 
                self.kmeans.labels_,
                self.kmeans.cluster_centers_
            )
            
            return {
                "n_clusters": self.n_clusters,
                "cluster_labels": self.kmeans.labels_.tolist(),
                "cluster_centers": self.kmeans.cluster_centers_.tolist(),
                "cluster_descriptions": cluster_descriptions,
                "silhouette_score": self.silhouette_score,
                "method": "kmeans"
            }
    
    def predict(self, user_profile: Dict) -> int:
        """
        Predict cluster for a single user profile.
        
        Args:
            user_profile: User profile dictionary
            
        Returns:
            Cluster label (-1 for noise in DBSCAN)
        """
        if not self.is_fitted:
            return 0
        
        # Extract features
        features = self._extract_features([user_profile])
        features_scaled = self.scaler.transform(features)
        
        # Predict cluster
        if self.clustering_method == "dbscan":
            # DBSCAN predict (using nearest neighbor approximation)
            # For simplicity, we'll use K-means as fallback for prediction
            # since DBSCAN doesn't have native predict
            if hasattr(self.kmeans, 'cluster_centers_'):
                cluster = self.kmeans.predict(features_scaled)[0]
            else:
                cluster = 0
        else:
            cluster = self.kmeans.predict(features_scaled)[0]
        
        return int(cluster)
    
    def _generate_cluster_descriptions(self, features: np.ndarray, 
                                       labels: np.ndarray, 
                                       centers: np.ndarray) -> Dict:
        """
        Generate human-readable descriptions for each cluster.
        
        Args:
            features: Scaled feature matrix
            labels: Cluster labels
            centers: Cluster centers
            
        Returns:
            Dictionary mapping cluster IDs to descriptions
        """
        descriptions = {}
        
        # Feature names for description
        feature_names = [
            "avg_value", "std_value", "median_value", "p95_value",
            "avg_hour", "std_hour", "transactions_per_day", "total_transactions"
        ]
        
        for cluster_id in range(self.n_clusters):
            # Get cluster center
            center = centers[cluster_id]
            
            # Get cluster members
            cluster_mask = labels == cluster_id
            cluster_size = np.sum(cluster_mask)
            
            # Generate description based on center values
            description_parts = []
            
            if center[0] > 0.5:  # High average value
                description_parts.append("high_value")
            elif center[0] < -0.5:  # Low average value
                description_parts.append("low_value")
            
            if center[4] > 0.5:  # Late transactions (high hour)
                description_parts.append("night_user")
            elif center[4] < -0.5:  # Early transactions (low hour)
                description_parts.append("morning_user")
            
            if center[6] > 0.5:  # High frequency
                description_parts.append("high_frequency")
            elif center[6] < -0.5:  # Low frequency
                description_parts.append("low_frequency")
            
            if not description_parts:
                description_parts.append("average_user")
            
            descriptions[str(cluster_id)] = {
                "name": "_".join(description_parts),
                "size": int(cluster_size),
                "characteristics": {
                    feature_names[i]: float(center[i]) 
                    for i in range(len(feature_names))
                }
            }
        
        return descriptions
    
    def _generate_dbscan_descriptions(self, features: np.ndarray, 
                                     labels: np.ndarray) -> Dict:
        """
        Generate human-readable descriptions for DBSCAN clusters.
        
        Args:
            features: Scaled feature matrix
            labels: Cluster labels (-1 = noise, >=0 = cluster)
            
        Returns:
            Dictionary mapping cluster IDs to descriptions
        """
        descriptions = {}
        
        # Feature names for description
        feature_names = [
            "avg_value", "std_value", "median_value", "p95_value",
            "avg_hour", "std_hour", "transactions_per_day", "total_transactions"
        ]
        
        # Handle noise cluster (-1)
        if -1 in labels:
            noise_count = np.sum(labels == -1)
            descriptions["-1"] = {
                "name": "outlier",
                "size": int(noise_count),
                "characteristics": "Noise/outlier - does not belong to any cluster"
            }
        
        # Handle actual clusters
        unique_labels = set(labels)
        unique_labels.discard(-1)
        
        for cluster_id in unique_labels:
            # Get cluster members
            cluster_mask = labels == cluster_id
            cluster_size = np.sum(cluster_mask)
            cluster_features = features[cluster_mask]
            
            # Calculate cluster center (mean of features)
            center = np.mean(cluster_features, axis=0)
            
            # Generate description based on center values
            description_parts = []
            
            if center[0] > 0.5:  # High average value
                description_parts.append("high_value")
            elif center[0] < -0.5:  # Low average value
                description_parts.append("low_value")
            
            if center[4] > 0.5:  # Late transactions (high hour)
                description_parts.append("night_user")
            elif center[4] < -0.5:  # Early transactions (low hour)
                description_parts.append("morning_user")
            
            if center[6] > 0.5:  # High frequency
                description_parts.append("high_frequency")
            elif center[6] < -0.5:  # Low frequency
                description_parts.append("low_frequency")
            
            if not description_parts:
                description_parts.append("average_user")
            
            descriptions[str(cluster_id)] = {
                "name": "_".join(description_parts),
                "size": int(cluster_size),
                "characteristics": {
                    feature_names[i]: float(center[i]) 
                    for i in range(len(feature_names))
                }
            }
        
        return descriptions
    
    def get_cluster_anomaly_score(self, user_profile: Dict, 
                                  cluster_id: int) -> float:
        """
        Calculate anomaly score based on distance to cluster center.
        
        Args:
            user_profile: User profile dictionary
            cluster_id: Cluster ID to compare against
            
        Returns:
            Anomaly score (0.0 to 1.0, higher = more anomalous)
        """
        if not self.is_fitted:
            return 0.0
        
        # Extract features
        features = self._extract_features([user_profile])
        features_scaled = self.scaler.transform(features)
        
        # Get cluster center
        center = self.kmeans.cluster_centers_[cluster_id]
        
        # Calculate distance to center
        distance = np.linalg.norm(features_scaled[0] - center)
        
        # Normalize to 0-1 range (assuming max distance ~5 for standardized data)
        anomaly_score = min(distance / 5.0, 1.0)
        
        return float(anomaly_score)
