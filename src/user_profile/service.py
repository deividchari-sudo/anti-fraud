"""
User Profile Service for Behavioral Profiling.
Calculates and manages user behavioral profiles from transaction history.
"""

from datetime import datetime
from typing import List, Dict, Optional, Any
import numpy as np
from collections import Counter, defaultdict

from .models import (
    UserProfile,
    Statistics,
    ValueStatistics,
    HourStatistics,
    FrequencyStatistics,
    Destinations,
    DestinationInfo,
    Canais,
    CanalInfo,
    Produtos,
    ProdutoInfo,
    AnomalyResult,
    BehavioralAnalysis
)
from .anomaly_detector import AnomalyDetector
from ..repositories import UserProfileRepository
from ..crypto import get_cpf_hasher
from .temporal_features import TemporalFeatureExtractor
from .clustering import UserClusterer


class UserProfileService:
    """Service for managing user behavioral profiles."""
    
    MIN_TRANSACTIONS_FOR_PROFILE = 30  # Minimum transactions to build reliable profile
    
    def __init__(self, repository: UserProfileRepository):
        """
        Initialize user profile service.
        
        Args:
            repository: Repository for profile persistence
        """
        self.repository = repository
        self.anomaly_detector = AnomalyDetector()
        self.cpf_hasher = get_cpf_hasher()
        self.temporal_extractor = TemporalFeatureExtractor()
        self.clusterer = UserClusterer(n_clusters=5)
    
    def _hash_cpf(self, cpf: str) -> str:
        """Hash CPF for storage (LGPD compliance)."""
        return self.cpf_hasher.hash_cpf(cpf)
    
    def train_clustering(self) -> Dict:
        """
        Train clustering model on all existing profiles.
        
        Returns:
            Dictionary with clustering information
        """
        # Get all profiles
        all_profiles = self.repository.list_all_profiles()
        
        if not all_profiles:
            return {"error": "No profiles available for clustering"}
        
        # Convert to list of dicts
        profile_list = list(all_profiles.values())
        
        # Train clustering
        clustering_info = self.clusterer.fit(profile_list)
        
        # Update profiles with cluster labels
        for cpf, profile_data in all_profiles.items():
            cluster_label = self.clusterer.predict(profile_data)
            # Update profile with cluster_id
            profile_data["cluster_id"] = cluster_label
            self.repository.save_profile(cpf, profile_data)
        
        return clustering_info
    
    def assign_cluster(self, profile: UserProfile) -> int:
        """
        Assign cluster to a user profile.
        
        Args:
            profile: UserProfile object
            
        Returns:
            Cluster ID
        """
        if not self.clusterer.is_fitted:
            return 0
        
        profile_dict = profile.dict()
        cluster_id = self.clusterer.predict(profile_dict)
        return cluster_id
    
    def calculate_profile(self, transactions: List[dict], cpf: str) -> UserProfile:
        """
        Calculate user profile from transaction history.
        
        Args:
            transactions: List of transaction dictionaries
            cpf: User CPF (will be hashed for storage)
            
        Returns:
            UserProfile with calculated statistics
        """
        if not transactions:
            raise ValueError("Cannot calculate profile from empty transaction list")
        
        # Hash CPF for storage
        hashed_cpf = self._hash_cpf(cpf)
        
        # Extract values
        valores = [float(t.get("valor", 0)) for t in transactions if t.get("valor", 0) > 0]
        horas = []
        timestamps = []
        
        for t in transactions:
            timestamp = t.get("timestamp", "")
            if timestamp:
                try:
                    dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    horas.append(dt.hour)
                    timestamps.append(dt)
                except (ValueError, TypeError):
                    pass
        
        # Calculate value statistics
        valor_stats = self._calculate_value_statistics(valores)
        
        # Calculate hour statistics
        hora_stats = self._calculate_hour_statistics(horas)
        
        # Calculate frequency statistics
        freq_stats = self._calculate_frequency_statistics(timestamps)
        
        # Calculate destinations
        destinations = self._calculate_destinations(transactions)
        
        # Calculate channels
        canais = self._calculate_canais(transactions)
        
        # Calculate products
        produtos = self._calculate_produtos(transactions)
        
        now = datetime.utcnow()
        
        # Extract temporal features
        temporal_features = self.temporal_extractor.extract_all_temporal_features(transactions)
        
        profile = UserProfile(
            cpf=hashed_cpf,  # Store hashed CPF
            created_at=now,
            last_updated=now,
            transaction_count=len(transactions),
            is_cold_start=len(transactions) < self.MIN_TRANSACTIONS_FOR_PROFILE,
            statistics=Statistics(
                valor=valor_stats,
                hora=hora_stats,
                frequencia=freq_stats
            ),
            destinations=destinations,
            canais=canais,
            produtos=produtos,
            temporal_features=temporal_features
        )
        
        # Assign cluster if clustering model is fitted
        if self.clusterer.is_fitted:
            profile.cluster_id = self.assign_cluster(profile)
        
        return profile
    
    def _calculate_value_statistics(self, valores: List[float]) -> ValueStatistics:
        """Calculate statistical measures for transaction values."""
        if not valores:
            return ValueStatistics(
                mean=0.0, std=0.0, median=0.0,
                p25=0.0, p75=0.0, p95=0.0,
                min=0.0, max=0.0
            )
        
        valores_array = np.array(valores)
        
        return ValueStatistics(
            mean=float(np.mean(valores_array)),
            std=float(np.std(valores_array)),
            median=float(np.median(valores_array)),
            p25=float(np.percentile(valores_array, 25)),
            p75=float(np.percentile(valores_array, 75)),
            p95=float(np.percentile(valores_array, 95)),
            min=float(np.min(valores_array)),
            max=float(np.max(valores_array))
        )
    
    def _calculate_hour_statistics(self, horas: List[int]) -> HourStatistics:
        """Calculate statistical measures for transaction hours."""
        if not horas:
            return HourStatistics(
                mean=12.0, std=4.5, median=12.0,
                p25=10.0, p75=14.0
            )
        
        horas_array = np.array(horas)
        
        return HourStatistics(
            mean=float(np.mean(horas_array)),
            std=float(np.std(horas_array)),
            median=float(np.median(horas_array)),
            p25=float(np.percentile(horas_array, 25)),
            p75=float(np.percentile(horas_array, 75))
        )
    
    def _calculate_frequency_statistics(self, timestamps: List[datetime]) -> FrequencyStatistics:
        """Calculate frequency statistics."""
        if not timestamps:
            return FrequencyStatistics(
                transactions_per_day_mean=0.0,
                transactions_per_day_std=0.0,
                days_active=0
            )
        
        # Calculate unique days
        unique_days = len(set(dt.date() for dt in timestamps))
        
        # Calculate transactions per day
        if unique_days > 0:
            transactions_per_day = len(timestamps) / unique_days
        else:
            transactions_per_day = 0.0
        
        # Simple std estimation (will be refined with more data)
        std_estimate = transactions_per_day * 0.3  # 30% variation assumption
        
        return FrequencyStatistics(
            transactions_per_day_mean=round(transactions_per_day, 2),
            transactions_per_day_std=round(std_estimate, 2),
            days_active=unique_days
        )
    
    def _calculate_destinations(self, transactions: List[dict]) -> Destinations:
        """Calculate destination statistics."""
        cpf_counts = defaultdict(int)
        banco_counts = defaultdict(int)
        cpf_last_seen = {}
        
        for t in transactions:
            receiver = t.get("receiver", {})
            cpf = receiver.get("cpfReceiver")
            banco = receiver.get("banco")
            timestamp = t.get("timestamp", "")
            
            if cpf:
                cpf_counts[cpf] += 1
                if timestamp:
                    try:
                        dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                        cpf_last_seen[cpf] = dt
                    except (ValueError, TypeError):
                        pass
            
            if banco:
                banco_counts[str(banco)] += 1
        
        # Get top 5 common CPFs
        top_cpfs = dict(sorted(cpf_counts.items(), key=lambda x: x[1], reverse=True)[:5])
        
        # Get top 5 common bancos
        top_bancos = dict(sorted(banco_counts.items(), key=lambda x: x[1], reverse=True)[:5])
        
        # Convert to DestinationInfo
        common_cpfs = {
            cpf: DestinationInfo(count=count, last_seen=cpf_last_seen.get(cpf))
            for cpf, count in top_cpfs.items()
        }
        
        common_bancos = {
            banco: {"count": count}
            for banco, count in top_bancos.items()
        }
        
        return Destinations(
            common_cpfs=common_cpfs,
            common_bancos=common_bancos
        )
    
    def _calculate_canais(self, transactions: List[dict]) -> Canais:
        """Calculate channel usage statistics."""
        canal_counts = Counter()
        
        for t in transactions:
            canal = t.get("canal", "").lower()
            if canal:
                canal_counts[canal] += 1
        
        total = sum(canal_counts.values()) if canal_counts else 1
        
        # Default values for missing channels
        app_count = canal_counts.get("app", 0)
        web_count = canal_counts.get("web", 0)
        api_count = canal_counts.get("api", 0)
        
        return Canais(
            app=CanalInfo(count=app_count, percentage=round(app_count / total, 2)),
            web=CanalInfo(count=web_count, percentage=round(web_count / total, 2)),
            api=CanalInfo(count=api_count, percentage=round(api_count / total, 2))
        )
    
    def _calculate_produtos(self, transactions: List[dict]) -> Produtos:
        """Calculate product usage statistics."""
        produto_counts = Counter()
        
        for t in transactions:
            produto = t.get("produto", "").lower()
            if produto:
                produto_counts[produto] += 1
        
        total = sum(produto_counts.values()) if produto_counts else 1
        
        # Default values for missing products
        pix_count = produto_counts.get("pix", 0)
        ted_count = produto_counts.get("ted", 0)
        boleto_count = produto_counts.get("boleto", 0)
        autenticacao_count = produto_counts.get("autenticacao", 0)
        
        produtos = Produtos(
            pix=ProdutoInfo(count=pix_count, percentage=round(pix_count / total, 2)),
            ted=ProdutoInfo(count=ted_count, percentage=round(ted_count / total, 2)),
            boleto=ProdutoInfo(count=boleto_count, percentage=round(boleto_count / total, 2))
        )
        
        if autenticacao_count > 0:
            produtos.autenticacao = ProdutoInfo(
                count=autenticacao_count,
                percentage=round(autenticacao_count / total, 2)
            )
        
        return produtos
    
    def get_or_create_profile(self, cpf: str) -> UserProfile:
        """
        Get existing profile or create global profile (cold start).
        
        Args:
            cpf: User CPF (will be hashed for lookup)
            
        Returns:
            UserProfile (existing or global profile for cold start)
        """
        hashed_cpf = self._hash_cpf(cpf)
        profile = self.repository.load_profile(hashed_cpf)
        
        if profile is None:
            # Cold start: use global profile
            global_profile = self.repository.load_global_profile()
            if global_profile:
                profile = UserProfile(
                    cpf=hashed_cpf,  # Use hashed CPF
                    created_at=datetime.utcnow(),
                    last_updated=datetime.utcnow(),
                    transaction_count=0,
                    is_cold_start=True,
                    statistics=global_profile.statistics,
                    destinations=global_profile.destinations,
                    canais=global_profile.canais,
                    produtos=global_profile.produtos
                )
            else:
                # Fallback: create minimal profile
                profile = self._create_minimal_profile(hashed_cpf)
        
        return profile
    
    def _create_minimal_profile(self, cpf: str) -> UserProfile:
        """Create minimal profile when no global profile exists."""
        now = datetime.utcnow()
        
        return UserProfile(
            cpf=cpf,
            created_at=now,
            last_updated=now,
            transaction_count=0,
            is_cold_start=True,
            statistics=Statistics(
                valor=ValueStatistics(
                    mean=450.0, std=250.0, median=350.0,
                    p25=150.0, p75=600.0, p95=1200.0,
                    min=0.0, max=5000.0
                ),
                hora=HourStatistics(
                    mean=13.0, std=4.5, median=13.0,
                    p25=10.0, p75=16.0
                ),
                frequencia=FrequencyStatistics(
                    transactions_per_day_mean=1.8,
                    transactions_per_day_std=1.2,
                    days_active=0
                )
            ),
            destinations=Destinations(),
            canais=Canais(
                app=CanalInfo(count=0, percentage=0.65),
                web=CanalInfo(count=0, percentage=0.25),
                api=CanalInfo(count=0, percentage=0.10)
            ),
            produtos=Produtos(
                pix=ProdutoInfo(count=0, percentage=0.55),
                ted=ProdutoInfo(count=0, percentage=0.25),
                boleto=ProdutoInfo(count=0, percentage=0.15)
            )
        )
    
    def update_profile(self, cpf: str, transaction: dict):
        """
        Update profile with new transaction (incremental update).
        
        Args:
            cpf: User CPF (will be hashed for lookup)
            transaction: New transaction
        """
        hashed_cpf = self._hash_cpf(cpf)
        profile = self.repository.load_profile(hashed_cpf)
        
        if profile is None:
            # Create new profile from single transaction
            profile = self.calculate_profile([transaction], cpf)
        else:
            # Incremental update (simplified - in production would use more sophisticated methods)
            profile.transaction_count += 1
            profile.last_updated = datetime.utcnow()
            
            # Update cold start status
            if profile.transaction_count >= self.MIN_TRANSACTIONS_FOR_PROFILE:
                profile.is_cold_start = False
        
        self.repository.save_profile(hashed_cpf, profile)
    
    def detect_anomaly(self, transaction: dict, profile: UserProfile) -> AnomalyResult:
        """
        Detect anomalies in transaction based on user profile.
        
        Args:
            transaction: Transaction dictionary
            profile: User profile
            
        Returns:
            AnomalyResult with detected anomalies
        """
        return self.anomaly_detector.detect(transaction, profile)
    
    def analyze_transaction(self, transaction: dict) -> BehavioralAnalysis:
        """
        Analyze transaction with behavioral profiling.
        
        Args:
            transaction: Transaction dictionary
            
        Returns:
            BehavioralAnalysis for API response
        """
        cpf = transaction.get("sender", {}).get("cpfSender")
        
        if not cpf:
            return BehavioralAnalysis(
                has_profile=False,
                transaction_count=0,
                is_cold_start=True,
                is_anomaly=False,
                anomaly_score=0.0,
                anomalies=[]
            )
        
        profile = self.get_or_create_profile(cpf)
        anomaly_result = self.detect_anomaly(transaction, profile)
        
        return BehavioralAnalysis(
            has_profile=not profile.is_cold_start,
            transaction_count=profile.transaction_count,
            is_cold_start=profile.is_cold_start,
            is_anomaly=anomaly_result.is_anomaly,
            anomaly_score=anomaly_result.anomaly_score,
            anomalies=anomaly_result.anomalies
        )
