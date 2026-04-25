"""
Configuration management for the fraud detection system.
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # API Settings
    app_name: str = "Fraud Detection API"
    app_version: str = "1.0.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    
    # Model Settings
    model_path: str = "models/fraud_model.pkl"
    feature_names_path: str = "models/feature_names.json"
    threshold: float = 0.5
    
    # Feature Engineering Settings
    max_history_size: int = 100  # Max transactions per CPF in cache
    
    # Logging Settings
    log_level: str = "INFO"
    audit_log_path: str = "logs/audit.log"
    
    # Data Settings
    dataset_path: str = "dataset_transacoes_expanded.csv"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
