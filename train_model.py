import pandas as pd
import json
from pathlib import Path

from src.model import FraudDetectionModel
from src.feature_engineering import FeatureEngineer
from src.repositories import TransactionRepository, CSVTransactionRepository


def load_and_prepare_data(repository: TransactionRepository = None) -> pd.DataFrame:
    """Load and prepare data for training."""
    # Use dependency injection or default repository
    repo = repository or CSVTransactionRepository('dataset_transacoes_expanded.csv')
    
    print("Loading data from dataset_transacoes_expanded.csv...")
    df = repo.load_dataset()
    
    print(f"Total samples: {len(df)}")
    print(f"Fraud distribution:\n{df['fraudResult'].value_counts()}")
    
    # Parse JSON payload
    print("Parsing JSON payloads...")
    payloads = []
    for payload_str in df['payload']:
        try:
            payload = json.loads(payload_str)
            payloads.append(payload)
        except:
            payloads.append({})
    
    # Create feature engineer
    feature_engineer = FeatureEngineer()
    
    # Extract features for all samples
    print("Extracting features...")
    features_list = []
    for payload in payloads:
        features = feature_engineer.extract_features(payload)
        features_list.append(features)
    
    # Create features DataFrame
    features_df = pd.DataFrame(features_list)
    
    # One-hot encode categorical variables
    print("Encoding categorical variables...")
    categorical_cols = ['canal', 'produto', 'jornada', 'direcao']
    for col in categorical_cols:
        if col in features_df.columns:
            dummies = pd.get_dummies(features_df[col], prefix=col)
            features_df = pd.concat([features_df, dummies], axis=1)
            features_df = features_df.drop(col, axis=1)
    
    # Combine with target
    df_prepared = pd.concat([features_df, df['fraudResult']], axis=1)
    
    # Add original payload for reference
    df_prepared['payload'] = df['payload']
    
    print(f"Prepared dataset shape: {df_prepared.shape}")
    print(f"Features: {list(df_prepared.columns)}")
    
    return df_prepared


def main():
    """Main training function."""
    # Load and prepare data
    df = load_and_prepare_data()
    
    # Initialize model (don't load existing model for training)
    model = FraudDetectionModel(model_path=None)
    
    # Train model
    metrics = model.train(df)
    
    # Print feature importance
    print("\n=== Top 10 Feature Importance ===")
    feature_importance = model.get_feature_importance()
    for i, (feature, importance) in enumerate(list(feature_importance.items())[:10], 1):
        print(f"{i}. {feature}: {importance:.4f}")
    
    # Update feature engineer with model's feature names
    feature_engineer = FeatureEngineer()
    feature_engineer.set_feature_names(model.feature_names)
    
    print("\n=== Training Complete ===")
    print(f"Model saved to: {model.model_path}")
    print(f"Feature names saved for inference")


if __name__ == "__main__":
    main()
