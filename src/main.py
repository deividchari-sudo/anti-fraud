from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import time
import json
from typing import Dict, Any, List, Optional

from src.models import TransactionRequest, FraudPrediction, HealthResponse
from src.feature_engineering import FeatureEngineer
from src.model import FraudDetectionModel
from src.repositories import ModelRepository, JoblibModelRepository
from src.rule_engine import RuleParser, RuleEvaluator, Rule
from config import settings

app = FastAPI(
    title="Fraud Detection API",
    description="API para detecção de fraude em tempo real para transações bancárias",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
feature_engineer = FeatureEngineer()

# Rule Engine - for natural language rules
rule_parser = RuleParser()
rule_evaluator = RuleEvaluator()

# Dependency injection for model repository
model_repository = JoblibModelRepository(
    settings.model_path,
    settings.feature_names_path
)

model = FraudDetectionModel(
    model_path=settings.model_path,
    model_repository=model_repository
)
model_loaded = False

# Try to load model on startup
try:
    model.load_model()
    feature_engineer.set_feature_names(model.feature_names)
    model_loaded = True
    print("Model loaded successfully on startup")
except Exception as e:
    print(f"Could not load model on startup: {e}")
    print("Please train the model first using train_model.py")


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        model_loaded=model_loaded,
        version="1.0.0"
    )


@app.post("/predict", response_model=FraudPrediction)
async def predict_fraud(request: TransactionRequest):
    """Predict if a transaction is fraudulent with audit explanation."""
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    start_time = time.time()
    
    try:
        # First, evaluate against rules
        transaction_dict = request.payload.model_dump()
        rule_result = rule_evaluator.evaluate(transaction_dict)
        
        # If rules marked as fraud, return immediately
        if rule_result['is_fraud_by_rules']:
            return FraudPrediction(
                transaction_id=request.payload.id,
                fraud_probability=1.0,
                is_fraud=True,
                confidence="high",
                processing_time_ms=float((time.time() - start_time) * 1000),
                timestamp=datetime.now().isoformat(),
                explanation={
                    "type": "rule_based",
                    "matched_rules": rule_result['matched_rules'],
                    "reason": "Transaction matched one or more fraud rules"
                }
            )
        
        # If rules marked as legitimate, return immediately
        if rule_result['is_legitimate_by_rules']:
            return FraudPrediction(
                transaction_id=request.payload.id,
                fraud_probability=0.0,
                is_fraud=False,
                confidence="high",
                processing_time_ms=float((time.time() - start_time) * 1000),
                timestamp=datetime.now().isoformat(),
                explanation={
                    "type": "rule_based",
                    "matched_rules": rule_result['matched_rules'],
                    "reason": "Transaction matched whitelist rule"
                }
            )
        
        # Extract features from payload
        features = feature_engineer.extract_features(transaction_dict)
        
        # Prepare DataFrame
        df = feature_engineer.prepare_dataframe(features)
        
        # Make prediction with explanation
        fraud_probability, is_fraud, explanation = model.predict(df, transaction_id=request.payload.id)
        
        # Determine confidence level
        if fraud_probability >= 0.8:
            confidence = "high"
        elif fraud_probability >= 0.5:
            confidence = "medium"
        else:
            confidence = "low"
        
        processing_time = (time.time() - start_time) * 1000
        
        response = FraudPrediction(
            transaction_id=request.payload.id,
            fraud_probability=float(fraud_probability),
            is_fraud=bool(is_fraud),
            confidence=confidence,
            processing_time_ms=float(processing_time),
            timestamp=datetime.now().isoformat(),
            explanation=explanation
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_fraud_batch(requests: list[TransactionRequest]):
    """Predict fraud for multiple transactions (batch processing)."""
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first."
        )
    
    start_time = time.time()
    results = []
    
    try:
        for request in requests:
            payload_dict = request.payload.model_dump()
            features = feature_engineer.extract_features(payload_dict)
            features_df = feature_engineer.prepare_dataframe(features)
            fraud_probability, is_fraud, explanation = model.predict(features_df, transaction_id=request.payload.id)
            
            confidence = "high" if fraud_probability >= 0.8 else "medium" if fraud_probability >= 0.5 else "low"
            
            results.append({
                "transaction_id": request.payload.id,
                "fraud_probability": round(float(fraud_probability), 4),
                "is_fraud": bool(is_fraud),
                "confidence": confidence,
                "explanation": explanation
            })
        
        processing_time_ms = (time.time() - start_time) * 1000
        
        return {
            "results": results,
            "total_transactions": len(requests),
            "processing_time_ms": round(processing_time_ms, 2),
            "avg_time_per_transaction": round(processing_time_ms / len(requests), 2)
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error during batch prediction: {str(e)}"
        )


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model."""
    if not model_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded."
        )
    
    try:
        feature_importance = model.get_feature_importance()
        # Convert numpy types to Python native types
        top_features = {
            k: float(v) for k, v in list(feature_importance.items())[:10]
        }
        
        return {
            "model_type": "XGBoost",
            "feature_count": len(model.feature_names),
            "threshold": float(model.threshold),
            "top_features": top_features
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error getting model info: {str(e)}"
        )


from pydantic import BaseModel


class RuleCreateRequest(BaseModel):
    rule_text: str
    name: Optional[str] = None
    description: Optional[str] = None


# Rule Management Endpoints

@app.post("/rules")
async def create_rule(request: RuleCreateRequest):
    """Create a new rule from natural language text."""
    try:
        rule = rule_parser.parse(request.rule_text, request.name, request.description)
        rule_evaluator.add_rule(rule)
        
        return {
            "rule_id": rule.id,
            "name": rule.name,
            "description": rule.description,
            "original_text": rule.original_text,
            "action": rule.action.value,
            "conditions_count": len(rule.conditions),
            "enabled": rule.enabled,
            "priority": rule.priority
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error parsing rule: {str(e)}")


@app.get("/rules")
async def list_rules():
    """List all rules."""
    rules = rule_evaluator.get_all_rules()
    
    return {
        "total_rules": len(rules),
        "rules": [
            {
                "rule_id": rule.id,
                "name": rule.name,
                "description": rule.description,
                "original_text": rule.original_text,
                "action": rule.action.value,
                "conditions_count": len(rule.conditions),
                "enabled": rule.enabled,
                "priority": rule.priority
            }
            for rule in rules
        ]
    }


@app.get("/rules/{rule_id}")
async def get_rule(rule_id: str):
    """Get a specific rule by ID."""
    rule = rule_evaluator.get_rule(rule_id)
    
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    
    return {
        "rule_id": rule.id,
        "name": rule.name,
        "description": rule.description,
        "original_text": rule.original_text,
        "action": rule.action.value,
        "conditions": [
            {
                "field": condition.field.value,
                "operator": condition.operator.value,
                "value": condition.value
            }
            for condition in rule.conditions
        ],
        "enabled": rule.enabled,
        "priority": rule.priority
    }


@app.delete("/rules/{rule_id}")
async def delete_rule(rule_id: str):
    """Delete a rule by ID."""
    success = rule_evaluator.remove_rule(rule_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Rule not found")
    
    return {"message": "Rule deleted successfully"}


@app.post("/rules/{rule_id}/enable")
async def enable_rule(rule_id: str):
    """Enable a rule by ID."""
    rule = rule_evaluator.get_rule(rule_id)
    
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    
    rule.enabled = True
    
    return {"message": "Rule enabled successfully"}


@app.post("/rules/{rule_id}/disable")
async def disable_rule(rule_id: str):
    """Disable a rule by ID."""
    rule = rule_evaluator.get_rule(rule_id)
    
    if not rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    
    rule.enabled = False
    
    return {"message": "Rule disabled successfully"}


@app.post("/rules/evaluate")
async def evaluate_rules(transaction: dict):
    """Evaluate a transaction against all rules."""
    result = rule_evaluator.evaluate(transaction)
    return result


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
