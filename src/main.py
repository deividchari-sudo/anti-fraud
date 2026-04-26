import time
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import settings
from src.feature_engineering import FeatureEngineer
from src.model import FraudDetectionModel
from src.models import FraudPrediction, HealthResponse, TransactionRequest
from src.repositories import JoblibModelRepository, SQLiteUserProfileRepository
from src.rule_engine import RuleEvaluator, RuleParser
from src.user_profile import UserProfileService
from src.crypto import get_cpf_hasher

app = FastAPI(
    title="Fraud Detection API",
    description="API para detecção de fraude em tempo real para transações bancárias",
    version="1.0.0",
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
    settings.model_path, settings.feature_names_path
)

model = FraudDetectionModel(
    model_path=settings.model_path, model_repository=model_repository
)
model_loaded = False

# Behavioral Profiling Service
user_profile_repository = SQLiteUserProfileRepository()
user_profile_service = UserProfileService(user_profile_repository)

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
    return HealthResponse(status="healthy", model_loaded=model_loaded, version="1.0.0")


@app.post("/predict", response_model=FraudPrediction)
async def predict_fraud(request: TransactionRequest):
    """Predict if a transaction is fraudulent with audit explanation."""
    if not model_loaded:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Please train the model first."
        )

    start_time = time.time()

    try:
        # First, evaluate against rules
        transaction_dict = request.payload.model_dump()
        rule_result = rule_evaluator.evaluate(transaction_dict)

        # If rules marked as fraud, return immediately
        if rule_result["is_fraud_by_rules"]:
            return FraudPrediction(
                transaction_id=request.payload.id,
                fraud_probability=1.0,
                is_fraud=True,
                confidence="high",
                processing_time_ms=float((time.time() - start_time) * 1000),
                timestamp=datetime.now().isoformat(),
                explanation={
                    "type": "rule_based",
                    "matched_rules": rule_result["matched_rules"],
                    "reason": "Transaction matched one or more fraud rules",
                },
            )

        # If rules marked as legitimate, return immediately
        if rule_result["is_legitimate_by_rules"]:
            return FraudPrediction(
                transaction_id=request.payload.id,
                fraud_probability=0.0,
                is_fraud=False,
                confidence="high",
                processing_time_ms=float((time.time() - start_time) * 1000),
                timestamp=datetime.now().isoformat(),
                explanation={
                    "type": "rule_based",
                    "matched_rules": rule_result["matched_rules"],
                    "reason": "Transaction matched whitelist rule",
                },
            )

        # Extract features from payload
        features = feature_engineer.extract_features(transaction_dict)

        # Prepare DataFrame
        df = feature_engineer.prepare_dataframe(features)

        # Make prediction with explanation
        fraud_probability, is_fraud, explanation = model.predict(
            df, transaction_id=request.payload.id
        )

        # Determine confidence level
        if fraud_probability >= 0.8:
            confidence = "high"
        elif fraud_probability >= 0.5:
            confidence = "medium"
        else:
            confidence = "low"

        processing_time = (time.time() - start_time) * 1000

        # Behavioral Profiling Analysis
        behavioral_analysis = user_profile_service.analyze_transaction(transaction_dict)

        # Update profile asynchronously (non-blocking)
        try:
            user_profile_service.update_profile(
                transaction_dict.get("sender", {}).get("cpfSender", ""),
                transaction_dict
            )
        except Exception as e:
            print(f"Warning: Could not update user profile: {e}")

        response = FraudPrediction(
            transaction_id=request.payload.id,
            fraud_probability=float(fraud_probability),
            is_fraud=bool(is_fraud),
            confidence=confidence,
            processing_time_ms=float(processing_time),
            timestamp=datetime.now().isoformat(),
            explanation=explanation,
            behavioral_analysis=behavioral_analysis.dict()
        )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/batch")
async def predict_fraud_batch(requests: list[TransactionRequest]):
    """Predict fraud for multiple transactions (batch processing)."""
    if not model_loaded:
        raise HTTPException(
            status_code=503, detail="Model not loaded. Please train the model first."
        )

    start_time = time.time()
    results = []

    try:
        for request in requests:
            payload_dict = request.payload.model_dump()
            features = feature_engineer.extract_features(payload_dict)
            features_df = feature_engineer.prepare_dataframe(features)
            fraud_probability, is_fraud, explanation = model.predict(
                features_df, transaction_id=request.payload.id
            )

            confidence = (
                "high"
                if fraud_probability >= 0.8
                else "medium" if fraud_probability >= 0.5 else "low"
            )

            results.append(
                {
                    "transaction_id": request.payload.id,
                    "fraud_probability": round(float(fraud_probability), 4),
                    "is_fraud": bool(is_fraud),
                    "confidence": confidence,
                    "explanation": explanation,
                }
            )

        processing_time_ms = (time.time() - start_time) * 1000

        return {
            "results": results,
            "total_transactions": len(requests),
            "processing_time_ms": round(processing_time_ms, 2),
            "avg_time_per_transaction": round(processing_time_ms / len(requests), 2),
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error during batch prediction: {str(e)}"
        )


@app.get("/model/info")
async def model_info():
    """Get information about the loaded model."""
    if not model_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded.")

    try:
        feature_importance = model.get_feature_importance()
        # Convert numpy types to Python native types
        top_features = {k: float(v) for k, v in list(feature_importance.items())[:10]}

        return {
            "model_type": "XGBoost",
            "feature_count": len(model.feature_names),
            "threshold": float(model.threshold),
            "top_features": top_features,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error getting model info: {str(e)}"
        )


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
            "priority": rule.priority,
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
                "priority": rule.priority,
            }
            for rule in rules
        ],
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
                "value": condition.value,
            }
            for condition in rule.conditions
        ],
        "enabled": rule.enabled,
        "priority": rule.priority,
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


# Behavioral Profiling Endpoints


@app.get("/profile/{cpf}")
async def get_user_profile(cpf: str):
    """Get user behavioral profile (CPF will be hashed for lookup)."""
    try:
        profile = user_profile_service.get_or_create_profile(cpf)
        return profile.dict()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting profile: {str(e)}")


class AnomalyFeedbackRequest(BaseModel):
    transaction_id: str
    cpf: str
    is_true_anomaly: bool
    analyst_id: str
    notes: Optional[str] = None
    anomaly_type: Optional[str] = None


@app.post("/feedback/anomaly")
async def submit_anomaly_feedback(request: AnomalyFeedbackRequest):
    """Submit feedback about anomaly detection from analysts (CPF will be hashed)."""
    try:
        # Hash CPF for storage
        cpf_hasher = get_cpf_hasher()
        hashed_cpf = cpf_hasher.hash_cpf(request.cpf)
        
        # Load existing feedback
        import json
        from pathlib import Path
        
        feedback_file = Path("data/anomaly_feedback.json")
        if feedback_file.exists():
            with open(feedback_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            data = {"feedback": [], "metadata": {"total_feedback": 0, "last_updated": datetime.utcnow().isoformat()}}
        
        # Add new feedback
        feedback_entry = {
            "transaction_id": request.transaction_id,
            "cpf": hashed_cpf,  # Store hashed CPF
            "timestamp": datetime.utcnow().isoformat(),
            "is_true_anomaly": request.is_true_anomaly,
            "analyst_id": request.analyst_id,
            "notes": request.notes,
            "anomaly_type": request.anomaly_type
        }
        
        data["feedback"].append(feedback_entry)
        data["metadata"]["total_feedback"] = len(data["feedback"])
        data["metadata"]["last_updated"] = datetime.utcnow().isoformat()
        
        # Save feedback
        with open(feedback_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)
        
        return {"status": "recorded", "transaction_id": request.transaction_id, "cpf_hashed": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error recording feedback: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
