from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime


class Sender(BaseModel):
    banco: Optional[int] = None
    agencia: Optional[str] = None
    nuConta: Optional[int] = None
    cpfSender: Optional[str] = None


class Receiver(BaseModel):
    banco: Optional[int] = None
    agencia: Optional[str] = None
    nuConta: Optional[int] = None
    cpfReceiver: Optional[str] = None


class ExtraInfo(BaseModel):
    codigo_barra: Optional[str] = None
    motivo_acesso: Optional[str] = None


class TransactionPayload(BaseModel):
    id: str
    timestamp: str
    canal: str
    produto: str
    jornada: str
    direcao: str
    sender: Sender
    receiver: Receiver
    valor: float
    extra_info: ExtraInfo


class TransactionRequest(BaseModel):
    payload: TransactionPayload


class FraudPrediction(BaseModel):
    transaction_id: str
    fraud_probability: float
    is_fraud: bool
    confidence: str
    processing_time_ms: float
    timestamp: str
    explanation: Optional[Dict[str, Any]] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
