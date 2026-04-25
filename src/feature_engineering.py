import math
from datetime import datetime
from typing import Any, Dict

import numpy as np
import pandas as pd


class FeatureEngineer:
    def __init__(self):
        self.feature_names = []
        # Cache para histórico de transações (simulado)
        self.transaction_history = {}

    def extract_features(self, payload: Dict[str, Any]) -> pd.Series:
        """Extract features from transaction payload."""
        features = {}

        # Basic features
        features["canal"] = payload.get("canal", "unknown")
        features["produto"] = payload.get("produto", "unknown")
        features["jornada"] = payload.get("jornada", "unknown")
        features["direcao"] = payload.get("direcao", "saida")
        features["valor"] = payload.get("valor", 0.0)

        # Sender features
        sender = payload.get("sender", {})
        features["sender_banco"] = sender.get("banco", 0)
        features["sender_agencia"] = self._encode_agencia(sender.get("agencia", "0001"))
        features["sender_conta"] = sender.get("nuConta", 0)

        # Receiver features
        receiver = payload.get("receiver", {})
        features["receiver_banco"] = receiver.get("banco", 0)
        features["receiver_agencia"] = self._encode_agencia(
            receiver.get("agencia", "0001")
        )
        features["receiver_conta"] = receiver.get("nuConta", 0)

        # Temporal features (basic + cyclical encoding)
        timestamp_str = payload.get("timestamp", "")
        if timestamp_str:
            try:
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                features["hora_do_dia"] = timestamp.hour
                features["dia_da_semana"] = timestamp.weekday()
                features["fim_de_semana"] = 1 if timestamp.weekday() >= 5 else 0
                features["horario_noturno"] = (
                    1 if timestamp.hour >= 22 or timestamp.hour < 6 else 0
                )

                # Cyclical encoding for temporal features
                features["hora_sin"] = math.sin(2 * math.pi * timestamp.hour / 24)
                features["hora_cos"] = math.cos(2 * math.pi * timestamp.hour / 24)
                features["dia_semana_sin"] = math.sin(
                    2 * math.pi * timestamp.weekday() / 7
                )
                features["dia_semana_cos"] = math.cos(
                    2 * math.pi * timestamp.weekday() / 7
                )

                # Additional temporal features
                features["dia_do_mes"] = timestamp.day
                features["mes_do_ano"] = timestamp.month
                features["inicio_mes"] = 1 if timestamp.day <= 5 else 0
                features["fim_mes"] = 1 if timestamp.day >= 25 else 0
            except Exception:
                features["hora_do_dia"] = 12
                features["dia_da_semana"] = 0
                features["fim_de_semana"] = 0
                features["horario_noturno"] = 0
                features["hora_sin"] = 0
                features["hora_cos"] = 0
                features["dia_semana_sin"] = 0
                features["dia_semana_cos"] = 0
                features["dia_do_mes"] = 15
                features["mes_do_ano"] = 6
                features["inicio_mes"] = 0
                features["fim_mes"] = 0
        else:
            features["hora_do_dia"] = 12
            features["dia_da_semana"] = 0
            features["fim_de_semana"] = 0
            features["horario_noturno"] = 0
            features["hora_sin"] = 0
            features["hora_cos"] = 0
            features["dia_semana_sin"] = 0
            features["dia_semana_cos"] = 0
            features["dia_do_mes"] = 15
            features["mes_do_ano"] = 6
            features["inicio_mes"] = 0
            features["fim_mes"] = 0

        # Value features (enhanced)
        valor = features["valor"]
        features["valor_log"] = np.log1p(valor) if valor > 0 else 0
        features["valor_maior_1000"] = 1 if valor > 1000 else 0
        features["valor_maior_5000"] = 1 if valor > 5000 else 0
        features["valor_maior_10000"] = 1 if valor > 10000 else 0
        features["valor_zscore"] = (
            valor / 5000 if valor > 0 else 0
        )  # Normalizado por valor médio
        features["faixa_valor_baixa"] = 1 if valor <= 100 else 0
        features["faixa_valor_media"] = 1 if 100 < valor <= 1000 else 0
        features["faixa_valor_alta"] = 1 if 1000 < valor <= 5000 else 0
        features["faixa_valor_muito_alta"] = 1 if valor > 5000 else 0

        # Product-specific features
        produto = features["produto"]
        features["is_pix"] = 1 if produto == "pix" else 0
        features["is_ted"] = 1 if produto == "ted" else 0
        features["is_boleto"] = 1 if produto == "boleto" else 0
        features["is_autenticacao"] = 1 if produto == "autenticacao" else 0

        # Channel-specific features
        canal = features["canal"]
        features["is_app"] = 1 if canal == "app" else 0
        features["is_web"] = 1 if canal == "web" else 0
        features["is_api"] = 1 if canal == "api" else 0

        # Journey-specific features
        jornada = features["jornada"]
        features["is_login"] = 1 if jornada == "login_perfil" else 0
        features["is_transferencia"] = 1 if jornada == "transferencia" else 0
        features["is_pix_troco"] = 1 if jornada == "pix_troco" else 0
        features["is_pix_saque"] = 1 if jornada == "pix_saque" else 0
        features["is_estorno"] = 1 if jornada == "estorno" else 0

        # Behavioral features (simulated with cache)
        cpf_sender = sender.get("cpfSender", "")
        if cpf_sender:
            features = self._extract_behavioral_features(
                features, cpf_sender, timestamp_str
            )
        else:
            # Default values when no CPF
            features["transacoes_ultimas_1h"] = 0
            features["transacoes_ultimas_24h"] = 0
            features["valor_total_ultimas_24h"] = 0
            features["valor_medio_ultimas_24h"] = 0
            features["nova_relacao"] = 0
            features["dispositivo_distinto"] = 0

        # Geolocation-based features (simulated based on bank codes)
        sender_banco = features["sender_banco"]
        receiver_banco = features["receiver_banco"]
        features["cross_border"] = (
            1
            if sender_banco and receiver_banco and sender_banco != receiver_banco
            else 0
        )
        features["mesmo_banco"] = (
            1
            if sender_banco and receiver_banco and sender_banco == receiver_banco
            else 0
        )
        features["banco_diferente_sender"] = (
            1
            if receiver_banco and receiver_banco > 0 and receiver_banco != sender_banco
            else 0
        )

        # Network features (simulated)
        features["grau_sender"] = 1  # Placeholder - would require graph analysis
        features["grau_receiver"] = 1  # Placeholder - would require graph analysis

        # Fraud pattern features
        features["horario_atipico"] = features["horario_noturno"]
        features["transacao_fora_horario_comercial"] = (
            1 if features["hora_do_dia"] < 8 or features["hora_do_dia"] >= 18 else 0
        )
        features["valor_atipico"] = features["valor_maior_5000"]
        features["multiplos_dispositivos"] = (
            0  # Placeholder - would require device tracking
        )

        return pd.Series(features)

    def _encode_agencia(self, agencia: str) -> int:
        """Encode agency as integer."""
        try:
            return int(agencia) if agencia else 0
        except Exception:
            return 0

    def _extract_behavioral_features(
        self, features: dict, cpf_sender: str, timestamp_str: str
    ) -> dict:
        """Extract behavioral features based on transaction history."""
        # Initialize history for this CPF if not exists
        if cpf_sender not in self.transaction_history:
            self.transaction_history[cpf_sender] = []

        # Add current transaction to history
        current_time = (
            datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            if timestamp_str
            else datetime.now()
        )
        self.transaction_history[cpf_sender].append(
            {
                "timestamp": current_time,
                "valor": features["valor"],
                "receiver_banco": features["receiver_banco"],
            }
        )

        # Keep only last 100 transactions to prevent memory issues
        if len(self.transaction_history[cpf_sender]) > 100:
            self.transaction_history[cpf_sender] = self.transaction_history[cpf_sender][
                -100:
            ]

        # Calculate behavioral features
        history = self.transaction_history[cpf_sender]

        # Transactions in last 1 hour
        one_hour_ago = current_time - pd.Timedelta(hours=1)
        transacoes_1h = [t for t in history if t["timestamp"] >= one_hour_ago]
        features["transacoes_ultimas_1h"] = len(transacoes_1h)

        # Transactions in last 24 hours
        one_day_ago = current_time - pd.Timedelta(days=1)
        transacoes_24h = [t for t in history if t["timestamp"] >= one_day_ago]
        features["transacoes_ultimas_24h"] = len(transacoes_24h)

        # Total and average value in last 24h
        if transacoes_24h:
            valores = [t["valor"] for t in transacoes_24h]
            features["valor_total_ultimas_24h"] = sum(valores)
            features["valor_medio_ultimas_24h"] = sum(valores) / len(valores)
        else:
            features["valor_total_ultimas_24h"] = 0
            features["valor_medio_ultimas_24h"] = 0

        # New relationship check (first transaction to this receiver)
        receiver_banco = features["receiver_banco"]
        if receiver_banco and receiver_banco > 0:
            previous_receivers = [t["receiver_banco"] for t in history[:-1]]
            features["nova_relacao"] = (
                1 if receiver_banco not in previous_receivers else 0
            )
        else:
            features["nova_relacao"] = 0

        # Distinct device (simulated - would require device tracking)
        features["dispositivo_distinto"] = 0  # Placeholder

        return features

    def prepare_dataframe(self, features: pd.Series) -> pd.DataFrame:
        """Convert features to DataFrame for model prediction."""
        df = pd.DataFrame([features])

        # One-hot encoding for categorical variables
        categorical_cols = ["canal", "produto", "jornada", "direcao"]
        for col in categorical_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df, dummies], axis=1)
                df = df.drop(col, axis=1)

        # Ensure all expected features are present
        # This will be updated after training to match model features
        return df

    def get_feature_names(self) -> list:
        """Get list of feature names after encoding."""
        return self.feature_names

    def set_feature_names(self, feature_names: list):
        """Set feature names from trained model."""
        self.feature_names = feature_names
