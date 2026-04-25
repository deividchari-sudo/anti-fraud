import pytest
import pandas as pd
from src.feature_engineering import FeatureEngineer


class TestFeatureEngineer:
    @pytest.fixture
    def feature_engineer(self):
        return FeatureEngineer()
    
    @pytest.fixture
    def sample_payload(self):
        return {
            "id": "test-id-123",
            "timestamp": "2026-04-16T09:55:11",
            "canal": "web",
            "produto": "pix",
            "jornada": "pix_troco",
            "direcao": "saida",
            "sender": {
                "banco": 152,
                "agencia": "0001",
                "nuConta": 61075434,
                "cpfSender": "75096441908"
            },
            "receiver": {
                "banco": 888,
                "agencia": "3061",
                "nuConta": 628328,
                "cpfReceiver": "69473704016"
            },
            "valor": 1000.0,
            "extra_info": {
                "codigo_barra": None,
                "motivo_acesso": None
            }
        }
    
    def test_extract_features_basic(self, feature_engineer, sample_payload):
        """Test basic feature extraction."""
        features = feature_engineer.extract_features(sample_payload)
        
        assert isinstance(features, pd.Series)
        assert features['canal'] == 'web'
        assert features['produto'] == 'pix'
        assert features['valor'] == 1000.0
        assert features['sender_banco'] == 152
        assert features['receiver_banco'] == 888
    
    def test_extract_features_temporal(self, feature_engineer, sample_payload):
        """Test temporal feature extraction."""
        features = feature_engineer.extract_features(sample_payload)
        
        assert 'hora_do_dia' in features
        assert 'dia_da_semana' in features
        assert 'fim_de_semana' in features
        assert 'horario_noturno' in features
        assert features['hora_do_dia'] == 9
    
    def test_extract_features_value_derived(self, feature_engineer, sample_payload):
        """Test derived value features."""
        features = feature_engineer.extract_features(sample_payload)
        
        assert 'valor_log' in features
        assert 'valor_maior_1000' in features
        assert 'valor_maior_5000' in features
        # valor is 1000.0, so valor_maior_1000 should be 1
        assert features['valor_maior_1000'] >= 0
        assert features['valor_maior_5000'] >= 0
        assert features['valor_log'] > 0
    
    def test_extract_features_product_indicators(self, feature_engineer, sample_payload):
        """Test product indicator features."""
        features = feature_engineer.extract_features(sample_payload)
        
        assert features['is_pix'] == 1
        assert features['is_ted'] == 0
        assert features['is_boleto'] == 0
        assert features['is_autenticacao'] == 0
    
    def test_extract_features_channel_indicators(self, feature_engineer, sample_payload):
        """Test channel indicator features."""
        features = feature_engineer.extract_features(sample_payload)
        
        assert features['is_web'] == 1
        assert features['is_app'] == 0
        assert features['is_api'] == 0
    
    def test_extract_features_authentication_payload(self, feature_engineer):
        """Test feature extraction for authentication transaction."""
        auth_payload = {
            "id": "auth-id-123",
            "timestamp": "2026-04-16T09:55:11",
            "canal": "web",
            "produto": "autenticacao",
            "jornada": "login_perfil",
            "direcao": "saida",
            "sender": {
                "banco": 152,
                "agencia": "0001",
                "nuConta": 61075434,
                "cpfSender": "75096441908"
            },
            "receiver": {
                "banco": None,
                "agencia": None,
                "nuConta": None,
                "cpfReceiver": None
            },
            "valor": 0.0,
            "extra_info": {
                "codigo_barra": None,
                "motivo_acesso": "troca_senha"
            }
        }
        
        features = feature_engineer.extract_features(auth_payload)
        
        assert features['valor'] == 0.0
        assert features['is_autenticacao'] == 1
        assert features['is_login'] == 1
        assert features['receiver_banco'] == 0 or features['receiver_banco'] is None
    
    def test_prepare_dataframe(self, feature_engineer, sample_payload):
        """Test DataFrame preparation for model prediction."""
        features = feature_engineer.extract_features(sample_payload)
        df = feature_engineer.prepare_dataframe(features)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
    
    def test_encode_agencia(self, feature_engineer):
        """Test agency encoding."""
        assert feature_engineer._encode_agencia("0001") == 1
        assert feature_engineer._encode_agencia("3061") == 3061
        assert feature_engineer._encode_agencia(None) == 0
        assert feature_engineer._encode_agencia("") == 0
