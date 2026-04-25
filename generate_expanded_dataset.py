import pandas as pd
import numpy as np
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List
import random


class DatasetExpander:
    def __init__(self, original_csv: str):
        """Initialize with original dataset."""
        self.df = pd.read_csv(original_csv)
        self.canais = ['web', 'app', 'api']
        self.produtos = ['pix', 'ted', 'boleto', 'autenticacao', 'financeiro_generico']
        self.jornadas = {
            'pix': ['pix_troco', 'pix_saque', 'transferencia', 'estorno'],
            'ted': ['transferencia'],
            'boleto': ['transferencia'],
            'autenticacao': ['login_perfil'],
            'financeiro_generico': ['transferencia']
        }
        
        # Analyze original patterns
        self.fraud_rate = self.df['fraudResult'].mean()
        print(f"Original fraud rate: {self.fraud_rate:.4f}")
        
        # Parse patterns from original data
        self._analyze_patterns()
    
    def _analyze_patterns(self):
        """Analyze patterns from original dataset."""
        print("Analyzing patterns from original dataset...")
        
        payloads = []
        for payload_str in self.df['payload']:
            try:
                payload = json.loads(payload_str)
                payloads.append(payload)
            except:
                payloads.append({})
        
        self.payloads_df = pd.DataFrame(payloads)
        
        # Extract patterns
        self.canal_distribution = self.payloads_df['canal'].value_counts(normalize=True).to_dict()
        self.produto_distribution = self.payloads_df['produto'].value_counts(normalize=True).to_dict()
        
        # Value distribution
        valores = self.payloads_df['valor'].values
        self.valor_mean = np.mean(valores)
        self.valor_std = np.std(valores)
        self.valor_min = np.min(valores)
        self.valor_max = np.max(valores)
        
        # Bank codes
        bancos_sender = self.payloads_df['sender'].apply(lambda x: json.loads(x) if isinstance(x, str) else x).apply(
            lambda x: x.get('banco', 0) if isinstance(x, dict) else 0
        )
        self.bancos_sender = bancos_sender.unique()
        
        print(f"Canal distribution: {self.canal_distribution}")
        print(f"Produto distribution: {self.produto_distribution}")
        print(f"Valor - Mean: {self.valor_mean:.2f}, Std: {self.valor_std:.2f}")
    
    def _generate_random_payload(self, is_fraud: bool = False) -> Dict:
        """Generate a random transaction payload based on patterns."""
        # Sample canal
        canal = np.random.choice(
            list(self.canal_distribution.keys()),
            p=list(self.canal_distribution.values())
        )
        
        # Sample produto
        produto = np.random.choice(
            list(self.produto_distribution.keys()),
            p=list(self.produto_distribution.values())
        )
        
        # Sample jornada based on produto
        jornada = np.random.choice(self.jornadas.get(produto, ['transferencia']))
        
        # Generate timestamp (random in last 30 days)
        days_ago = np.random.randint(0, 30)
        hours_ago = np.random.randint(0, 24)
        timestamp = datetime.now() - timedelta(days=days_ago, hours=hours_ago)
        
        # Generate value (log-normal distribution)
        if produto == 'autenticacao':
            valor = 0.0
        else:
            valor = np.random.lognormal(
                mean=np.log(self.valor_mean + 1),
                sigma=0.5
            )
            valor = min(valor, self.valor_max * 2)  # Cap at 2x max
            valor = max(valor, 10)  # Minimum 10
        
        # Fraud patterns: higher values, unusual hours
        if is_fraud:
            valor = valor * np.random.uniform(1.5, 3.0)  # Fraudulent transactions tend to be higher
            if np.random.random() > 0.5:
                timestamp = timestamp.replace(hour=np.random.randint(22, 24) or np.random.randint(0, 6))
        
        # Generate sender
        sender_banco = int(np.random.choice(self.bancos_sender))
        sender = {
            "banco": sender_banco,
            "agencia": f"{np.random.randint(1, 9999):04d}",
            "nuConta": int(np.random.randint(10000000, 99999999)),
            "cpfSender": f"{random.randint(10000000000, 99999999999)}"
        }
        
        # Generate receiver (null for autenticacao)
        if produto == 'autenticacao':
            receiver = {
                "banco": None,
                "agencia": None,
                "nuConta": None,
                "cpfReceiver": None
            }
        else:
            receiver_banco = int(np.random.choice(self.bancos_sender))
            receiver = {
                "banco": receiver_banco,
                "agencia": f"{np.random.randint(1, 9999):04d}",
                "nuConta": int(np.random.randint(10000000, 99999999)),
                "cpfReceiver": f"{random.randint(10000000000, 99999999999)}"
            }
        
        # Extra info
        extra_info = {
            "codigo_barra": None,
            "motivo_acesso": "troca_senha" if produto == 'autenticacao' else None
        }
        
        payload = {
            "id": str(uuid.uuid4()),
            "timestamp": timestamp.isoformat(),
            "canal": canal,
            "produto": produto,
            "jornada": jornada,
            "direcao": "saida",
            "sender": sender,
            "receiver": receiver,
            "valor": float(round(valor, 2)),
            "extra_info": extra_info
        }
        
        return payload
    
    def generate_expanded_dataset(self, target_size: int = 50000) -> pd.DataFrame:
        """Generate expanded dataset with target_size samples."""
        print(f"Generating expanded dataset with {target_size} samples...")
        
        # Calculate number of fraud and legitimate samples
        n_fraud = int(target_size * self.fraud_rate)
        n_legit = target_size - n_fraud
        
        print(f"Target: {n_legit} legitimate, {n_fraud} fraud samples")
        
        # Generate samples
        new_rows = []
        
        # Generate legitimate samples
        for _ in range(n_legit):
            payload = self._generate_random_payload(is_fraud=False)
            new_rows.append({
                'payload': json.dumps(payload),
                'fraudResult': 0
            })
        
        # Generate fraud samples
        for _ in range(n_fraud):
            payload = self._generate_random_payload(is_fraud=True)
            new_rows.append({
                'payload': json.dumps(payload),
                'fraudResult': 1
            })
        
        # Combine with original dataset
        expanded_df = pd.concat([self.df, pd.DataFrame(new_rows)], ignore_index=True)
        
        # Shuffle
        expanded_df = expanded_df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Expanded dataset size: {len(expanded_df)}")
        print(f"Fraud distribution:\n{expanded_df['fraudResult'].value_counts()}")
        
        return expanded_df
    
    def save_dataset(self, df: pd.DataFrame, output_path: str):
        """Save expanded dataset to CSV."""
        df.to_csv(output_path, index=False)
        print(f"Dataset saved to {output_path}")


def main():
    """Main function to generate expanded dataset."""
    expander = DatasetExpander('dataset_transacoes.csv')
    
    # Generate expanded dataset
    expanded_df = expander.generate_expanded_dataset(target_size=50000)
    
    # Save expanded dataset
    expander.save_dataset(expanded_df, 'dataset_transacoes_expanded.csv')
    
    print("\n=== Dataset Expansion Complete ===")
    print(f"Original: 10,000 samples")
    print(f"Expanded: 50,000 samples")
    print(f"Output: dataset_transacoes_expanded.csv")


if __name__ == "__main__":
    main()
