"""
Cryptographic utilities for data protection and LGPD compliance.
"""

import hashlib
import os
from typing import Optional


class CPFHasher:
    """Handles CPF hashing for LGPD compliance."""
    
    def __init__(self, salt: Optional[str] = None):
        """
        Initialize CPF hasher with salt.
        
        Args:
            salt: Salt for hashing. If None, generates random salt.
        """
        self.salt = salt or os.urandom(32).hex()
    
    def hash_cpf(self, cpf: str) -> str:
        """
        Hash CPF using SHA-256 with salt.
        
        Args:
            cpf: CPF string (can include formatting)
            
        Returns:
            Hashed CPF (hex string)
        """
        # Remove formatting
        clean_cpf = cpf.replace(".", "").replace("-", "")
        
        # Hash with salt
        salted_cpf = f"{clean_cpf}{self.salt}"
        hashed = hashlib.sha256(salted_cpf.encode()).hexdigest()
        
        return hashed
    
    def hash_cpf_batch(self, cpfs: list) -> dict:
        """
        Hash multiple CPFs.
        
        Args:
            cpfs: List of CPF strings
            
        Returns:
            Dictionary mapping original CPF -> hashed CPF
        """
        return {cpf: self.hash_cpf(cpf) for cpf in cpfs}
    
    def get_salt(self) -> str:
        """Get the salt used for hashing."""
        return self.salt


# Singleton instance for application-wide use
_cpf_hasher: Optional[CPFHasher] = None


def get_cpf_hasher(salt: Optional[str] = None) -> CPFHasher:
    """
    Get or create CPF hasher singleton.
    
    Args:
        salt: Salt for hashing (only used on first call)
        
    Returns:
        CPFHasher instance
    """
    global _cpf_hasher
    if _cpf_hasher is None:
        _cpf_hasher = CPFHasher(salt)
    return _cpf_hasher
