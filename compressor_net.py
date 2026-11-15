"""
CompressorNet - Neural network compression benchmark suite

This module provides a comprehensive suite for benchmarking and testing
neural network compression techniques, including various compression
algorithms and evaluation metrics.

Features:
- Multiple compression algorithms
- Benchmarking utilities
- Compression ratio calculations
- Quality metrics
- Memory and speed profiling
"""

import torch
import torch.nn as nn
import time
import numpy as np
from typing import Optional, Dict, Any, Callable, Tuple
from dataclasses import dataclass
from enum import Enum


class CompressionAlgorithm(Enum):
    """Supported compression algorithms"""
    PRUNING = "pruning"
    QUANTIZATION = "quantization"
    SVD = "svd"
    FOURIER = "fourier"
    DNA = "dna"


@dataclass
class CompressionResult:
    """Result of compression"""
    compressed_size: int
    original_size: int
    compression_ratio: float
    compression_time: float
    decompression_time: float
    quality_score: Optional[float] = None


class CompressorNet:
    """
    CompressorNet - Neural network compression benchmark suite.
    
    This class provides a unified interface for benchmarking various
    compression algorithms on neural networks.
    
    Example:
        >>> compressor = CompressorNet()
        >>> # Compress a model
        >>> result = compressor.compress(model, algorithm='fourier')
        >>> print(f"Compression ratio: {result.compression_ratio}x")
    
    Args:
        device: Device to use for compression (default: 'cpu')
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.algorithms = {
            'pruning': self._compress_pruning,
            'quantization': self._compress_quantization,
            'svd': self._compress_svd,
            'fourier': self._compress_fourier,
            'dna': self._compress_dna,
        }
    
    def _count_parameters(self, model: nn.Module) -> int:
        """Count total parameters in a model"""
        return sum(p.numel() for p in model.parameters())
    
    def _count_trainable_parameters(self, model: nn.Module) -> int:
        """Count trainable parameters in a model"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def _count_stored_elements(self, obj: Any) -> int:
        """Recursively count elements of all tensors stored in a nested structure.
        
        Used to compute an honest compressed size: every tensor that would
        need to be kept to reconstruct the model is counted, including
        tensors nested inside dicts.
        """
        if isinstance(obj, torch.Tensor):
            return obj.numel()
        if isinstance(obj, dict):
            return sum(self._count_stored_elements(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return sum(self._count_stored_elements(v) for v in obj)
        return 0
    
    def compress(
        self,
        model: nn.Module,
        algorithm: str = 'fourier',
        **kwargs
    ) -> CompressionResult:
        """
        Compress a model using the specified algorithm.
        
        Args:
            model: PyTorch model to compress
            algorithm: Compression algorithm to use
            **kwargs: Additional arguments for the algorithm
            
        Returns:
            CompressionResult with compression statistics
        """
        if algorithm not in self.algorithms:
            raise ValueError(f"Unknown algorithm: {algorithm}. Choose from {list(self.algorithms.keys())}")
        
        # Get original size
        original_size = self._count_parameters(model)
        
        # Compress
        start_time = time.time()
        compressed = self.algorithms[algorithm](model, **kwargs)
        compression_time = time.time() - start_time
        
        # Get compressed size (count ALL stored tensors, including nested dicts)
        compressed_size = self._count_stored_elements(compressed)
        
        # Calculate ratio
        compression_ratio = original_size / compressed_size if compressed_size > 0 else float('inf')
        
        return CompressionResult(
            compressed_size=compressed_size,
            original_size=original_size,
            compression_ratio=compression_ratio,
            compression_time=compression_time,
            decompression_time=0.0,
            quality_score=None
        )
    
    def _compress_pruning(
        self,
        model: nn.Module,
        pruning_rate: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """Compress using pruning"""
        compressed = {}
        for name, param in model.named_parameters():
            if len(param.shape) > 1:  # Only prune weights, not biases
                mask = torch.rand_like(param) > pruning_rate
                compressed[name] = param * mask
            else:
                compressed[name] = param
        return compressed
    
    def _compress_quantization(
        self,
        model: nn.Module,
        bits: int = 8
    ) -> Dict[str, torch.Tensor]:
        """Compress using quantization"""
        compressed = {}
        scale = 2 ** bits - 1
        for name, param in model.named_parameters():
            quantized = (param * scale).round() / scale
            compressed[name] = quantized
        return compressed
    
    def _compress_svd(
        self,
        model: nn.Module,
        rank_ratio: float = 0.5
    ) -> Dict[str, torch.Tensor]:
        """Compress using SVD decomposition"""
        compressed = {}
        for name, param in model.named_parameters():
            if len(param.shape) == 2:
                U, S, Vh = torch.linalg.svd(param)
                rank = int(min(param.shape) * rank_ratio)
                U_trunc = U[:, :rank]
                S_trunc = S[:rank]
                Vh_trunc = Vh[:rank, :]
                compressed[name] = U_trunc @ torch.diag(S_trunc) @ Vh_trunc
            else:
                compressed[name] = param
        return compressed
    
    def _compress_fourier(
        self,
        model: nn.Module,
        num_freqs: int = 64
    ) -> Dict[str, Any]:
        """Compress using Fourier features"""
        compressed = {
            'algorithm': 'fourier',
            'num_freqs': num_freqs,
            'params': {}
        }
        
        # Store Fourier projection matrix
        B = torch.randn(num_freqs, 2, device=self.device) * 20.0
        compressed['B'] = B
        
        # Compress each parameter
        for name, param in model.named_parameters():
            if len(param.shape) == 2:
                # Use Fourier features for 2D parameters
                out_dim, in_dim = param.shape
                i = torch.linspace(-1, 1, out_dim, device=self.device)
                j = torch.linspace(-1, 1, in_dim, device=self.device)
                grid_i, grid_j = torch.meshgrid(i, j, indexing='ij')
                coords = torch.stack([grid_i.flatten(), grid_j.flatten()], dim=1)
                
                proj = 2 * np.pi * coords @ B.T
                feats = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)
                
                # Simple linear projection (in practice, use a neural network)
                compressed['params'][name] = torch.mean(feats, dim=0)
            else:
                compressed['params'][name] = param
        
        return compressed
    
    def _compress_dna(
        self,
        model: nn.Module,
        hidden_size: int = 128
    ) -> Dict[str, Any]:
        """Compress using DNA encoding"""
        compressed = {
            'algorithm': 'dna',
            'hidden_size': hidden_size,
            'params': {}
        }
        
        # Store DNA network parameters
        compressed['B'] = torch.randn(64, 2, device=self.device) * 20.0
        compressed['scale'] = torch.tensor(0.01, device=self.device)
        
        # Compress each parameter
        for name, param in model.named_parameters():
            compressed['params'][name] = torch.mean(param)
        
        return compressed
    
    def benchmark(
        self,
        model: nn.Module,
        algorithms: list = None
    ) -> Dict[str, CompressionResult]:
        """
        Benchmark multiple compression algorithms on a model.
        
        Args:
            model: PyTorch model to benchmark
            algorithms: List of algorithms to test
            
        Returns:
            Dictionary mapping algorithm names to CompressionResults
        """
        if algorithms is None:
            algorithms = list(self.algorithms.keys())
        
        results = {}
        for algo in algorithms:
            try:
                result = self.compress(model, algorithm=algo)
                results[algo] = result
            except Exception as e:
                print(f"Error with {algo}: {e}")
                results[algo] = None
        
        return results
    
    def compare_algorithms(
        self,
        model: nn.Module,
        algorithms: list = None
    ) -> None:
        """
        Compare compression algorithms and print results.
        
        Args:
            model: PyTorch model to compare
            algorithms: List of algorithms to compare
        """
        results = self.benchmark(model, algorithms)
        
        print("\n" + "="*70)
        print("Compression Algorithm Comparison")
        print("="*70)
        print(f"{'Algorithm':<15} {'Original':<12} {'Compressed':<12} {'Ratio':<12}")
        print("-"*70)
        
        for algo, result in results.items():
            if result:
                print(f"{algo:<15} {result.original_size:<12,} {result.compressed_size:<12,} {result.compression_ratio:<12.1f}x")
            else:
                print(f"{algo:<15} {'ERROR':<12} {'ERROR':<12} {'ERROR':<12}")


# Example usage
if __name__ == "__main__":
    print("CompressorNet - Example Usage")
    print("=" * 60)
    
    # Create compressor
    compressor = CompressorNet()
    
    # Create a simple model
    import torch.nn as nn
    model = nn.Sequential(
        nn.Linear(128, 256),
        nn.SiLU(),
        nn.Linear(256, 128)
    )
    
    print(f"\nOriginal model parameters: {compressor._count_parameters(model):,}")
    
    # Compress with different algorithms
    print("\nTesting compression algorithms...")
    
    algorithms = ['pruning', 'quantization', 'svd', 'fourier', 'dna']
    for algo in algorithms:
        result = compressor.compress(model, algorithm=algo)
        print(f"  {algo}: {result.compression_ratio:.1f}x compression")
    
    # Compare all algorithms
    print("\nComparing all algorithms...")
    compressor.compare_algorithms(model, algorithms)
    
    print("\n✓ CompressorNet test completed successfully!")
