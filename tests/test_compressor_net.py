"""
Tests for CompressorNet module
"""

import pytest
import torch
import torch.nn as nn
from compressor_net import CompressorNet, CompressionAlgorithm, CompressionResult


class TestCompressorNetInitialization:
    """Test CompressorNet initialization"""
    
    def test_default_initialization(self):
        """Test default initialization"""
        compressor = CompressorNet()
        assert compressor.device == 'cpu'
        assert len(compressor.algorithms) == 5
    
    def test_custom_device(self):
        """Test custom device initialization"""
        compressor = CompressorNet(device='cpu')
        assert compressor.device == 'cpu'


class TestCompressorNetParameterCounting:
    """Test parameter counting"""
    
    def test_count_parameters(self):
        """Test counting parameters"""
        compressor = CompressorNet()
        
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        
        count = compressor._count_parameters(model)
        # 10*20 + 20 + 20*10 + 10 = 200 + 20 + 200 + 10 = 430
        assert count == 430
    
    def test_count_trainable_parameters(self):
        """Test counting trainable parameters"""
        compressor = CompressorNet()
        
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        
        count = compressor._count_trainable_parameters(model)
        assert count == 430


class TestCompressorNetCompression:
    """Test compression algorithms"""
    
    def test_pruning_compression(self):
        """Test pruning compression"""
        compressor = CompressorNet()
        
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        
        result = compressor.compress(model, algorithm='pruning', pruning_rate=0.5)
        assert isinstance(result, CompressionResult)
        assert result.original_size == 430
        assert result.compression_ratio > 0
    
    def test_quantization_compression(self):
        """Test quantization compression"""
        compressor = CompressorNet()
        
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        
        result = compressor.compress(model, algorithm='quantization', bits=8)
        assert isinstance(result, CompressionResult)
        assert result.original_size == 430
        assert result.compression_ratio > 0
    
    def test_svd_compression(self):
        """Test SVD compression"""
        compressor = CompressorNet()
        
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        
        result = compressor.compress(model, algorithm='svd', rank_ratio=0.5)
        assert isinstance(result, CompressionResult)
        assert result.original_size == 430
        assert result.compression_ratio > 0
    
    def test_fourier_compression(self):
        """Test Fourier compression"""
        compressor = CompressorNet()
        
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        
        result = compressor.compress(model, algorithm='fourier', num_freqs=64)
        assert isinstance(result, CompressionResult)
        assert result.original_size == 430
        assert result.compression_ratio > 0
    
    def test_dna_compression(self):
        """Test DNA compression"""
        compressor = CompressorNet()
        
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        
        result = compressor.compress(model, algorithm='dna', hidden_size=128)
        assert isinstance(result, CompressionResult)
        assert result.original_size == 430
        assert result.compression_ratio > 0


class TestCompressorNetBenchmark:
    """Test benchmarking"""
    
    def test_benchmark_single(self):
        """Test benchmarking single algorithm"""
        compressor = CompressorNet()
        
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        
        results = compressor.benchmark(model, algorithms=['pruning'])
        assert 'pruning' in results
        assert results['pruning'] is not None
    
    def test_benchmark_multiple(self):
        """Test benchmarking multiple algorithms"""
        compressor = CompressorNet()
        
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        
        algorithms = ['pruning', 'quantization']
        results = compressor.benchmark(model, algorithms=algorithms)
        
        assert len(results) == 2
        assert 'pruning' in results
        assert 'quantization' in results


class TestCompressorNetCompare:
    """Test comparison"""
    
    def test_compare_algorithms(self):
        """Test comparing algorithms"""
        compressor = CompressorNet()
        
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 10)
        )
        
        # This should not raise an error
        compressor.compare_algorithms(model, algorithms=['pruning', 'quantization'])


class TestCompressionResult:
    """Test CompressionResult"""
    
    def test_result_creation(self):
        """Test creating CompressionResult"""
        result = CompressionResult(
            compressed_size=100,
            original_size=1000,
            compression_ratio=10.0,
            compression_time=0.1,
            decompression_time=0.05,
            quality_score=0.95
        )
        
        assert result.compressed_size == 100
        assert result.original_size == 1000
        assert result.compression_ratio == 10.0
        assert result.compression_time == 0.1
        assert result.decompression_time == 0.05
        assert result.quality_score == 0.95


class TestCompressionAlgorithm:
    """Test CompressionAlgorithm enum"""
    
    def test_enum_values(self):
        """Test enum values"""
        assert CompressionAlgorithm.PRUNING.value == "pruning"
        assert CompressionAlgorithm.QUANTIZATION.value == "quantization"
        assert CompressionAlgorithm.SVD.value == "svd"
        assert CompressionAlgorithm.FOURIER.value == "fourier"
        assert CompressionAlgorithm.DNA.value == "dna"


class TestErrorHandling:
    """Test error handling"""
    
    def test_invalid_algorithm(self):
        """Test invalid algorithm raises error"""
        compressor = CompressorNet()
        
        model = nn.Sequential(nn.Linear(10, 10))
        
        with pytest.raises(ValueError):
            compressor.compress(model, algorithm='invalid')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
