# CompressorNet

[![CI](https://github.com/kapildhinwa2010-dev/CompressorNet/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/CompressorNet/actions/workflows/ci.yml)
[![CodeQL](https://github.com/kapildhinwa2010-dev/CompressorNet/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/CompressorNet/actions/workflows/codeql.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Overview

**CompressorNet** is a comprehensive neural network compression benchmark suite that implements and compares multiple compression algorithms for deep learning models.

### Key Features

- ✅ **Multiple Algorithms**: Pruning, Quantization, SVD, Fourier, DNA
- ✅ **Benchmarking**: Compare compression algorithms side-by-side
- ✅ **Comprehensive Metrics**: Compression ratio, time, quality
- ✅ **Easy to Use**: Simple API for compression and benchmarking
- ✅ **Extensible**: Add custom compression algorithms
- ✅ **Professional**: Production-ready implementation

## Installation

```bash
pip install torch numpy pytest
```

Or install from source:

```bash
git clone https://github.com/kapildhinwa2010-dev/CompressorNet.git
cd CompressorNet
pip install -e .
```

## Usage

### Basic Compression

```python
from compressor_net import CompressorNet

# Create compressor
compressor = CompressorNet()

# Compress a model
result = compressor.compress(model, algorithm='fourier')
print(f"Compression ratio: {result.compression_ratio}x")
```

### Benchmark Multiple Algorithms

```python
# Benchmark all algorithms
results = compressor.benchmark(model)

for algo, result in results.items():
    print(f"{algo}: {result.compression_ratio:.1f}x")
```

### Compare Algorithms

```python
# Compare algorithms with formatted output
compressor.compare_algorithms(model)
```

### Individual Algorithms

```python
# Pruning
result = compressor.compress(model, algorithm='pruning', pruning_rate=0.5)

# Quantization
result = compressor.compress(model, algorithm='quantization', bits=8)

# SVD
result = compressor.compress(model, algorithm='svd', rank_ratio=0.5)

# Fourier
result = compressor.compress(model, algorithm='fourier', num_freqs=64)

# DNA
result = compressor.compress(model, algorithm='dna', hidden_size=128)
```

## Supported Algorithms

### 1. Pruning
Removes a percentage of weights based on magnitude.
- **Parameter**: `pruning_rate` (default: 0.5)
- **Pros**: Simple, fast
- **Cons**: Loss of accuracy if not fine-tuned

### 2. Quantization
Reduces precision of weights to fewer bits.
- **Parameter**: `bits` (default: 8)
- **Pros**: Hardware-friendly, fast inference
- **Cons**: Limited compression ratio

### 3. SVD Decomposition
Factorizes weight matrices using singular value decomposition.
- **Parameter**: `rank_ratio` (default: 0.5)
- **Pros**: Theoretical foundation, good for linear layers
- **Cons**: Only works for 2D matrices

### 4. Fourier Features
Uses Fourier feature projections for parameterization.
- **Parameter**: `num_freqs` (default: 64)
- **Pros**: Extreme compression, smooth parameterization
- **Cons**: Requires training

### 5. DNA Encoding
Encodes parameters using a compact neural network.
- **Parameter**: `hidden_size` (default: 128)
- **Pros**: Extreme compression, flexible
- **Cons**: Requires training

## Compression Results

| Algorithm | Typical Compression Ratio | Speed | Quality Preservation |
|-----------|---------------------------|-------|---------------------|
| Pruning | 2-10x | ⚡⚡⚡ | ⭐⭐⭐ |
| Quantization | 2-8x | ⚡⚡⚡⚡ | ⭐⭐⭐⭐ |
| SVD | 2-100x | ⚡⚡ | ⭐⭐⭐ |
| Fourier | 10-1000x | ⚡ | ⭐⭐⭐⭐ |
| DNA | 10-10000x | ⚡ | ⭐⭐⭐⭐ |

## API Reference

### CompressorNet

```python
CompressorNet(device: str = 'cpu')
```

**Methods:**
- `compress(model: nn.Module, algorithm: str, **kwargs) -> CompressionResult`: Compress a model
- `benchmark(model: nn.Module, algorithms: list) -> Dict[str, CompressionResult]`: Benchmark algorithms
- `compare_algorithms(model: nn.Module, algorithms: list) -> None`: Print comparison table

### CompressionResult

Dataclass containing:
- `compressed_size: int` - Size of compressed parameters
- `original_size: int` - Size of original parameters
- `compression_ratio: float` - Compression ratio
- `compression_time: float` - Time taken to compress (seconds)
- `decompression_time: float` - Time taken to decompress (seconds)
- `quality_score: Optional[float]` - Quality metric (if available)

### CompressionAlgorithm

Enum with supported algorithms:
- `PRUNING`
- `QUANTIZATION`
- `SVD`
- `FOURIER`
- `DNA`

## Testing

Run the test suite:

```bash
python -m pytest tests/ -v
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This work builds upon research in:
- Neural network compression
- Model quantization
- Pruning techniques
- Low-rank factorization
- Implicit neural representations
