# CompressorNet

[![CI](https://github.com/kapildhinwa2010-dev/CompressorNet/actions/workflows/ci.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/CompressorNet/actions/workflows/ci.yml)
[![CI](https://github.com/kapildhinwa2010-dev/CompressorNet/actions/workflows/codeql.yml/badge.svg)](https://github.com/kapildhinwa2010-dev/CompressorNet/actions/workflows/codeql.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)

Compare five neural-network compression algorithms on one API.

## What it does

CompressorNet implements pruning, quantization, SVD, Fourier features
and DNA-style generator encoding behind one compress()/benchmark() API
and reports size ratio and timing per algorithm.

## Installation

    pip install torch numpy pytest

## Usage

    from compressor_net import CompressorNet

    compressor = CompressorNet()
    result = compressor.compress(model, algorithm="svd", rank_ratio=0.5)
    print(result.compression_ratio)

    results = compressor.benchmark(model)
    compressor.compare_algorithms(model)

## Algorithms

- pruning (rate): zero smallest-magnitude weights
- quantization (bits): lower-precision weights
- svd (rank_ratio): low-rank factorization of 2-D matrices
- fourier (num_freqs): Fourier-feature generator per matrix
- dna (hidden_size): compact-MLP generator per matrix

## API

- CompressorNet(device="cpu"): compress, benchmark, compare_algorithms
- CompressionResult: compressed_size, original_size, compression_ratio,
  compression_time, decompression_time, quality_score

## Honest results and limitations

- Pruning, quantization and SVD are standard and behave as expected.
- The fourier and dna entries are structural (they report param-count
  ratios), not measured quality: no fine-tuning or accuracy evaluation
  is included, so their "quality" column in any comparison is a
  placeholder, not a result.
- quality_score is None for algorithms without an explicit error
  measure; the suite does not evaluate task accuracy at all.
- Ratios for generator methods assume you keep only the generator;
  that is only valid if you accept its reconstruction error.

## Negative results (from the source experiments)

- The reserv.dean-part-2 log records that generator-based compression
  (fourier/dna) did not reconstruct trained weights acceptably without
  per-layer fitting, and the related carrier-mode recall failed about
  half of noise trials. The robust design kept in the toolkit is the
  Reed-Solomon slot lattice (ECC-Memory).

## License

MIT - see [LICENSE](LICENSE).
