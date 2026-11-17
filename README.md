# CompressorNet

CompressorNet is an experimental Python prototype for applying several parameter transformations to a PyTorch model and comparing the number of tensor elements retained by those transformations. It is useful as a small API and algorithm demonstration, not as a validated model-compression or benchmarking system.

## Status

Early-stage prototype. The implementation can calculate parameter counts and return transformed parameter dictionaries, but it does not provide a general model serializer, a decompressor, accuracy or reconstruction-quality evaluation, or a reliable measure of storage or inference savings. In particular, the reported compression ratio is based on tensor element counts, not bytes on disk or in memory.

## Requirements

- Python 3.10 or newer
- PyTorch 2.0 or newer
- NumPy 1.24 or newer

The source module can be run directly from a checkout. Install the declared dependencies and run its example:

```bash
python -m pip install -r requirements.txt
python compressor_net.py
```

There is no separate command-line interface. `pyproject.toml` declares project metadata, but the implementation is a single root-level Python module; the commands above use that module in the checkout rather than relying on a packaged CLI.

## Usage

Import the module from the repository root and pass a PyTorch model to `compress`:

```python
import torch.nn as nn
from compressor_net import CompressorNet

model = nn.Sequential(nn.Linear(16, 8))
compressor = CompressorNet()
result = compressor.compress(model, algorithm="svd", rank_ratio=0.5)

print(result.original_size)
print(result.compressed_size)
print(result.compression_ratio)
```

`compress` returns a `CompressionResult` with element counts, an elapsed compression time, a `decompression_time` field set to `0.0`, and a `quality_score` field set to `None`. The ratio is `original_size / compressed_size`; it should not be interpreted as a byte-level compression ratio or as evidence of preserved model behavior.

The module also provides:

- `benchmark(model, algorithms=None)`, which calls `compress` for the selected algorithms (all five by default) and returns a mapping of names to results or `None` for caught failures.
- `compare_algorithms(model, algorithms=None)`, which prints the element-count comparison.
- `CompressionAlgorithm` and `CompressionResult` types.

Available algorithm names and parameters are `pruning` (`pruning_rate`, default `0.5`), `quantization` (`bits`, default `8`), `svd` (`rank_ratio`, default `0.5`), `fourier` (`num_freqs`, default `64`), and `dna` (`hidden_size`, default `128`).

## What the implementation does—and does not do

- **Pruning:** applies a random mask to multidimensional parameters and returns dense tensors, rather than pruning the smallest-magnitude weights or storing a sparse representation. One-dimensional parameters are kept unchanged.
- **Quantization:** rounds scaled parameter values but keeps the results as ordinary tensors; it does not pack values into a lower-bit storage format.
- **SVD:** truncates a matrix's singular-value decomposition and immediately reconstructs a dense matrix. It does not retain compact factors as the compressed representation.
- **Fourier:** creates a random projection matrix and stores mean Fourier features for two-dimensional parameters; other parameters are retained. No fitting, reconstruction, or quality evaluation is implemented.
- **DNA:** stores a random tensor, a scalar, and the mean of each parameter tensor. `hidden_size` is recorded as metadata; there is no trained generator or decoding path.

Consequently, the element counts for Fourier and DNA do not establish that their representations can reconstruct a model or preserve its predictions. Compression timing is recorded for the transformation only; decompression time and quality are not measured. The module does not evaluate accuracy, reconstruction error, inference speed, or serialized file size. Results can also depend on random masks or projections.

## Tests

Run the test suite from the repository root:

```bash
python -m pytest -q tests
```

On the reviewed commit, this command completed with **15 passed**. The repository also defines a GitHub Actions workflow that runs tests and an import check on Python 3.10, 3.11, and 3.12; this statement describes the workflow configuration, not the status of a remote CI run.

## License

The repository includes an MIT License. See [LICENSE](LICENSE) for its full terms and the copyright notice.