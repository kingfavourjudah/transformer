# Transformer - GPT Character-Level Model

A Rust implementation of a character-level GPT (Generative Pre-trained Transformer) model using PyTorch bindings (tch-rs). This project demonstrates transformer architecture with self-attention mechanisms for sequence prediction tasks.

## Features

- **Transformer Architecture**: Implementation of multi-head self-attention and feed-forward layers
- **Character-Level Modeling**: Trains on character sequences for next-character prediction
- **Synthetic Data Generation**: Built-in data generation for testing and experimentation
- **CLI Interface**: Command-line arguments for configurable training parameters
- **Visualization**: Loss curve plotting using the Plotters library
- **Comprehensive Testing**: Unit tests for model architecture and data generation

## Architecture

The model implements a simplified GPT architecture with:
- Token embeddings
- Multi-head self-attention mechanism (Query, Key, Value projections)
- Feed-forward neural network with ReLU activation
- Layer normalization
- Output projection layer for vocabulary predictions

## Requirements

### System Dependencies
- **Rust**: Edition 2024 (nightly toolchain recommended)
- **LibTorch**: Version 2.7.1 or compatible
  - Download from [PyTorch official website](https://pytorch.org/get-started/locally/)
  - Extract to a known location (e.g., `~/Programs/libtorch`)

### Rust Dependencies
- `tch` 0.18.0 - PyTorch bindings for Rust
- `rand` 0.8.5 - Random number generation
- `plotters` 0.3.7 - Data visualization
- `clap` 4.5.42 - Command-line argument parsing

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/kingfavourjudah/transformer.git
   cd transformer
   ```

2. **Install LibTorch**:
   - Download LibTorch 2.7.1 for your platform
   - Extract to a directory (e.g., `~/Programs/libtorch`)
   - The project is configured to use `/Users/favourjudah/Programs/libtorch` by default

3. **Configure LibTorch path** (if different from default):

   Edit `.cargo/config.toml` and update the path:
   ```toml
   [env]
   DYLD_LIBRARY_PATH = "/your/path/to/libtorch/lib"
   ```

4. **Build the project**:
   ```bash
   cargo build --release
   ```

## Usage

### Basic Training

Run with default parameters:
```bash
cargo run --release
```

### Custom Parameters

```bash
cargo run --release -- [OPTIONS]
```

**Available Options**:
- `-s, --samples <SAMPLES>` - Number of training samples (default: 1000)
- `-n, --seq-len <SEQ_LEN>` - Sequence length of training data (default: 5)
- `-e, --epochs <EPOCHS>` - Number of training epochs (default: 500)
- `-d, --hidden-dim <HIDDEN_DIM>` - Hidden dimension size (default: 128)
- `-l, --lr <LR>` - Learning rate (default: 0.001)

### Example Commands

Train with custom parameters:
```bash
cargo run --release -- --samples 2000 --epochs 1000 --seq-len 10 --hidden-dim 256 --lr 0.0005
```

Quick test run:
```bash
cargo run --release -- --samples 100 --epochs 50 --seq-len 5
```

### Output

The program will:
1. Generate synthetic training data
2. Train the model and display loss per epoch
3. Evaluate the model on test data
4. Generate a loss curve plot saved as `loss_plot.png`

Example output:
```
Epoch: 1, Loss: 4.3324
Epoch: 2, Loss: 0.8370
Epoch: 3, Loss: 0.6714
...
Evaluation accuracy: 0.8500
```

## Project Structure

```
transformer/
├── src/
│   ├── main.rs           # Entry point and training loop
│   ├── model.rs          # Transformer model architecture
│   ├── data.rs           # Synthetic data generation
│   ├── evaluate.rs       # Model evaluation utilities
│   ├── cli.rs            # Command-line argument parsing
│   ├── lib.rs            # Library root
│   └── tests/
│       ├── mod.rs
│       ├── model_tests.rs   # Model architecture tests
│       └── data_tests.rs    # Data generation tests
├── .cargo/
│   └── config.toml       # Build configuration
├── Cargo.toml            # Project dependencies
├── README.md
└── LICENSE
```

## Testing

Run the test suite:
```bash
cargo test
```

Run tests with output:
```bash
cargo test -- --nocapture
```

## Configuration

The `.cargo/config.toml` file contains important build settings:
- C++17 standard for LibTorch compatibility
- LibTorch version check bypass (for version 2.7.1)
- Dynamic library path for runtime linking

## Troubleshooting

### Library Not Found Errors

If you encounter `dyld: Library not loaded` errors:
1. Verify LibTorch is installed correctly
2. Update `DYLD_LIBRARY_PATH` in `.cargo/config.toml`
3. On Linux, use `LD_LIBRARY_PATH` instead of `DYLD_LIBRARY_PATH`

### Compilation Errors

If you see C++ standard errors:
- Ensure `.cargo/config.toml` has `CXXFLAGS = "-std=c++17"`
- Verify LibTorch version compatibility

### Version Mismatch

The project uses `LIBTORCH_BYPASS_VERSION_CHECK = "1"` to work with LibTorch 2.7.1. If you encounter issues, ensure your LibTorch version is 2.5.1 or later.

## Performance Notes

- Training time depends on the number of samples, epochs, and hidden dimensions
- GPU support is available through tch-rs if CUDA-enabled LibTorch is installed
- Release builds (`--release`) are significantly faster than debug builds

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [tch-rs](https://github.com/LaurentMazare/tch-rs) - Rust bindings for PyTorch
- Inspired by the original Transformer architecture from "Attention Is All You Need"
- Character-level modeling approach based on Andrej Karpathy's work

## Author

Lucky Nweke ([@kingfavourjudah](https://github.com/kingfavourjudah))

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [tch-rs Documentation](https://docs.rs/tch/latest/tch/)
