# DiffProt

The implementation for "Self-Interpretable Graph Fraud Detection via Diffusion-Based Prototypes".

## Requirements

- Python 3.11.5
- PyTorch 2.1.0
- DGL 1.1.2
- NumPy, SciPy, SymPy, scikit-learn

## Usage
```bash
python main.py --dataset amazon/yelp/tfinance
```

## Configuration

Each dataset has a corresponding configuration file in the `configs/` directory.

## Project Structure

```
DiffProt/
├── main.py              # Main training script
├── DiffProt.py          # Model definition
├── dataset.py           # Dataset loader
├── utils.py             # Utility functions
├── configs/             # Configuration files
│   ├── amazon.json
│   ├── yelp.json
│   └── tfinance.json
└── README.md
```

## Acknowledgement

This code is built upon the [BWGNN repository](https://github.com/squareRoot3/Rethinking-Anomaly-Detection).
