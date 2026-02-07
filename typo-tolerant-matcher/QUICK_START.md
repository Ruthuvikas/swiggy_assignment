# Quick Start Guide

## Installation

```bash
cd typo-tolerant-matcher
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run Demo

```bash
cd src
python inference.py
```

This will:
1. Load the trained model
2. Show 5 typo-tolerance examples
3. Benchmark inference speed (500 items)

## Train from Scratch

```bash
cd src
python train.py
```

Training takes ~5-10 minutes on CPU.

## Test Individual Queries

```python
from inference import FastInference

# Load model
matcher = FastInference('../models/best_model.pth')

# Score a query against targets
query = "chiken biryani"
targets = ["Chicken Biryani", "Paneer Tikka", "Dal Makhani"]

results = matcher.score_one_to_many(query, targets)
for target, score in results:
    print(f"{target}: {score:.3f}")
```

## Model Info

- **Size**: 838KB (0.27MB float32)
- **Parameters**: 70,145
- **Speed**: ~2,500 queries/sec on CPU
- **Handles**: Typos, transliterations, Hindi/English/Hinglish

## Files

- `src/model.py` - Model architecture
- `src/train.py` - Training script
- `src/inference.py` - Demo and benchmarking
- `models/best_model.pth` - Trained weights
- `README.md` - Full documentation
- `docs/DATA.md` - Data documentation
- `SUBMISSION_SUMMARY.md` - Project summary
