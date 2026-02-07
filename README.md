# Typo-Tolerant Fuzzy Matcher

A lightweight Transformer model for handling misspelled and transliterated food delivery search queries.

**Swiggy AI Engineer Assignment — Challenge 2C**

---

## Results

| Metric | Requirement | Achieved |
|--------|------------|----------|
| Accuracy | 95% | **95.09%** |
| Model Size | <20 MB | 2.0 MB |
| Parameters | <10M | 88,609 |
| Inference Speed | <100ms (CPU, 500 items) | **0.72ms** (with embedding cache) |
| Languages | Hindi/English/Hinglish | Supported |

---

## Quick Start

```bash
cd typo-tolerant-matcher
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cd src
python inference_transformer.py
```

This runs 5 qualitative examples and a speed benchmark.

---

## Model Architecture

Transformer Encoder with a Siamese network for pairwise similarity scoring.

```
Character Input (max 50 chars)
    ↓
Embedding (64-dim) + Positional Encoding
    ↓
Transformer Encoder (3 layers, 4 heads, FFN dim=256)
    ↓
Mean Pooling → L2 Normalized (64-dim)
    ↓
MLP Similarity Scorer → Score (0-1)
```

- **Parameters**: 88,609
- **Model Size**: 2.0 MB (float32)
- **Architecture**: d_model=64, heads=4, layers=3

---

## Training Data

**Sources** (as per assignment guidelines):
1. **Kaggle datasets**: Swiggy Bangalore (10K restaurants), Indian Food 101 (255 dishes) — extracted 413 unique dishes
2. **LLM-generated**: 177 common Indian dishes covering major cuisines

**Total**: 553 unique dishes → 3,117 training examples (2,227 positive with typos, 890 negative pairs)

**Training config**: Contrastive loss (margin=0.25), AdamW (lr=5e-4), batch size 32, early stopped at epoch 41.

---

## Qualitative Examples

| Query (with typo) | Top Match | Score |
|---|---|---|
| "chiken biryani" | Chicken Biryani | 97.1% |
| "panner tikka" | Paneer Tikka | 98.0% |
| "buter chiken" | Butter Chicken | 97.6% |
| "masla dosa" | Masala Dosa | 95.6% |
| "dal makhni" | Dal Makhani | 97.6% |

All correct dishes ranked in the top 3 with >95% confidence.

---

## Performance

With pre-computed target embeddings (the realistic production scenario — dish names are static):

- **1 query vs 500 targets**: 0.72ms on CPU
- **Throughput**: ~700K items/sec
- **Pre-compute cost** (one-time at startup): ~89ms

The key insight: target dish embeddings are computed once at startup. At query time, only the user's query goes through the encoder (~0.5ms), then a lightweight MLP scores it against all cached embeddings (~0.05ms).

---

## Technical Approach

**Why Transformer over CNN?** The CNN baseline (83.12%) only captures local n-grams. The Transformer's self-attention sees the full sequence, which matters for matching "chiken biryani" to "Chicken Biryani" where the error context spans multiple characters. This gave a +12% accuracy improvement.

**Why character-level?** Character-level tokenization naturally handles typos, transliterations, and new dish names without any vocabulary limitations.

**Why contrastive loss?** MSE treats all errors equally. Contrastive loss explicitly pulls matching pairs together and pushes non-matching pairs apart, giving better discrimination.

---

## Project Structure

```
typo-tolerant-matcher/
├── README.md
├── TECHNICAL_REPORT.md
├── docs/
│   └── DATA.md
├── src/
│   ├── model_transformer.py
│   ├── train_transformer.py
│   ├── inference_transformer.py
│   ├── generate_more_data.py
│   └── dataset.py
├── models/
│   ├── transformer_final.pth
│   └── cnn_final.pth
├── data/
│   ├── raw/
│   └── processed/
│       └── training_data_llm.json
└── requirements.txt
```

---

## Reproducibility

```bash
cd src
python generate_more_data.py    # generate training data
python train_transformer.py     # train from scratch (~20 min on CPU)
python inference_transformer.py # run demo + benchmark
```

---

## Model Evolution

| Attempt | Architecture | Data | Accuracy |
|---------|-------------|------|----------|
| 1 | CNN | 1,540 real examples | 83.12% |
| 2 | Transformer | 2,162 examples (+ LLM data) | 92.92% |
| 3 | Transformer | 3,117 examples (expanded) | **95.09%** |

Key improvements: switching to Transformer, expanding training data by 44%, adding diverse typo patterns (5 types including transliteration).

---

## Production Considerations

**Optimizations applied**:
- Pre-computed target embeddings (247x speedup)
- NumPy-based batch tokenizer (6x faster encoding)
- `torch.inference_mode()` instead of `torch.no_grad()`

**Further options** (not implemented):
- ONNX export for 2-3x faster encoder
- INT8 quantization (model → ~0.16MB)
- GPU inference (<20ms)

**Requirements**: Python 3.9+, PyTorch 2.0+, any modern CPU, <100MB RAM.

---

## Documentation

- [Technical Report](typo-tolerant-matcher/TECHNICAL_REPORT.md) — detailed analysis, CNN vs Transformer comparison, design decisions
- [Data Documentation](typo-tolerant-matcher/docs/DATA.md) — data sources, generation process, preprocessing
