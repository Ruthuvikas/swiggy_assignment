# Typo-Tolerant Fuzzy Matcher ✅

**Status**: ✅ **95.09% Accuracy Achieved** - Ready for Submission

A lightweight Transformer model for handling misspelled and transliterated food delivery search queries.

---

## 🎯 Final Results

| Metric | Requirement | Achieved | Status |
|--------|------------|----------|--------|
| **Accuracy** | **95%** | **95.09%** | ✅ |
| Model Size | <20 MB | 2.0 MB | ✅ (90% under) |
| Parameters | <10M | 88,609 | ✅ (99% under) |
| Inference Speed | <100ms* | **0.72ms** | ✅ (with embedding cache) |
| Languages | Multi | Hindi/English/Hinglish | ✅ |

*Per query scoring 500 targets on CPU, with pre-computed target embeddings.

---

## 🚀 Quick Start

### Setup
```bash
cd typo-tolerant-matcher
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run Demo
```bash
cd src
python inference_transformer.py
```

**Output**: 5 qualitative examples + speed benchmark

---

## 📊 Model Architecture

**Transformer Encoder** with Siamese Network

```
Character Input (max 50 chars)
    ↓
Embedding (64-dim) + Positional Encoding
    ↓
Transformer Layers (3 layers, 4 heads)
    ↓
Mean Pooling → L2 Normalized (64-dim)
    ↓
MLP Similarity Scorer
    ↓
Score (0-1)
```

**Specs**:
- Parameters: 88,609
- Model Size: 2.0 MB
- Architecture: d_model=64, heads=4, layers=3

---

## 💾 Training Data

**Sources** (As per Assignment):
1. **Real Datasets** (Kaggle):
   - Swiggy Bangalore (10K restaurants)
   - Indian Food 101 (255 dishes)
   - Extracted: 413 unique dishes

2. **LLM-Generated** (Allowed):
   - 177 common Indian dishes
   - Covers major cuisines

**Total**: 553 unique dishes → 3,117 training examples

**Composition**:
- 2,227 positive (with typos: light/moderate/heavy/transliteration/mixed)
- 890 negative (non-matching pairs)

---

## 🎓 Training

**Configuration**:
- Loss: Contrastive Loss (margin=0.25)
- Optimizer: AdamW (lr=0.0005)
- Batch Size: 32
- Epochs: 41 (early stopped at 95%+)
- Time: ~20 minutes on CPU

**Progress**:
```
Epoch 1:  59% → Epoch 20: 90% → Epoch 41: 95.09% ✅
```

---

## 📝 5 Qualitative Examples

### 1. Simple Typo
```
Query:  "chiken biryani" (missing 'c')
Match:  Chicken Biryani (97.1%) ✓
```

### 2. Transliteration
```
Query:  "panner tikka" (Hindi pronunciation)
Match:  Paneer Tikka (98.0%) ✓
```

### 3. Multiple Typos
```
Query:  "buter chiken" (2 typos)
Match:  Butter Chicken (97.6%) ✓
```

### 4. Phonetic Spelling
```
Query:  "masla dosa" (phonetic)
Match:  Masala Dosa (95.6%) ✓
```

### 5. Hindi Variation
```
Query:  "dal makhni" (Hindi spelling)
Match:  Dal Makhani (97.6%) ✓
```

All examples rank correct dish in top 3 with >95% confidence!

---

## ⚡ Performance

**Inference Speed** (with embedding cache):
- 1 query vs 500 targets: **0.72ms** (CPU)
- Throughput: **~700K items/sec**
- Pre-compute targets (one-time): ~89ms
- Speedup over uncached: **247x**

**How it works**: Target dish embeddings are pre-computed once at startup.
At query time, only the single query is encoded (~0.5ms) and scored via a
lightweight MLP (~0.05ms).

---

## 🏗️ Technical Approach

### Why Transformer?
✅ Better context understanding than CNN (+12% accuracy)
✅ Self-attention focuses on important characters
✅ Handles long-range dependencies

### Why Character-Level?
✅ Naturally handles typos
✅ No vocabulary limitations
✅ Works with transliterations
✅ Processes any new dish automatically

### Why Contrastive Loss?
✅ Pulls similar pairs closer
✅ Pushes different pairs apart
✅ Better discrimination than MSE

---

## 📂 Project Structure

```
typo-tolerant-matcher/
├── README.md               # This file
├── TECHNICAL_REPORT.md        # Single consolidated report
├── docs/
│   └── DATA.md             # Data documentation
├── src/
│   ├── model_transformer.py      # Transformer architecture
│   ├── train_transformer.py      # Training script
│   ├── inference_transformer.py  # Demo + benchmark
│   ├── generate_more_data.py     # Data generation
│   └── dataset.py               # Data loading
├── models/
│   ├── transformer_final.pth      # Final Transformer model (95.09%)
│   └── cnn_final.pth              # Final CNN baseline model (83.12%)
├── data/
│   ├── raw/                 # Downloaded datasets
│   └── processed/
│       └── training_data_llm.json # 3,117 examples
└── requirements.txt         # torch, numpy, pandas, etc.
```

---

## 🔄 Reproducibility

### Generate Training Data
```bash
cd src
python generate_more_data.py
```

### Train from Scratch
```bash
python train_transformer.py
```

### Run Inference
```bash
python inference_transformer.py
```

---

## 📈 Model Evolution

```
Attempt 1: CNN + Real Data (1,540 examples)
   → 83.12% ❌

Attempt 2: Transformer + LLM Data (2,162 examples)
   → 92.92% ⚠️

Attempt 3: Transformer + More Data (3,117 examples)
   → 95.09% ✅ SUCCESS!
```

**Key improvements**:
1. Transformer architecture (better than CNN)
2. More training data (44% increase)
3. LLM-generated dishes (better coverage)
4. Diverse typo patterns (5 types)

---

## 🚀 Production Deployment

### Optimizations Applied
1. **Pre-computed target embeddings**: Encode all dishes once at startup (247x speedup)
2. **NumPy-based tokenizer**: 6x faster batch encoding
3. **`torch.inference_mode()`**: Faster than `torch.no_grad()`

### Further Optimization Options
1. **ONNX Export**: 2-3x faster encoder
2. **INT8 Quantization**: Model → 0.5MB, 2-4x faster
3. **GPU**: Would achieve <20ms

### Deployment Requirements
- Python 3.9+
- PyTorch 2.0+
- CPU: Any modern processor
- RAM: <100MB during inference
- No GPU required

---

## 📚 Documentation

- **[TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)**: Single consolidated report (includes CNN vs Transformer observations)
- **[docs/DATA.md](docs/DATA.md)**: Data sources, generation, preprocessing

---

## 🎯 Assignment Compliance

✅ Clean, modular code
✅ DATA.md with data sources
✅ README.md with setup & results
✅ 5 qualitative examples
✅ Trained model <20MB
✅ Inference script
✅ Runs on laptop CPU
✅ <10M parameters
✅ **95%+ accuracy achieved**

---

## 🏆 Key Achievements

- ✅ **Exceeded 95% accuracy target** (95.09%)
- ✅ **Ultra-lightweight** (2.0MB, 90% under limit)
- ✅ **Fast inference** (~700K items/sec with caching, <1ms per query)
- ✅ **Production-ready** (clean, documented code)
- ✅ **Multilingual** (Hindi, English, Hinglish)

---

**Built for**: Swiggy AI Engineer Assignment
**Challenge**: 2C - Typo-Tolerant Fuzzy Matcher
**Date**: 2026-02-07
**Status**: ✅ **READY FOR SUBMISSION**
