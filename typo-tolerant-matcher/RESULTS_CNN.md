# Results: Character-Level CNN Approach

## Model Architecture

**Type**: Character-Level CNN with Siamese Network + Contrastive Loss

**Architecture:**
```
Character Embedding (vocab=128, dim=32)
    ↓
Multi-Scale CNN (kernels: 2, 3, 4 | filters: 128 each)
    ↓
Max Pooling + Concatenation (384 dims)
    ↓
Dropout (0.5) + Linear (384 → 64)
    ↓
L2 Normalization → Embeddings
    ↓
MLP Similarity Scorer (128 → 32 → 1)
```

## Model Stats

- **Parameters**: 70,145
- **Model Size**: 0.84 MB (well under 20MB limit)
- **Training Device**: CPU
- **Loss Function**: Contrastive Loss (margin=0.3)
- **Optimizer**: AdamW (lr=0.001, weight_decay=0.01)

## Training Data

**Source**: Real datasets only
- Indian Food 101: 255 dishes
- Swiggy Bangalore: 192 categories
- **Total unique dishes**: 350 (after filtering)

**Training Examples**: 1,540
- Positive (typos): 543 (simple, moderate, multiple, transliteration)
- Negative: 997 (easy + hard negatives)

**Split**: 80/20 train/validation
- Training: 1,232 examples
- Validation: 308 examples

## Results

### Final Performance

| Metric | Value |
|--------|-------|
| **Best Val Accuracy** | **83.12%** |
| Train Accuracy | 96.7% |
| Val Loss | 0.059 |
| Training Epochs | 100 |
| Training Time | ~15 minutes (CPU) |

### Performance Over Time

- Epoch 1: Val Acc = 49.06%
- Epoch 10: Val Acc = 70.44%
- Epoch 20: Val Acc = 78.30%
- Epoch 50: Val Acc = 82.15%
- Epoch 100: Val Acc = 83.12%

### Inference Speed

- **Batch of 500 items**: 200ms (CPU)
- **Per item**: 0.4ms
- **Throughput**: 2,498 queries/sec

## Analysis

### Strengths
✅ Very lightweight (0.84MB)
✅ Fast inference (2,498/sec)
✅ Naturally handles character-level typos
✅ No vocabulary limitations

### Limitations
❌ Accuracy plateaued at ~83%
❌ Overfitting (96.7% train vs 83% val)
❌ Limited by small dataset (350 dishes, 1540 examples)
❌ Character-level may miss word-level semantics

### Why 83% (not 95%+)

1. **Small Dataset**
   - Only 350 unique dishes from real datasets
   - 308 validation examples too small
   - Limited typo variations

2. **Model Capacity**
   - CNN may be too simple for complex patterns
   - Character-level loses word-level context
   - No attention mechanism

3. **Data Quality**
   - Real datasets had noise (locations, categories mixed with dishes)
   - Limited coverage of Indian food diversity

## Sample Predictions

### Good Examples (Correct)
```
Query: "chickne tikkaa" → "Chicken Tikka" (Score: 0.94) ✓
Query: "biryaani" → "Biryani" (Score: 0.92) ✓
Query: "panner" → "Paneer" (Score: 0.89) ✓
```

### Failure Cases
```
Query: "briyani" → "Butter Chicken" (Score: 0.78) ✗ (Should match "Biryani")
Query: "tikka masala" → "Tikka" (Score: 0.55) ✗ (Partial match confusion)
```

## Conclusion

**Character-Level CNN achieved 83.12% accuracy**, which is:
- Better than baseline (49%)
- Meets size/speed constraints
- But **falls short of 95% target**

**Next Steps:**
- Try Transformer architecture (better context modeling)
- Generate more diverse training data (LLM-generated)
- Increase dataset to 5000+ examples

---

*Model saved as: models/best_model_v2.pth*
*Date: 2026-02-07*
