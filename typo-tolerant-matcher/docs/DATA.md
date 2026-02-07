# Data Documentation

## Data Sources

### 1. Synthetic Training Data

**Source**: Programmatically generated using domain knowledge

**Generation Method**:
- Base vocabulary: 50+ popular Indian dishes
- Typo generation algorithms
- Transliteration patterns
- Negative sampling

### 2. Base Dish Vocabulary

**Curated list of popular dishes**:
- North Indian: Butter Chicken, Paneer Tikka, Dal Makhani, etc.
- South Indian: Masala Dosa, Idli Sambar, Uttapam, etc.
- Indo-Chinese: Chicken Manchurian, Hakka Noodles, Chilli Chicken, etc.
- Biryani variants: Hyderabadi, Chicken, Mutton, Veg, etc.
- Street food: Pav Bhaji, Chole Bhature, Samosa, etc.
- Desserts: Gulab Jamun, Rasgulla, Jalebi, etc.

**Total**: 50+ unique dishes

## Data Generation Strategy

### Typo Generation

**Algorithm**: `TypoGenerator` class in `data_generator.py`

**Typo Types**:

1. **Character Substitution** (40% of typos)
   - Keyboard proximity: `c → k`, `i → o`
   - Phonetic similarity: `s → c`, `k → q`
   - Transliteration: `a → aa`, `i → ee`

2. **Character Deletion** (25% of typos)
   - Random character removal
   - Example: `chicken → chiken`

3. **Character Swapping** (20% of typos)
   - Adjacent character transposition
   - Example: `biryani → biriyani`

4. **Character Duplication** (15% of typos)
   - Double characters
   - Example: `butter → buttter`

### Transliteration Patterns

**Common Hindi-English variations**:
- `paneer`: panir, panner, panier
- `biryani`: biriyani, biriani, biryaani, briyani
- `chicken`: chiken, chikin, chikn, cheeken
- `masala`: masalla, masaala, masla
- `tikka`: tika, teeka, tikaa
- `tandoori`: tanduri, tandori, tandhuri
- `naan`: nan, na'an
- `roti`: rotti, rotee, roty
- `dal`: daal, dhal, dhaal

### Training Data Composition

**Positive Examples** (60%):
- Exact matches: 50 examples
- Single typo: 100 examples
- Multiple typos: 50 examples
- Transliterations: 50 examples

**Negative Examples** (40%):
- Complete non-matches: 100 examples
- Partial matches (shared words): 50 examples

**Total**: ~400 training examples

### Data Format

```json
{
  "query": "chiken biryani",
  "target": "Chicken Biryani",
  "label": 0.95,
  "category": "typo"
}
```

**Fields**:
- `query`: User search query (with typos/transliterations)
- `target`: Correct dish name
- `label`: Similarity score (0.0 to 1.0)
  - 1.0: Exact match
  - 0.9-0.99: Typo/transliteration
  - 0.3-0.6: Partial match
  - 0.0-0.2: No match
- `category`: Type of example (exact, typo, transliteration, negative)

## Preprocessing

### Text Normalization

1. **Lowercase**: All text converted to lowercase
2. **Character Encoding**: ASCII characters (0-127)
3. **Padding**: Fixed length sequences (max 50 chars)
4. **Unknown Characters**: Mapped to `<UNK>` token

### Tokenization

**Character-Level Tokenization**:
- Vocabulary size: 128 (ASCII)
- Special tokens: `<PAD>` (0), `<UNK>` (1)
- No word boundaries needed
- Handles any Unicode by ASCII approximation

## Data Quality

### Validation

✓ **Typo Realism**: Based on common user errors
✓ **Transliteration Coverage**: Major Hindi-English patterns
✓ **Label Consistency**: Scores reflect actual similarity
✓ **Class Balance**: Mix of positive and negative examples

### Limitations

- **Limited to common dishes**: May not cover regional specialties
- **Synthetic data**: Not based on real user queries
- **English/Hinglish only**: No Devanagari script support
- **No context**: Doesn't consider user preferences or location

## Future Data Improvements

1. **Real User Queries**: Collect from production logs (if available)
2. **Regional Variations**: Add more regional dish names
3. **Context**: Include user history and location
4. **Multi-lingual**: Support for more Indian languages
5. **Seasonal Dishes**: Special occasion foods

## LLM-Generated Data

**Not used in this implementation** to keep it fully deterministic and reproducible.

Alternative approach (if needed):
```python
# Prompt for LLM data generation
"""
Generate 100 realistic food delivery app search queries
that Indian users would type. Cover these categories equally:
- Exact dish names
- Misspelled dish names
- Cuisine types
- Dietary preferences (veg, keto, jain)
- Occasion-based (party, quick lunch)
- Attribute-based (spicy, less oily)
- Hinglish queries
- Vague/exploratory queries

Format as JSON: [{
  "query": "...",
  "category": "...",
  "expected_keywords": [...]
}]
"""
```

## Data Statistics

- **Training Examples**: ~320
- **Validation Examples**: ~80
- **Average Query Length**: 15-20 characters
- **Vocabulary Coverage**: 50+ unique dishes
- **Typo Rate**: 20-30% of training examples
- **Transliteration Rate**: 15% of training examples

## Reproducibility

**Seed**: Random seed set for reproducibility
**Generation Script**: `src/data_generator.py`
**Output**: `data/processed/training_data.json`

To regenerate data:
```bash
cd src
python data_generator.py
```
