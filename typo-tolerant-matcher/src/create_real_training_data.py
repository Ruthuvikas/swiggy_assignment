import pandas as pd
import random
import json
from typing import List, Dict
import sys
sys.path.append('.')
from data_generator import TypoGenerator


def extract_dishes_from_datasets():
    """Extract all unique dish names from datasets"""

    # Load datasets
    swiggy = pd.read_csv('../data/raw/Swiggy Bangalore.csv')
    indian_food = pd.read_csv('../data/raw/indian_food.csv')

    dishes = set()

    # 1. From Indian Food 101 - these are verified dish names
    print("Extracting from Indian Food 101...")
    for name in indian_food['name'].dropna():
        dish = str(name).strip()
        if len(dish) > 2 and len(dish) < 50:
            dishes.add(dish)

    print(f"  Found {len(dishes)} dishes from Indian Food 101")

    # 2. From Swiggy Category - these are cuisine/dish types
    print("Extracting from Swiggy categories...")
    category_dishes = set()
    for cat in swiggy['Category'].dropna():
        # Categories are like "Biryani, North Indian, Chinese"
        items = str(cat).split(',')
        for item in items:
            item = item.strip()
            # Only add if it looks like a dish/cuisine (no special chars, reasonable length)
            if (3 <= len(item) <= 40 and
                not any(c in item for c in ['@', '(', ')', '[', ']', '{', '}']) and
                not item.replace(' ', '').replace('-', '').isdigit()):
                category_dishes.add(item)

    print(f"  Found {len(category_dishes)} categories from Swiggy")
    dishes.update(category_dishes)

    return sorted(list(dishes))


def create_training_pairs(dishes: List[str], num_examples: int = 5000):
    """Create training pairs with typos and negatives"""

    typo_gen = TypoGenerator()
    pairs = []

    # Limit dishes to reasonable food-related ones
    food_keywords = ['biryani', 'chicken', 'paneer', 'rice', 'curry', 'masala',
                     'dal', 'naan', 'roti', 'tikka', 'kebab', 'dosa', 'idli',
                     'tandoori', 'butter', 'fried', 'chilli', 'manchurian',
                     'pizza', 'burger', 'pasta', 'salad', 'soup', 'roll',
                     'fish', 'mutton', 'prawn', 'egg', 'veg', 'aloo', 'gobi',
                     'palak', 'chana', 'rajma', 'chole', 'pav', 'bhaji',
                     'paratha', 'samosa', 'pakora', 'vada', 'uttapam',
                     'gulab', 'rasgulla', 'jalebi', 'halwa', 'kheer',
                     'momo', 'thali', 'korma', 'vindaloo', 'fry']

    # Strict filtering
    filtered_dishes = []
    for dish in dishes:
        dish_lower = dish.lower()
        # Must contain food keyword AND not have special chars
        if (any(kw in dish_lower for kw in food_keywords) and
            not '@' in dish and not '(' in dish and not ')' in dish and
            len(dish) >= 3 and len(dish) <= 40):
            filtered_dishes.append(dish)

    print(f"Filtered to {len(filtered_dishes)} food-related dishes from {len(dishes)}")

    # If too few, add all dishes without special chars
    if len(filtered_dishes) < 200:
        for dish in dishes:
            if (not '@' in dish and not '(' in dish and not ')' in dish and
                3 <= len(dish) <= 40 and dish.replace(' ', '').isalpha()):
                filtered_dishes.append(dish)

    # Remove duplicates
    filtered_dishes = list(set(filtered_dishes))
    print(f"After cleanup: {len(filtered_dishes)} dishes")

    # Generate positive examples (with typos) - MULTIPLE variations per dish for better coverage
    num_positives = int(num_examples * 0.6)

    # Generate multiple typo variations for each dish to increase diversity
    for dish in filtered_dishes:
        for variation_num in range(4):  # 4 variations per dish
            if len(pairs) >= num_positives:
                break
            # Generate different types of typos with multiple variations
            typo_type = ['simple', 'moderate', 'multiple', 'transliteration'][variation_num % 4]

            if typo_type == 'simple':
                # Light typo (one word affected)
                words = dish.split()
                if words:
                    typo_idx = random.randint(0, len(words) - 1)
                    words[typo_idx] = typo_gen.add_typo(words[typo_idx], 0.4)
                query = ' '.join(words).lower()
                score = 0.97

            elif typo_type == 'moderate':
                # Moderate typos (multiple words, light errors)
                words = dish.split()
                typo_words = [typo_gen.add_typo(w, 0.35) for w in words]
                query = ' '.join(typo_words).lower()
                score = 0.93

            elif typo_type == 'multiple':
                # Heavy typos (all words affected)
                words = dish.split()
                typo_words = [typo_gen.add_typo(w, 0.6) for w in words]
                query = ' '.join(typo_words).lower()
                score = 0.88

            else:  # transliteration
                query = typo_gen.add_transliteration(dish.lower())
                score = 0.95

            if query != dish.lower():  # Only add if there's actual difference
                pairs.append({
                    'query': query,
                    'target': dish,
                    'label': score,
                    'category': 'positive_' + typo_type
                })

    # Generate hard negative examples (similar but different dishes)
    num_hard_negatives = int(num_examples * 0.2)

    for _ in range(num_hard_negatives):
        dish1, dish2 = random.sample(filtered_dishes, 2)

        # Calculate word overlap for partial match score
        words1 = set(dish1.lower().split())
        words2 = set(dish2.lower().split())
        overlap = len(words1 & words2)

        if overlap > 0:
            score = min(0.4, 0.2 + overlap * 0.1)  # Partial match
            pairs.append({
                'query': dish1.lower(),
                'target': dish2,
                'label': score,
                'category': 'hard_negative'
            })

    # Generate easy negative examples (completely different)
    num_easy_negatives = int(num_examples * 0.2)

    for _ in range(num_easy_negatives):
        dish1, dish2 = random.sample(filtered_dishes, 2)

        # Only if completely different
        words1 = set(dish1.lower().split())
        words2 = set(dish2.lower().split())

        if len(words1 & words2) == 0:
            pairs.append({
                'query': dish1.lower(),
                'target': dish2,
                'label': 0.0,
                'category': 'easy_negative'
            })

    random.shuffle(pairs)
    return pairs


if __name__ == "__main__":
    print("Extracting dishes from datasets...")
    dishes = extract_dishes_from_datasets()
    print(f"Found {len(dishes)} unique dishes")

    print("\nCreating training pairs...")
    training_data = create_training_pairs(dishes, num_examples=5000)
    print(f"Generated {len(training_data)} training examples")

    # Show distribution
    categories = {}
    for item in training_data:
        cat = item['category']
        categories[cat] = categories.get(cat, 0) + 1

    print("\nCategory distribution:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    # Save
    output_path = '../data/processed/training_data_real.json'
    with open(output_path, 'w') as f:
        json.dump(training_data, f, indent=2)

    print(f"\nSaved to {output_path}")

    # Show examples
    print("\nSample examples:")
    for i, item in enumerate(random.sample(training_data, 10)):
        print(f"{i+1}. Query: '{item['query']}' -> Target: '{item['target']}' | Score: {item['label']:.2f} | {item['category']}")
