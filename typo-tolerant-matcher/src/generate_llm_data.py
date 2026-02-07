"""
Generate comprehensive synthetic training data using LLM prompts
As allowed by the assignment: "LLM-generated synthetic data"
"""

import json
import random
from typing import List, Dict
import pandas as pd
from data_generator import TypoGenerator


# LLM-Generated Indian Food Dishes (simulating LLM output)
# In production, this would come from Claude/GPT API
LLM_GENERATED_DISHES = [
    # North Indian
    "Butter Chicken", "Chicken Tikka Masala", "Paneer Butter Masala",
    "Dal Makhani", "Palak Paneer", "Kadai Paneer", "Malai Kofta",
    "Aloo Gobi", "Bhindi Masala", "Baingan Bharta", "Chana Masala",
    "Rajma Chawal", "Chole Bhature", "Pav Bhaji", "Samosa", "Pakora",
    "Tandoori Chicken", "Chicken Korma", "Rogan Josh", "Nihari",
    "Biryani", "Chicken Biryani", "Mutton Biryani", "Hyderabadi Biryani",
    "Veg Biryani", "Egg Biryani", "Naan", "Garlic Naan", "Butter Naan",
    "Roti", "Tandoori Roti", "Aloo Paratha", "Gobi Paratha", "Paneer Paratha",

    # South Indian
    "Masala Dosa", "Plain Dosa", "Rava Dosa", "Set Dosa", "Mysore Masala Dosa",
    "Idli", "Medu Vada", "Sambar Vada", "Idli Sambar", "Uttapam",
    "Onion Uttapam", "Tomato Uttapam", "Pongal", "Upma", "Pesarattu",
    "Appam", "Puttu", "Rasam", "Sambar", "Curd Rice", "Lemon Rice",
    "Bisi Bele Bath", "Puliyogare", "Coconut Chutney", "Tomato Chutney",

    # Chinese/Indo-Chinese
    "Chicken Fried Rice", "Veg Fried Rice", "Egg Fried Rice", "Schezwan Rice",
    "Hakka Noodles", "Chow Mein", "Singapore Noodles", "Manchurian",
    "Chicken Manchurian", "Veg Manchurian", "Gobi Manchurian",
    "Chilli Chicken", "Chilli Paneer", "Dragon Chicken", "Honey Chicken",
    "Spring Roll", "Veg Spring Roll", "Chicken Spring Roll",
    "Momos", "Veg Momos", "Chicken Momos", "Fried Momos", "Steamed Momos",
    "Hot and Sour Soup", "Manchow Soup", "Sweet Corn Soup",

    # Street Food
    "Pani Puri", "Bhel Puri", "Sev Puri", "Dahi Puri", "Ragda Pattice",
    "Vada Pav", "Misal Pav", "Pav Bhaji", "Dabeli", "Kachori",
    "Aloo Tikki", "Papdi Chaat", "Dahi Bhalla", "Gol Gappa",

    # Sweets/Desserts
    "Gulab Jamun", "Rasgulla", "Rasmalai", "Jalebi", "Kheer",
    "Gajar Halwa", "Moong Dal Halwa", "Badam Halwa", "Kulfi",
    "Falooda", "Rabri", "Phirni", "Shrikhand", "Sandesh",
    "Ladoo", "Besan Ladoo", "Motichoor Ladoo", "Boondi Ladoo",
    "Barfi", "Kaju Katli", "Peda", "Mysore Pak",

    # Beverages
    "Masala Chai", "Filter Coffee", "Cold Coffee", "Lassi",
    "Mango Lassi", "Sweet Lassi", "Salt Lassi", "Buttermilk",
    "Jaljeera", "Nimbu Pani", "Aam Panna", "Thandai",

    # Continental/Western
    "Margherita Pizza", "Pepperoni Pizza", "Veg Pizza", "Chicken Pizza",
    "Cheese Pizza", "Pasta Alfredo", "Pasta Arrabiata", "Mac and Cheese",
    "Penne Pasta", "Spaghetti", "Lasagna", "Burger", "Cheese Burger",
    "Chicken Burger", "Veg Burger", "French Fries", "Sandwich",
    "Club Sandwich", "Grilled Sandwich", "Caesar Salad", "Greek Salad",

    # Regional Specialties
    "Dhokla", "Khandvi", "Thepla", "Fafda", "Khaman", "Handvo",
    "Misal Pav", "Vada Pav", "Poha", "Upma", "Sabudana Khichdi",
    "Daal Baati Churma", "Litti Chokha", "Tunday Kabab", "Galouti Kabab",
    "Fish Curry", "Prawn Curry", "Pomfret Fry", "Fish Tikka",
    "Chicken Tikka", "Seekh Kebab", "Tangdi Kebab", "Paneer Tikka",
]


def generate_comprehensive_training_data(num_examples: int = 8000) -> List[Dict]:
    """Generate comprehensive training data using LLM-generated dishes"""

    typo_gen = TypoGenerator()
    pairs = []

    print(f"Total LLM-generated dishes: {len(LLM_GENERATED_DISHES)}")

    # Also include real dataset dishes
    try:
        swiggy = pd.read_csv('../data/raw/Swiggy Bangalore.csv')
        indian_food = pd.read_csv('../data/raw/indian_food.csv')

        real_dishes = set()
        for name in indian_food['name'].dropna():
            real_dishes.add(str(name).strip())

        for cat in swiggy['Category'].dropna():
            items = str(cat).split(',')
            for item in items:
                item = item.strip()
                if 3 <= len(item) <= 40 and not any(c in item for c in ['@', '(', ')', '[', ']']):
                    real_dishes.add(item)

        all_dishes = list(set(LLM_GENERATED_DISHES) | real_dishes)
        print(f"Combined with real datasets: {len(all_dishes)} unique dishes")
    except:
        all_dishes = LLM_GENERATED_DISHES
        print("Using LLM-generated dishes only")

    # Generate positive examples with typos (65% - more positive examples for better learning)
    num_positives = int(num_examples * 0.65)

    # Generate MANY variations per dish (at least 10 per dish)
    variations_per_dish = 10

    print(f"Generating {variations_per_dish} variations per dish...")

    positive_count = 0
    for dish in all_dishes:
        for var_num in range(variations_per_dish):
            if positive_count >= num_positives:
                break

            # More diverse typo variations
            typo_type = random.choice(['light', 'light', 'moderate', 'moderate', 'heavy', 'transliteration', 'mixed', 'combo'])

        if typo_type == 'light':
            # Single light typo
            words = dish.split()
            if words:
                idx = random.randint(0, len(words) - 1)
                words[idx] = typo_gen.add_typo(words[idx], 0.3)
            query = ' '.join(words).lower()
            score = 0.98

        elif typo_type == 'moderate':
            # Moderate typos
            words = dish.split()
            typo_words = [typo_gen.add_typo(w, 0.4) for w in words]
            query = ' '.join(typo_words).lower()
            score = 0.94

        elif typo_type == 'heavy':
            # Heavy typos
            words = dish.split()
            typo_words = [typo_gen.add_typo(w, 0.6) for w in words]
            query = ' '.join(typo_words).lower()
            score = 0.87

        elif typo_type == 'transliteration':
            query = typo_gen.add_transliteration(dish.lower())
            score = 0.96

        elif typo_type == 'mixed':
            # Mix of typos and transliteration
            query = typo_gen.add_transliteration(dish.lower())
            words = query.split()
            typo_words = [typo_gen.add_typo(w, 0.3) for w in words]
            query = ' '.join(typo_words)
            score = 0.90

        else:  # combo - extreme variations
            # Multiple transformations
            query = typo_gen.add_transliteration(dish.lower())
            words = query.split()
            typo_words = [typo_gen.add_typo(w, 0.5) for w in words]
            query = ' '.join(typo_words)
            score = 0.85

            if query != dish.lower():
                pairs.append({
                    'query': query,
                    'target': dish,
                    'label': score,
                    'category': f'positive_{typo_type}'
                })
                positive_count += 1

    # Generate hard negative examples (20%)
    num_hard_negatives = int(num_examples * 0.2)

    for _ in range(num_hard_negatives):
        dish1, dish2 = random.sample(all_dishes, 2)

        words1 = set(dish1.lower().split())
        words2 = set(dish2.lower().split())
        overlap = len(words1 & words2)

        if overlap > 0:
            score = min(0.45, 0.15 + overlap * 0.15)
            pairs.append({
                'query': dish1.lower(),
                'target': dish2,
                'label': score,
                'category': 'hard_negative'
            })

    # Generate easy negative examples (20%)
    num_easy_negatives = int(num_examples * 0.2)

    for _ in range(num_easy_negatives):
        dish1, dish2 = random.sample(all_dishes, 2)

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
    print("Generating comprehensive LLM + Real dataset training data...")
    print("=" * 70)

    training_data = generate_comprehensive_training_data(num_examples=5000)

    print(f"\nGenerated {len(training_data)} training examples")

    # Show distribution
    categories = {}
    for item in training_data:
        cat = item['category']
        categories[cat] = categories.get(cat, 0) + 1

    print("\nCategory distribution:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    # Save
    output_path = '../data/processed/training_data_llm.json'
    with open(output_path, 'w') as f:
        json.dump(training_data, f, indent=2)

    print(f"\nSaved to {output_path}")

    # Show examples
    print("\nSample examples:")
    for i, item in enumerate(random.sample(training_data, 15)):
        print(f"{i+1}. '{item['query']}' → '{item['target']}' | {item['label']:.2f} | {item['category']}")
