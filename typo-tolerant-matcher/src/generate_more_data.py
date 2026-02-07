"""Generate maximum training data - simplified and correct"""
import json
import random
from typing import List, Dict
import pandas as pd
from data_generator import TypoGenerator

# Load LLM dishes
LLM_DISHES = [
    "Butter Chicken", "Chicken Tikka Masala", "Paneer Butter Masala",
    "Dal Makhani", "Palak Paneer", "Kadai Paneer", "Malai Kofta",
    "Aloo Gobi", "Bhindi Masala", "Baingan Bharta", "Chana Masala",
    "Rajma Chawal", "Chole Bhature", "Pav Bhaji", "Samosa", "Pakora",
    "Tandoori Chicken", "Chicken Korma", "Rogan Josh", "Nihari",
    "Biryani", "Chicken Biryani", "Mutton Biryani", "Hyderabadi Biryani",
    "Veg Biryani", "Egg Biryani", "Naan", "Garlic Naan", "Butter Naan",
    "Roti", "Tandoori Roti", "Aloo Paratha", "Gobi Paratha", "Paneer Paratha",
    "Masala Dosa", "Plain Dosa", "Rava Dosa", "Set Dosa", "Mysore Masala Dosa",
    "Idli", "Medu Vada", "Sambar Vada", "Idli Sambar", "Uttapam",
    "Onion Uttapam", "Tomato Uttapam", "Pongal", "Upma", "Pesarattu",
    "Chicken Fried Rice", "Veg Fried Rice", "Egg Fried Rice", "Schezwan Rice",
    "Hakka Noodles", "Chow Mein", "Singapore Noodles", "Manchurian",
    "Chicken Manchurian", "Veg Manchurian", "Gobi Manchurian",
    "Chilli Chicken", "Chilli Paneer", "Dragon Chicken", "Honey Chicken",
    "Spring Roll", "Veg Spring Roll", "Chicken Spring Roll",
    "Momos", "Veg Momos", "Chicken Momos", "Fried Momos", "Steamed Momos",
    "Pani Puri", "Bhel Puri", "Sev Puri", "Dahi Puri", "Ragda Pattice",
    "Vada Pav", "Misal Pav", "Dabeli", "Kachori", "Aloo Tikki",
    "Gulab Jamun", "Rasgulla", "Rasmalai", "Jalebi", "Kheer",
    "Gajar Halwa", "Moong Dal Halwa", "Badam Halwa", "Kulfi",
    "Masala Chai", "Filter Coffee", "Cold Coffee", "Lassi",
    "Mango Lassi", "Sweet Lassi", "Salt Lassi", "Buttermilk",
    "Margherita Pizza", "Pepperoni Pizza", "Veg Pizza", "Chicken Pizza",
    "Pasta Alfredo", "Pasta Arrabiata", "Mac and Cheese",
    "Burger", "Cheese Burger", "Chicken Burger", "Veg Burger",
    "French Fries", "Sandwich", "Club Sandwich", "Grilled Sandwich",
    "Caesar Salad", "Greek Salad", "Dhokla", "Khandvi", "Thepla",
    "Fish Curry", "Prawn Curry", "Pomfret Fry", "Fish Tikka",
    "Seekh Kebab", "Tangdi Kebab", "Paneer Tikka"
]

# Load real datasets
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

    ALL_DISHES = list(set(LLM_DISHES) | real_dishes)
except:
    ALL_DISHES = LLM_DISHES

print(f"Total dishes: {len(ALL_DISHES)}")

typo_gen = TypoGenerator()
pairs = []

# Generate 10 positive variations per dish
print("Generating positive examples...")
for dish in ALL_DISHES:
    typo_types = ['light', 'moderate', 'heavy', 'transliteration', 'mixed'] * 2  # 10 variations

    for typo_type in typo_types:
        if typo_type == 'light':
            words = dish.split()
            if words:
                idx = random.randint(0, len(words) - 1)
                words[idx] = typo_gen.add_typo(words[idx], 0.3)
            query = ' '.join(words).lower()
            score = 0.98

        elif typo_type == 'moderate':
            words = dish.split()
            typo_words = [typo_gen.add_typo(w, 0.4) for w in words]
            query = ' '.join(typo_words).lower()
            score = 0.94

        elif typo_type == 'heavy':
            words = dish.split()
            typo_words = [typo_gen.add_typo(w, 0.6) for w in words]
            query = ' '.join(typo_words).lower()
            score = 0.87

        elif typo_type == 'transliteration':
            query = typo_gen.add_transliteration(dish.lower())
            score = 0.96

        else:  # mixed
            query = typo_gen.add_transliteration(dish.lower())
            words = query.split()
            typo_words = [typo_gen.add_typo(w, 0.4) for w in words]
            query = ' '.join(typo_words)
            score = 0.90

        if query != dish.lower():
            pairs.append({
                'query': query,
                'target': dish,
                'label': score,
                'category': f'positive_{typo_type}'
            })

print(f"Generated {len(pairs)} positive examples")

# Generate negative examples (40% of positives)
print("Generating negative examples...")
num_negatives = int(len(pairs) * 0.4)

for _ in range(num_negatives):
    dish1, dish2 = random.sample(ALL_DISHES, 2)

    words1 = set(dish1.lower().split())
    words2 = set(dish2.lower().split())
    overlap = len(words1 & words2)

    if overlap > 0:
        score = min(0.45, 0.15 + overlap * 0.15)
        category = 'hard_negative'
    else:
        score = 0.0
        category = 'easy_negative'

    pairs.append({
        'query': dish1.lower(),
        'target': dish2,
        'label': score,
        'category': category
    })

random.shuffle(pairs)

print(f"\nTotal training examples: {len(pairs)}")

# Show distribution
categories = {}
for item in pairs:
    cat = item['category']
    categories[cat] = categories.get(cat, 0) + 1

print("\nCategory distribution:")
for cat, count in sorted(categories.items()):
    print(f"  {cat}: {count}")

# Save
with open('../data/processed/training_data_llm.json', 'w') as f:
    json.dump(pairs, f, indent=2)

print(f"\nSaved to ../data/processed/training_data_llm.json")

# Show examples
print("\nSample examples:")
for i, item in enumerate(random.sample(pairs[:100], min(10, len(pairs)))):
    print(f"{i+1}. '{item['query']}' → '{item['target']}' | {item['label']:.2f} | {item['category']}")
