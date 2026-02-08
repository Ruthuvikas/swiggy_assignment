import json
import random
import pandas as pd
from typing import List, Dict, Tuple


class TypoGenerator:
    """Generate realistic typos for training data"""

    # Common character substitutions (keyboard proximity, phonetic, transliteration)
    SUBSTITUTIONS = {
        'a': ['a', 'e', 'aa', '@'],
        'e': ['e', 'a', 'i', '3'],
        'i': ['i', 'e', 'y', '1'],
        'o': ['o', 'a', 'u', '0'],
        'u': ['u', 'o', 'oo'], 
        'c': ['c', 'k', 's'],
        'k': ['k', 'c', 'q', 'kk'],
        's': ['s', 'c', 'z', 'ss'],
        'z': ['z', 's'],
        'p': ['p', 'b', 'pp'],
        'b': ['b', 'p', 'v', 'bb'],
        't': ['t', 'd', 'tt'],
        'd': ['d', 't', 'dd'],
        'n': ['n', 'm', 'nn'],
        'm': ['m', 'n', 'mm'],
        'r': ['r', 'rr'],
        'l': ['l', 'll'],
        'h': ['h', ''],
    }

    @staticmethod
    def add_typo(word: str, typo_rate: float = 0.2) -> str:
        """Add random typo to a word"""
        if random.random() > typo_rate or len(word) < 2:
            return word

        word = list(word.lower())
        typo_type = random.choice(['substitute', 'delete', 'swap', 'duplicate', 'insert'])

        if typo_type == 'substitute' and len(word) > 0:
            idx = random.randint(0, len(word) - 1)
            char = word[idx]
            if char in TypoGenerator.SUBSTITUTIONS:
                word[idx] = random.choice(TypoGenerator.SUBSTITUTIONS[char])
            else:
                # Random adjacent keyboard key
                word[idx] = random.choice('abcdefghijklmnopqrstuvwxyz')

        elif typo_type == 'delete' and len(word) > 2:
            idx = random.randint(0, len(word) - 1)
            word.pop(idx)

        elif typo_type == 'swap' and len(word) > 1:
            idx = random.randint(0, len(word) - 2)
            word[idx], word[idx + 1] = word[idx + 1], word[idx]

        elif typo_type == 'duplicate' and len(word) > 0:
            idx = random.randint(0, len(word) - 1)
            word.insert(idx, word[idx])

        elif typo_type == 'insert' and len(word) > 0:
            # Insert random character
            idx = random.randint(0, len(word))
            word.insert(idx, random.choice('aeiou'))

        return ''.join(word)

    @staticmethod
    def add_transliteration(text: str) -> str:
        """Add Hindi transliteration patterns"""
        transliterations = {
            'chicken': ['chiken', 'chikin', 'chikn', 'cheeken'],
            'biryani': ['biryani', 'biriyani', 'biriani', 'biryaani', 'briyani'],
            'paneer': ['paneer', 'panir', 'panner', 'panier'],
            'butter': ['butter', 'buter', 'buttar', 'butar'],
            'masala': ['masala', 'masalla', 'masaala', 'masla'],
            'tikka': ['tikka', 'tika', 'teeka', 'tikaa'],
            'tandoori': ['tandoori', 'tanduri', 'tandori', 'tandhuri'],
            'naan': ['naan', 'nan', 'naan', 'na\'an'],
            'roti': ['roti', 'rotti', 'rotee', 'roty'],
            'dal': ['dal', 'daal', 'dhal', 'dhaal'],
        }

        words = text.lower().split()
        for i, word in enumerate(words):
            for correct, variants in transliterations.items():
                if word == correct:
                    words[i] = random.choice(variants)
                    break

        return ' '.join(words)


def generate_food_queries() -> List[Dict]:
    """Generate realistic food delivery search queries"""

    dishes = [
        "Chicken Biryani", "Paneer Tikka", "Butter Chicken", "Dal Makhani",
        "Tandoori Chicken", "Masala Dosa", "Hyderabadi Biryani", "Mutton Rogan Josh",
        "Palak Paneer", "Chana Masala", "Fish Curry", "Prawn Masala",
        "Veg Fried Rice", "Chicken Fried Rice", "Hakka Noodles",
        "Paneer Butter Masala", "Kadai Paneer", "Malai Kofta",
        "Aloo Gobi", "Baingan Bharta", "Bhindi Masala",
        "Chicken 65", "Chicken Manchurian", "Chilli Chicken",
        "Egg Curry", "Egg Biryani", "Omelette",
        "Pav Bhaji", "Chole Bhature", "Rajma Chawal",
        "Idli Sambar", "Medu Vada", "Uttapam",
        "Paratha", "Aloo Paratha", "Gobi Paratha",
        "Samosa", "Kachori", "Pakora",
        "Gulab Jamun", "Rasgulla", "Jalebi",
        "Chicken Kebab", "Seekh Kebab", "Paneer Tikka",
        "Caesar Salad", "Greek Salad", "Fruit Salad",
        "Margherita Pizza", "Pepperoni Pizza", "Veg Pizza",
        "Chicken Burger", "Veg Burger", "Cheese Burger",
        "Pasta Alfredo", "Pasta Arrabiata", "Mac and Cheese",
        "Spring Roll", "Momos", "Dim Sum"
    ]

    queries = []
    typo_gen = TypoGenerator()

    for dish in dishes:
        # Exact match
        queries.append({
            "query": dish.lower(),
            "target": dish,
            "label": 1.0,
            "category": "exact"
        })

        # Typo variations
        for _ in range(2):
            typo_query = ' '.join([typo_gen.add_typo(word, 0.3) for word in dish.lower().split()])
            queries.append({
                "query": typo_query,
                "target": dish,
                "label": 0.95,
                "category": "typo"
            })

        # Transliteration
        trans_query = typo_gen.add_transliteration(dish.lower())
        if trans_query != dish.lower():
            queries.append({
                "query": trans_query,
                "target": dish,
                "label": 0.95,
                "category": "transliteration"
            })

    # Add negative examples (non-matching pairs)
    for _ in range(len(queries) // 2):
        dish1, dish2 = random.sample(dishes, 2)
        # Check if they share common words (partial match)
        words1 = set(dish1.lower().split())
        words2 = set(dish2.lower().split())
        overlap = len(words1 & words2)

        if overlap > 0:
            label = 0.3 + (overlap * 0.2)  # Partial match
        else:
            label = 0.0  # No match

        queries.append({
            "query": dish1.lower(),
            "target": dish2,
            "label": label,
            "category": "negative"
        })

    random.shuffle(queries)
    return queries


if __name__ == "__main__":
    queries = generate_food_queries()
    print(f"Generated {len(queries)} training examples")

    # Show some examples
    for q in queries[:10]:
        print(f"Query: '{q['query']}' -> Target: '{q['target']}' | Score: {q['label']:.2f} | Category: {q['category']}")
