import torch
import time
import numpy as np
from typing import List, Tuple
from model import TypoTolerantMatcher
from dataset import CharTokenizer


class FastInference:
    """Optimized inference for batch scoring"""
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.device = device

        # Load model
        checkpoint = torch.load(model_path, map_location=device)
        config = checkpoint['config']

        self.model = TypoTolerantMatcher(
            vocab_size=config['vocab_size'],
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            output_dim=config['output_dim']
        )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()

        self.tokenizer = CharTokenizer(max_len=config.get('max_len', 50))

        print(f"Model loaded on {device}")
        print(f"Model size: {sum(p.numel() for p in self.model.parameters()) * 4 / 1024 / 1024:.2f} MB")

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        """Encode multiple texts to tensor"""
        encoded = [self.tokenizer.encode(text) for text in texts]
        return torch.tensor(encoded, dtype=torch.long).to(self.device)

    def score_batch(self, queries: List[str], targets: List[str]) -> np.ndarray:
        """
        Score a batch of query-target pairs

        Args:
            queries: List of query strings
            targets: List of target strings

        Returns:
            Array of similarity scores
        """
        with torch.no_grad():
            query_tensor = self.encode_texts(queries)
            target_tensor = self.encode_texts(targets)

            scores = self.model(query_tensor, target_tensor)

            return scores.cpu().numpy()

    def score_one_to_many(self, query: str, targets: List[str]) -> List[Tuple[str, float]]:
        """
        Score one query against many targets (typical search scenario)

        Args:
            query: Single query string
            targets: List of target strings

        Returns:
            List of (target, score) tuples sorted by score
        """
        queries = [query] * len(targets)
        scores = self.score_batch(queries, targets)

        results = list(zip(targets, scores))
        results.sort(key=lambda x: x[1], reverse=True)

        return results

    def benchmark(self, num_items: int = 500):
        """Benchmark inference speed"""
        print(f"\nBenchmarking with {num_items} items...")

        # Generate dummy data
        queries = [f"chicken biryani {i}" for i in range(num_items)]
        targets = [f"Chicken Biryani Special {i}" for i in range(num_items)]

        # Warmup
        _ = self.score_batch(queries[:10], targets[:10])

        # Benchmark
        start = time.time()
        scores = self.score_batch(queries, targets)
        elapsed = (time.time() - start) * 1000  # Convert to ms

        print(f"✓ Scored {num_items} items in {elapsed:.2f}ms")
        print(f"  Average: {elapsed/num_items:.3f}ms per item")
        print(f"  Throughput: {num_items/(elapsed/1000):.0f} items/sec")

        return elapsed


def demo():
    """Demo the inference"""
    print("Loading model...")
    inference = FastInference('../models/best_model.pth', device='cpu')

    # Example queries with typos
    test_cases = [
        ("chiken biryani", ["Chicken Biryani", "Chicken Tikka", "Fish Biryani", "Veg Biryani"]),
        ("panner tikka", ["Paneer Tikka", "Paneer Butter Masala", "Chicken Tikka", "Tandoori Paneer"]),
        ("buter chiken", ["Butter Chicken", "Chicken Curry", "Butter Naan", "Chicken Biryani"]),
        ("dal makhni", ["Dal Makhani", "Dal Tadka", "Rajma Masala", "Chole Masala"]),
        ("tanduri roti", ["Tandoori Roti", "Tandoori Chicken", "Butter Roti", "Garlic Naan"]),
    ]

    print("\n" + "="*70)
    print("DEMO: Typo-Tolerant Fuzzy Matching")
    print("="*70)

    for query, targets in test_cases:
        print(f"\nQuery: '{query}'")
        print("-" * 50)

        results = inference.score_one_to_many(query, targets)

        for target, score in results:
            bar = "█" * int(score * 20)
            print(f"  {score:.3f} {bar:20s} {target}")

    # Benchmark
    inference.benchmark(500)


if __name__ == "__main__":
    demo()
