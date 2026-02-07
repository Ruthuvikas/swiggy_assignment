import torch
import torch.nn as nn
import torch.nn.functional as F


class CharCNN(nn.Module):
    """Character-level CNN encoder for handling typos and transliterations"""
    def __init__(self, vocab_size=128, embedding_dim=32, hidden_dim=128, output_dim=64):
        super(CharCNN, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Multiple filter sizes to capture different n-gram patterns
        self.conv1 = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=2, padding=1)
        self.conv2 = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(embedding_dim, hidden_dim, kernel_size=4, padding=2)

        self.fc = nn.Linear(hidden_dim * 3, output_dim)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embedding_dim)
        embedded = embedded.transpose(1, 2)  # (batch, embedding_dim, seq_len)

        # Apply convolutions
        conv1_out = F.relu(self.conv1(embedded))
        conv2_out = F.relu(self.conv2(embedded))
        conv3_out = F.relu(self.conv3(embedded))

        # Max pooling
        pool1 = F.max_pool1d(conv1_out, conv1_out.size(2)).squeeze(2)
        pool2 = F.max_pool1d(conv2_out, conv2_out.size(2)).squeeze(2)
        pool3 = F.max_pool1d(conv3_out, conv3_out.size(2)).squeeze(2)

        # Concatenate
        pooled = torch.cat([pool1, pool2, pool3], dim=1)
        pooled = self.dropout(pooled)

        # Final projection
        output = self.fc(pooled)
        output = F.normalize(output, p=2, dim=1)  # L2 normalize for cosine similarity

        return output


class TypoTolerantMatcher(nn.Module):
    """Typo-tolerant fuzzy matcher for food delivery queries"""
    def __init__(self, vocab_size=128, embedding_dim=32, hidden_dim=128, output_dim=64):
        super(TypoTolerantMatcher, self).__init__()

        self.encoder = CharCNN(vocab_size, embedding_dim, hidden_dim, output_dim)

        # Similarity scoring layer (optional, can use cosine directly)
        self.similarity_fc = nn.Sequential(
            nn.Linear(output_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, query, target):
        """
        Args:
            query: (batch, seq_len) - encoded query string
            target: (batch, seq_len) - encoded target string
        Returns:
            similarity score between 0 and 1
        """
        query_embed = self.encoder(query)
        target_embed = self.encoder(target)

        # Concatenate for similarity scoring
        combined = torch.cat([query_embed, target_embed], dim=1)
        score = self.similarity_fc(combined)

        return score.squeeze(1)

    def encode(self, text):
        """Encode text to embedding vector"""
        return self.encoder(text)

    def score_batch(self, query_embeds, target_embeds):
        """Fast batch scoring using pre-computed embeddings"""
        combined = torch.cat([query_embeds, target_embeds], dim=1)
        scores = self.similarity_fc(combined)
        return scores.squeeze(1)


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model size
    model = TypoTolerantMatcher(vocab_size=128, embedding_dim=32, hidden_dim=128, output_dim=64)
    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Estimated size: {count_parameters(model) * 4 / 1024 / 1024:.2f} MB (float32)")
