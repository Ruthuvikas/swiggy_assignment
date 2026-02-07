import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    def __init__(self, d_model, max_len=50):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class TransformerEncoder(nn.Module):
    """Transformer encoder for character sequences"""
    def __init__(self, vocab_size=128, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, output_dim=64):
        super(TransformerEncoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.fc = nn.Linear(d_model, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, src_key_padding_mask=None):
        """
        Args:
            x: (batch, seq_len)
            src_key_padding_mask: (batch, seq_len) - True for padding positions
        """
        # Embedding
        embedded = self.embedding(x) * math.sqrt(self.embedding.embedding_dim)
        embedded = self.pos_encoder(embedded)

        # Transformer
        if src_key_padding_mask is None:
            src_key_padding_mask = (x == 0)  # Padding mask

        transformed = self.transformer(embedded, src_key_padding_mask=src_key_padding_mask)

        # Mean pooling (ignore padding)
        mask = (~src_key_padding_mask).unsqueeze(-1).float()
        pooled = (transformed * mask).sum(dim=1) / mask.sum(dim=1)

        # Project to output dimension
        output = self.fc(self.dropout(pooled))
        output = F.normalize(output, p=2, dim=1)

        return output


class TransformerTypoMatcher(nn.Module):
    """Transformer-based typo-tolerant matcher"""
    def __init__(self, vocab_size=128, d_model=64, nhead=4, num_layers=2,
                 dim_feedforward=128, output_dim=64):
        super(TransformerTypoMatcher, self).__init__()

        self.encoder = TransformerEncoder(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            output_dim=output_dim
        )

        # Similarity scoring
        self.similarity_fc = nn.Sequential(
            nn.Linear(output_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, query, target):
        """
        Args:
            query: (batch, seq_len)
            target: (batch, seq_len)
        Returns:
            similarity scores (batch,)
        """
        query_embed = self.encoder(query)
        target_embed = self.encoder(target)

        combined = torch.cat([query_embed, target_embed], dim=1)
        score = self.similarity_fc(combined)

        return score.squeeze(1)

    def encode(self, text):
        """Encode text to embedding"""
        return self.encoder(text)


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model
    model = TransformerTypoMatcher(
        vocab_size=128,
        d_model=64,
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        output_dim=64
    )

    print(f"Model parameters: {count_parameters(model):,}")
    print(f"Estimated size: {count_parameters(model) * 4 / 1024 / 1024:.2f} MB")

    # Test forward pass
    batch_size = 8
    seq_len = 50
    query = torch.randint(0, 128, (batch_size, seq_len))
    target = torch.randint(0, 128, (batch_size, seq_len))

    output = model(query, target)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.3f}, {output.max():.3f}]")
