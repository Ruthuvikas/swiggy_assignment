import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import json
import numpy as np
from tqdm import tqdm
import os

from model_transformer import TransformerTypoMatcher, count_parameters
from dataset import CharTokenizer, FoodMatchDataset, collate_fn


class ContrastiveLoss(nn.Module):
    """Contrastive loss for similarity learning"""
    def __init__(self, margin=0.3):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, predictions, labels):
        is_similar = (labels > 0.5).float()
        loss_similar = is_similar * torch.pow(1 - predictions, 2)
        loss_dissimilar = (1 - is_similar) * torch.pow(torch.clamp(predictions - self.margin, min=0.0), 2)
        loss = loss_similar + loss_dissimilar
        return loss.mean()


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        query = batch['query'].to(device)
        target = batch['target'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()

        outputs = model(query, target)
        loss = criterion(outputs, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        predictions = (outputs > 0.5).float()
        binary_labels = (labels > 0.5).float()
        correct += (predictions == binary_labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})

    return total_loss / len(dataloader), correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            query = batch['query'].to(device)
            target = batch['target'].to(device)
            labels = batch['label'].to(device)

            outputs = model(query, target)
            loss = criterion(outputs, labels)

            total_loss += loss.item()

            predictions = (outputs > 0.5).float()
            binary_labels = (labels > 0.5).float()
            correct += (predictions == binary_labels).sum().item()
            total += labels.size(0)

    return total_loss / len(dataloader), correct / total


def main():
    config = {
        'vocab_size': 128,
        'd_model': 64,
        'nhead': 4,
        'num_layers': 3,  # Increased layers
        'dim_feedforward': 256,  # Increased capacity
        'output_dim': 64,
        'max_len': 50,
        'batch_size': 32,
        'num_epochs': 150,
        'learning_rate': 0.0005,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'margin': 0.25
    }

    print(f"Using device: {config['device']}")
    print("\n" + "="*70)
    print("TRANSFORMER MODEL TRAINING")
    print("="*70)

    # Load LLM-generated training data
    print("\nLoading LLM-generated training data...")
    with open('../data/processed/training_data_llm.json', 'r') as f:
        queries = json.load(f)

    print(f"Loaded {len(queries)} examples")

    categories = {}
    for item in queries:
        cat = item['category']
        categories[cat] = categories.get(cat, 0) + 1

    print("\nData distribution:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")

    # Create tokenizer
    tokenizer = CharTokenizer(max_len=config['max_len'])

    # Create datasets
    dataset = FoodMatchDataset(queries, tokenizer)

    # Split
    train_size = int(0.85 * len(dataset))  # 85/15 split for more training data
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                             shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                           shuffle=False, collate_fn=collate_fn)

    print(f"\nTrain: {train_size} | Val: {val_size}")

    # Create model
    model = TransformerTypoMatcher(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_layers=config['num_layers'],
        dim_feedforward=config['dim_feedforward'],
        output_dim=config['output_dim']
    )

    print(f"\n{'='*70}")
    print(f"Model: Transformer")
    print(f"Parameters: {count_parameters(model):,}")
    print(f"Estimated size: {count_parameters(model) * 4 / 1024 / 1024:.2f} MB")
    print(f"{'='*70}\n")

    model = model.to(config['device'])

    # Loss and optimizer
    criterion = ContrastiveLoss(margin=config['margin'])
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15)

    # Training loop
    best_val_acc = 0.0

    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, config['device'])
        val_loss, val_acc = validate(model, val_loader, criterion, config['device'])

        scheduler.step(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('../models', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': config
            }, '../models/best_model_transformer.pth')
            print(f"✓ Saved best model (Val Acc: {val_acc:.4f})")

        # Early stopping if we achieve 95%+
        if val_acc >= 0.95:
            print(f"\n{'='*70}")
            print(f"🎉 ACHIEVED 95%+ ACCURACY!")
            print(f"   Val Acc: {val_acc:.4f} ({val_acc*100:.2f}%)")
            print(f"{'='*70}\n")
            break

    print("\n" + "="*70)
    print("TRAINING COMPLETE")
    print("="*70)
    print(f"Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print("="*70)


if __name__ == "__main__":
    main()
