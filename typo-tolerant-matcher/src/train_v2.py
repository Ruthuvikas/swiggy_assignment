import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import json
import numpy as np
from tqdm import tqdm
import os

from model import TypoTolerantMatcher, count_parameters
from dataset import CharTokenizer, FoodMatchDataset, collate_fn


class ContrastiveLoss(nn.Module):
    """Contrastive loss for similarity learning"""
    def __init__(self, margin=0.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, predictions, labels):
        """
        predictions: model output scores (0-1)
        labels: ground truth scores (0-1)
        """
        # Convert labels to binary (similar/dissimilar)
        is_similar = (labels > 0.5).float()

        # For similar pairs: penalize low scores
        # For dissimilar pairs: penalize high scores
        loss_similar = is_similar * torch.pow(1 - predictions, 2)
        loss_dissimilar = (1 - is_similar) * torch.pow(torch.clamp(predictions - self.margin, min=0.0), 2)

        loss = loss_similar + loss_dissimilar
        return loss.mean()


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
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

        # Forward pass
        outputs = model(query, target)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()

        # Calculate accuracy (threshold at 0.5)
        predictions = (outputs > 0.5).float()
        binary_labels = (labels > 0.5).float()
        correct += (predictions == binary_labels).sum().item()
        total += labels.size(0)

        pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})

    return total_loss / len(dataloader), correct / total


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

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

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(binary_labels.cpu().numpy())

    return total_loss / len(dataloader), correct / total


def main():
    # Configuration
    config = {
        'vocab_size': 128,
        'embedding_dim': 32,
        'hidden_dim': 128,
        'output_dim': 64,
        'max_len': 50,
        'batch_size': 32,  # Smaller batch for better gradients
        'num_epochs': 100,
        'learning_rate': 0.001,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'margin': 0.3  # Contrastive loss margin
    }

    print(f"Using device: {config['device']}")

    # Load real training data
    print("Loading training data...")
    with open('../data/processed/training_data_real.json', 'r') as f:
        queries = json.load(f)

    print(f"Loaded {len(queries)} examples")

    # Show distribution
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

    # Split into train/val
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                             shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                           shuffle=False, collate_fn=collate_fn)

    # Create model
    model = TypoTolerantMatcher(
        vocab_size=config['vocab_size'],
        embedding_dim=config['embedding_dim'],
        hidden_dim=config['hidden_dim'],
        output_dim=config['output_dim']
    )

    print(f"\nModel parameters: {count_parameters(model):,}")
    print(f"Estimated size: {count_parameters(model) * 4 / 1024 / 1024:.2f} MB\n")

    model = model.to(config['device'])

    # Loss and optimizer - Using Contrastive Loss
    criterion = ContrastiveLoss(margin=config['margin'])
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=0.01)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                                     patience=10)

    # Training loop
    best_val_acc = 0.0

    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, config['device'])
        val_loss, val_acc = validate(model, val_loader, criterion, config['device'])

        scheduler.step(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model based on accuracy
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
            }, '../models/best_model_v2.pth')
            print(f"✓ Saved best model (Val Acc: {val_acc:.4f})")

        # Early stopping if we achieve 95%+
        if val_acc >= 0.95:
            print(f"\n🎉 Achieved 95%+ accuracy! Val Acc: {val_acc:.4f}")
            break

    print("\nTraining complete!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
