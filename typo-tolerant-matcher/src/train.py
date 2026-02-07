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
from data_generator import generate_food_queries


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
    # Configuration
    config = {
        'vocab_size': 128,
        'embedding_dim': 32,
        'hidden_dim': 128,
        'output_dim': 64,
        'max_len': 50,
        'batch_size': 64,
        'num_epochs': 50,
        'learning_rate': 0.001,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    print(f"Using device: {config['device']}")

    # Generate training data
    print("Generating training data...")
    queries = generate_food_queries()
    print(f"Generated {len(queries)} examples")

    # Save generated data
    os.makedirs('../data/processed', exist_ok=True)
    with open('../data/processed/training_data.json', 'w') as f:
        json.dump(queries, f, indent=2)

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

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")

        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, config['device'])
        val_loss, val_acc = validate(model, val_loader, criterion, config['device'])

        scheduler.step(val_loss)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs('../models', exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': config
            }, '../models/best_model.pth')
            print("✓ Saved best model")

    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
