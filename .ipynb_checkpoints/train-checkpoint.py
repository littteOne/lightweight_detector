# train.py
import torch
import torch.nn as nn
from config import config
from model import get_model
from dataset import get_loaders
from utils import evaluate

def main():
    device = torch.device(config['device'] if torch.cuda.is_available() else "cpu")

    train_loader, val_loader = get_loaders(
        config['data_dir'],
        batch_size=config['batch_size'],
        img_size=config['img_size']
    )

    model = get_model(config['model_name'], freeze=config['freeze'])
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['lr']
    )

    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        acc = evaluate(model, val_loader, device)
        print(f"[Epoch {epoch+1}] Loss: {total_loss:.4f} | Val Acc: {acc:.4f}")

    torch.save(model.state_dict(), "light_model.pth")

if __name__ == "__main__":
    main()
