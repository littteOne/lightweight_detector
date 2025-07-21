# utils.py
from sklearn.metrics import accuracy_score
import torch

def evaluate(model, val_loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for imgs, labs in val_loader:
            imgs = imgs.to(device)
            out = model(imgs).argmax(dim=1).cpu()
            preds.extend(out)
            labels.extend(labs)
    acc = accuracy_score(labels, preds)
    return acc
