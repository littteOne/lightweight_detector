# dataset.py
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_loaders(data_dir, batch_size=32, img_size=224):
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    train_data = datasets.ImageFolder(f"{data_dir}/train", transform=transform)
    val_data = datasets.ImageFolder(f"{data_dir}/val", transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)

    return train_loader, val_loader
