# config.py
config = {
    "data_dir": "mydataset",        # 数据集根目录
    "model_name": "resnet18",       # 或 'mobilenet_v2'
    "batch_size": 32,
    "epochs": 20,
    "lr": 1e-4,
    "freeze": True,
    "img_size": 224,
    "device": "cuda"
}
