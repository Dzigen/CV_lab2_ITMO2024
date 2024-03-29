from ultralytics import YOLO
import torch
import torchvision


print(torch.__version__)
print(torchvision.__version__)

BASE_DIR = '/home/ubuntu/ImgGen/lab2'
EPOCHS = 100
BATCH_SIZE = 64
SAVE_DIR = f"{BASE_DIR}/save_dir"


model = YOLO(f"{BASE_DIR}/yolov8n.pt")
print("Number of parameters: ", sum([p.numel() for p in model.parameters()]))

results = model.train(data=f'{BASE_DIR}/data/config.yaml', epochs=EPOCHS,
                      save_dir=SAVE_DIR, batch=BATCH_SIZE)

resuts = model.eval()