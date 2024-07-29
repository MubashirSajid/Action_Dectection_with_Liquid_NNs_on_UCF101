import os
import cv2
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import numpy as np
import torch.nn as nn
import time
import pytorch_lightning as pl
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from collections import deque
from ncps.wirings import AutoNCP
from ncps.torch import LTC
import json

with open('class_mapping.json', 'r') as f:
    class_mapping = json.load(f)
class_mapping_int_keys = {v: k for k, v in class_mapping.items()}
with open('class_mapping_int_keys.json', 'w') as f:
    json.dump(class_mapping_int_keys, f)

# Load the class mapping with integer keys
def load_class_mapping_int_keys(mapping_path):
    with open(mapping_path, 'r') as f:
        class_mapping = json.load(f)
    return {int(k): v for k, v in class_mapping.items()}

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Action Classification Model
class ActionClassificationModel(pl.LightningModule):
    def __init__(self, model, lr=0.001):
        super(ActionClassificationModel, self).__init__()
        self.model = model
        self.lr = lr
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.model(x)
        y_hat = y_hat.mean(dim=1)  # Average over the sequence length dimension

        loss = self.criterion(y_hat.view(-1, 101), y)  # 101 classes for UCF101
        acc = (y_hat.argmax(1) == y).float().mean()  # Change to argmax(1) to match the new shape
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.model(x)
        y_hat = y_hat.mean(dim=1)  # Average over the sequence length dimension
        loss = self.criterion(y_hat.view(-1, 101), y)
        acc = (y_hat.argmax(1) == y).float().mean()  # Change to argmax(1) to match the new shape
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.model(x)
        y_hat = y_hat.mean(dim=1)  # Average over the sequence length dimension
        loss = self.criterion(y_hat.view(-1, 101), y)
        acc = (y_hat.argmax(1) == y).float().mean()  # Change to argmax(1) to match the new shape
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

# Feature Extractor
class FeatureExtractor(nn.Module):
    def __init__(self, original_model):
        super(FeatureExtractor, self).__init__()
        # Exclude layers after the second layer of ResNet18
        self.features = nn.Sequential(
            *list(original_model.children())[:7]  # Up to layer2 (BasicBlock)
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Add an adaptive pooling layer

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)  # Flatten the output to a 1D vector
        return x

# Initialize the modified ResNet model
original_model = resnet18(pretrained=True)
resnet_model = FeatureExtractor(original_model)
resnet_model.eval()

# Load model
def load_model(model_path, input_size, wiring, num_classes):
    ltc_model = LTC(input_size, wiring, batch_first=True)
    model = ActionClassificationModel(ltc_model)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# Real-time classification
def real_time_classification():
    model_path = 'action_classification_model_seq64.pth'
    input_size = 256
    wiring = AutoNCP(128, 101)
    num_classes = 101

    model = load_model(model_path, input_size, wiring, num_classes)
    class_mapping = load_class_mapping_int_keys('class_mapping_int_keys.json')

    cap = cv2.VideoCapture(0)  # Open the default webcam
    frames = deque(maxlen=64)  # Use deque to maintain a sliding window of frames
    action = "N/A"  # Initialize action with a default value

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
        img_t = preprocess(Image.fromarray(frame_rgb)).unsqueeze(0)
        with torch.no_grad():
            features = resnet_model(img_t).squeeze(0)
        frames.append(features)

        if len(frames) == 64:
            frame_sequence = torch.stack(list(frames))  # Convert deque to tensor

            frame_sequence = frame_sequence.unsqueeze(0)  # Add batch dimension

            with torch.no_grad():
                output, _ = model(frame_sequence)
                output = output.mean(dim=1)  # Average over the sequence length dimension
                prediction = output.argmax(dim=1).item()

            action = class_mapping[prediction]
            print(f"Predicted action: {action}")

        # Display the action prediction on the frame
        cv2.putText(frame, f"Predicted action: {action}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Webcam', frame)  # Display the frame in a window

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_classification()
