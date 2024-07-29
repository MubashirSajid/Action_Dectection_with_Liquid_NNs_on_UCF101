import os
import cv2
import numpy as np
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, Subset
from collections import deque
from ncps.wirings import AutoNCP
from ncps.torch import LTC

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

# Custom Convolutional Feature Extractor
class ConvFeatureExtractor(nn.Module):
    def __init__(self):
        super(ConvFeatureExtractor, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = torch.flatten(x, 1)  # Flatten the output to a 1D vector
        return x

# Initialize the custom feature extractor
feature_extractor = ConvFeatureExtractor()
feature_extractor.eval()

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
        acc = (y_hat.argmax(1) == y).float().mean()
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.model(x)
        y_hat = y_hat.mean(dim=1)  # Average over the sequence length dimension
        loss = self.criterion(y_hat.view(-1, 101), y)
        acc = (y_hat.argmax(1) == y).float().mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self.model(x)
        y_hat = y_hat.mean(dim=1)  # Average over the sequence length dimension
        loss = self.criterion(y_hat.view(-1, 101), y)
        acc = (y_hat.argmax(1) == y).float().mean()
        self.log('test_loss', loss, prog_bar=True)
        self.log('test_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.lr)

# Load model
# def load_model(model_path, input_size, wiring, num_classes):
#     ltc_model = LTC(input_size, wiring, batch_first=True)
#     model = ActionClassificationModel(ltc_model)
#     model.load_state_dict(torch.load(model_path))
#     model.eval()
#     return model

def load_model(model_path, input_size, wiring, num_classes):
    ltc_model = LTC(input_size, wiring, batch_first=True)
    model = ActionClassificationModel(ltc_model)

    # Load the state dictionary
    state_dict = torch.load(model_path)

    # Rename keys to match the model definition
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('ltc_model.'):
            new_key = k.replace('ltc_model.', 'model.')
        else:
            new_key = k
        new_state_dict[new_key] = v

    model.load_state_dict(new_state_dict)
    model.eval()
    return model

# Generate and overlay activation map
# def overlay_activation(frame, activation_map):
#     activation_map = activation_map.squeeze().cpu().detach().numpy()
#     activation_map = cv2.resize(activation_map, (frame.shape[1], frame.shape[0]))
#     activation_map = (activation_map - activation_map.min()) / (activation_map.max() - activation_map.min())
#     activation_map = (activation_map * 255).astype(np.uint8)
#     activation_map = cv2.applyColorMap(activation_map, cv2.COLORMAP_JET)
#     overlayed_frame = cv2.addWeighted(frame, 0.6, activation_map, 0.4, 0)
#     return overlayed_frame

# Real-time classification with attention map
def real_time_classification():
    model_path = 'action_classification_model_seq64_noRes.pth'
    input_size = 256
    wiring = AutoNCP(128, 101)
    num_classes = 101

    model = load_model(model_path, input_size, wiring, num_classes)
    feature_extractor = ConvFeatureExtractor().eval()  # Initialize the custom feature extractor
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
            features = feature_extractor(img_t).squeeze(0)
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

        # Overlay attention map on the frame (here we assume the last feature map is the attention map for simplicity)
        # activation_map = frames[-1].view(8, 32)  # Reshape the last feature map to match the frame dimensions
        # frame_with_attention = overlay_activation(frame, activation_map)

        # Display the action prediction on the frame
        cv2.putText(frame, f"Predicted action: {action}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Webcam', frame)  # Display the frame with attention map in a window

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    real_time_classification()
