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
from torch.nn.utils.rnn import pad_sequence
from collections import deque
from ncps.wirings import AutoNCP
from ncps.torch import LTC
import os
import torch
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
import numpy as np
import torch.nn as nn

# Frame Extraction Function
# def extract_frames_from_videos(video_folder, output_folder, frame_rate=1):
#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)
    
#     for class_folder in os.listdir(video_folder):
#         class_path = os.path.join(video_folder, class_folder)
#         if os.path.isdir(class_path):
#             for video_file in os.listdir(class_path):
#                 video_path = os.path.join(class_path, video_file)
#                 class_output_folder = os.path.join(output_folder, class_folder)
#                 if not os.path.exists(class_output_folder):
#                     os.makedirs(class_output_folder)
                
#                 cap = cv2.VideoCapture(video_path)
#                 count = 0
#                 success = True
                
#                 while success:
#                     success, frame = cap.read()
#                     if count % frame_rate == 0 and success:
#                         frame_filename = os.path.join(class_output_folder, f"{os.path.splitext(video_file)[0]}_frame_{count}.jpg")
#                         cv2.imwrite(frame_filename, frame)
#                     count += 1

#                 cap.release()

# # Example usage for frame extraction
# video_folder = '/mnt/nvme0n1/khubaib_mubashir/Liquid_NNs/UCF-101'
# output_folder = '/mnt/nvme0n1/khubaib_mubashir/Liquid_NNs/Processing_frames/output_frames_variable_seq'
# extract_frames_from_videos(video_folder, output_folder, frame_rate=30)

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Custom Feature Extractor using ResNet
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
feature_extractor = FeatureExtractor(original_model)
feature_extractor.eval()

# Function to extract features from frames
def extract_features_from_frames(frame_folder, feature_folder):
    if not os.path.exists(feature_folder):
        os.makedirs(feature_folder)

    for class_folder in os.listdir(frame_folder):
        class_path = os.path.join(frame_folder, class_folder)
        if os.path.isdir(class_path):
            class_feature_folder = os.path.join(feature_folder, class_folder)
            if not os.path.exists(class_feature_folder):
                os.makedirs(class_feature_folder)

            for frame_filename in os.listdir(class_path):
                frame_path = os.path.join(class_path, frame_filename)
                if os.path.isfile(frame_path):
                    img = Image.open(frame_path).convert('RGB')
                    img_t = preprocess(img).unsqueeze(0)
                    with torch.no_grad():
                        features = feature_extractor(img_t).squeeze(0)
                    feature_filename = os.path.join(class_feature_folder, frame_filename.replace('.jpg', '.npy'))
                    np.save(feature_filename, features.numpy())

# Example usage for feature extraction
frame_folder = '/mnt/nvme0n1/khubaib_mubashir/Liquid_NNs/Processing_frames/output_frames_variable_seq'
features_folder = '/mnt/nvme0n1/khubaib_mubashir/Liquid_NNs/Processing_frames/output_features_variable_seq'
extract_features_from_frames(frame_folder, features_folder)

# Dataset Preparation
class UCFFeatureDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

def create_sequences(feature_folder):
    sequences_x = []
    sequences_y = []
    class_mapping = {class_name: idx for idx, class_name in enumerate(sorted(os.listdir(feature_folder)))}
    class_mapping_path = 'class_mapping.json'
    with open(class_mapping_path, 'w') as f:
        json.dump(class_mapping, f)
    print(f"Class mapping: {class_mapping}")

    for class_folder in os.listdir(feature_folder):
        class_path = os.path.join(feature_folder, class_folder)
        label = class_mapping[class_folder]
        file_list = sorted(os.listdir(class_path), key=lambda x: int(x.split('_')[-1].split('.')[0]))

        features = [np.load(os.path.join(class_path, file)) for file in file_list]

        sequences_x.append(np.stack(features))
        sequences_y.append(label)

    return sequences_x, sequences_y

# Example usage
sequences_x, sequences_y = create_sequences(features_folder)
print("Number of sequences:", len(sequences_x))

class VariableLengthDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = torch.tensor(self.sequences[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return sequence, label

dataset = VariableLengthDataset(sequences_x, sequences_y)

total_size = len(dataset)
val_size = int(total_size * 0.1)
test_size = int(total_size * 0.1)
train_size = total_size - val_size - test_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

def collate_fn(batch):
    sequences, labels = zip(*batch)
    sequences_padded = pad_sequence(sequences, batch_first=True)
    return sequences_padded, torch.tensor(labels, dtype=torch.long)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

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

def main():
    class Args:
        model = "ltc"
        log = 1
        size = 128
        epochs = 200
        batch_size = 16
    args = Args()

    if args.model == "ltc":
        wiring = AutoNCP(args.size, 101)  # 101 output classes for UCF101
        ltc_model = LTC(256, wiring, batch_first=True)
        model = ActionClassificationModel(ltc_model, lr=0.01)
    else:
        raise ValueError(f"Unknown model type '{args.model}'")

    trainer = pl.Trainer(
        max_epochs=args.epochs,
        log_every_n_steps=args.log,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu'
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    # Save the trained model
    torch.save(model.state_dict(), 'action_classification_model_variable_seq.pth')

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()

    # Calculate and print the duration
    duration = end_time - start_time
    print(f"Time taken to complete the process: {duration:.2f} seconds")
