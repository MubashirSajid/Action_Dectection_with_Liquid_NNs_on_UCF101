import os
import cv2

import os
import cv2

import os
import cv2

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

# # Example usage
# video_folder = 'UCF-101'
# output_folder = 'Processing_frames/output_frames_testing'
# extract_frames_from_videos(video_folder, output_folder, frame_rate=30)




# # def extract_frames_from_videos(video_folder, output_folder, frame_rate=1):
# #     if not os.path.exists(output_folder):
# #         os.makedirs(output_folder)
    
# #     for class_folder in os.listdir(video_folder):
# #         class_path = os.path.join(video_folder, class_folder)
# #         if os.path.isdir(class_path):
# #             processed_people = set()  # To keep track of processed people for each action
            
# #             for video_file in os.listdir(class_path):
# #                 video_path = os.path.join(class_path, video_file)
# #                 person_id = video_file.split('_')[2]  # Extract the person identifier (g number)
                
# #                 if person_id in processed_people:
# #                     continue  # Skip this video if this person is already processed
                
# #                 processed_people.add(person_id)  # Mark this person as processed
# #                 class_output_folder = os.path.join(output_folder, class_folder)
# #                 if not os.path.exists(class_output_folder):
# #                     os.makedirs(class_output_folder)
                
# #                 cap = cv2.VideoCapture(video_path)
# #                 count = 0
# #                 success = True
                
# #                 while success:
# #                     success, frame = cap.read()
# #                     if count % frame_rate == 0 and success:
# #                         frame_filename = os.path.join(class_output_folder, f"{os.path.splitext(video_file)[0]}_frame_{count}.jpg")
# #                         cv2.imwrite(frame_filename, frame)
# #                     count += 1

# #                 cap.release()

# # Example usage
# video_folder = 'UCF-101'
# output_folder = 'Processing_frames/output_frames_testing'
# extract_frames_from_videos(video_folder, output_folder, frame_rate=30)

# import os
# import torch
# import torchvision.transforms as transforms
# from torchvision.models import resnet18
# from PIL import Image
# import numpy as np
# import torch.nn as nn

# # Load a pre-trained ResNet model and modify it to extract features from a layer that outputs 256 features
# class FeatureExtractor(nn.Module):
#     def __init__(self, original_model):
#         super(FeatureExtractor, self).__init__()
#         # Exclude layers after the second layer of ResNet18
#         self.features = nn.Sequential(
#             *list(original_model.children())[:7]  # Up to layer2 (BasicBlock)
#         )
#         self.pool = nn.AdaptiveAvgPool2d((1, 1))  # Add an adaptive pooling layer

#     def forward(self, x):
#         x = self.features(x)
#         x = self.pool(x)
#         x = torch.flatten(x, 1)  # Flatten the output to a 1D vector
#         return x

# # Initialize the modified ResNet model
# original_model = resnet18(pretrained=True)
# model = FeatureExtractor(original_model)
# model.eval()

# # Transformation for input images
# preprocess = transforms.Compose([
#     transforms.Resize(256),
#     transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
# ])

# def extract_features(image_path, model, preprocess):
#     img = Image.open(image_path).convert('RGB')
#     img_t = preprocess(img)
#     batch_t = torch.unsqueeze(img_t, 0)

#     with torch.no_grad():
#         features = model(batch_t)
#     return features.squeeze().numpy()

# # Example usage
# frame_folder = 'Processing_frames/output_frames_testing'
# features_folder = 'Processing_frames/output_features_testing'
# os.makedirs(features_folder, exist_ok=True)

# for class_folder in os.listdir(frame_folder):
#     class_path = os.path.join(frame_folder, class_folder)
#     if os.path.isdir(class_path):  # Check if it's a directory
#         class_features_folder = os.path.join(features_folder, class_folder)
#         os.makedirs(class_features_folder, exist_ok=True)
        
#         for frame_filename in os.listdir(class_path):
#             frame_path = os.path.join(class_path, frame_filename)
#             if os.path.isfile(frame_path):  # Check if it's a file
#                 features = extract_features(frame_path, model, preprocess)
#                 np.save(os.path.join(class_features_folder, frame_filename.replace('.jpg', '.npy')), features)
#                 print("Feature size:", features.shape)  # Should print (256,)
import os
import numpy as np
import json

def create_sequences(feature_folder, seq_len=32):
    sequences_x = []
    sequences_y = []
    class_mapping = {class_name: idx for idx, class_name in enumerate(sorted(os.listdir(feature_folder)))}
        # Save the class-to-index mapping
    class_mapping_path = 'class_mapping.json'
    with open(class_mapping_path, 'w') as f:
        json.dump(class_mapping, f)

    print(f"Class mapping: {class_mapping}")

    for class_folder in os.listdir(feature_folder):
        class_path = os.path.join(feature_folder, class_folder)
        label = class_mapping[class_folder]
        file_list = sorted(os.listdir(class_path), key=lambda x: int(x.split('_')[-1].split('.')[0]))

        features = [np.load(os.path.join(class_path, file)) for file in file_list]

        for i in range(0, len(features) - seq_len + 1, seq_len):
            sequences_x.append(np.stack(features[i:i + seq_len]))
            sequences_y.append(label)

    return np.array(sequences_x), np.array(sequences_y)
features_folder = 'Processing_frames/output_features_testing'
# Example usage
sequences_x, sequences_y = create_sequences(features_folder, seq_len=64)
print("Sequence shape:", sequences_x.shape)
from torch.utils.data import Dataset, DataLoader
import torch
class UCFFeatureDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.tensor(self.sequences[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.long)

dataset = UCFFeatureDataset(sequences_x, sequences_y)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# dataset = UCFFeatureDataset(sequences_x, sequences_y)
    
# Split dataset into train, validation, and test sets
total_size = len(dataset)
val_size = int(total_size * 0.1)
test_size = int(total_size * 0.1)
train_size = total_size - val_size - test_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)



import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
from ncps.wirings import AutoNCP
from ncps.torch import LTC
from torch.utils.data import DataLoader
import torch

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
    torch.save(model.state_dict(), 'action_classification_model_seq64.pth')
import time
if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()

    # Calculate and print the duration
    duration = end_time - start_time
    print(f"Time taken to complete the process: {duration:.2f} seconds")











# import torch.nn as nn
# import torch
# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, num_layers=2):
#         super(LSTMModel, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
    
#     def forward(self, x):
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         out, _ = self.lstm(x, (h0, c0))
#         out = self.fc(out[:, -1, :])
#         return out
    

# import pytorch_lightning as pl
# import torch.optim as optim

# class ActionClassificationLSTMModel(pl.LightningModule):
#     def __init__(self, model, lr=0.001):
#         super(ActionClassificationLSTMModel, self).__init__()
#         self.model = model
#         self.lr = lr
#         self.criterion = nn.CrossEntropyLoss()

#     def forward(self, x):
#         return self.model(x)

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self.model(x)
#         loss = self.criterion(y_hat, y)
#         acc = (y_hat.argmax(1) == y).float().mean()
#         self.log('train_loss', loss, prog_bar=True)
#         self.log('train_acc', acc, prog_bar=True)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self.model(x)
#         loss = self.criterion(y_hat, y)
#         acc = (y_hat.argmax(1) == y).float().mean()
#         self.log('val_loss', loss, prog_bar=True)
#         self.log('val_acc', acc, prog_bar=True)
#         return loss

#     def test_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self.model(x)
#         loss = self.criterion(y_hat, y)
#         acc = (y_hat.argmax(1) == y).float().mean()
#         self.log('test_loss', loss, prog_bar=True)
#         self.log('test_acc', acc, prog_bar=True)
#         return loss

#     def configure_optimizers(self):
#         return optim.Adam(self.parameters(), lr=self.lr)
# import time
# def train_lstm_model():
#     input_size = 256 # Size of the feature vector from ResNet
#     hidden_size = 128
#     output_size = 101  # Number of classes in UCF-101
#     num_layers = 2
#     lr = 0.001
#     epochs = 200
#     batch_size = 16

#     lstm_model = LSTMModel(input_size, hidden_size, output_size, num_layers)
#     model = ActionClassificationLSTMModel(lstm_model, lr=lr)

#     trainer = pl.Trainer(
#         max_epochs=epochs,
#         accelerator='gpu' if torch.cuda.is_available() else 'cpu'
#     )

#     trainer.fit(model, train_loader, val_loader)
#     trainer.test(model, test_loader)

# # Train the LSTM model
# start_time = time.time()
# train_lstm_model()
# end_time = time.time()
# # Calculate and print the duration
# duration = end_time - start_time
# print(f"Time taken to complete the process: {duration:.2f} seconds")




# import torch.nn as nn
# import pytorch_lightning as pl
# import torch.optim as optim
# import time



# class GRUModel(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, num_layers=2):
#         super(GRUModel, self).__init__()
#         self.hidden_size = hidden_size
#         self.num_layers = num_layers
#         self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
#         self.fc = nn.Linear(hidden_size, output_size)
    
#     def forward(self, x):
#         h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
#         out, _ = self.gru(x, h0)
#         out = self.fc(out[:, -1, :])
#         return out
# class ActionClassificationGRUModel(pl.LightningModule):
#     def __init__(self, model, lr=0.001):
#         super(ActionClassificationGRUModel, self).__init__()
#         self.model = model
#         self.lr = lr
#         self.criterion = nn.CrossEntropyLoss()

#     def forward(self, x):
#         return self.model(x)

#     def training_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self.model(x)
#         loss = self.criterion(y_hat, y)
#         acc = (y_hat.argmax(1) == y).float().mean()
#         self.log('train_loss', loss, prog_bar=True)
#         self.log('train_acc', acc, prog_bar=True)
#         return loss

#     def validation_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self.model(x)
#         loss = self.criterion(y_hat, y)
#         acc = (y_hat.argmax(1) == y).float().mean()
#         self.log('val_loss', loss, prog_bar=True)
#         self.log('val_acc', acc, prog_bar=True)
#         return loss

#     def test_step(self, batch, batch_idx):
#         x, y = batch
#         y_hat = self.model(x)
#         loss = self.criterion(y_hat, y)
#         acc = (y_hat.argmax(1) == y).float().mean()
#         self.log('test_loss', loss, prog_bar=True)
#         self.log('test_acc', acc, prog_bar=True)
#         return loss

#     def configure_optimizers(self):
#         return optim.Adam(self.parameters(), lr=self.lr)
# def train_gru_model():
#     input_size = 256
#     hidden_size = 128
#     output_size = 101
#     num_layers = 2
#     lr = 0.001
#     epochs = 200
#     batch_size = 16

#     gru_model = GRUModel(input_size, hidden_size, output_size, num_layers)
#     model = ActionClassificationGRUModel(gru_model, lr=lr)

#     trainer = pl.Trainer(
#         max_epochs=epochs,
#         accelerator='gpu' if torch.cuda.is_available() else 'cpu'
#     )

#     trainer.fit(model, train_loader, val_loader)
#     trainer.test(model, test_loader)


# start_time = time.time()
# # Train the GRU model
# train_gru_model()
# end_time = time.time()
# # Calculate and print the duration
# duration = end_time - start_time
# print(f"Time taken to complete the process: {duration:.2f} seconds")