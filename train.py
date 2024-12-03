import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import VADDataset
from model import VADClipModel
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error

class RMSELoss(nn.Module):
    def __init__(self):
        super(RMSELoss, self).__init__()

    def forward(self, y_pred, y_true):
        return torch.sqrt(mean_squared_error(y_pred.cpu().detach().numpy(), y_true.cpu().detach().numpy()))

def train_model(model, train_loader, optimizer, criterion_center, criterion_class, device):
    model.train()
    running_loss = 0.0
    for waveforms, centers in train_loader:
        waveforms = waveforms.to(device)
        centers = centers.to(device)
        
        optimizer.zero_grad()
        
        center_pred, class_pred = model(waveforms)
        loss_center = criterion_center(center_pred.squeeze(), centers)
        loss_class = criterion_class(class_pred, centers)
        loss = loss_center + loss_class
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    return running_loss / len(train_loader)

def get_train_loader(data_dir, txt_dir, batch_size=32):
    dataset = VADDataset(data_dir, txt_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VADClipModel(device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    criterion_center = RMSELoss()
    criterion_class = nn.CrossEntropyLoss()
    train_loader = get_train_loader('data/train/wav160', 'data/train/txt')
    
    for epoch in range(10):  # Train for 10 epochs
        train_loss = train_model(model, train_loader, optimizer, criterion_center, criterion_class, device)
        print(f'Epoch {epoch+1}/{10}, Loss: {train_loss}')
    
    return model
