import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from model import VADClipModel
from dataset import VADDataset, load_data

def train_model(model, loader, optimizer, center_criterion, class_criterion, device, epochs=5):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for waveform, center_sets, labels in loader:
            waveform, center_sets, labels = waveform.to(device), center_sets.to(device), labels.to(device)

            optimizer.zero_grad()
            pred_center, pred_class = model(waveform)
            mask = (center_sets != -1.0)  

            loss_center = center_criterion(pred_center[mask], center_sets[mask])
            loss_class = class_criterion(pred_class, labels)
            loss = loss_center + loss_class

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(loader)}")

def predict_model(model, loader, device, output_path="my_submission.csv"):
    model.eval()
    predictions = []

    with torch.no_grad():
        for waveform, _, _ in loader:
            waveform = waveform.to(device)
            pred_center, pred_class = model(waveform)
            pred_class = torch.argmax(pred_class, dim=1)

            for center, cls in zip(pred_center.cpu().numpy(), pred_class.cpu().numpy()):
                if cls == 3:  
                    predictions.append(center)
                else:
                    predictions.append(-1.0)

    import pandas as pd
    df = pd.DataFrame({
        "Id": list(range(len(predictions))),
        "Center": np.array(predictions)
    })
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")
