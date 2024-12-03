import torch
from train import train
from dataset import VADDataset
import pandas as pd
import numpy as np

def main():
    model = train()
    
    model.eval()
    test_dataset = VADDataset('data/test/wav160')
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

    y_pred = []
    
    for waveform, _ in test_loader:
        waveform = waveform.to(device)
        
        with torch.no_grad():
            center_pred, class_pred = model(waveform)
        
        # Get the predicted center (if class is "speech", else -1)
        center = center_pred.item() if class_pred.argmax() == 3 else -1.0
        y_pred.append(center)

    df = pd.DataFrame({
        "Id": np.arange(len(y_pred)).astype(int),
        "Center": np.array(y_pred).astype(float),
    })
    df.to_csv("result_vad.csv", index=False)
    print("Prediction saved to result_vad.csv")

if __name__ == "__main__":
    main()
