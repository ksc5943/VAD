import torch
import torch.nn as nn
import clip
from transformers import CLIPProcessor, CLIPModel

class VADClipModel(nn.Module):
    def __init__(self, device):
        super(VADClipModel, self).__init__()
        self.device = device
        self.clip_model = clip.load("ViT-B/32", device)[0]
        self.fc_center = nn.Linear(512, 1)  # For predicting the center value (speech only)
        self.fc_class = nn.Linear(512, 4)  # For predicting the sound class (silence, car, dog, speech)

    def forward(self, waveform):
        waveform = waveform.to(self.device)
        image_features = self.clip_model.encode_image(waveform)
        
        center_pred = self.fc_center(image_features)
        class_pred = self.fc_class(image_features)
        
        return center_pred, class_pred
