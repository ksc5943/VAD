import torch
import torch.nn as nn
import clip

class VADClipModel(nn.Module):
    def __init__(self, num_classes=4, device="cuda"):
        super(VADClipModel, self).__init__()
        self.device = device
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device)
        self.waveform_fc = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(256),  
            nn.Flatten()
        )

        self.fc_center = nn.Linear(512, 1)  
        self.fc_class = nn.Linear(512, num_classes)

    def forward(self, x):
        waveform_embedding = self.waveform_fc(x).unsqueeze(0)  
        
        image_features = self.clip_model.encode_image(waveform_embedding)

        center_pred = self.fc_center(image_features).squeeze()
        class_pred = self.fc_class(image_features)
        
        return center_pred, class_pred
