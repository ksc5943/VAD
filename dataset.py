import os
import torchaudio
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class VADDataset(Dataset):
    def __init__(self, wav_paths, center_sets, labels, transform=None):
        self.wav_paths = wav_paths
        self.center_sets = center_sets
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):
        waveform, sample_rate = torchaudio.load(self.wav_paths[idx])
        
        if self.transform:
            waveform = self.transform(waveform)
        
        label = self.labels[idx]
        center = self.center_sets[idx]

        return waveform, center, label

def load_data(data_dir, has_txt=True):
    wav_paths = []
    center_sets = []
    labels = []
    
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith(".wav"):
                wav_path = os.path.join(root, file)
                txt_path = wav_path.replace('.wav', '.txt')
                
                if has_txt and os.path.exists(txt_path):
                    with open(txt_path, "r") as f:
                        content = f.read().split("\t")
                        onset, offset, sound_class = float(content[0]), float(content[1]), content[2]
                        center = (onset + offset) / 2.0 if sound_class == "speech" else -1.0
                        label = {"silence": 0, "car": 1, "dog": 2, "speech": 3}[sound_class]
                else:
                    center = -1.0
                    label = 0  

                wav_paths.append(wav_path)
                center_sets.append(center)
                labels.append(label)
    
    return wav_paths, center_sets, labels