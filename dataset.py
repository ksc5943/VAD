import os
import torch
import torchaudio
from torch.utils.data import Dataset

class VADDataset(Dataset):
    def __init__(self, data_dir, txt_dir=None):
        self.data_dir = data_dir
        self.txt_dir = txt_dir
        self.files = []
        
        # Load .wav files and corresponding .txt files (if provided)
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.wav'):
                    wav_path = os.path.join(root, file)
                    txt_path = os.path.join(self.txt_dir, file.replace('.wav', '.txt')) if txt_dir else None
                    self.files.append((wav_path, txt_path))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        wav_path, txt_path = self.files[idx]
        waveform, sample_rate = torchaudio.load(wav_path)
        
        if txt_path:
            with open(txt_path, 'r') as f:
                onset, offset, sound_class = f.read().split('\t')
                onset, offset = float(onset), float(offset)
                if sound_class == 'speech':
                    center = (onset + offset) / 2.0
                else:
                    center = -1.0
            return waveform, center
        else:
            return waveform, -1.0  # Return -1.0 for test data (no center)
