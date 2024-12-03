import os
import torch
from torch.utils.data import Dataset
from torchaudio import load

class VADDataset(Dataset):
    def __init__(self, wav_dir, txt_dir=None, processor=None, is_test=False):
        self.wav_dir = wav_dir
        self.txt_dir = txt_dir
        self.processor = processor
        self.is_test = is_test

        self.file_paths = []
        self.labels = []
        self.load_data()

    def load_data(self):
        for root, _, files in os.walk(self.wav_dir):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    self.file_paths.append(file_path)
                    
                    if self.txt_dir:
                        txt_file = os.path.join(self.txt_dir, file.replace('.wav', '.txt'))
                        if os.path.exists(txt_file):
                            with open(txt_file, 'r') as f:
                                content = f.read()
                                onset, offset, sound_class = content.split('\t')
                                if sound_class == 'speech':
                                    center = (float(onset) + float(offset)) / 2.0
                                    self.labels.append(center)
                                else:
                                    self.labels.append(-1.0)
                    else:
                        self.labels.append(-1.0)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        waveform, sample_rate = load(file_path)
        
        if self.processor:
            waveform = self.processor(waveform)
        
        return waveform, label
