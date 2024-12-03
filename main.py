import torch
from torch.utils.data import DataLoader
from dataset import load_data, VADDataset
from model import VADClipModel
from train import train_model, predict_model

def main():
    train_dir = "data/train"
    test_dir = "data/test"

    train_files, train_centers, train_labels = load_data(train_dir)
    test_files, _, _ = load_data(test_dir, has_txt=False)  # 테스트 데이터는 txt 없음

    train_dataset = VADDataset(train_files, train_centers, train_labels, transform=None)
    test_dataset = VADDataset(test_files, [-1.0] * len(test_files), [0] * len(test_files), transform=None)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VADClipModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    center_criterion = nn.MSELoss()
    class_criterion = nn.CrossEntropyLoss()

    train_model(model, train_loader, optimizer, center_criterion, class_criterion, device)
    predict_model(model, test_loader, device)

if __name__ == "__main__":
    main()