# spds_baseline.py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2470, 0.2435, 0.2616))
])

train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


class SPDSExtractor(nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        feats = []

        for p in [2, 4, 8]:
            patches = x.unfold(2, p, p//2).unfold(3, p, p//2)
            patches = patches.contiguous().view(B, C, -1, p*p)

            mean = patches.mean(-1)
            std = patches.std(-1) + 1e-6

            if p == 2:
                dets = torch.det(patches[:, :, :, :4].reshape(B*C, -1, 2, 2).double()).float()
                dets = torch.sign(dets) * torch.abs(dets)**0.7
                dets = dets.view(B, C, -1).mean(dim=1, keepdim=True)
                combined = torch.cat([mean, std, dets], dim=1)
            else:
                combined = torch.cat([mean, std], dim=1)

            feats.append(combined.mean(-1))

        return torch.cat(feats, dim=1)


class SPDSNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor = SPDSExtractor()
        self.classifier = nn.Sequential(
            nn.Linear(19, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        x = self.extractor(x)
        return self.classifier(x)


model = SPDSNet()
optimizer = optim.Adam(model.parameters(), lr=0.003)
criterion = nn.CrossEntropyLoss()

print("training started")
model.train()

for epoch in range(20):
    correct = total = 0
    for i, (x, y) in enumerate(train_loader):
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        pred = out.argmax(1)
        correct += (pred == y).sum().item()
        total += y.size(0)

        if i % 100 == 0:
            print(f"epoch {epoch+1} - batch {i} - loss {loss.item():.4f}")

    acc = 100.0 * correct / total
    print(f"EPOCH {epoch+1} -> {acc:.2f}%")

print("finished.")