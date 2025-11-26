#Phase 3 
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

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
            stride = p // 2
            patches = x.unfold(2, p, stride).unfold(3, p, stride)  # (B,C,H',W',p,p)
            patches = patches.contiguous().view(B, C, -1, p*p)      # (B,C,N,p*p)

            mean = patches.mean(-1)      # (B,C,N)
            std  = patches.std(-1) + 1e-6

            if p == 2:
                # Signed determinants
                raw_det = torch.det(patches[:, :, :, :4].reshape(B*C, -1, 2, 2).double()).float()
                det = torch.sign(raw_det) * torch.abs(raw_det) ** 0.7
                det = det.view(B, C, -1).mean(1, keepdim=True)        # (B,1,N)
                combined = torch.cat([mean, std, det], dim=1)         # (B,7,N)
            else:
                combined = torch.cat([mean, std], dim=1)              # (B,6,N)

            # Reshape back to spatial map
            s = int(combined.shape[-1] ** 0.5)
            spatial = combined.view(B, -1, s, s)  # (B, channels, s, s)
            feats.append(spatial)

        # Concatenate all scales → (B, total_ch, smallest_s, smallest_s)
        return torch.cat(feats, dim=1)  # (B, ~19, 15, 15) for CIFAR-32


class SimilarityHead(nn.Module):
    def forward(self, x):
        # x: (B, C, H, W)
        row_var = x.var(dim=2, keepdim=True)   # variance across columns
        col_var = x.var(dim=3, keepdim=True)   # variance across rows
        return torch.cat([row_var.mean([2,3]), col_var.mean([2,3])], dim=1)  # (B,2)


class SPDSNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.extractor = SPDSExtractor()
        self.sim_head = SimilarityHead()
        self.order_head = nn.Conv2d(19, 4, kernel_size=3, padding=1)  # predicts 4 directions

        self.classifier = nn.Sequential(
            nn.Linear(19 * 15 * 15 + 2, 512),   # spatial feats + similarity
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x, return_aux=False):
        spatial = self.extractor(x)           # (B,19,15,15)
        sim_feat = self.sim_head(spatial)     # (B,2)
        order_logits = self.order_head(spatial).mean([2,3])  # (B,4)

        flattened = spatial.flatten(1)        # (B,19*225)
        final_feat = torch.cat([flattened, sim_feat], dim=1)

        cls_out = self.classifier(final_feat)

        if return_aux:
            return cls_out, order_logits
        return cls_out



def create_order_labels(batch_size, device):
    # 0=up, 1=right, 2=down, 3=left → random per batch
    return torch.randint(0, 4, (batch_size,), device=device)

def apply_shift(x, direction):
    # x: (B,C,H,W)
    if direction == 0:   return F.pad(x[..., 1:, :], (0,0,0,1))   # up
    if direction == 1:   return F.pad(x[..., :, 1:], (0,1,0,0))   # right
    if direction == 2:   return F.pad(x[..., :-1, :], (0,0,1,0))  # down
    if direction == 3:   return F.pad(x[..., :, :-1], (1,0,0,0))  # left
    return x


model = SPDSNet().cuda() if torch.cuda.is_available() else SPDSNet()
optimizer = optim.Adam(model.parameters(), lr=0.003)
ce_loss = nn.CrossEntropyLoss()
order_loss_fn = nn.CrossEntropyLoss()

print("PHASE 2 TRAINING STARTED — LET'S BREAK 70%")
model.train()

for epoch in range(30):
    total_correct = total = 0
    epoch_loss = 0.0

    for i, (img, label) in enumerate(train_loader):
        img = img.cuda() if torch.cuda.is_available() else img
        label = label.cuda() if torch.cuda.is_available() else label

        # Random direction for order loss
        direction = create_order_labels(img.size(0), img.device)
        shifted = apply_shift(img, direction)

        optimizer.zero_grad()
        cls_out, order_out = model(shifted, return_aux=True)

        ce = ce_loss(cls_out, label)
        order_loss = order_loss_fn(order_out, direction)
        loss = ce + 0.5 * order_loss

        loss.backward()
        optimizer.step()

        pred = cls_out.argmax(1)
        total_correct += (pred == label).sum().item()
        total += label.size(0)
        epoch_loss += loss.item()

        if i % 100 == 0:
            print(f"Epoch {epoch+1} [{i}] | Loss: {loss.item():.4f} | CE: {ce.item():.3f} | Order: {order_loss.item():.3f}")

    acc = 100.0 * total_correct / total
    print(f"EPOCH {epoch+1} COMPLETE → ACCURACY: {acc:.2f}%")

