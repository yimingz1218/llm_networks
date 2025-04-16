import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

class KappaDataset(Dataset):
    def __init__(self, embeddings, l, i, j, z1, z2):
        # embeddings: N×d tensor, l,i,j,z1,z2: each length-N arrays
        self.w = embeddings
        self.l = l.float()
        self.i = i.float()
        self.j = j.float()
        # map z1,z2 ∈ {0,1,2} for T,H,D
        self.z1 = z1.long()
        self.z2 = z2.long()
    def __len__(self):
        return len(self.w)
    def __getitem__(self, idx):
        prop = torch.stack([self.i[idx]/self.l[idx].clamp(min=1),
                            self.j[idx]/self.l[idx].clamp(min=1),
                            1 - (self.i[idx]+self.j[idx]).clamp(max=self.l[idx])/self.l[idx].clamp(min=1)])
        onehot_z1 = torch.nn.functional.one_hot(self.z1[idx], num_classes=3).float()
        x = torch.cat([self.w[idx], prop, onehot_z1], dim=0)
        return x, self.z2[idx]

class KappaNet(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 3)
        )
    def forward(self, x):
        return self.net(x)  # logits

# --- training setup ---
def train_model(dataset, lr=1e-3, batch_size=128, epochs=20, weight_decay=1e-4):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = KappaNet(in_dim=dataset.w.shape[1] + 3 + 3)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in loader:
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
        print(f"Epoch {epoch+1}/{epochs} — loss: {total_loss/len(dataset):.4f}")
    return model

# Usage:
# ds = KappaDataset(embs, l, i, j, z1, z2)
# model = train_model(ds)
