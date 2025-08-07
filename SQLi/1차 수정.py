import torch
import torch.nn as nn
import numpy as np
import random
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# 기본 설정
latent_dim = 100
seq_len = 200  # SQL query 길이 고정
vocab_size = 256  # byte 단위로 처리할 경우 256

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 1. CSV 데이터 로드 및 전처리
class SQLDataset(Dataset):
    def __init__(self, csv_path):
        df = pd.read_csv(csv_path)
        self.queries = df[df['Label'] == 1]['Query'].astype(str).tolist()

        # Preprocessing: 바이트로 인코딩 후 패딩
        self.encoded = []
        for q in self.queries:
            b = list(q.encode('utf-8'))[:seq_len]
            b += [0] * (seq_len - len(b))  # zero padding
            one_hot = np.eye(vocab_size)[b]  # one-hot 인코딩
            self.encoded.append(one_hot)

    def __len__(self):
        return len(self.encoded)

    def __getitem__(self, idx):
        return torch.tensor(self.encoded[idx], dtype=torch.float32)

# 2. 모델 정의 (Generator / Discriminator)
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, seq_len * vocab_size),
            nn.Sigmoid()
        )

    def forward(self, z):
        out = self.net(z)
        return out.view(-1, seq_len, vocab_size)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len * vocab_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.view(-1, seq_len * vocab_size)
        return self.net(x)

# 3. 유전 변이 함수 (선택사항, 활용 안할 수도 있음)
def genetic_mutation(output_seq, mutation_rate=0.1):
    mutated = output_seq.clone()
    for i in range(mutated.shape[0]):
        for j in range(mutated.shape[1]):
            if random.random() < mutation_rate:
                idx = random.randint(0, vocab_size - 1)
                mutated[i, j] = 0
                mutated[i, j, idx] = 1  # one-hot 변이
    return mutated

# 4. 학습 준비
dataset = SQLDataset('/content/drive/MyDrive/doyean/Modified_SQL_Dataset.csv')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

G = Generator().to(device)
D = Discriminator().to(device)

criterion = nn.BCELoss()
optimizer_G = torch.optim.Adam(G.parameters(), lr=0.0002)
optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0002)

# 5. 학습 루프
epochs = 1000
for epoch in range(epochs):
    for real_samples in dataloader:
        real_samples = real_samples.to(device)
        batch_size = real_samples.size(0)

        # 진짜/가짜 라벨
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # ---------------------
        # 1. Discriminator 학습
        # ---------------------
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_samples = G(z)

        D_real = D(real_samples)
        D_fake = D(fake_samples.detach())

        d_loss_real = criterion(D_real, real_labels)
        d_loss_fake = criterion(D_fake, fake_labels)
        d_loss = d_loss_real + d_loss_fake

        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # ---------------------
        # 2. Generator 학습
        # ---------------------
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_samples = G(z)
        D_fake = D(fake_samples)
        g_loss = criterion(D_fake, real_labels)  # Generator는 D를 속이고 싶어함

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

    if epoch%5 == 0:
      print(f"[Epoch {epoch+1}/{epochs}] D_loss: {d_loss.item():.4f} | G_loss: {g_loss.item():.4f}")
