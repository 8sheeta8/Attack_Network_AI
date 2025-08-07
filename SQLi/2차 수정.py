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
n_critic = 5  # Discriminator를 Generator보다 더 자주 학습
lambda_gp = 10  # Gradient Penalty 가중치

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
            nn.Sigmoid()  # 출력은 여전히 [0,1] 범위로 유지 (one-hot 유사)
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
            nn.Linear(256, 1)
            # WGAN에서는 Sigmoid 제거
        )

    def forward(self, x):
        x = x.view(-1, seq_len * vocab_size)
        return self.net(x)

# 3. Gradient Penalty 계산
def compute_gradient_penalty(D, real_samples, fake_samples):
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, 1).to(device)
    alpha = alpha.expand(real_samples.size())  # [batch_size, seq_len, vocab_size]

    # 실제와 가짜 샘플 사이의 보간
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)

    # Gradient 계산
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(d_interpolates.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    # Gradient Penalty: ||gradient||_2 - 1 의 L2 norm
    gradients = gradients.view(batch_size, -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_gp
    return gradient_penalty

# 4. 학습 준비
dataset = SQLDataset('/content/drive/MyDrive/doyean/Modified_SQL_Dataset.csv')
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

G = Generator().to(device)
D = Discriminator().to(device)

# WGAN에서는 RMSprop 또는 Adam with beta1=0이 일반적
optimizer_G = torch.optim.Adam(G.parameters(), lr=0.0001, betas=(0.0, 0.9))
optimizer_D = torch.optim.Adam(D.parameters(), lr=0.0001, betas=(0.0, 0.9))

# 5. 학습 루프
epochs = 1000
for epoch in range(epochs):
    for i, real_samples in enumerate(dataloader):
        real_samples = real_samples.to(device)
        batch_size = real_samples.size(0)

        # ---------------------
        # 1. Discriminator 학습 (n_critic 번)
        # ---------------------
        for _ in range(n_critic):
            # Noise 생성
            z = torch.randn(batch_size, latent_dim).to(device)
            fake_samples = G(z).detach()

            # Wasserstein loss
            d_real = D(real_samples).mean()
            d_fake = D(fake_samples).mean()
            gp = compute_gradient_penalty(D, real_samples, fake_samples)

            # Discriminator loss
            d_loss = -d_real + d_fake + gp

            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            # Gradient clipping (WGAN 안정성을 위해)
            for p in D.parameters():
                p.data.clamp_(-0.01, 0.01)

        # ---------------------
        # 2. Generator 학습
        # ---------------------
        z = torch.randn(batch_size, latent_dim).to(device)
        fake_samples = G(z)
        g_loss = -D(fake_samples).mean()  # Generator는 D의 출력 최대화

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

    if epoch % 5 == 0:
        print(f"[Epoch {epoch+1}/{epochs}] D_loss: {d_loss.item():.4f} | G_loss: {g_loss.item():.4f}")
