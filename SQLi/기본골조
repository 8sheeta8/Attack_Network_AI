import torch
import torch.nn as nn
import numpy as np
import random

latent_dim = 100
seq_len = 50  
vocab_size = 256  

# 생성기: latent z → SQL 시퀀스
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

# 판별기: SQL 시퀀스 → 진짜/가짜 판별
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

# GA 개체 구조 
def genetic_mutation(output_seq, mutation_rate=0.1):
    mutated = output_seq.clone()
    for i in range(mutated.shape[0]):
        for j in range(mutated.shape[1]):
            if random.random() < mutation_rate:
                idx = random.randint(0, vocab_size - 1)
                mutated[i, j] = 0
                mutated[i, idx] = 1  # one-hot 변이
    return mutated
