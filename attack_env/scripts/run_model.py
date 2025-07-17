import pandas as pd
import torch
import os
import sys

# 1. 공격 유형에 따라 모델 선택
def select_model(pcap_filename):
    if any(x in pcap_filename.lower() for x in ["web", "xss", "sql"]):
        return "model/web_model.pt"
    else:
        return "model/non_web_model.pt"

# 2. 실행 인자 확인
if len(sys.argv) != 2:
    print("Usage: python3 run_model.py <pcap_or_npy_filename>")
    sys.exit(1)

filename = sys.argv[1]
model_path = select_model(filename)
print(f"[INFO] Selected model: {model_path}")

# 3. flow.csv 로드
df = pd.read_csv("flow.csv")
X = torch.tensor(df.values, dtype=torch.float32)

# 4. 모델 로드
model = torch.load(model_path)
model.eval()

# 5. 예측 수행
with torch.no_grad():
    y_pred = model(X)

# 6. 이진 분류 처리
if y_pred.shape[1] == 2:
    preds = torch.argmax(y_pred, dim=1)
else:
    preds = (torch.sigmoid(y_pred) > 0.5).long().view(-1)

# 7. 결과 저장
os.makedirs("logs", exist_ok=True)
with open("logs/result.txt", "w") as f:
    for i, label in enumerate(preds):
        result = "MALICIOUS" if label == 1 else "BENIGN"
        f.write(f"Flow {i}: {result}\n")

print("[INFO] 탐지 완료 → logs/result.txt 저장됨.")

