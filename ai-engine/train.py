import os
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import ast
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(current_dir, "..", "data", "cocktail_training_dataset.csv")
vocab_path = os.path.join(current_dir, "..", "data", "ingredient_vocab.csv")
model_save_path = os.path.join(current_dir, "..", "model", "cocktail_model.pth")

df = pd.read_csv(dataset_path).fillna(0)
vocab_df = pd.read_csv(vocab_path).fillna(0)
vocab_df.columns = vocab_df.columns.str.strip()

def safe_eval(x):
    try:
        res = ast.literal_eval(str(x))
        return res if isinstance(res, list) else []
    except:
        return []

df['ingredient_ids'] = df['ingredient_ids'].apply(safe_eval)
df['volumes_ml'] = df['volumes_ml'].apply(safe_eval)
df['target_flavors'] = df['target_flavors'].apply(safe_eval)

df = df[df['target_flavors'].map(len) == 12].reset_index(drop=True)
print(f"[*] 유효 학습 데이터: {len(df)}개")

# 임베딩 가중치 초기화
RAW_FLAVOR_COLS = ['Sweet', 'Sour', 'Bitter', 'Umami_Salty', 'Fruity', 'Citrus', 'Floral', 'Herbal', 'Spicy', 'Woody_Smoky', 'Body', 'Fizzy']

max_id_in_data = 0
for ids in df['ingredient_ids']:
    if ids: max_id_in_data = max(max_id_in_data, max(ids))

max_id_in_vocab = vocab_df['ingredient_id'].max()
vocab_size = int(max(max_id_in_data, max_id_in_vocab) + 100)

init_weights = np.zeros((vocab_size, 12))
for _, row in vocab_df.iterrows():
    idx = int(row['ingredient_id'])
    if idx < vocab_size:
        init_weights[idx] = row[RAW_FLAVOR_COLS].values.astype(float)

class CocktailFlavorPredictor(nn.Module):
    def __init__(self, vocab_size, init_weights):
        super().__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(init_weights), freeze=False)
        self.regressor = nn.Sequential(
            nn.Linear(12, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 12)
        )

    def forward(self, ids, vols):
        embedded = self.embedding(ids)
        weighted = embedded * vols.unsqueeze(-1)
        combined = torch.sum(weighted, dim=1)
        return self.regressor(combined)

model = CocktailFlavorPredictor(vocab_size, init_weights)

optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-5)
criterion = nn.MSELoss()

print("[*] 학습 시작 (nan 방지 모드)...")
epochs = 10
max_len = 10 

for epoch in range(epochs):
    total_loss = 0
    for _, row in df.iterrows():
        ids = row['ingredient_ids']
        vols = row['volumes_ml']
        target = row['target_flavors']
        
        p_ids = torch.tensor([ids + [0]*(max_len-len(ids))], dtype=torch.long)
        p_vols = torch.tensor([vols + [0.0]*(max_len-len(vols))], dtype=torch.float32) / 100.0
        target_t = torch.tensor([target], dtype=torch.float32)
        
        optimizer.zero_grad()
        output = model(p_ids, p_vols)
        
        loss = criterion(output, target_t)
        
        # 만약 loss가 nan이면 즉시 중단하여 원인 파악
        if torch.isnan(loss):
            print(f"[!] 에러: Epoch {epoch}에서 Loss가 nan이 되었습니다.")
            exit()
            
        loss.backward()
        
        # 기울기 클리핑
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
    
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(df):.6f}")

torch.save({
    'model_state_dict': model.state_dict(),
    'vocab_size': vocab_size
}, model_save_path)
print(f"[*] 학습 완료 및 모델 저장: {model_save_path}")