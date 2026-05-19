import os
import json
import random
import ast

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from model_arch import CocktailModel

# 환경 설정 및 하이퍼 파라미터
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else "."

# 정제된 데이터셋 경로
TRAIN_DATA_PATH = os.path.join(CURRENT_DIR, "..", "data", "train_dataset.csv")
VAL_DATA_PATH = os.path.join(CURRENT_DIR, "..", "data", "val_dataset.csv")
VOCAB_PATH = os.path.join(CURRENT_DIR, "..", "data", "ingredient_vocab.csv")
MODEL_SAVE_PATH = os.path.join(CURRENT_DIR, "..", "model", "cocktail_model.pth")
LOG_PATH = os.path.join(CURRENT_DIR, "..", "logs")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

FLAVOR_COLS = [
    'Sweet', 'Sour', 'Bitter', 'Umami_Salty', 
    'Fruity', 'Citrus', 'Floral', 'Herbal', 
    'Spicy', 'Woody_Smoky', 'Body', 'Fizzy'
]

# 제조 방식 카테고리 매핑 사전
METHOD_MAP = {'Build': 0, 'Shake': 1, 'Stir': 2, 'Blend': 3, 'Unknown': 4}

# 데이터셋 정의
class CocktailDataset(Dataset):
    def __init__(self, file_path, max_len=12):
        df = pd.read_csv(file_path).fillna(0)
        self.data = []
        
        for _, row in df.iterrows():
            try:
                # ast.literal_eval로 파싱 버그 원천 차단
                ids = ast.literal_eval(str(row['ingredient_ids']))
                vols = ast.literal_eval(str(row['volumes_ml']))
                flavors = ast.literal_eval(str(row['target_flavors']))
                
                # 정제된 데이터에 target_abv가 없으면 이전 코드의 target_proof 기반 역산 적용
                if 'target_abv' in row:
                    abv = float(row['target_abv'])
                else:
                    abv = float(row['target_proof']) / 2.0 if 'target_proof' in row else 0.0
                
                # Target 변수 구성: [ABV(1개) + Flavors(12개)] = 총 13차원
                target = [abv] + flavors
                target = np.array(target, dtype=np.float32)
                
                # 제조 방식 정수 인코딩
                method_str = row['method_category'] if 'method_category' in row else 'Build'
                method_idx = METHOD_MAP.get(method_str, 4)
                
                # Padding & Truncation
                ids = ids[:max_len]
                vols = vols[:max_len]
                pad_len = max_len - len(ids)
                
                ids += [0] * pad_len
                vols += [0.0] * pad_len
                
                # Volume 정규화 (100ml 기준 스케일링)
                vols = [v / 100.0 for v in vols]
                
                self.data.append({
                    'ids': torch.tensor(ids, dtype=torch.long),
                    'vols': torch.tensor(vols, dtype=torch.float32),
                    'method': torch.tensor(method_idx, dtype=torch.long),
                    'target': torch.tensor(target, dtype=torch.float32)
                })
            except Exception as e:
                continue
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

# 학습 및 검증 프로세스
def run():
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    os.makedirs(LOG_PATH, exist_ok=True)
    log_file = os.path.join(LOG_PATH, "train_log.txt")
    
    # Vocab 데이터 로드 및 가중치 초기화 행렬 구성
    vocab_df = pd.read_csv(VOCAB_PATH)
    vocab_size = int(vocab_df['ingredient_id'].max() + 100)
    init_weights = np.zeros((vocab_size, 13), dtype=np.float32)
    
    for _, row in vocab_df.iterrows():
        idx = int(row['ingredient_id'])
        if idx >= vocab_size:
            continue
        
        flavor_vector = row[FLAVOR_COLS].values.astype(np.float32)
        alcohol_flag = float(row['is_alcohol'])
        init_weights[idx] = np.append(flavor_vector, alcohol_flag)
    
    # DataLoader 정의
    train_dataset = CocktailDataset(TRAIN_DATA_PATH)
    val_dataset = CocktailDataset(VAL_DATA_PATH)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # 모델 및 최적화 도구 설정
    model = CocktailModel(vocab_size, init_weights).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, min_lr=1e-6)
    
    # 검증 데이터셋 수치 분석 결과에 입각하여 오차 0.5 넘었던 노드들과 Fizzy 타겟 강제 수동 가중치 지정
    # 모델 출력 순서: [0: ABV, 1: Sweet, 2: Sour, 3: Bitter, 4: Umami, 5: Fruity, 6: Citrus, 7: Floral, 8: Herbal, 9: Spicy, 10: Woody, 11: Body, 12: Fizzy]
    loss_weights = torch.ones(13, dtype=torch.float32).to(DEVICE)
    loss_weights[0] = 1.5
    loss_weights[1] = 1.8
    loss_weights[2] = 1.0
    loss_weights[3] = 1.2
    loss_weights[4] = 1.0
    loss_weights[5] = 1.8
    loss_weights[6] = 2.0
    loss_weights[7] = 1.2
    loss_weights[8] = 1.6
    loss_weights[9] = 1.2
    loss_weights[10] = 1.2
    loss_weights[11] = 1.2
    loss_weights[12] = 3.0
    
    base_criterion = nn.SmoothL1Loss(beta=0.5, reduction='none')
    
    epochs = 300
    best_mae = float('inf')
    patience = 30
    patience_counter = 0
    
    print(f"[*] Train Dataset Count: {len(train_dataset)}")
    print(f"[*] Val Dataset Count: {len(val_dataset)}")
    print(f"[*] 오차 지표 정량 분석 완료 기반 동적 복합 손실 벌점 행렬 활성화.")
    print(f"[*] 적용된 가중치 벡터:\n{loss_weights.cpu().numpy()}")
    
    if len(train_dataset) == 0:
        print("에러: 데이터셋 변환 중 데이터가 0개입니다.")
        return
    
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("TRAIN START\n\n")
    
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            ids = batch['ids'].to(DEVICE)
            vols = batch['vols'].to(DEVICE)
            methods = batch['method'].to(DEVICE)
            target = batch['target'].to(DEVICE)
            
            optimizer.zero_grad()
            output = model(ids, vols, methods)
            
            # 요소별 손실 계산 후 업데이트된 복합 멀티 밸런스 가중치 행렬 곱 연산 적용
            raw_loss = base_criterion(output, target)  
            weighted_loss = raw_loss * loss_weights    # 13개 노드 오차 수준별 벌점 가중 차등 분배
            loss = weighted_loss.mean()                
            
            if torch.isnan(loss):
                continue
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation Phase
        model.eval()
        val_loss, val_mae = 0.0, 0.0
        valid_batches = 0
        
        # 에포크 단위 검증 단계에서 각 노드별 MAE를 누적 수집하여, 
        # 실시간 모니터링 및 필요시 오차 가중치 추가 핸들링을 위한 통계 저장소 설계
        column_errors = torch.zeros(13).to(DEVICE)
        
        with torch.no_grad():
            for batch in val_loader:
                ids = batch['ids'].to(DEVICE)
                vols = batch['vols'].to(DEVICE)
                methods = batch['method'].to(DEVICE)
                target = batch['target'].to(DEVICE)
                
                output = model(ids, vols, methods)
                if torch.isnan(output).any():
                    continue
                
                raw_loss = base_criterion(output, target)
                weighted_loss = raw_loss * loss_weights
                loss = weighted_loss.mean()
                
                # 가중치 계산을 우회한 순수 절댓값 MAE 계산
                batch_mae_matrix = torch.abs(output - target)
                mae = batch_mae_matrix.mean()
                
                # 각 노드별 오차 누적합
                column_errors += batch_mae_matrix.mean(dim=0)
                
                val_loss += loss.item()
                val_mae += mae.item()
                valid_batches += 1
                
        avg_val_loss = val_loss / max(valid_batches, 1)
        avg_val_mae = val_mae / max(valid_batches, 1)
        
        # 에포크 평균 칼럼별 오차 계산 완료 부
        final_column_errors = (column_errors / max(valid_batches, 1)).cpu().numpy()
        
        scheduler.step(avg_val_mae)
        current_lr = optimizer.param_groups[0]['lr']
        
        log_text = f"Epoch {epoch+1:03d} | LR {current_lr:.6f} | Train Loss {avg_train_loss:.4f} | Val Loss {avg_val_loss:.4f} | Val MAE {avg_val_mae:.4f}"
        print(log_text)
        
        # 매 50 에포크마다 취약 노드 현황 모니터링 로그 출력 부 활성화
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f" [맛 노드별 실시간 MAE 현황]")
            print(f"   - Sweet: {final_column_errors[1]:.4f} | Sour: {final_column_errors[2]:.4f} | Bitter: {final_column_errors[3]:.4f}")
            print(f"   - Fruity: {final_column_errors[5]:.4f} | Citrus: {final_column_errors[6]:.4f} | Fizzy: {final_column_errors[12]:.4f}")
            
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_text + "\n")
            
        if avg_val_mae < best_mae:
            best_mae = avg_val_mae
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab_size': vocab_size,
                'init_weights': init_weights
            }, MODEL_SAVE_PATH)
            
            save_log = f"✅ Best model saved | MAE {best_mae:.4f}"
            print(save_log)
            with open(log_file, "a", encoding="utf-8") as f:
                f.write(save_log + "\n")
        else:
            # 조기 종료에 도달하기 전인 15 에포크 동안 진전이 없을 때, 오차가 높은 고질적 불량 칼럼의 손실 패널티를 20%씩 추가 인상하여 모델을 압박
            patience_counter += 1
            if patience_counter == 15:
                print("[가중치 동적 밸런싱 발동] 학습 정체 극복을 위해 오차 0.5 돌파 항목 가중치 강화")
                for i in range(13):
                    if final_column_errors[i] > 0.5:
                        loss_weights[i] *= 1.2
                print(f" 조정된 가중치 행렬:\n{loss_weights.cpu().numpy()}")
            
        if patience_counter >= patience:
            print("\n🛑 Early stopping triggered")
            break

if __name__ == "__main__":
    run()