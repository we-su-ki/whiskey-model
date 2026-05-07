import torch
import torch.nn as nn
import os

# 모델 구조 정의
class CocktailFlavorPredictor(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, 12)
        self.regressor = nn.Sequential(
            nn.Linear(12, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 12)
        )

    def forward(self, ids, vols):
        total_vol = torch.sum(vols, dim=1, keepdim=True) + 1e-8
        norm_vols = vols / total_vol

        embedded = self.embedding(ids)
        weighted = embedded * norm_vols.unsqueeze(-1)

        combined = torch.sum(weighted, dim=1)
        return self.regressor(combined)

# 모델 로드 함수
def load_trained_model(model_path):
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    vocab_size = checkpoint['vocab_size']
    
    model = CocktailFlavorPredictor(vocab_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # 추론 모드 전환
    return model

# 예측 실행
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    pth_path = os.path.join(current_dir, "..", "model", "cocktail_model.pth")

    if not os.path.exists(pth_path):
        print(f"[!] 모델 파일이 없습니다: {pth_path}")
    else:
        predictor = load_trained_model(pth_path)
        
        # 가상의 테스트 데이터
        test_ids = []
        test_vols = []
        
        # 패딩
        p_ids = torch.tensor([test_ids + [0]*(10-len(test_ids))], dtype=torch.long)
        p_vols = torch.tensor([test_vols + [0.0]*(10-len(test_vols))], dtype=torch.float32)
        
        # 맛 예측
        with torch.no_grad():
            prediction = predictor(p_ids, p_vols)
            result = prediction.squeeze().tolist()
        
        # 결과 출력
        FLAVORS = ['Sweet', 'Sour', 'Bitter', 'Umami_Salty', 'Fruity', 'Citrus', 'Floral', 'Herbal', 'Spicy', 'Woody_Smoky', 'Body', 'Fizzy']
        print("\n=== 칵테일 맛 예측 결과 ===")
        for name, score in zip(FLAVORS, result):
            print(f"{name}: {max(0.0, round(score, 2))}")
            # print(f"{name}: {score}")