from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import torch
import torch.nn as nn
import os

app = FastAPI()

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

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "..", "model", "cocktail_model.pth")

try:
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    vocab_size = checkpoint['vocab_size']
    model = CocktailFlavorPredictor(vocab_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print("모델 로드 성공!")
except FileNotFoundError:
    print(f"에러: {model_path} 파일을 찾을 수 없습니다.")
except Exception as e:
    print(f"모델 로드 중 오류 발생: {e}")

FLAVOR_NAMES = [
    'Sweet', 'Sour', 'Bitter', 'Umami_Salty', 
    'Fruity', 'Citrus', 'Floral', 'Herbal', 
    'Spicy', 'Woody_Smoky', 'Body', 'Fizzy'
]

class IngredientItem(BaseModel):
    id: int
    amount: float

class RecipeRequest(BaseModel):
    ingredients: List[IngredientItem]

@app.post("/predict")
async def predict(req: RecipeRequest):
    ing_ids = [i.id for i in req.ingredients]
    amounts = [i.amount for i in req.ingredients]

    max_len = 10
    ids_t = torch.tensor([ing_ids + [0]*(max_len - len(ing_ids))], dtype=torch.long)
    vols_t = torch.tensor([amounts + [0.0]*(max_len - len(amounts))], dtype=torch.float32)
    
    with torch.no_grad():
        pred = model(ids_t, vols_t).squeeze().tolist()

    result = {name: max(0.0, round(val, 2)) for name, val in zip(FLAVOR_NAMES, pred)}
    
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)