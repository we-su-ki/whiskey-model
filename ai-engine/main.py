import os
import torch
import torch.nn as nn

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

from model_arch import CocktailModel

app = FastAPI(
    title="AI Cocktail Flavor & ABV Predictor API",
    description="칵테일 레시피(재료 ID, 용량, 기법)를 기반으로 맛과 도수를 실시간 예측하는 AI 엔진",
    version="1.0.0"
)

# 디바이스 설정
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 제조 방식 카테고리 매핑 사전
METHOD_MAP = {'Build': 0, 'Shake': 1, 'Stir': 2, 'Blend': 3, 'Unknown': 4}
FLAVOR_NAMES = [
    'Sweet', 'Sour', 'Bitter', 'Umami_Salty', 
    'Fruity', 'Citrus', 'Floral', 'Herbal', 
    'Spicy', 'Woody_Smoky', 'Body', 'Fizzy'
]

# API 전용 서버 시동 시 모델 로드 부 (글로벌 싱글톤 패턴)
model = None

@app.on_event("startup")
def load_model_on_startup():
    global model
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(current_dir, "..", "model", "cocktail_model.pth")
    
    if not os.path.exists(model_path):
        print(f"❌ [에러] 모델 파일 부재: {model_path}")
        return

    try:
        print("🔄 AI 서빙 엔진 인스턴스를 메모리에 로드하는 중...")
        # 최신 PyTorch 버전 대응을 위한 weights_only=False 안전 주입
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
        vocab_size = checkpoint['vocab_size']
        init_weights = checkpoint['init_weights']
        
        # 구조 일치 및 가중치 입히기
        model = CocktailModel(vocab_size, init_weights)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(DEVICE)
        model.eval()
        print("✅ AI 예측 엔진 서빙 준비 완료!")
    except Exception as e:
        print(f"❌ 가중치 로드 중 결함 발생: {e}")

# 0.5 단위 서비스 정제용 함수
def round_to_nearest_05(score: float) -> float:
    rounded = round(score * 2) / 2
    return max(0.0, min(10.0, rounded))

# Pydantic 요청/응답 스키마 선언
class IngredientItem(BaseModel):
    id: int
    amount: float  # ml 단위 수치

class RecipeRequest(BaseModel):
    ingredients: List[IngredientItem]
    method: str  # 예: "Shake", "Stir", "Build", "Blend"

# 추론 엔드포인트 구현
@app.post("/predict", summary="칵테일 분석 및 예측 실행")
async def predict(req: RecipeRequest):
    global model
    if model is None:
        raise HTTPException(status_code=503, detail="AI 모델 엔진이 준비되지 않았습니다.")
    
    if not req.ingredients:
        raise HTTPException(status_code=400, detail="최소 1개 이상의 재료가 필요합니다.")

    # 입력값 전처리 데이터 파싱
    ing_ids = [i.id for i in req.ingredients]
    amounts = [i.amount for i in req.ingredients]
    
    max_len = 12  # 학습 스펙 패딩 고정 길이 조정 (12개)
    if len(ing_ids) > max_len:
        ing_ids = ing_ids[:max_len]
        amounts = amounts[:max_len]
        
    pad_len = max_len - len(ing_ids)
    
    padded_ids = ing_ids + [0] * pad_len
    # ML 단위 정규화 스케일링 스페이스 (/100.0) 보정
    padded_vols = [v / 100.0 for v in amounts] + [0.0] * pad_len
    
    # 제조 기법 인덱스 변환 (대소문자 방어 코딩 포함)
    method_norm = req.method.strip().capitalize()
    method_idx = METHOD_MAP.get(method_norm, 4) # 맵에 없으면 Unknown(4) 처리
    
    try:
        # 텐서 배치화 생성 및 디바이스 밀어넣기
        ids_t = torch.tensor([padded_ids], dtype=torch.long).to(DEVICE)
        vols_t = torch.tensor([padded_vols], dtype=torch.float32).to(DEVICE)
        methods_t = torch.tensor([method_idx], dtype=torch.long).to(DEVICE)
        
        # AI 예측 연산 개시
        with torch.no_grad():
            prediction = model(ids_t, vols_t, methods_t)
            result = prediction.squeeze().cpu().tolist()
            
        # 결과 패키징 가공 분리
        predicted_abv = max(0.0, round(result[0], 2)) # 알코올 도수는 소수점 둘째자리 일반 반올림
        predicted_flavors = result[1:]
        
        # 0.5 단위 맵 정제 기술 적용
        flavor_output = {
            name: round_to_nearest_05(val) 
            for name, val in zip(FLAVOR_NAMES, predicted_flavors)
        }
        
        # 최종 JSON API 규격 리턴
        return {
            "status": "success",
            "data": {
                "abv": predicted_abv,
                "flavors": flavor_output
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference 도중 커널 에러 발생: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)