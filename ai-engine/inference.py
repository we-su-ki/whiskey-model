import os
import torch
import torch.nn as nn

from model_arch import CocktailModel

# 디바이스 설정
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 제조 방식 카테고리 매핑 사전
METHOD_MAP = {'Build': 0, 'Shake': 1, 'Stir': 2, 'Blend': 3, 'Unknown': 4}
FLAVORS = [
    'Sweet', 'Sour', 'Bitter', 'Umami_Salty', 
    'Fruity', 'Citrus', 'Floral', 'Herbal', 
    'Spicy', 'Woody_Smoky', 'Body', 'Fizzy'
]

# 모델 로드 함수
def load_trained_model(model_path):
    # 가중치 파일 열기
    checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    vocab_size = checkpoint['vocab_size']
    init_weights = checkpoint['init_weights']
    
    # 올바른 아키텍처로 선언 후 가중치 주입
    model = CocktailModel(vocab_size, init_weights)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE)
    model.eval() # 추론 모드 고정
    return model

# 입력 파싱 및 패딩 전처리 함수
def preprocess_input(ingredients, method_str, max_len=12):
    """
    ingredients: [ {"id": 930, "ml": 45.0}, {"id": 126, "ml": 15.0} ... ] 형태의 리스트
    method_str: "Shake", "Build" 등의 문자열
    """
    # 입력값 전처리 데이터 파싱
    ing_ids = [item['id'] for item in ingredients]
    vols = [item['ml'] for item in ingredients]
    
    pad_len = max_len - len(ing_ids)
    
    # 패딩 처리
    padded_ids = ing_ids[:max_len] + [0] * pad_len
    padded_vols = vols[:max_len] + [0.0] * pad_len
    
    #  ML 단위 정규화 스케일링 (/100.0)
    padded_vols = [v / 100.0 for v in padded_vols]
    
    # 제조 방식 인덱싱
    method_idx = METHOD_MAP.get(method_str, 4)
    
    # 텐서 변환 및 디바이스 할당 (Batch 차원 추가)
    ids_tensor = torch.tensor([padded_ids], dtype=torch.long).to(DEVICE)
    vols_tensor = torch.tensor([padded_vols], dtype=torch.float32).to(DEVICE)
    method_tensor = torch.tensor([method_idx], dtype=torch.long).to(DEVICE)
    
    return ids_tensor, vols_tensor, method_tensor

# 실행부
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else "."
    pth_path = os.path.join(current_dir, "..", "model", "cocktail_model.pth")

    if not os.path.exists(pth_path):
        print(f"[!] 모델 파일이 없습니다: {pth_path}\n먼저 train.py를 성공적으로 끝내주세요.")
    else:
        print("🔄 가중치 로드 및 인공지능 서빙 엔진 시동 중...")
        predictor = load_trained_model(pth_path)
        print("✅ 모델 로드 완료!")
        
        # 가상의 가이드 테스트 데이터 (API 요청 예시 규격)
        # 예: id:1 Dry cider 45ml + id:198 Mango 30ml (제조방식: Shake)
        # sample_ingredients = [
        #     {"id": 1, "ml": 45.0}, 
        #     {"id": 30, "ml": 30.0}
        # ]
        # sample_method = "Shake"
        
        sample_ingredients = [
            {"id": 0, "ml": 0}
        ]
        sample_method = "Shake"
        sample_ingredients = [
            {
            "id": 1348,
            "ml": 50.0
            }, 
            {
            "id": 927,
            "ml": 25.0
            }, 
            {
            "id": 577,
            "ml": 5.0
            },
            {
            "id": 264,
            "ml": 0.1
            }
        ]
        sample_method = "Stir"
        
        # 전처리 실행
        p_ids, p_vols, p_methods = preprocess_input(sample_ingredients, sample_method)
        
        # 맛 및 도수 추론 실행
        with torch.no_grad():
            prediction = predictor(p_ids, p_vols, p_methods)
            result = prediction.squeeze().cpu().tolist()
        
        # 결과 출력 (정답 API 명세 포맷 데이터 매핑)
        predicted_abv = result[0]        # 0번 인덱스는 도수(ABV)
        predicted_flavors = result[1:]   # 1번부터 끝까지는 12개 맛 점수
        
        print("\n" + "="*30)
        print("        AI 칵테일 예측 결과")
        print("="*30)
        print(f"예상 알코올 도수 (ABV): {round(predicted_abv, 2)}%")
        print("-" * 30)
        for name, score in zip(FLAVORS, predicted_flavors):
            print(f"• {name:<12} : {round(score, 2)}점 / 10.0점")
            # print(f"• {name:<12} : {round(score)}점 / 10.0점")
        print("="*30)