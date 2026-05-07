# 🥃 whiskey-model (Repository Description)

### 🌟 Project Summary
사용자가 입력한 위스키 및 재료 레시피를 바탕으로 12가지 맛 지표(Flavor Profile)를 예측하는 딥러닝 기반 AI 서빙 서버입니다. PyTorch로 학습된 모델을 FastAPI를 통해 고성능 API로 제공합니다.

### 🛠 Tech Stack
- Framework: FastAPI (Python 3.10+)
- Deep Learning: PyTorch (Embedding + MLP Regression)
- Data Analysis: Pandas, Scikit-learn
- Serving: Uvicorn

### 🏗 핵심 기능 (Core Logic)
- 재료 임베딩 (Ingredient Embedding): 각 재료의 고유 특성을 고차원 벡터로 변환하여 학습된 맛의 관계성을 파악합니다.
- 가중 합산 (Weighted Aggregation): 레시피에 포함된 재료의 용량(ml)을 가중치로 사용하여 전체적인 맛의 밸런스를 계산합니다.
- 맛 예측 (Flavor Prediction): 단맛, 신맛, 쓴맛, 바디감 등 12가지 지표를 0~10 사이의 수치로 추론합니다.

## 📂 Directory Structure
```Plaintext
whiskey-model/
├── models/
│   └── cocktail_model.pth       # 학습된 PyTorch 모델 가중치
├── data/
│   └── ingredient_vocab.csv      # 재료 ID 및 초기 맛 데이터 사전
├── main.py                       # API 엔드포인트 및 서버 실행 파일
├── model_arch.py                 # 모델 클래스(Architecture) 정의
├── requirements.txt              # 의존성 라이브러리 목록
└── .gitignore                    # 불필요한 파일 제외 설정
```

## 🚀 Quick Start
```Bash
# 1. 라이브러리 설치
pip install -r requirements.txt

# 2. 서버 실행
python main.py
```

---