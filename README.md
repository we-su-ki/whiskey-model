# 🥃 whiskey-model (Repository Description)

### 🌟 Project Summary
사용자가 입력한 위스키 및 재료 레시피를 바탕으로 알코올 도수(ABV) 및 12가지 맛 지표(Flavor Profile)를 정밀하게 예측하는 딥러닝 기반 AI 서빙 서버입니다. PyTorch로 학습된 Multi-Head Attention 기반 회귀 모델을 FastAPI를 통해 고성능 API로 제공합니다.

### 🛠 Tech Stack
- Framework: FastAPI (Python 3.10+)
- Deep Learning: PyTorch (Embedding + Multi-Head Attention + MLP Regression)
- Data Analysis: Pandas, NumPy
- Serving: Uvicorn

### 🏗 핵심 기능 (Core Logic)
- 재료 임베딩 & 어텐션 (Embedding & Attention): 각 재료 고유의 맛 프로필과 알코올 유무를 벡터화하고, Multi-Head Attention 구조를 통해 재료 간의 미세한 상호작용과 시너지를 추론합니다.
- 용량 기반 가중 합산 (Weighted Aggregation): 레시피에 들어가는 재료별 용량($ml$)의 스퀘어루트 값(sqrt)을 소프트 가중치로 사용하여 맛의 전체적인 지배력을 계산합니다.
- 복합 손실 벌점 알고리즘 (Adaptive Loss Weighting): Baseline 오차 분석 결과에 따라 예측이 어려운 핵심 맛 노드(Citrus, Fruity, Sweet 등)와 탄산감(Fizzy)에 차등 손실 벌점을 부여하여 높은 일반화 성능을 보장합니다.

## 📂 Directory Structure
```Plaintext
whiskey-model/
├── ai-engine/
│   ├── main.py            # FastAPI API 엔드포인트 및 서버 구동 파일
│   ├── inference.py       # 단일 레시피 덤프 기반 맛 프로필 추론 로직
│   ├── model_arch.py      # CocktailModel 신경망 아키텍처 정의 클래스
│   └── train.py           # 오차 기반 동적 벌점 제어가 포함된 모델 학습 스크립트
│
├── model/                 # 학습 완료된 가중치 저장 폴더
│   └── cocktail_model.pth # 검증 오차(Val MAE) 최적 시점(Best)의 모델 파라미터
│
├── data/                  # 전처리와 패딩이 완료되어 즉시 학습 가능한 정제 CSV 폴더
│
├── raw_data/              # 원본 소스 데이터 수집 및 데이터 가공 전 원시 데이터베이스
│   └── build_dataset.py   # 로우 데이터를 학습용 포맷 데이터셋으로 변환하는 스크립트
│
├── logs/                  # 학습 에포크별 Train/Val Loss 및 MAE 기록 로그 폴더
│
├── requirements.txt       # 프로젝트 의존성 라이브러리 목록
└── .gitignore
```

## 🚀 Quick Start

1. 가상환경 구축 및 의존성 라이브러리 설치
```
# 레포지토리 클론 후 루트 디렉토리에서 실행
pip install -r requirements.txt
```

2. AI 모델 학습 및 가중치 생성
```
python ai-engine/train.py
```

3. FastAPI 추론 서버 실행
```
python ai-engine/main.py
```

---