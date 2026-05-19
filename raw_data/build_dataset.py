import os
import json
import random
import pandas as pd
from sklearn.model_selection import train_test_split

# 경로 정의 및 파일 로드 설정
current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else "."
cocktails_path = os.path.join(current_dir, "cocktail_rows.csv")
ingredients_path = os.path.join(current_dir, "ingredients_rows.csv")

print("🔄 원본 데이터셋 로드 중...")
df_cocktails = pd.read_csv(cocktails_path)
df_ingredients = pd.read_csv(ingredients_path)

# 맛 지표 컬럼 정의
flavor_cols = ['Sweet', 'Sour', 'Bitter', 'Umami_Salty', 'Fruity', 'Citrus', 
               'Floral', 'Herbal', 'Spicy', 'Woody_Smoky', 'Body', 'Fizzy']

# 출력 저장 디렉토리 생성
data_dir = os.path.join(current_dir, "..", "data")
os.makedirs(data_dir, exist_ok=True)

# 모델 임베딩용 ingredient_vocab.csv 생성
def make_ingredient_vocab(df_i):
    print("[STEP 1] 재료 가중치 뼈대 사전(ingredient_vocab.csv) 생성 시작...")
    
    # 모델 학습 가중치 빌드에 필요한 컬럼 추출 및 카피
    # 원본 'id'를 학습 코드 스펙인 'ingredient_id'로 정제
    vocab_df = df_i[['id', 'name'] + flavor_cols + ['is_alcohol']].copy()
    vocab_df.rename(columns={'id': 'ingredient_id'}, inplace=True)
    
    vocab_save_path = os.path.join(data_dir, "ingredient_vocab.csv")
    vocab_df.to_csv(vocab_save_path, index=False)
    print(f"✅ ingredient_vocab.csv 생성 완료! (총 재료: {len(vocab_df)}개)")

# API 대응용 칵테일 데이터 정제 및 증강 함수
def create_augmented_dataset_v2(df_c, df_i, n_aug=5):
    print("[STEP 2] 칵테일 데이터셋 정제 및 데이터 증강(Augmentation) 시작...")
    final_data = []
    
    for _, row in df_c.iterrows():
        try:
            # 1. 재료 정보 JSON 파싱
            ing_raw = row['ingredients_ml'].replace("'", '"')
            ing_list = json.loads(ing_raw)
            
            ing_ids, volumes = [], []
            total_liquid_volume = 0.0
            
            for item in ing_list:
                matched = df_i[df_i['name'] == item['name']]
                if not matched.empty:
                    ing_id = int(matched.iloc[0]['id'])
                    vol = float(item['ml'])
                    
                    ing_ids.append(ing_id)
                    volumes.append(vol)
                    total_liquid_volume += vol

            if not ing_ids or total_liquid_volume == 0: 
                continue

            # 확실한 Target ABV 산출
            orig_proof = float(row['proof_inside_bracket_proof']) if not pd.isna(row['proof_inside_bracket_proof']) else 0.0 # proof가 유실된 경우 0으로 들어가는 버그 방지
            target_abv = round(orig_proof / 2.0, 2)
            
            # 데이터 베이스 객체 생성
            base_sample = {
                'method_category': row['method_category'] if not pd.isna(row['method_category']) else 'Unknown',
                'ingredient_ids': ing_ids,
                'volumes_ml': volumes,
                'target_abv': target_abv,
                'target_flavors': row[flavor_cols].values.tolist()
            }
            
            # 원본 추가
            final_data.append(base_sample)
            
            # 증강 실행 (순서 뒤섞기 및 미세 용량 노이즈 추가)
            for _ in range(n_aug):
                combined = list(zip(ing_ids, volumes))
                random.shuffle(combined)
                new_ids, new_vols = zip(*combined)
                
                # ±5% Jittering 적용
                jittered_vols = [round(v * random.uniform(0.95, 1.05), 2) for v in new_vols]
                
                final_data.append({
                    'method_category': base_sample['method_category'],
                    'ingredient_ids': list(new_ids),
                    'volumes_ml': jittered_vols,
                    'target_abv': base_sample['target_abv'], # 지터링을 주어도 한 잔의 총 타겟 스펙은 유지
                    'target_flavors': base_sample['target_flavors']
                })
        except Exception as e:
            continue
            
    return pd.DataFrame(final_data)

# 메인 실행 파이프라인
if __name__ == "__main__":
    # 보카 사전 빌드
    make_ingredient_vocab(df_ingredients)
    print("-" * 50)
    
    # 데이터 정제 및 증강 실행
    df_full = create_augmented_dataset_v2(df_cocktails, df_ingredients, n_aug=5)
    
    # 훈련/검증 데이터셋 분리 (8:2 비율)
    train_df, val_df = train_test_split(df_full, test_size=0.2, random_state=42, shuffle=True)
    
    # 최종 결과 CSV 파일 저장
    train_df.to_csv(os.path.join(data_dir, 'train_dataset.csv'), index=False)
    val_df.to_csv(os.path.join(data_dir, 'val_dataset.csv'), index=False)
    
    print("-" * 50)
    print(f"모든 전처리 프로세스 성공적으로 완료!")
    print(f"생성된 총 데이터 수 (증강 포함): {len(df_full)}개")
    print(f"저장 완료: {os.path.abspath(data_dir)}")