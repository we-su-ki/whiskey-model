import torch
import torch.nn as nn

class CocktailModel(nn.Module):
    def __init__(self, vocab_size, init_weights):
        super().__init__()
        
        # 사전 학습된 임베딩 레이어
        base_weights = torch.FloatTensor(init_weights)
        self.embedding = nn.Embedding.from_pretrained(
            embeddings=base_weights, freeze=False, padding_idx=0
        )
        
        # 제조 방식 임베딩
        self.method_embedding = nn.Embedding(num_embeddings=5, embedding_dim=8)
        
        # 프로젝션 레이어
        self.embedding_proj = nn.Sequential(
            nn.Linear(13, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(0.10)
        )
        
        # 어텐션 레이어 및 피드 포워드
        self.attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True, dropout=0.10)
        self.norm1 = nn.LayerNorm(64)
        
        self.ff = nn.Sequential(
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, 64)
        )
        self.norm2 = nn.LayerNorm(64)
        
        # 회귀 분석을 위한 최종 MLP Head 
        # 풀링 벡터(64) + 메타 피처(2) + 제조방식 임베딩(8) = 총 74차원 입력
        self.head = nn.Sequential(
            nn.Linear(74, 256), nn.GELU(), nn.Dropout(0.15),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 13) # 최종 출력 차원: 13
        )
    
    def forward(self, ids, vols, methods):
        # Embedding & Projection
        emb = self.embedding(ids)
        emb = self.embedding_proj(emb)
        
        # 용량 가중치 부여 (소프트 가중치)
        weights = torch.sqrt(vols + 1e-6)
        emb = emb * weights.unsqueeze(-1)
        
        # Attention & Residual Blocks
        attn_out, _ = self.attn(emb, emb, emb)
        x = self.norm1(emb + attn_out)
        
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        # 패딩 마스킹 적용 및 풀링
        mask = (ids != 0).unsqueeze(-1).float()
        x = x * mask
        pooled = x.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        
        # 메타 피처 결합 (재료 개수 비율, 최대 용량)
        ingredient_count = (ids != 0).sum(dim=1, keepdim=True).float() / 12.0
        dominant_volume = vols.max(dim=1, keepdim=True)[0]
        
        # 제조 방식 임베딩 벡터 추출 및 결합
        m_emb = self.method_embedding(methods) # (batch_size, 8)
        pooled = torch.cat([pooled, ingredient_count, dominant_volume, m_emb], dim=1)
        
        # 최종 예측
        out = self.head(pooled)
        
        # ABV 예측 부위(0번 인덱스)와 맛 점수 부위(1~12번 인덱스) 분할 슬라이싱 처리
        abv_pred = torch.relu(out[:, 0:1]) # 도수는 음수가 나올 수 없음
        flavor_pred = torch.sigmoid(out[:, 1:]) * 10.0
        
        return torch.cat([abv_pred, flavor_pred], dim=1)