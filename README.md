# 트랜스포머 논문 기반 분석 및 구현 팀 과제

## 📋 프로젝트 개요

Transformer 논문 "Attention Is All You Need"를 분석하고, 핵심 구조를 PyTorch로 직접 구현하여 IMDB 영화 리뷰 감정 분류 작업을 수행한 프로젝트입니다.

---

## 📁 파일 구조

```
.
├── README.md                    # 프로젝트 개요 (본 문서)
├── Transformer_paper.md          # Transformer 논문 요약 및 핵심 이론 정리
├── Transformer_PyTorch구현.ipynb          # Transformer 변형 구조 및 응용 분석 (BERT, GPT, ViT)
├── Transformer_미니실험.ipynb               # 기본 Transformer 컴포넌트 구현 및 검증
└── Transformer_확장.md     # IMDB 감정 분류 전체 실험 코드
```

---

## 1. 논문 요약

### 핵심 내용
- **RNN/CNN 제거**: 순환(Recurrence) 및 합성곱(Convolution) 구조를 완전히 배제하고 오직 Attention 메커니즘만 사용
- **병렬화 가능**: 순차적 계산 제약을 제거하여 학습 속도 대폭 향상
- **장거리 의존성 해결**: 단어 간 거리와 무관하게 상수 시간 내 전역적 관계 파악

### 주요 구조
1. **Positional Encoding**: 삼각함수 기반 위치 정보 인코딩
2. **Scaled Dot-Product Attention**: Query, Key, Value를 활용한 상관관계 계산
3. **Multi-Head Attention**: 여러 관점에서 병렬로 정보 추출
4. **Encoder-Decoder**: 각 6개 레이어로 구성된 스택 구조

### 성능
- WMT 2014 영어-독일어 번역: **28.4 BLEU** (기존 대비 2 BLEU 향상)
- 학습 시간: 기존 모델 대비 **1/4 이하의 리소스**로 SOTA 달성

> **자세한 내용**: `tranformer_paper.md` 참고

---

## 2. 구현 설명

### 구현한 컴포넌트

#### 2.1 핵심 모듈 (`transformer.py`)
```python
1. CustomPositionalEncoding      # 위치 인코딩 (sin/cos 함수)
2. CustomAttention               # Scaled Dot-Product Attention
3. CustomMultiHeadAttention      # Multi-Head Attention
4. TransformerSentimentClassifier # 간단한 감정 분류기
```

#### 2.2 전체 시스템 (`transformer_미니실험.py`)
```python
1. Vocabulary                    # 단어-인덱스 매핑
2. IMDBDataset                   # 데이터셋 클래스
3. PositionalEncoding            # 위치 인코딩
4. MultiHeadAttention            # Multi-Head Self-Attention
5. FeedForwardNetwork            # Position-wise FFN
6. EncoderBlock                  # Attention + FFN + Residual + LayerNorm
7. TransformerEncoder            # Encoder Block 스택
8. SentimentClassifier           # 최종 분류 모델
```

### 주요 수식 구현

**Positional Encoding**:
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Scaled Dot-Product Attention**:
```
Attention(Q, K, V) = softmax(QK^T / √d_k) V
```

> **자세한 내용**: `transformer.py`, `transformer_미니실험.py` 참고

---

## 3. 실험 결과 요약

### 실험 설정
- **데이터셋**: IMDB Movie Reviews (50,000개)
- **Task**: 이진 감정 분류 (긍정/부정)
- **Split**: Train 25,000 / Test 25,000

### Baseline 모델 하이퍼파라미터
| 파라미터 | 값 |
|---------|-----|
| Vocabulary Size | 10,000 |
| Max Sequence Length | 256 |
| d_model | 128 |
| num_heads | 8 |
| num_layers | 3 |
| d_ff | 512 |
| Dropout | 0.1 |
| Learning Rate | 1e-4 |
| Batch Size | 32 |
| Epochs | 5 |

### 최종 성능

| 지표 | 값 |
|------|-----|
| **Test Accuracy** | **83.24%** |
| Train Accuracy | 84.60% |
| Precision | 83.35% |
| Recall | 83.24% |
| F1-Score | 83.23% |
| Parameters | 1,875,074 |

### 하이퍼파라미터 실험

| 모델 | d_model | Layers | Epochs | Test Acc |
|------|---------|--------|--------|----------|
| Baseline | 128 | 3 | 5 | **83.24%** ✅ |
| Large | 256 | 3 | 3 | 82.86% |

**발견사항**:
- 큰 모델이 항상 좋은 것은 아님
- IMDB는 비교적 단순한 태스크로 d_model=128로 충분
- 큰 모델은 수렴을 위해 더 많은 epoch 필요

> **자세한 내용**: `transformer_미니실험.py` 하단 최종 정리 섹션 참고

---

## 🔗 확장 및 응용

Transformer 아키텍처는 다양한 변형으로 발전:

- **BERT** (Encoder Only): 양방향 문맥 이해, 질문 답변/분류 태스크
- **GPT** (Decoder Only): 자기회귀 생성, 대화형 AI
- **Vision Transformer (ViT)**: 이미지를 패치로 분할하여 처리

> **자세한 내용**: `transformer_확장.py` 참고

---

