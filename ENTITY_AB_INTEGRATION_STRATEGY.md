# Entity Extraction A/B Integration Strategy
## 병합 전략 및 개선사항 종합 문서

**작성일**: 2024-12-16  
**목적**: `final_medical_ai_agent_entity_ab_addon` 스캐폴드를 현재 스캐폴드에 논리적으로 병합하고, 현재 스캐폴드의 강점을 살리면서 단점을 보완

---

## 1. 병합 개요

### 1.1 병합 대상 스캐폴드

| 항목 | 이전 스캐폴드 | 새 스캐폴드 (addon) |
|------|--------------|-------------------|
| 경로 | `C:\Users\KHIDI\Downloads\med_entity_ab_scaffold` | `C:\Users\KHIDI\Downloads\final_medical_ai_agent_entity_ab_addon` |
| 핵심 개선 | 3자 비교 기본 구조 | **라우터 패턴** + Agent 통합 최적화 |
| 주요 추가 | - | `extraction/entity_ab_router.py` (핵심) |

### 1.2 병합 원칙

1. **무결성 보장**: 현재 스캐폴드의 기존 Agent/RAG 코드 영향 최소화
2. **모듈성 강화**: 엔티티 추출을 "교체 가능한 부품"으로 설계
3. **차별점 강화**: 현재 스캐폴드의 강점 (RAGAS 평가, 멀티턴 대화, Agentic RAG) 유지
4. **단점 보완**: 엔티티 추출 A/B 테스트 자동화 및 평가 지표 강화

---

## 2. 현재 스캐폴드의 강점 분석

### 2.1 차별화된 강점 (유지 및 강화 대상)

#### A. RAGAS 기반 RAG 시스템 평가
- **위치**: `experiments/evaluation/ragas_metrics.py`
- **강점**:
  - LLM-as-a-Judge (GPT-4o-mini) 기반 자동 평가
  - Faithfulness, Answer Relevancy, Context Precision/Recall/Relevancy 5개 축
  - 멀티턴 대화 로그 평가 지원
- **개선 방향**: 엔티티 추출 품질이 RAG 성능에 미치는 영향 정량화
  - 예: MedCAT vs QuickUMLS로 추출한 엔티티를 검색 쿼리로 사용 → RAGAS 점수 비교

#### B. Agentic RAG + 3-Tier Memory
- **위치**: `agents/`, `memory/`
- **강점**:
  - 단순 RAG를 넘어선 동적 검색 및 자기개선 루프
  - 멀티턴 컨텍스트 관리 (short/long/episodic memory)
- **개선 방향**: 엔티티 추출 결과를 메모리 인덱싱에 활용
  - 예: 추출된 의료 엔티티(CUI)를 메모리 키로 사용 → 관련 대화 검색 정확도 향상

#### C. 멀티턴 대화 로그 생성 및 평가
- **위치**: `experiments/run_llm_vs_rag_comparison.py`
- **강점**:
  - LLM-only, Basic RAG, Corrective RAG 3자 비교 자동화
  - 턴별 로그 (turn_id, session_id, context, response) 체계적 저장
- **개선 방향**: 엔티티 추출 방법을 추가 변수로 포함
  - 예: `Basic_RAG_MedCAT` vs `Basic_RAG_QuickUMLS` 비교

#### D. 한국어 의료 도메인 특화
- **위치**: `extraction/multilingual_medcat.py`, `extraction/neural_translator.py`
- **강점**:
  - Helsinki-NLP 번역 모델 통합
  - 한국어 → 영어 번역 → MedCAT 추출 파이프라인
- **개선 방향**: KM-BERT NER로 한국어 직접 추출 vs 번역 기반 추출 비교
  - 예: "당뇨병" → KM-BERT: B-DISEASE, 번역: "diabetes" → MedCAT: C0011849

---

### 2.2 현재 스캐폴드의 한계점 (보완 대상)

| 한계점 | 현재 상태 | 개선 방안 (새 스캐폴드 반영) |
|--------|----------|---------------------------|
| **엔티티 추출 A/B 테스트 어려움** | MedCAT만 주로 사용, 비교 실험 수동 | `entity_ab_router.py`로 환경 변수 기반 자동 교체 |
| **NER 평가 지표 부재** | 엔티티 추출 정확도 미측정 | `src/med_entity_ab/metrics/ner_metrics.py` 통합 (strict/overlap F1, boundary IoU) |
| **KM-BERT NER 학습 스크립트 부재** | KM-BERT 사전학습 모델만 언급 | `scripts/train_kmbert_ner.py` 추가 (CoNLL 형식 지원) |
| **배치 비교 실험 자동화 부족** | 단일 질의 위주 테스트 | `cli/run_batch_compare.py` 통합 (JSONL 입력 → 모델별 예측 JSONL 출력) |
| **링킹 평가 지표 부재** | CUI 링킹 성능 미측정 | `src/med_entity_ab/metrics/linking_metrics.py` 통합 (Accuracy@k, MRR) |

---

## 3. 병합 전략: 논리적 구조 설계

### 3.1 아키텍처 레이어 분리

```
┌─────────────────────────────────────────────────────────────┐
│                    Agent Layer (기존 유지)                    │
│  - agents/medical_agent.py                                   │
│  - 멀티턴 대화, 메모리 관리, 도구 호출                          │
└─────────────────┬───────────────────────────────────────────┘
                  │ extract_for_agent(text)
┌─────────────────▼───────────────────────────────────────────┐
│          Entity Extraction Router (신규 추가)                │
│  - extraction/entity_ab_router.py                           │
│  - 환경 변수로 extractor 선택 (medcat/quickumls/kmbert_ner) │
│  - 캐싱, 메타데이터 (latency) 제공                            │
└─────────────────┬───────────────────────────────────────────┘
                  │
    ┌─────────────┼─────────────┐
    │             │             │
┌───▼───┐   ┌────▼────┐   ┌────▼────┐
│MedCAT │   │QuickUMLS│   │KM-BERT  │
│       │   │         │   │NER      │
└───────┘   └─────────┘   └─────────┘
```

### 3.2 파일 배치 전략

#### A. 신규 추가 파일 (addon에서 가져옴)

| 파일 | 위치 | 역할 |
|------|------|------|
| `entity_ab_router.py` | `extraction/` | **핵심**: Agent와 extractor 간 인터페이스 |
| `train_kmbert_ner.py` | `scripts/` | KM-BERT NER fine-tuning (CoNLL 형식) |
| `ENTITY_AB_INTEGRATION_STRATEGY.md` | 루트 | 본 문서 (병합 전략 설명) |

#### B. 기존 파일 유지 (이미 통합됨)

| 파일 | 위치 | 상태 |
|------|------|------|
| `src/med_entity_ab/` | 전체 | ✓ 이미 통합 완료 (이전 병합) |
| `cli/run_compare.py` | `cli/` | ✓ 이미 통합 완료 |
| `cli/run_batch_compare.py` | `cli/` | ✓ 이미 통합 완료 |
| `cli/evaluate_from_gold.py` | `cli/` | ✓ 이미 통합 완료 |

#### C. 중복 파일 처리

| 파일 | 이전 addon | 현재 스캐폴드 | 처리 방안 |
|------|-----------|--------------|----------|
| `requirements.txt` | 기본 의존성 | 더 포괄적 | **현재 유지** (이미 병합됨) |
| `README.md` | 엔티티 비교 위주 | 전체 시스템 설명 | **현재 유지** (엔티티 섹션 이미 추가됨) |
| `configs/default.yaml` | 엔티티 설정만 | - | **현재 유지** (이미 존재) |

---

## 4. 핵심 개선사항: `entity_ab_router.py`

### 4.1 설계 철학

**문제**: Agent 코드가 개별 extractor를 직접 import하면, A/B 테스트 시 코드 수정 필요  
**해결**: 라우터 패턴으로 추상화 → 환경 변수로 교체

### 4.2 주요 기능

```python
from extraction.entity_ab_router import extract_for_agent

# Agent 코드에서 사용
entities = extract_for_agent(user_text)  # 환경 변수 ENTITY_EXTRACTOR 자동 참조
```

#### 기능 1: 단일 extractor 사용
```python
from extraction.entity_ab_router import extract_entities

entities = extract_entities(text, extractor="medcat")
# → List[Entity]
```

#### 기능 2: 모든 extractor 동시 실행 (비교 모드)
```python
from extraction.entity_ab_router import extract_entities_all

all_results = extract_entities_all(text)
# → {"medcat": [Entity, ...], "quickumls": [Entity, ...], "kmbert_ner": [Entity, ...]}
```

#### 기능 3: 메타데이터 포함 (latency 측정)
```python
from extraction.entity_ab_router import extract_entities_with_metadata

result = extract_entities_with_metadata(text, extractor="medcat")
# → {"entities": [...], "latency_ms": 123.45, "extractor": "medcat", "num_entities": 5}
```

### 4.3 환경 변수 기반 A/B 테스트

**.env 파일 설정**:
```bash
# 실험 1: MedCAT
ENTITY_EXTRACTOR=medcat

# 실험 2: QuickUMLS
ENTITY_EXTRACTOR=quickumls

# 실험 3: KM-BERT NER
ENTITY_EXTRACTOR=kmbert_ner
```

**Agent 코드 수정 불필요**:
```python
# agents/medical_agent.py (변경 없음)
from extraction.entity_ab_router import extract_for_agent

entities = extract_for_agent(user_text)  # .env 파일만 바꾸면 extractor 교체
```

---

## 5. KM-BERT NER 학습 파이프라인

### 5.1 기존 문제점

- 현재 스캐폴드: KM-BERT 사전학습 모델만 언급, NER fine-tuning 스크립트 부재
- 새 addon: `scripts/train_kmbert_ner.py` 제공 (CoNLL 형식 지원)

### 5.2 개선된 학습 파이프라인

#### Step 1: 데이터 준비 (CoNLL 형식)
```
data/kbmc/train.conll:
당뇨	B-DISEASE
병	I-DISEASE
환자	O
입니다	O

(빈 줄로 문장 구분)
```

#### Step 2: Fine-tuning 실행
```bash
python scripts/train_kmbert_ner.py \
  --base_model madatnlp/km-bert \
  --train_file data/kbmc/train.conll \
  --valid_file data/kbmc/valid.conll \
  --test_file data/kbmc/test.conll \
  --output_dir models/kmbert_ner_kbmc \
  --epochs 10 \
  --batch_size 16
```

#### Step 3: 환경 변수 설정
```bash
# .env
KMBERT_NER_DIR=models/kmbert_ner_kbmc
```

#### Step 4: 즉시 사용
```bash
python cli/run_compare.py --text "당뇨병 환자입니다"
```

### 5.3 개선사항 (이전 스크립트 대비)

| 항목 | 이전 | 개선 |
|------|------|------|
| 데이터 형식 | 특정 데이터셋 의존 | **Dataset-agnostic** (CoNLL 범용) |
| 평가 지표 | 간단한 accuracy | **seqeval** 기반 entity-level F1 |
| 서브워드 처리 | 불명확 | 첫 서브워드만 라벨, 나머지 -100 |
| 테스트 세트 | 미지원 | `--test_file` 옵션 추가 |
| 출력 | 체크포인트만 | 테스트 결과 `test_metrics.txt` 저장 |

---

## 6. 평가 지표 강화

### 6.1 NER 평가 지표 (새로 추가)

**위치**: `src/med_entity_ab/metrics/ner_metrics.py`

| 지표 | 설명 | 사용 시점 |
|------|------|----------|
| **Strict Match F1** | (start, end, label) 완전 일치 | 정확한 경계 평가 |
| **Overlap F1** | span이 일부라도 겹치면 정답 | 완화된 평가 |
| **Boundary IoU** | 경계 품질 (IoU ≥ τ) | 경계 정확도 분석 |
| **Per-type F1** | 엔티티 타입별 F1 (DISEASE, DRUG 등) | 클래스 불균형 분석 |

**사용 예시**:
```bash
python cli/evaluate_from_gold.py \
  --gold_jsonl data/gold/gold.jsonl \
  --pred_jsonl outputs/run1/pred_kmbert_ner.jsonl \
  --mode strict  # or overlap
```

### 6.2 링킹 평가 지표 (새로 추가)

**위치**: `src/med_entity_ab/metrics/linking_metrics.py`

| 지표 | 설명 | 사용 시점 |
|------|------|----------|
| **Accuracy@1** | Top-1 CUI 정확도 | 단일 후보 평가 |
| **Accuracy@k** | Top-k 내 정답 포함 여부 | 다중 후보 평가 |
| **MRR (Mean Reciprocal Rank)** | 정답 순위의 역수 평균 | 순위 품질 평가 |
| **Normalization Rate** | CUI 부여 성공률 | 링킹 커버리지 |

**사용 예시**:
```bash
python cli/evaluate_from_gold.py \
  --gold_jsonl data/gold/gold.jsonl \
  --pred_jsonl outputs/run1/pred_quickumls.jsonl \
  --mode overlap \
  --linking \
  --k 5
```

### 6.3 Gold 없이 평가 (초기 탐색용)

**위치**: `src/med_entity_ab/metrics/agreement.py`

| 지표 | 설명 | 사용 시점 |
|------|------|----------|
| **Jaccard Similarity** | 엔티티 문자열 집합 유사도 | 모델 간 합의도 |
| **Span Overlap Ratio** | span 겹침 비율 | 경계 일치도 |
| **Stability** | 동일 질의 변형에서 엔티티 유지율 | 로버스트니스 평가 |

---

## 7. Agent 통합 시나리오

### 7.1 시나리오 1: 기본 통합 (최소 변경)

**목표**: Agent가 엔티티 추출을 사용하되, 코드 수정 최소화

**변경 사항**:
```python
# agents/medical_agent.py (기존)
from extraction.medcat2_adapter import extract_entities_medcat

entities = extract_entities_medcat(user_text)

# agents/medical_agent.py (개선)
from extraction.entity_ab_router import extract_for_agent

entities = extract_for_agent(user_text)  # 환경 변수로 제어
```

**장점**:
- 코드 1줄 변경으로 A/B 테스트 가능
- 기존 Agent 로직 영향 없음

---

### 7.2 시나리오 2: 메모리 인덱싱 강화

**목표**: 추출된 엔티티(CUI)를 메모리 키로 사용 → 관련 대화 검색 정확도 향상

**구현**:
```python
# agents/medical_agent.py
from extraction.entity_ab_router import extract_entities_with_metadata

# 엔티티 추출 + latency 측정
result = extract_entities_with_metadata(user_text, extractor="medcat")
entities = result['entities']

# 엔티티를 메모리 키로 사용
cui_keys = [ent.code for ent in entities if ent.code]
memory.index_by_cuis(turn_id, cui_keys)  # 메모리에 CUI 인덱스 추가

# 다음 턴에서 관련 대화 검색
related_turns = memory.search_by_cuis(cui_keys)
```

**장점**:
- 의료 개념 기반 대화 검색 (키워드 검색보다 정확)
- 멀티턴 일관성 향상 (동일 질병/약물 언급 시 이전 대화 참조)

---

### 7.3 시나리오 3: RAG 쿼리 강화

**목표**: 엔티티 추출 결과를 RAG 검색 쿼리로 활용 → 검색 정확도 향상

**구현**:
```python
# agents/medical_agent.py
from extraction.entity_ab_router import extract_entities

# 원본 질의
user_text = "당뇨병 환자인데 아스피린 먹어도 되나요?"

# 엔티티 추출
entities = extract_entities(user_text, extractor="medcat")

# 엔티티 기반 쿼리 확장
entity_terms = [ent.text for ent in entities]
expanded_query = f"{user_text} {' '.join(entity_terms)}"

# RAG 검색
contexts = retriever.search(expanded_query)
```

**장점**:
- 의료 용어 정규화 (예: "당뇨" → "diabetes mellitus")
- 검색 재현율 향상 (동의어/약어 처리)

---

### 7.4 시나리오 4: RAGAS 평가 연동

**목표**: 엔티티 추출 방법이 RAG 성능에 미치는 영향 정량화

**실험 설계**:
```bash
# 실험 1: MedCAT 기반 RAG
ENTITY_EXTRACTOR=medcat python experiments/run_llm_vs_rag_comparison.py \
  --output_dir outputs/rag_medcat

# 실험 2: QuickUMLS 기반 RAG
ENTITY_EXTRACTOR=quickumls python experiments/run_llm_vs_rag_comparison.py \
  --output_dir outputs/rag_quickumls

# 실험 3: KM-BERT NER 기반 RAG
ENTITY_EXTRACTOR=kmbert_ner python experiments/run_llm_vs_rag_comparison.py \
  --output_dir outputs/rag_kmbert_ner

# RAGAS 평가
python experiments/evaluate_llm_vs_rag.py \
  --log_dirs outputs/rag_medcat outputs/rag_quickumls outputs/rag_kmbert_ner
```

**평가 지표**:
| 지표 | 의미 | 예상 결과 |
|------|------|----------|
| **Faithfulness** | 응답이 검색 컨텍스트로 지지되는가 | 엔티티 정확도 ↑ → Faithfulness ↑ |
| **Answer Relevancy** | 질문-응답 의도 정합 | 엔티티 기반 쿼리 → Relevancy ↑ |
| **Context Precision** | 검색 컨텍스트 품질 | 정규화된 엔티티 → Precision ↑ |

**장점**:
- 엔티티 추출 방법의 downstream 영향 정량화
- 논문 기여: "엔티티 추출 품질이 RAG 성능에 미치는 영향 분석"

---

## 8. 실험 자동화 파이프라인

### 8.1 배치 비교 실험

**목표**: 대량의 질의에 대해 3자 비교 자동화

**입력**: `data/queries.jsonl`
```json
{"id": "q001", "text": "당뇨병 환자인데 아스피린 먹어도 되나요?"}
{"id": "q002", "text": "혈당이 250인데 응급실 가야 하나요?"}
```

**실행**:
```bash
python cli/run_batch_compare.py \
  --input data/queries.jsonl \
  --out_dir outputs/entity_ab_001
```

**출력**:
```
outputs/entity_ab_001/
  ├── pred_medcat.jsonl        # MedCAT 예측
  ├── pred_quickumls.jsonl     # QuickUMLS 예측
  ├── pred_kmbert_ner.jsonl    # KM-BERT NER 예측
  └── latency_log.csv          # 모델별 latency
```

---

### 8.2 Gold 기반 정량 평가

**목표**: Gold 라벨과 비교하여 NER/링킹 정확도 측정

**Gold 준비**: `data/gold/gold.jsonl`
```json
{
  "id": "q001",
  "text": "당뇨병 환자인데 아스피린 먹어도 되나요?",
  "entities": [
    {"start": 0, "end": 3, "text": "당뇨병", "label": "DISEASE", "code": "C0011849"},
    {"start": 11, "end": 15, "text": "아스피린", "label": "DRUG", "code": "C0004057"}
  ]
}
```

**평가**:
```bash
# NER 평가 (strict match)
python cli/evaluate_from_gold.py \
  --gold_jsonl data/gold/gold.jsonl \
  --pred_jsonl outputs/entity_ab_001/pred_kmbert_ner.jsonl \
  --mode strict

# NER 평가 (overlap)
python cli/evaluate_from_gold.py \
  --gold_jsonl data/gold/gold.jsonl \
  --pred_jsonl outputs/entity_ab_001/pred_kmbert_ner.jsonl \
  --mode overlap

# 링킹 평가 (CUI)
python cli/evaluate_from_gold.py \
  --gold_jsonl data/gold/gold.jsonl \
  --pred_jsonl outputs/entity_ab_001/pred_quickumls.jsonl \
  --mode overlap \
  --linking \
  --k 5
```

**출력 예시**:
```
=== NER Evaluation (strict) ===
Precision: 0.8523
Recall:    0.7891
F1:        0.8195

Per-type F1:
  DISEASE: 0.8912
  DRUG:    0.7823
  SYMPTOM: 0.7456

=== Linking Evaluation ===
Accuracy@1: 0.7234
Accuracy@5: 0.8912
MRR:        0.7891
Normalization Rate: 0.9123
```

---

## 9. 한국어 의료 도메인 특화 전략

### 9.1 현재 스캐폴드의 강점 활용

**기존 구조**:
```
한국어 질의 → Helsinki-NLP 번역 → 영어 → MedCAT → UMLS CUI
```

**문제점**:
- 번역 오류 누적
- 한국어 의료 용어 번역 품질 불안정

---

### 9.2 개선된 하이브리드 전략

**전략 1: 추출(Span)과 링킹(CUI) 분리**

```
한국어 질의
  ↓
KM-BERT NER (한국어 직접 추출)
  ↓
Span: "당뇨병", "아스피린"
  ↓
번역 (span만)
  ↓
"diabetes mellitus", "aspirin"
  ↓
QuickUMLS/MedCAT (링킹만)
  ↓
CUI: C0011849, C0004057
```

**장점**:
- 한국어 NER 정확도 향상 (KM-BERT는 한국어 의료 코퍼스로 사전학습)
- 번역 부담 감소 (전체 문장이 아닌 span만 번역)
- 링킹 정확도 유지 (UMLS는 영어 기반)

---

**전략 2: 앙상블**

```python
# extraction/hybrid_extractor.py (신규 제안)
from extraction.entity_ab_router import extract_entities

# 한국어 직접 추출
entities_kr = extract_entities(text, extractor="kmbert_ner")

# 번역 기반 추출
text_en = translate(text)
entities_en = extract_entities(text_en, extractor="medcat")

# 앙상블 (span overlap + CUI 투표)
entities_final = ensemble(entities_kr, entities_en)
```

**장점**:
- 두 방법의 강점 결합
- 신뢰도 높은 엔티티만 선택 (양쪽 모두 검출된 경우)

---

### 9.3 실험 설계

**비교 대상**:
1. **Baseline**: 번역 → MedCAT
2. **KM-BERT Only**: KM-BERT NER (링킹 없음)
3. **Hybrid**: KM-BERT (span) + 번역 + QuickUMLS (링킹)
4. **Ensemble**: KM-BERT + (번역 → MedCAT) 앙상블

**평가 지표**:
- NER F1 (한국어 gold 필요)
- 링킹 Accuracy@1 (CUI gold 필요)
- End-to-end RAG 성능 (RAGAS)

---

## 10. 구현 타임라인

### Phase 1: 핵심 통합 (완료)
- [x] `entity_ab_router.py` 추가
- [x] `train_kmbert_ner.py` 개선
- [x] 본 문서 작성

### Phase 2: Agent 연동 (1일)
- [ ] `agents/medical_agent.py`에 `extract_for_agent()` 적용
- [ ] 환경 변수 기반 A/B 테스트 검증
- [ ] 메모리 인덱싱에 CUI 활용

### Phase 3: 평가 파이프라인 (2일)
- [ ] Gold 데이터 준비 (최소 50개 질의)
- [ ] NER/링킹 평가 실행
- [ ] RAGAS 연동 실험 (엔티티 추출 방법별 RAG 성능 비교)

### Phase 4: 한국어 특화 (3일)
- [ ] KM-BERT NER fine-tuning (KBMC 데이터셋)
- [ ] 하이브리드 전략 구현
- [ ] 앙상블 전략 구현 및 평가

### Phase 5: 문서화 및 재현성 (1일)
- [ ] 실험 결과 정리 (`ENTITY_AB_RESULTS.md`)
- [ ] 환경 설정 가이드 업데이트 (`.env.example`)
- [ ] README 업데이트 (Quick Start 섹션)

---

## 11. 환경 변수 설정 가이드

### 11.1 필수 환경 변수

**.env 파일**:
```bash
# ========================================
# Entity Extraction Configuration
# ========================================

# 사용할 extractor 선택 (medcat|quickumls|kmbert_ner)
ENTITY_EXTRACTOR=medcat

# MedCAT 모델팩 경로
MEDCAT_MODELPACK=medcat2/mc_modelpack_snomed_int_16_mar_2022_25be3857ba34bdd5

# QuickUMLS 인덱스 경로 (빌드 필요)
QUICKUMLS_INDEX_DIR=data/quickumls_index

# KM-BERT NER 체크포인트 경로 (fine-tuning 후)
KMBERT_NER_DIR=models/kmbert_ner_kbmc

# Entity AB 설정 파일 (선택)
ENTITY_AB_CONFIG=configs/default.yaml

# ========================================
# 기존 Agent/RAG 설정 (유지)
# ========================================
OPENAI_API_KEY=sk-...
PINECONE_API_KEY=...
# ...
```

### 11.2 A/B 테스트 예시

**실험 1: MedCAT**
```bash
# .env
ENTITY_EXTRACTOR=medcat

# 실행
python agents/medical_agent.py --query "당뇨병 환자입니다"
```

**실험 2: QuickUMLS**
```bash
# .env
ENTITY_EXTRACTOR=quickumls

# 실행 (코드 수정 없음)
python agents/medical_agent.py --query "당뇨병 환자입니다"
```

**실험 3: KM-BERT NER**
```bash
# .env
ENTITY_EXTRACTOR=kmbert_ner

# 실행 (코드 수정 없음)
python agents/medical_agent.py --query "당뇨병 환자입니다"
```

---

## 12. 기대 효과 및 학술적 기여

### 12.1 시스템 개선 효과

| 항목 | 개선 전 | 개선 후 |
|------|---------|---------|
| **엔티티 추출 A/B 테스트** | 수동 코드 수정 | 환경 변수 1줄 변경 |
| **NER 평가** | 미측정 | Strict/Overlap F1, Boundary IoU |
| **링킹 평가** | 미측정 | Accuracy@k, MRR |
| **한국어 NER** | 번역 의존 | KM-BERT 직접 추출 |
| **배치 실험** | 단일 질의 위주 | JSONL 입력 자동화 |

### 12.2 학술적 기여 포인트

1. **재현 가능한 비교 실험 프레임워크**
   - MedCAT, QuickUMLS, KM-BERT NER 3자 비교
   - 환경 변수 기반 교체 → 코드 수정 없는 ablation study

2. **엔티티 추출 품질의 downstream 영향 정량화**
   - 엔티티 추출 방법 → RAG 검색 정확도 → RAGAS 점수
   - "엔티티 추출 정확도 10% 향상 → Faithfulness 5% 향상" 같은 인과 분석

3. **한국어 의료 도메인 특화 전략**
   - KM-BERT (한국어 직접) vs 번역 기반 비교
   - 하이브리드 전략 (KM-BERT span + UMLS 링킹)

4. **멀티턴 대화에서 엔티티 기반 메모리 인덱싱**
   - CUI 기반 대화 검색 → 멀티턴 일관성 향상
   - "이전에 당뇨병 언급했던 대화" 자동 검색

---

## 13. 결론

### 13.1 병합 완료 항목

✅ **핵심 파일 추가**:
- `extraction/entity_ab_router.py` (Agent 통합 레이어)
- `scripts/train_kmbert_ner.py` (KM-BERT NER fine-tuning)
- `ENTITY_AB_INTEGRATION_STRATEGY.md` (본 문서)

✅ **기존 강점 유지**:
- RAGAS 평가 시스템
- Agentic RAG + 3-Tier Memory
- 멀티턴 대화 로그 생성

✅ **단점 보완**:
- 엔티티 추출 A/B 테스트 자동화
- NER/링킹 평가 지표 추가
- 한국어 직접 추출 (KM-BERT NER)

### 13.2 다음 단계

1. **Agent 연동**: `agents/medical_agent.py`에 `extract_for_agent()` 적용
2. **Gold 데이터 준비**: 최소 50개 질의 + 엔티티 라벨
3. **RAGAS 연동 실험**: 엔티티 추출 방법별 RAG 성능 비교
4. **한국어 특화**: KM-BERT NER fine-tuning + 하이브리드 전략

### 13.3 핵심 메시지

> **"엔티티 추출을 교체 가능한 부품으로 설계하여, 코드 수정 없이 A/B 테스트 가능한 재현 가능한 실험 프레임워크 구축"**

이를 통해:
- **연구 생산성**: ablation study 자동화
- **재현성**: 환경 변수 기반 실험 설정
- **학술 기여**: 엔티티 추출 품질의 downstream 영향 정량화

---

**문서 끝**

