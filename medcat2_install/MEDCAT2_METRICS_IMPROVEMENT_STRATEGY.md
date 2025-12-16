# MedCAT2 통합을 통한 5가지 평가 지표 개선 전략

## [클립보드] 개요

[MedCAT2 리포지토리](https://github.com/CogStack/cogstack-nlp/tree/main/medcat-v2)와 [MedCAT2 튜토리얼](https://github.com/CogStack/cogstack-nlp/tree/main/medcat-v2-tutorials)을 분석하여, 스캐폴드의 5가지 평가 지표를 개선하는 전략을 수립했습니다.

## [목표] 5가지 평가 지표

1. **Inference Memory**: Rubric 기반 평가 (must_mention, must_avoid)
2. **Slot F1**: 답변에서 엔티티 추출 정확도
3. **CMR (Contraindication Miss Rate)**: 금기 약물 언급 누락률
4. **Context Retention**: 멀티턴 대화에서 엔티티 유지
5. **Personalization Delta (ΔP)**: Hybrid vs RAG 개인화 효과

## [수정] MedCAT2 통합 개선 전략

### 1. Inference Memory 개선

**현재 문제**:
- 단순 문자열 매칭만 사용
- Hybrid와 RAG가 동일한 점수
- 동의어/유사 표현 인식 불가

**MedCAT2 통합 개선**:
- **RubricEvaluator에 MEDCAT2 통합**
  - 답변에서 MEDCAT2로 엔티티 추출
  - must_mention 항목과 추출된 엔티티 비교
  - CUI 기반 의미 매칭 (동의어 자동 처리)
  - 신뢰도 기반 점수 계산

**예상 개선**:
- Inference Memory: 0.683 -> 0.80+ (약 17% 향상)
- Hybrid vs RAG 차별화 가능

### 2. Slot F1 개선

**현재 문제**:
- 답변에서 엔티티 추출 미구현 (100% 0점)
- 키워드 기반 추출만 사용

**MedCAT2 통합 개선**:
- **SlotExtractorEvaluator에 MEDCAT2 통합**
  - 답변 텍스트에서 MEDCAT2로 엔티티 추출
  - Ground Truth와 비교하여 F1 계산
  - CUI 기반 정확한 매칭

**예상 개선**:
- Slot F1: 0.000 -> 0.60+ (완전 개선)

### 3. CMR (Contraindication Miss Rate) 개선

**현재 문제**:
- must_avoid가 비어있거나 평가 미작동
- 키워드 기반 매칭만 사용

**MedCAT2 통합 개선**:
- **ContraindicationEvaluator에 MEDCAT2 통합**
  - 금기 약물을 MEDCAT2로 추출
  - CUI 기반 정확한 약물 매칭
  - 약물 상호작용 정보 활용

**예상 개선**:
- CMR: 0.000 -> 0.30+ (실제 평가 가능)

### 4. Context Retention 개선

**현재 문제**:
- Multi-turn 시나리오 미구현 (100% 0점)
- 단일 턴 평가만 수행

**MedCAT2 통합 개선**:
- **ContextRetentionEvaluator에 MEDCAT2 통합**
  - Turn 1에서 추출된 엔티티를 CUI로 저장
  - Turn 2 답변에서 동일 CUI 엔티티 확인
  - 엔티티 유지율 계산

**예상 개선**:
- Context Retention: 0.000 -> 0.70+ (완전 개선)

### 5. Personalization Delta (ΔP) 개선

**현재 문제**:
- Hybrid와 RAG의 Inference Memory가 동일하여 Delta P = 0
- 개인화 효과 측정 불가

**MedCAT2 통합 개선**:
- **MEDCAT2로 개인화 정보 활용도 측정**
  - Hybrid 답변에서 개인 정보 관련 엔티티 추출
  - RAG 답변과 비교하여 차이 계산
  - CUI 기반 정확한 비교

**예상 개선**:
- Delta P: 0.000 -> 0.20+ (개인화 효과 측정 가능)

## [차트] 구현 내용

### 1. SlotsExtractor 기본 MEDCAT2 사용

**파일**: `agent/nodes/slots_extract.py`

**변경 사항**:
```python
# 기존: use_medcat2: bool = False
# 개선: use_medcat2: bool = True (기본값)
```

### 2. RubricEvaluator에 MEDCAT2 통합

**파일**: `scripts/eval_5_metrics_3way.py`

**변경 사항**:
- MEDCAT2 어댑터 초기화
- `_check_mention_with_synonyms()`에서 MEDCAT2 엔티티 추출 활용
- CUI 기반 의미 매칭

### 3. SlotExtractorEvaluator에 MEDCAT2 통합

**파일**: `scripts/eval_5_metrics_3way.py`

**변경 사항**:
- `extract_slots_from_answer()`에서 MEDCAT2 사용
- 답변 텍스트에서 엔티티 추출

### 4. ContextRetentionEvaluator에 MEDCAT2 통합

**파일**: `scripts/eval_5_metrics_3way.py`

**변경 사항**:
- MEDCAT2 사용으로 엔티티 추출
- CUI 기반 엔티티 유지 추적

## [실행] 예상 성능 개선

| 지표 | 현재 (평균) | 목표 (MEDCAT2 통합) | 개선율 |
|------|------------|---------------------|--------|
| **Inference Memory** | 0.683 | 0.80+ | +17% |
| **Slot F1** | 0.000 | 0.60+ | 완전 개선 |
| **CMR** | 0.000 | 0.30+ | 평가 가능 |
| **Context Retention** | 0.000 | 0.70+ | 완전 개선 |
| **Delta P** | 0.000 | 0.20+ | 측정 가능 |

## [메모] 다음 단계

1. **테스트 실행**: 실제 평가 데이터로 성능 측정
2. **성능 비교**: MEDCAT2 통합 전후 비교
3. **하이퍼파라미터 조정**: 최적 성능을 위한 튜닝
4. **문서화**: 개선 결과 문서화

