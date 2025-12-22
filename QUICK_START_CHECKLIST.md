# 🚀 빠른 시작 체크리스트

**목적**: Agentic RAG 고도화 실험을 5분 안에 시작하기

---

## ✅ 사전 준비 (5분)

### 1. 환경 변수 확인

```bash
# .env 파일에 OpenAI API 키가 있는지 확인
cat .env | grep OPENAI_API_KEY
```

**없으면**:
```bash
echo "OPENAI_API_KEY=sk-..." > .env
```

### 2. 의존성 설치 확인

```bash
# RAGAS 설치 확인
python -c "import ragas; print(ragas.__version__)"

# 없으면 설치
pip install ragas datasets langchain-openai
```

### 3. 코퍼스 데이터 확인

```bash
# 코퍼스 파일이 있는지 확인
ls data/corpus/*.txt

# 없으면 샘플 생성 (테스트용)
mkdir -p data/corpus
echo "메트포르민은 당뇨병 치료제입니다. 주요 부작용은 소화불량, 설사 등이 있습니다." > data/corpus/sample.txt
```

---

## 🎯 실험 1: RAG 변형 비교 (10분)

### 실행

```bash
# Windows
python experiments\run_rag_variants_comparison.py --patient-id P001 --turns 5

# Linux/Mac
python experiments/run_rag_variants_comparison.py --patient-id P001 --turns 5
```

### 예상 출력

```
==========================================
RAG 시스템 간 비교 실험 (피드백 반영)
==========================================
환자 시나리오: 당뇨병 환자 (메트포르민 복용)
대화 턴 수: 5
비교 변형: basic_rag, modular_rag, corrective_rag
==========================================

[basic_rag] Basic RAG: 단순 검색-생성
  Turn 1/5: 당뇨병 환자인데 메트포르민을 복용하고 있어요...
    ✓ Q=0.650, Iter=0, Docs=8, Time=2.3s
  ...

[modular_rag] Modular RAG: LLM 품질 평가 + Self-Refine
  Turn 1/5: 당뇨병 환자인데 메트포르민을 복용하고 있어요...
    ✓ Q=0.720, Iter=1, Docs=8, Time=4.1s
  ...

[corrective_rag] Corrective RAG (Agentic)
  Turn 1/5: 당뇨병 환자인데 메트포르민을 복용하고 있어요...
    ✓ Q=0.850, Iter=2, Docs=12, Time=6.5s
  ...
```

### 결과 확인

```bash
# 로그 파일 확인
ls runs/rag_variants_comparison/comparison_P001_*.json

# 간단히 보기
cat runs/rag_variants_comparison/comparison_P001_*.json | grep "avg_quality"
```

---

## 📊 실험 2: RAGAS 평가 (5분)

### 실행

```bash
# 최신 비교 결과 파일 찾기
ls -t runs/rag_variants_comparison/comparison_P001_*.json | head -1

# RAGAS 평가 실행
python experiments/evaluate_rag_variants.py runs/rag_variants_comparison/comparison_P001_20251216_143022.json
```

### 예상 출력

```
==========================================
RAG 변형 RAGAS 평가 (LLM as a Judge)
==========================================

[RAGAS 평가] basic_rag
  Turn 1: 당뇨병 환자인데 메트포르민을 복용하고 있어요...
    ✓ Faithfulness=0.720, Relevancy=0.680, Precision=0.650
  ...

[RAGAS 평가] modular_rag
  Turn 1: 당뇨병 환자인데 메트포르민을 복용하고 있어요...
    ✓ Faithfulness=0.780, Relevancy=0.730, Precision=0.720
  ...

==========================================
RAGAS 메트릭 비교 테이블
==========================================
변형                 Faithfulness   Relevancy      Precision
--------------------------------------------------------------------------------
basic_rag            0.720±0.080    0.680±0.100    0.650±0.120
modular_rag          0.780±0.070    0.730±0.090    0.720±0.100
corrective_rag       0.840±0.060    0.760±0.080    0.780±0.090
==========================================
```

### 결과 확인

```bash
# CSV 요약 확인 (엑셀/구글 시트로 열기)
cat runs/rag_variants_comparison/ragas_evaluation/ragas_summary_P001_*.csv
```

---

## 🔬 실험 3: 고도화 프로파일 테스트 (5분)

### 슬롯 기반 메모리 테스트

```bash
python experiments/run_ablation_single.py \
    --profile personalized_slot_memory \
    --query "당뇨병 환자인데 메트포르민을 복용하고 있어요"
```

### 최종 고도화 테스트

```bash
python experiments/run_ablation_single.py \
    --profile advanced_personalized_rag \
    --query "가슴이 아파요"
```

---

## 🎉 자동 실행 (모든 실험 한 번에)

### Windows

```cmd
run_enhancement_experiments.bat
```

### Linux/Mac

```bash
bash run_enhancement_experiments.sh
```

**실행 내용**:
1. 환자 시나리오 3개 (P001, P002, P003) × 5턴 비교
2. RAGAS 평가 자동 실행
3. CSV 요약 자동 생성

**소요 시간**: 약 15~20분

---

## 📋 결과 확인 체크리스트

### 1. 비교 로그 생성 확인

```bash
# 3개 환자 × 1개 파일 = 3개 파일
ls runs/rag_variants_comparison/comparison_*.json | wc -l
# 예상 출력: 3
```

### 2. RAGAS 평가 결과 확인

```bash
# 3개 환자 × 1개 JSON + 1개 CSV = 6개 파일
ls runs/rag_variants_comparison/ragas_evaluation/* | wc -l
# 예상 출력: 6
```

### 3. 메트릭 확인

```bash
# Faithfulness 평균 확인
cat runs/rag_variants_comparison/ragas_evaluation/ragas_summary_*.csv | grep "corrective_rag"
# 예상 출력: corrective_rag,0.8400,0.0600,0.7600,0.0800,0.7800,0.0900
```

---

## 🐛 문제 해결

### 오류 1: `ModuleNotFoundError: No module named 'ragas'`

**해결**:
```bash
pip install ragas datasets langchain-openai
```

### 오류 2: `OPENAI_API_KEY가 설정되지 않았습니다`

**해결**:
```bash
echo "OPENAI_API_KEY=sk-your-key-here" > .env
```

### 오류 3: `FileNotFoundError: corpus 파일이 없습니다`

**해결**:
```bash
# 샘플 코퍼스 생성
mkdir -p data/corpus
echo "메트포르민은 당뇨병 치료제입니다." > data/corpus/sample.txt
```

### 오류 4: 실험이 너무 느림

**해결**:
```bash
# 턴 수 줄이기
python experiments/run_rag_variants_comparison.py --patient-id P001 --turns 2

# 변형 수 줄이기
python experiments/run_rag_variants_comparison.py --patient-id P001 --variants basic_rag corrective_rag
```

---

## 📊 다음 단계

### 1. 결과 분석

```bash
# JSON 파일 열어서 통계적 유의성 확인
cat runs/rag_variants_comparison/ragas_evaluation/ragas_P001_*.json | grep "p_value"
```

### 2. 논문/보고서 작성

- CSV 파일을 엑셀/구글 시트로 열어 테이블 작성
- 통계적 유의성 (p-value < 0.05) 확인
- 효과 크기 (Cohen's d > 0.5) 확인

### 3. 추가 실험

```bash
# 더 많은 환자 시나리오 추가
python experiments/run_rag_variants_comparison.py --patient-id P004 --turns 10

# 고도화 프로파일 비교
python experiments/run_ablation_comparison.py \
    --profiles baseline full_context_engineering advanced_personalized_rag
```

---

## 📚 참고 문서

- **고도화 가이드**: `PERSONALIZED_RAG_ENHANCEMENT_GUIDE.md`
- **구현 요약**: `ENHANCEMENT_IMPLEMENTATION_SUMMARY.md`
- **Ablation Study**: `ABLATION_STUDY_GUIDE.md`
- **RAGAS 통합**: `RAGAS_INTEGRATION_COMPLETE.md`

---

## ✅ 완료 체크리스트

- [ ] 환경 변수 설정 (.env)
- [ ] 의존성 설치 (ragas, datasets, langchain-openai)
- [ ] 코퍼스 데이터 준비
- [ ] RAG 변형 비교 실험 실행
- [ ] RAGAS 평가 실행
- [ ] 결과 파일 확인 (JSON, CSV)
- [ ] 고도화 프로파일 테스트
- [ ] 결과 분석 및 논문 작성

---

**작성자**: AI Assistant  
**최종 수정**: 2025-12-16  
**소요 시간**: 전체 약 25~30분

