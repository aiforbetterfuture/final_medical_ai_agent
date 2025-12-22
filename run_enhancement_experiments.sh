#!/bin/bash
# Agentic RAG 고도화 실험 자동 실행 스크립트
# 사용법: bash run_enhancement_experiments.sh

set -e  # 오류 시 중단

echo "=========================================="
echo "Agentic RAG 고도화 실험 자동 실행"
echo "=========================================="
echo ""

# 색상 정의
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ============================================================
# Phase 1: RAG 변형 비교 (피드백 반영)
# ============================================================

echo -e "${BLUE}[Phase 1] RAG 변형 비교 실험${NC}"
echo ""

# 환자 시나리오 3개 실행
for PATIENT in P001 P002 P003; do
    echo -e "${GREEN}[실행] 환자 시나리오: $PATIENT${NC}"
    
    python experiments/run_rag_variants_comparison.py \
        --patient-id $PATIENT \
        --turns 5 \
        --variants basic_rag modular_rag corrective_rag
    
    echo ""
done

echo -e "${GREEN}✓ Phase 1 완료: RAG 변형 비교 로그 생성${NC}"
echo ""

# ============================================================
# Phase 2: RAGAS 평가
# ============================================================

echo -e "${BLUE}[Phase 2] RAGAS 평가 (LLM as a Judge)${NC}"
echo ""

# 최신 비교 결과 파일 찾기
COMPARISON_DIR="runs/rag_variants_comparison"

if [ ! -d "$COMPARISON_DIR" ]; then
    echo -e "${YELLOW}⚠ 경고: 비교 결과 디렉토리가 없습니다: $COMPARISON_DIR${NC}"
    exit 1
fi

# 각 환자별 최신 파일 평가
for PATIENT in P001 P002 P003; do
    LATEST_FILE=$(ls -t $COMPARISON_DIR/comparison_${PATIENT}_*.json 2>/dev/null | head -1)
    
    if [ -z "$LATEST_FILE" ]; then
        echo -e "${YELLOW}⚠ 경고: $PATIENT 비교 결과 파일이 없습니다${NC}"
        continue
    fi
    
    echo -e "${GREEN}[평가] $PATIENT: $LATEST_FILE${NC}"
    
    python experiments/evaluate_rag_variants.py "$LATEST_FILE"
    
    echo ""
done

echo -e "${GREEN}✓ Phase 2 완료: RAGAS 평가 결과 생성${NC}"
echo ""

# ============================================================
# Phase 3: 결과 요약
# ============================================================

echo -e "${BLUE}[Phase 3] 결과 요약${NC}"
echo ""

# CSV 파일 병합 (간단한 요약)
RAGAS_DIR="$COMPARISON_DIR/ragas_evaluation"

if [ -d "$RAGAS_DIR" ]; then
    echo -e "${GREEN}[요약] RAGAS 평가 결과:${NC}"
    echo ""
    
    for CSV_FILE in $RAGAS_DIR/ragas_summary_*.csv; do
        if [ -f "$CSV_FILE" ]; then
            echo "파일: $(basename $CSV_FILE)"
            cat "$CSV_FILE"
            echo ""
        fi
    done
    
    echo -e "${GREEN}✓ Phase 3 완료: 결과 요약 출력${NC}"
else
    echo -e "${YELLOW}⚠ 경고: RAGAS 평가 디렉토리가 없습니다${NC}"
fi

echo ""

# ============================================================
# 완료 메시지
# ============================================================

echo "=========================================="
echo -e "${GREEN}✅ 모든 실험 완료!${NC}"
echo "=========================================="
echo ""
echo "결과 위치:"
echo "  - 비교 로그: $COMPARISON_DIR/comparison_*.json"
echo "  - RAGAS 평가: $RAGAS_DIR/ragas_*.json"
echo "  - CSV 요약: $RAGAS_DIR/ragas_summary_*.csv"
echo ""
echo "다음 단계:"
echo "  1. CSV 파일을 엑셀/구글 시트로 열어 테이블 작성"
echo "  2. JSON 파일에서 통계적 유의성 확인 (p-value, Cohen's d)"
echo "  3. 논문/보고서 작성"
echo ""

