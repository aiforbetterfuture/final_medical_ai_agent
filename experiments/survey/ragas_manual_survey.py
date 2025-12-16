"""
RAGAS 수동 설문조사 생성기

목적:
- RAGAS 자동 평가가 시간이 많이 걸릴 경우 대체 방안
- 대화 로그를 읽어 수동 평가용 설문지 생성
- Markdown 체크박스 형식

사용법:
    python experiments/survey/ragas_manual_survey.py --log-dir experiments/comparison_logs/20251216_llm_vs_rag
"""

import json
import sys
import argparse
from pathlib import Path
from typing import List, Dict

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# ============================================================
# 설문 템플릿
# ============================================================

SURVEY_TEMPLATE = """
# RAGAS 평가 설문 (턴 {turn})

## 질문
{question}

## 답변
{answer}

## 검색된 문서
{contexts}

---

### 1. Faithfulness (근거 충실도)
답변이 검색된 문서의 내용과 얼마나 일치하나요?

[ ] 1점: 전혀 일치하지 않음 (심각한 환각)
[ ] 2점: 일부만 일치
[ ] 3점: 대체로 일치하나 일부 근거 부족
[ ] 4점: 잘 일치함
[ ] 5점: 완벽히 일치함

### 2. Answer Relevancy (답변 관련성)
답변이 질문과 얼마나 관련이 있나요?

[ ] 1점: 전혀 관련 없음
[ ] 2점: 일부만 관련
[ ] 3점: 대체로 관련 있으나 누락 있음
[ ] 4점: 잘 관련됨
[ ] 5점: 완벽히 관련됨

### 3. Context Precision (문서 정확도)
검색된 문서가 질문에 얼마나 적합한가요?

[ ] 1점: 전혀 적합하지 않음
[ ] 2점: 일부만 적합
[ ] 3점: 대체로 적합
[ ] 4점: 매우 적합
[ ] 5점: 완벽히 적합

---

"""

# ============================================================
# 대화 로그 읽기
# ============================================================

def read_jsonl(file_path: Path) -> List[Dict]:
    """JSONL 파일 읽기"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data


# ============================================================
# 설문지 생성
# ============================================================

def generate_survey_forms(log_dir: Path, output_dir: Path):
    """
    각 시스템별 설문지 생성
    
    Args:
        log_dir: 로그 디렉토리
        output_dir: 설문지 출력 디렉토리
    """
    variants = ['llm_only', 'basic_rag', 'corrective_rag']
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for variant_name in variants:
        variant_dir = log_dir / variant_name
        
        if not variant_dir.exists():
            print(f"경고: {variant_name} 디렉토리를 찾을 수 없습니다: {variant_dir}")
            continue
        
        print(f"\n[{variant_name.upper()}] 설문지 생성 중...")
        
        # 모든 환자 로그 읽기
        all_turns = []
        for log_file in variant_dir.glob('*.jsonl'):
            patient_logs = read_jsonl(log_file)
            all_turns.extend(patient_logs)
        
        # 설문지 파일 생성
        survey_file = output_dir / f"survey_{variant_name}.md"
        
        with open(survey_file, 'w', encoding='utf-8') as f:
            # 헤더
            f.write(f"# RAGAS 수동 평가 설문지\n\n")
            f.write(f"**시스템**: {variant_name}\n")
            f.write(f"**총 턴 수**: {len(all_turns)}\n\n")
            f.write(f"---\n\n")
            
            # 각 턴별 설문
            for turn_data in all_turns:
                # 문서 포맷팅
                if turn_data.get('contexts'):
                    contexts_text = '\n'.join([f"- {ctx[:200]}..." for ctx in turn_data['contexts']])
                else:
                    contexts_text = "- (검색된 문서 없음)"
                
                survey = SURVEY_TEMPLATE.format(
                    turn=turn_data.get('turn', '?'),
                    question=turn_data.get('question', ''),
                    answer=turn_data.get('answer', ''),
                    contexts=contexts_text
                )
                f.write(survey)
        
        print(f"  저장: {survey_file}")
        print(f"  턴 수: {len(all_turns)}")


# ============================================================
# 메인 실행
# ============================================================

def main():
    """메인 실행"""
    parser = argparse.ArgumentParser(description='RAGAS 수동 설문조사 생성')
    parser.add_argument('--log-dir', type=str, required=True,
                        help='로그 디렉토리')
    parser.add_argument('--output-dir', type=str,
                        default='experiments/survey/forms',
                        help='설문지 출력 디렉토리')
    
    args = parser.parse_args()
    
    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)
    
    if not log_dir.exists():
        print(f"오류: 로그 디렉토리를 찾을 수 없습니다: {log_dir}")
        return
    
    print(f"{'='*80}")
    print("RAGAS 수동 설문조사 생성")
    print(f"{'='*80}")
    print(f"로그 디렉토리: {log_dir}")
    print(f"출력 디렉토리: {output_dir}\n")
    
    # 설문지 생성
    generate_survey_forms(log_dir, output_dir)
    
    print(f"\n{'='*80}")
    print("✓ 설문지 생성 완료!")
    print(f"{'='*80}")
    print(f"\n다음 단계:")
    print(f"1. {output_dir} 디렉토리의 설문지를 열어 평가 수행")
    print(f"2. 체크박스에 [x] 표시")
    print(f"3. python experiments/survey/analyze_survey_results.py --survey-dir {output_dir}")


if __name__ == '__main__':
    main()

