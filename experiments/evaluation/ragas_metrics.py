"""
RAGAS 평가지표 계산 모듈 (RAGAS 0.4.x 호환)

메트릭:
- faithfulness: 근거 문서와의 일치도
- answer_relevancy: 질문과의 관련성
- context_precision: 검색된 문서의 정확도
- context_recall: 검색된 문서의 재현율
- context_relevancy: 검색된 문서의 관련성
"""

import os
import logging
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path

# 로거 설정
logger = logging.getLogger(__name__)

# RAGAS 임포트 시도 (설치되지 않았을 경우를 대비한 안전장치)
try:
    from ragas import evaluate
    from ragas.metrics import (
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
        context_relevancy
    )
    from datasets import Dataset
    HAS_RAGAS = True
    import ragas
    RAGAS_VERSION = ragas.__version__
    logger.info(f"RAGAS {RAGAS_VERSION} API를 사용합니다.")
except ImportError as e:
    HAS_RAGAS = False
    RAGAS_VERSION = "N/A"
    logger.warning(f"RAGAS가 설치되지 않았습니다: {e}")


def calculate_ragas_metrics(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: Optional[str] = None
) -> Optional[Dict[str, float]]:
    """
    RAGAS 메트릭 계산 (기본 2개 메트릭)
    
    Args:
        question: 사용자 질문
        answer: 생성된 답변
        contexts: 검색된 문서 리스트
        ground_truth: 정답 (선택사항)
    
    Returns:
        메트릭 점수 딕셔너리 또는 None (실패 시)
    """
    if not HAS_RAGAS:
        logger.warning("RAGAS가 설치되지 않아 메트릭을 계산할 수 없습니다.")
        return None
    
    # 빈 contexts 처리
    if not contexts or all(not ctx.strip() for ctx in contexts):
        logger.warning("contexts가 비어있어 RAGAS 메트릭을 계산할 수 없습니다.")
        contexts = [""]

    try:
        # 1. 데이터 준비 (HuggingFace Dataset 포맷)
        data_dict = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],  # contexts는 리스트의 리스트여야 함
        }
        
        if ground_truth:
            data_dict["ground_truth"] = [ground_truth]

        dataset = Dataset.from_dict(data_dict)

        # 2. LLM 및 임베딩 모델 설정 (RAGAS가 사용할 모델 명시)
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from dotenv import load_dotenv

        # .env 파일 로드
        project_root = Path(__file__).parent.parent.parent
        env_path = project_root / '.env'
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)

        # OpenAI API 키 확인
        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            logger.error("OPENAI_API_KEY가 설정되지 않았습니다.")
            return None

        # OpenAI 모델 설정
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_key)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_key)

        # 3. 메트릭 정의 (faithfulness와 answer_relevancy만 사용)
        metrics = [
            faithfulness,
            answer_relevancy
        ]

        # 4. 평가 실행
        # raise_exceptions=False로 설정하여 개별 메트릭 실패가 전체를 멈추지 않게 함
        results = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False
        )

        # 5. 결과 변환
        # RAGAS 0.4.x는 EvaluationResult 객체를 반환
        final_scores = {}

        # EvaluationResult 객체를 딕셔너리로 변환
        if hasattr(results, 'to_pandas'):
            df = results.to_pandas()
            if 'faithfulness' in df.columns:
                final_scores['faithfulness'] = float(df['faithfulness'].iloc[0])
            if 'answer_relevancy' in df.columns:
                final_scores['answer_relevance'] = float(df['answer_relevancy'].iloc[0])
        elif isinstance(results, dict):
            final_scores = results
        else:
            logger.warning(f"예상치 못한 결과 타입: {type(results)}")
            return None

        logger.info(f"RAGAS 메트릭 계산 완료: {final_scores}")
        return final_scores

    except Exception as e:
        logger.error(f"RAGAS 메트릭 계산 중 오류 발생: {e}", exc_info=True)
        return None


def calculate_ragas_metrics_full(
    question: str,
    answer: str,
    contexts: List[str],
    ground_truth: Optional[str] = None
) -> Optional[Dict[str, float]]:
    """
    RAGAS 전체 메트릭 계산 (5개 메트릭)
    
    Args:
        question: 사용자 질문
        answer: 생성된 답변
        contexts: 검색된 문서 리스트
        ground_truth: 정답 (context_recall 계산에 필요)
    
    Returns:
        {
            'faithfulness': 0.85,
            'answer_relevancy': 0.78,
            'context_precision': 0.82,
            'context_recall': 0.75,  # ground_truth 있을 때만
            'context_relevancy': 0.80
        }
    """
    if not HAS_RAGAS:
        logger.warning("RAGAS가 설치되지 않아 메트릭을 계산할 수 없습니다.")
        return None
    
    # 빈 contexts 처리
    if not contexts or all(not ctx.strip() for ctx in contexts):
        logger.warning("contexts가 비어있어 RAGAS 메트릭을 계산할 수 없습니다.")
        contexts = ["No context available"]

    try:
        # 1. 데이터 준비
        data_dict = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],
        }
        
        # ground_truth 있으면 context_recall 계산 가능
        if ground_truth:
            data_dict["ground_truth"] = [ground_truth]

        dataset = Dataset.from_dict(data_dict)

        # 2. LLM 및 임베딩 모델 설정
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from dotenv import load_dotenv

        project_root = Path(__file__).parent.parent.parent
        env_path = project_root / '.env'
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)

        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            logger.error("OPENAI_API_KEY가 설정되지 않았습니다.")
            return None

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_key)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_key)

        # 3. 메트릭 정의 (전체 메트릭)
        metrics = [
            faithfulness,
            answer_relevancy,
            context_precision,
            context_relevancy
        ]
        
        # context_recall은 ground_truth 필요
        if ground_truth:
            metrics.append(context_recall)

        # 4. 평가 실행 (LLM as a Judge)
        results = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,  # GPT-4o-mini (LLM as a Judge)
            embeddings=embeddings,
            raise_exceptions=False
        )

        # 5. 결과 변환
        final_scores = {}

        if hasattr(results, 'to_pandas'):
            df = results.to_pandas()
            
            # 각 메트릭 추출
            if 'faithfulness' in df.columns:
                final_scores['faithfulness'] = float(df['faithfulness'].iloc[0])
            if 'answer_relevancy' in df.columns:
                final_scores['answer_relevancy'] = float(df['answer_relevancy'].iloc[0])
            if 'context_precision' in df.columns:
                final_scores['context_precision'] = float(df['context_precision'].iloc[0])
            if 'context_recall' in df.columns:
                final_scores['context_recall'] = float(df['context_recall'].iloc[0])
            if 'context_relevancy' in df.columns:
                final_scores['context_relevancy'] = float(df['context_relevancy'].iloc[0])
        elif isinstance(results, dict):
            final_scores = results
        else:
            logger.warning(f"예상치 못한 결과 타입: {type(results)}")
            return None

        logger.info(f"RAGAS 전체 메트릭 계산 완료: {final_scores}")
        return final_scores

    except Exception as e:
        logger.error(f"RAGAS 전체 메트릭 계산 중 오류 발생: {e}", exc_info=True)
        return None


def calculate_ragas_metrics_batch(
    questions: List[str],
    answers: List[str],
    contexts_list: List[List[str]],
    ground_truths: Optional[List[str]] = None
) -> Optional[pd.DataFrame]:
    """
    배치 RAGAS 메트릭 계산
    
    Args:
        questions: 질문 리스트
        answers: 답변 리스트
        contexts_list: 검색된 문서 리스트의 리스트
        ground_truths: 정답 리스트 (선택사항)
    
    Returns:
        메트릭 점수 DataFrame 또는 None (실패 시)
    """
    if not HAS_RAGAS:
        logger.warning("RAGAS가 설치되지 않아 메트릭을 계산할 수 없습니다.")
        return None
    
    if len(questions) != len(answers) or len(questions) != len(contexts_list):
        logger.error("입력 리스트의 길이가 일치하지 않습니다.")
        return None

    try:
        # 데이터 준비
        data_dict = {
            "question": questions,
            "answer": answers,
            "contexts": contexts_list,
        }
        
        if ground_truths:
            data_dict["ground_truth"] = ground_truths

        dataset = Dataset.from_dict(data_dict)

        # 모델 설정
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings
        from dotenv import load_dotenv

        project_root = Path(__file__).parent.parent.parent
        env_path = project_root / '.env'
        if env_path.exists():
            load_dotenv(dotenv_path=env_path)

        openai_key = os.getenv('OPENAI_API_KEY')
        if not openai_key:
            logger.error("OPENAI_API_KEY가 설정되지 않았습니다.")
            return None

        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=openai_key)
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=openai_key)

        # 평가 실행
        results = evaluate(
            dataset=dataset,
            metrics=[faithfulness, answer_relevancy],
            llm=llm,
            embeddings=embeddings,
            raise_exceptions=False
        )

        # 결과 변환
        if hasattr(results, 'to_pandas'):
            return results.to_pandas()
        else:
            logger.warning(f"예상치 못한 결과 타입: {type(results)}")
            return None

    except Exception as e:
        logger.error(f"배치 RAGAS 메트릭 계산 중 오류 발생: {e}", exc_info=True)
        return None
