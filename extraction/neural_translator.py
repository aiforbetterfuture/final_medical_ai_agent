"""
신경망 기반 다국어 번역 모듈 (Helsinki-NLP)

transformers 라이브러리를 사용하여 한영/영한 번역을 수행합니다.
MedCAT2와 함께 사용하여 한국어 의료 엔티티 추출 정확도를 향상시킵니다.
"""

import os
import logging
from typing import Optional, Dict, Any
from functools import lru_cache

logger = logging.getLogger(__name__)

# 전역 번역기 캐시 (싱글톤 패턴)
_translator_cache: Dict[str, Any] = {}


def _get_device() -> str:
    """사용 가능한 디바이스 반환 (GPU 우선)"""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


class NeuralTranslator:
    """
    Helsinki-NLP opus-mt 모델 기반 신경망 번역기
    
    특징:
    - 한영/영한 양방향 번역 지원
    - GPU 가속 지원 (CUDA, MPS)
    - 배치 번역 지원
    - 모델 지연 로딩 (필요할 때만 로드)
    """
    
    # 모델 설정
    # 한영 번역: Helsinki-NLP/opus-mt-ko-en (확인됨)
    # 영한 번역: Helsinki-NLP/opus-mt-en-ko는 존재하지 않으므로 대체 모델 사용
    KO2EN_MODEL = "Helsinki-NLP/opus-mt-ko-en"
    EN2KO_MODEL = None  # 영한 번역 모델이 Hugging Face에 없음, 사용하지 않음
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """싱글톤 패턴"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        use_gpu: bool = True,
        max_length: int = 512,
        lazy_load: bool = True
    ):
        """
        Args:
            use_gpu: GPU 사용 여부
            max_length: 최대 번역 길이
            lazy_load: 지연 로딩 여부 (True면 실제 번역 시 모델 로드)
        """
        if self._initialized:
            return
        
        self.use_gpu = use_gpu
        self.max_length = max_length
        self.device = _get_device() if use_gpu else "cpu"
        
        self._ko2en_pipeline = None
        self._en2ko_pipeline = None
        self._transformers_available = None
        
        if not lazy_load:
            self._load_pipelines()
        
        self._initialized = True
        logger.info(f"[NeuralTranslator] 초기화 완료 (device={self.device}, lazy_load={lazy_load})")
    
    def _check_transformers(self) -> bool:
        """transformers 라이브러리 사용 가능 여부 확인"""
        if self._transformers_available is not None:
            return self._transformers_available
        
        try:
            import transformers
            self._transformers_available = True
            logger.info(f"[NeuralTranslator] transformers {transformers.__version__} 사용 가능")
        except ImportError:
            self._transformers_available = False
            logger.warning("[NeuralTranslator] transformers 패키지가 설치되지 않았습니다. pip install transformers 실행 필요")
        
        return self._transformers_available
    
    def _load_pipelines(self):
        """번역 파이프라인 로드"""
        if not self._check_transformers():
            return
        
        try:
            from transformers import pipeline
            
            # 한영 번역기
            if self._ko2en_pipeline is None:
                logger.info(f"[NeuralTranslator] 한영 번역 모델 로딩: {self.KO2EN_MODEL}")
                self._ko2en_pipeline = pipeline(
                    "translation",
                    model=self.KO2EN_MODEL,
                    device=0 if self.device == "cuda" else -1
                )
                logger.info("[NeuralTranslator] 한영 번역 모델 로드 완료")
            
            # 영한 번역기 (모델이 없으므로 건너뜀)
            if self.EN2KO_MODEL is None:
                logger.info("[NeuralTranslator] 영한 번역 모델이 설정되지 않았습니다. (Helsinki-NLP/opus-mt-en-ko는 Hugging Face에 존재하지 않음)")
                self._en2ko_pipeline = None
            elif self._en2ko_pipeline is None:
                try:
                    logger.info(f"[NeuralTranslator] 영한 번역 모델 로딩: {self.EN2KO_MODEL}")
                    self._en2ko_pipeline = pipeline(
                        "translation",
                        model=self.EN2KO_MODEL,
                        device=0 if self.device == "cuda" else -1
                    )
                    logger.info("[NeuralTranslator] 영한 번역 모델 로드 완료")
                except Exception as e:
                    logger.warning(f"[NeuralTranslator] 영한 번역 모델 로드 실패: {e}")
                    self._en2ko_pipeline = None
                
        except Exception as e:
            logger.error(f"[NeuralTranslator] 번역 모델 로드 실패: {e}")
            # 한영 번역기만 실패한 경우에만 None으로 설정
            if self._ko2en_pipeline is not None:
                # 한영은 성공했지만 영한만 실패한 경우
                logger.warning("[NeuralTranslator] 한영 번역기는 사용 가능하지만 영한 번역기는 사용할 수 없습니다.")
            else:
                self._ko2en_pipeline = None
            self._en2ko_pipeline = None
    
    @property
    def is_available(self) -> bool:
        """번역기 사용 가능 여부"""
        return self._check_transformers()
    
    def translate_ko2en(self, text: str) -> str:
        """
        한국어 → 영어 번역
        
        Args:
            text: 한국어 텍스트
            
        Returns:
            영어로 번역된 텍스트 (실패 시 원본 반환)
        """
        if not text or not text.strip():
            return text
        
        # 이미 영어인지 간단히 확인
        if self._is_mostly_english(text):
            return text
        
        if not self._check_transformers():
            logger.debug("[NeuralTranslator] transformers 없음, 원본 반환")
            return text
        
        # 지연 로딩
        if self._ko2en_pipeline is None:
            self._load_pipelines()
        
        if self._ko2en_pipeline is None:
            logger.warning("[NeuralTranslator] 한영 번역기 로드 실패, 원본 반환")
            return text
        
        try:
            result = self._ko2en_pipeline(
                text,
                max_length=self.max_length,
                truncation=True
            )
            translated = result[0]["translation_text"]
            logger.debug(f"[NeuralTranslator] 한영 번역: {text[:50]}... → {translated[:50]}...")
            return translated
        except Exception as e:
            logger.error(f"[NeuralTranslator] 한영 번역 오류: {e}")
            return text
    
    def translate_en2ko(self, text: str) -> str:
        """
        영어 → 한국어 번역
        
        주의: Helsinki-NLP/opus-mt-en-ko 모델이 Hugging Face에 존재하지 않으므로
        현재는 원본 텍스트를 반환합니다. 필요시 다른 번역 모델로 대체 가능합니다.
        
        Args:
            text: 영어 텍스트
            
        Returns:
            한국어로 번역된 텍스트 (실패 시 원본 반환)
        """
        if not text or not text.strip():
            return text
        
        # 영한 번역 모델이 없으므로 원본 반환
        if self.EN2KO_MODEL is None:
            logger.debug("[NeuralTranslator] 영한 번역 모델이 설정되지 않음, 원본 반환")
            return text
        
        if not self._check_transformers():
            return text
        
        # 지연 로딩
        if self._en2ko_pipeline is None:
            self._load_pipelines()
        
        if self._en2ko_pipeline is None:
            logger.debug("[NeuralTranslator] 영한 번역기 로드 실패, 원본 반환")
            return text
        
        try:
            result = self._en2ko_pipeline(
                text,
                max_length=self.max_length,
                truncation=True
            )
            translated = result[0]["translation_text"]
            logger.debug(f"[NeuralTranslator] 영한 번역: {text[:50]}... → {translated[:50]}...")
            return translated
        except Exception as e:
            logger.error(f"[NeuralTranslator] 영한 번역 오류: {e}")
            return text
    
    def _is_mostly_english(self, text: str) -> bool:
        """텍스트가 대부분 영어인지 확인 (간단한 휴리스틱)"""
        if not text:
            return False
        
        # 한글 문자 비율 계산
        korean_chars = sum(1 for c in text if '\uac00' <= c <= '\ud7a3' or '\u1100' <= c <= '\u11ff')
        total_alpha = sum(1 for c in text if c.isalpha())
        
        if total_alpha == 0:
            return False
        
        korean_ratio = korean_chars / total_alpha
        return korean_ratio < 0.1  # 10% 미만이면 영어로 간주
    
    def batch_translate_ko2en(self, texts: list) -> list:
        """
        배치 한영 번역
        
        Args:
            texts: 한국어 텍스트 리스트
            
        Returns:
            영어로 번역된 텍스트 리스트
        """
        if not texts:
            return []
        
        if not self._check_transformers():
            return texts
        
        if self._ko2en_pipeline is None:
            self._load_pipelines()
        
        if self._ko2en_pipeline is None:
            return texts
        
        try:
            results = self._ko2en_pipeline(
                texts,
                max_length=self.max_length,
                truncation=True,
                batch_size=len(texts)
            )
            return [r["translation_text"] for r in results]
        except Exception as e:
            logger.error(f"[NeuralTranslator] 배치 번역 오류: {e}")
            return texts


# 싱글톤 인스턴스 접근 함수
def get_neural_translator(lazy_load: bool = True) -> NeuralTranslator:
    """전역 NeuralTranslator 인스턴스 반환"""
    return NeuralTranslator(lazy_load=lazy_load)


# 편의 함수
def neural_translate_ko2en(text: str) -> str:
    """한국어 → 영어 번역 (편의 함수)"""
    translator = get_neural_translator()
    return translator.translate_ko2en(text)


def neural_translate_en2ko(text: str) -> str:
    """영어 → 한국어 번역 (편의 함수)"""
    translator = get_neural_translator()
    return translator.translate_en2ko(text)

