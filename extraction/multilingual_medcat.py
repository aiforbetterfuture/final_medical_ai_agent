"""
다국어 MedCAT2 파이프라인

한국어 입력을 자동으로 감지하고 영어로 번역한 후 MedCAT2로 엔티티를 추출합니다.
추출된 엔티티는 다시 한국어로 번역되어 반환됩니다.

구조:
[사용자 입력 ko] 
   └─(langdetect: 'ko')→ [ko → en 번역]
           └─→ MedCAT2.get_entities(text_en)
                    └─→ (concept_id, cui, semantic_type 등)
                              └─(설명만 en → ko 번역)
"""

import os
import logging
from typing import Dict, List, Any, Optional, Tuple

logger = logging.getLogger(__name__)


def _detect_language(text: str) -> str:
    """
    텍스트의 언어를 감지
    
    Args:
        text: 입력 텍스트
        
    Returns:
        언어 코드 ('ko', 'en', 'unknown')
    """
    if not text or not text.strip():
        return "unknown"
    
    # langdetect 사용 시도
    try:
        from langdetect import detect
        lang = detect(text)
        return lang
    except ImportError:
        logger.debug("[언어감지] langdetect 패키지 없음, 휴리스틱 사용")
    except Exception as e:
        logger.debug(f"[언어감지] langdetect 오류: {e}")
    
    # 휴리스틱: 한글 문자 비율로 판단
    korean_chars = sum(1 for c in text if '\uac00' <= c <= '\ud7a3' or '\u1100' <= c <= '\u11ff')
    total_chars = len(text.replace(" ", ""))
    
    if total_chars == 0:
        return "unknown"
    
    korean_ratio = korean_chars / total_chars
    
    if korean_ratio > 0.3:
        return "ko"
    elif korean_ratio < 0.1:
        return "en"
    else:
        return "mixed"


class MultilingualMedCAT:
    """
    다국어 MedCAT2 엔티티 추출기
    
    특징:
    1. 자동 언어 감지 (langdetect 또는 휴리스틱)
    2. 한영 신경망 번역 (Helsinki-NLP opus-mt)
    3. 사전 기반 번역 폴백
    4. UMLS CUI + 한국어 이름 동시 반환
    
    출력 구조:
    {
        "cui": "C0011849",
        "pretty_name_en": "Diabetes mellitus",
        "pretty_name_ko": "당뇨병",
        "semantic_type": "Disease or Syndrome",
        "source_text": "당뇨가 있고",
        "translated_text": "I have diabetes",
        "confidence": 0.95,
        "span_start_en": 7,
        "span_end_en": 15
    }
    """
    
    _instance = None
    
    def __new__(cls, *args, **kwargs):
        """싱글톤 패턴"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(
        self,
        use_neural_translation: bool = True,
        use_dict_translation: bool = True,
        medcat_model_path: Optional[str] = None
    ):
        """
        Args:
            use_neural_translation: 신경망 번역 사용 여부 (Helsinki-NLP)
            use_dict_translation: 사전 기반 번역 사용 여부 (폴백)
            medcat_model_path: MedCAT2 모델 경로
        """
        if self._initialized:
            return
        
        self.use_neural_translation = use_neural_translation
        self.use_dict_translation = use_dict_translation
        
        # 번역기 초기화
        self._neural_translator = None
        self._dict_translator = None
        
        # MedCAT2 어댑터 (지연 로딩)
        self._medcat_adapter = None
        self._medcat_model_path = medcat_model_path
        
        self._initialized = True
        logger.info("[MultilingualMedCAT] 초기화 완료")
    
    def _get_neural_translator(self):
        """신경망 번역기 지연 로딩"""
        if self._neural_translator is None and self.use_neural_translation:
            try:
                from .neural_translator import NeuralTranslator
                self._neural_translator = NeuralTranslator(lazy_load=True)
                logger.info("[MultilingualMedCAT] 신경망 번역기 초기화")
            except Exception as e:
                logger.warning(f"[MultilingualMedCAT] 신경망 번역기 초기화 실패: {e}")
                self._neural_translator = None
        return self._neural_translator
    
    def _get_dict_translator(self):
        """사전 기반 번역기 지연 로딩"""
        if self._dict_translator is None and self.use_dict_translation:
            try:
                # 프로젝트 루트의 korean_translator 사용
                import sys
                from pathlib import Path
                project_root = Path(__file__).parent.parent
                if str(project_root) not in sys.path:
                    sys.path.insert(0, str(project_root))
                
                from korean_translator import KoreanTranslator
                self._dict_translator = KoreanTranslator()
                logger.info("[MultilingualMedCAT] 사전 기반 번역기 초기화")
            except Exception as e:
                logger.warning(f"[MultilingualMedCAT] 사전 기반 번역기 초기화 실패: {e}")
                self._dict_translator = None
        return self._dict_translator
    
    def _get_medcat_adapter(self):
        """MedCAT2 어댑터 지연 로딩"""
        if self._medcat_adapter is None:
            try:
                from .medcat2_adapter import MedCAT2Adapter
                self._medcat_adapter = MedCAT2Adapter(model_path=self._medcat_model_path)
                logger.info("[MultilingualMedCAT] MedCAT2 어댑터 초기화")
            except Exception as e:
                logger.warning(f"[MultilingualMedCAT] MedCAT2 어댑터 초기화 실패: {e}")
                self._medcat_adapter = None
        return self._medcat_adapter
    
    def translate_to_english(self, text: str) -> Tuple[str, str]:
        """
        텍스트를 영어로 번역
        
        Args:
            text: 입력 텍스트 (한국어 또는 영어)
            
        Returns:
            (번역된 영어 텍스트, 사용된 번역 방법)
        """
        if not text or not text.strip():
            return "", "none"
        
        # 언어 감지
        lang = _detect_language(text)
        
        # 이미 영어면 그대로 반환
        if lang == "en":
            return text, "passthrough"
        
        translated = text
        method = "none"
        
        # 1. 사전 기반 번역 (먼저 적용하여 의료 용어 정확도 향상)
        dict_translator = self._get_dict_translator()
        if dict_translator:
            translated = dict_translator.translate_to_english(text)
            method = "dictionary"
        
        # 2. 신경망 번역 (추가 번역 또는 대체)
        neural_translator = self._get_neural_translator()
        if neural_translator and neural_translator.is_available:
            # 사전 번역 결과에서 아직 한국어가 많이 남아있으면 신경망 번역 적용
            remaining_lang = _detect_language(translated)
            if remaining_lang in ("ko", "mixed"):
                translated = neural_translator.translate_ko2en(translated)
                method = "neural" if method == "none" else "dictionary+neural"
        
        return translated, method
    
    def translate_to_korean(self, text: str) -> str:
        """
        영어 텍스트를 한국어로 번역
        
        Args:
            text: 영어 텍스트
            
        Returns:
            한국어로 번역된 텍스트
        """
        if not text or not text.strip():
            return ""
        
        # 사전 기반 역번역
        dict_translator = self._get_dict_translator()
        if dict_translator:
            # 역사전 생성
            reverse_dict = {v.lower(): k for k, v in dict_translator.term_dict.items()}
            
            # 사전 매칭
            result = text
            for eng, kor in reverse_dict.items():
                if eng in result.lower():
                    # 대소문자 무시 치환
                    import re
                    result = re.sub(re.escape(eng), kor, result, flags=re.IGNORECASE)
            
            if result != text:
                return result
        
        # 신경망 번역
        neural_translator = self._get_neural_translator()
        if neural_translator and neural_translator.is_available:
            return neural_translator.translate_en2ko(text)
        
        return text
    
    def extract_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        다국어 텍스트에서 의료 엔티티 추출
        
        Args:
            text: 입력 텍스트 (한국어 또는 영어)
            
        Returns:
            {
                "conditions": [...],
                "symptoms": [...],
                "medications": [...],
                "vitals": [...],
                "labs": [],
                "metadata": {
                    "original_text": "...",
                    "translated_text": "...",
                    "detected_language": "ko",
                    "translation_method": "dictionary+neural"
                }
            }
        """
        if not text or not text.strip():
            return {
                "conditions": [],
                "symptoms": [],
                "medications": [],
                "vitals": [],
                "labs": [],
                "metadata": {}
            }
        
        original_text = text
        detected_lang = _detect_language(text)
        
        # 1. 영어로 번역
        translated_text, translation_method = self.translate_to_english(text)
        
        # 2. MedCAT2로 엔티티 추출
        medcat = self._get_medcat_adapter()
        if medcat is None or medcat._model is None:
            logger.warning("[MultilingualMedCAT] MedCAT2 사용 불가, 빈 결과 반환")
            return {
                "conditions": [],
                "symptoms": [],
                "medications": [],
                "vitals": [],
                "labs": [],
                "metadata": {
                    "original_text": original_text,
                    "translated_text": translated_text,
                    "detected_language": detected_lang,
                    "translation_method": translation_method,
                    "error": "MedCAT2 not available"
                }
            }
        
        # MedCAT2 추출
        entities = medcat.extract_entities(translated_text)
        
        # 3. 한국어 이름 추가 (원래 한국어 입력인 경우)
        if detected_lang in ("ko", "mixed"):
            entities = self._add_korean_names(entities)
        
        # 4. 메타데이터 추가
        entities["metadata"] = {
            "original_text": original_text,
            "translated_text": translated_text,
            "detected_language": detected_lang,
            "translation_method": translation_method
        }
        
        return entities
    
    def _add_korean_names(self, entities: Dict[str, List[Dict]]) -> Dict[str, List[Dict]]:
        """추출된 엔티티에 한국어 이름 추가"""
        for category in ["conditions", "symptoms", "medications"]:
            for ent in entities.get(category, []):
                english_name = ent.get("name", "")
                if english_name:
                    # 영어 이름 보존
                    ent["pretty_name_en"] = english_name
                    # 한국어 번역 추가
                    korean_name = self.translate_to_korean(english_name)
                    ent["pretty_name_ko"] = korean_name
                    # 기본 이름을 한국어로 설정
                    if korean_name != english_name:
                        ent["name"] = korean_name
        
        return entities
    
    def get_entities_multilingual(self, text: str) -> Dict[str, Any]:
        """
        ChatGPT 제안 형식의 다국어 엔티티 추출
        
        출력 구조 (각 엔티티):
        {
            "cui": "C0011849",
            "pretty_name_en": "Diabetes mellitus",
            "pretty_name_ko": "당뇨병",
            "semantic_type": "Disease or Syndrome",
            "source_text": "당뇨가 있고",
            "translated_text": "I have diabetes",
            "confidence": 0.95,
            "span_start_en": 7,
            "span_end_en": 15
        }
        """
        if not text or not text.strip():
            return {"entities": [], "metadata": {}}
        
        original_text = text
        detected_lang = _detect_language(text)
        translated_text, translation_method = self.translate_to_english(text)
        
        # MedCAT2 raw 엔티티 추출
        medcat = self._get_medcat_adapter()
        if medcat is None or medcat._model is None:
            return {
                "entities": [],
                "metadata": {
                    "original_text": original_text,
                    "translated_text": translated_text,
                    "detected_language": detected_lang,
                    "translation_method": translation_method,
                    "error": "MedCAT2 not available"
                }
            }
        
        try:
            raw_result = medcat._model.get_entities(translated_text)
            raw_entities = raw_result.get("entities", {})
        except Exception as e:
            logger.error(f"[MultilingualMedCAT] MedCAT2 추출 오류: {e}")
            return {
                "entities": [],
                "metadata": {
                    "original_text": original_text,
                    "translated_text": translated_text,
                    "detected_language": detected_lang,
                    "translation_method": translation_method,
                    "error": str(e)
                }
            }
        
        # 엔티티 변환
        entities = []
        for ent in raw_entities.values() if isinstance(raw_entities, dict) else raw_entities:
            if not isinstance(ent, dict):
                continue
            
            english_name = ent.get("pretty_name", ent.get("source_value", ""))
            korean_name = self.translate_to_korean(english_name) if detected_lang in ("ko", "mixed") else english_name
            
            entity_data = {
                "cui": ent.get("cui", ""),
                "pretty_name_en": english_name,
                "pretty_name_ko": korean_name,
                "semantic_type": self._get_semantic_type(ent),
                "source_text": original_text,
                "translated_text": translated_text,
                "confidence": ent.get("acc", 0.0),
                "span_start_en": ent.get("start", 0),
                "span_end_en": ent.get("end", 0),
                "type_ids": ent.get("type_ids", []),
                "icd10": ent.get("icd10", []),
                "meta_anns": ent.get("meta_anns", {})
            }
            entities.append(entity_data)
        
        return {
            "entities": entities,
            "metadata": {
                "original_text": original_text,
                "translated_text": translated_text,
                "detected_language": detected_lang,
                "translation_method": translation_method,
                "entity_count": len(entities)
            }
        }
    
    def _get_semantic_type(self, entity: Dict) -> str:
        """엔티티의 의미 유형 추출"""
        # tui 필드에서 추출
        tui = entity.get("tui", [])
        if tui:
            # TUI to semantic type 매핑 (일부)
            tui_map = {
                "T047": "Disease or Syndrome",
                "T048": "Mental or Behavioral Dysfunction",
                "T049": "Cell or Molecular Dysfunction",
                "T184": "Sign or Symptom",
                "T121": "Pharmacologic Substance",
                "T200": "Clinical Drug"
            }
            for t in tui:
                if t in tui_map:
                    return tui_map[t]
        
        # type_ids에서 추론
        type_ids = entity.get("type_ids", [])
        if type_ids:
            return "Medical Entity"
        
        return "Unknown"


# 싱글톤 인스턴스 접근 함수
def get_multilingual_medcat(
    use_neural_translation: bool = True,
    use_dict_translation: bool = True
) -> MultilingualMedCAT:
    """전역 MultilingualMedCAT 인스턴스 반환"""
    return MultilingualMedCAT(
        use_neural_translation=use_neural_translation,
        use_dict_translation=use_dict_translation
    )


# 편의 함수 (ChatGPT 제안 형식)
def extract_medcat_entities_multilingual(text: str) -> Dict[str, Any]:
    """
    다국어 텍스트에서 MedCAT2 엔티티 추출 (편의 함수)
    
    Args:
        text: 한국어 또는 영어 텍스트
        
    Returns:
        {
            "entities": [...],
            "metadata": {...}
        }
    """
    extractor = get_multilingual_medcat()
    return extractor.get_entities_multilingual(text)


# LangGraph 노드용 함수
def preprocess_text_for_medcat(text: str) -> Tuple[str, Dict[str, Any]]:
    """
    LangGraph 노드에서 사용할 전처리 함수
    
    사용자 질의를 번역하고 MedCAT2용으로 준비합니다.
    
    Args:
        text: 원본 텍스트
        
    Returns:
        (번역된 영어 텍스트, 메타데이터)
    """
    extractor = get_multilingual_medcat()
    translated, method = extractor.translate_to_english(text)
    
    metadata = {
        "original_text": text,
        "translated_text": translated,
        "detected_language": _detect_language(text),
        "translation_method": method
    }
    
    return translated, metadata

