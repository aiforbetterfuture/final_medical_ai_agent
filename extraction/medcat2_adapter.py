"""
MedCAT2 어댑터 (다국어 지원)
- MedCAT2 모델 로드
- 엔티티 추출
- UMLS CUI 매핑
- 한국어 입력 자동 번역 및 엔티티 추출

다국어 파이프라인:
[사용자 입력 ko] 
   └─(langdetect)→ [ko → en 번역]
           └─→ MedCAT2.get_entities(text_en)
                    └─→ (concept_id, cui, semantic_type 등)
                              └─(설명만 en → ko 번역)
"""

import os
import re
import logging
from typing import Dict, Any, List, Optional, Tuple
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# .env 파일 자동 로드
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)


def _parse_vitals_labs(text: str) -> Dict[str, List[Dict[str, Any]]]:
    """한글/영문 뒤섞인 수치 파싱(혈압/A1c/FPG)"""
    t = text.lower()
    out = {"vitals": [], "labs": []}
    
    # 혈압 파싱: 140/90 또는 140/90 mmHg
    m = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})\s*(mmhg)?", t)
    if m:
        sbp = float(m.group(1))
        dbp = float(m.group(2))
        if sbp < dbp:
            sbp, dbp = dbp, sbp
        out["vitals"].append({"name": "SBP", "value": sbp, "unit": "mmHg"})
        out["vitals"].append({"name": "DBP", "value": dbp, "unit": "mmHg"})
    
    # A1c 파싱
    m = re.search(r"(a1c|당화혈색소|hba1c)\s*[:는은]?\s*(\d+(?:\.\d+)?)\s*%", t, re.IGNORECASE)
    if m:
        out["labs"].append({"name": "A1c", "value": float(m.group(2)), "unit": "%"})
    
    # 공복혈당 파싱
    m = re.search(r"(fpg|공복혈당|혈당)\s*[: ]?(\d+(?:\.\d+)?)\s*(mg/dl|mg)?", t)
    if m:
        out["labs"].append({"name": "FPG", "value": float(m.group(2)), "unit": "mg/dL"})
    
    return out


class MedCAT2Adapter:
    """
    MEDCAT2 어댑터 클래스
    
    환경 변수:
        MEDCAT2_MODEL_PATH: 모델 팩 파일 경로 (필수)
    """
    
    _instance = None
    _model = None
    
    def __new__(cls, model_path: Optional[str] = None):
        """싱글톤 패턴"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, model_path: Optional[str] = None):
        """초기화 (싱글톤이므로 한 번만 실행)"""
        if self._model is not None:
            return  # 이미 로드됨
        
        self.model_path = model_path or os.getenv("MEDCAT2_MODEL_PATH")
        
        if not self.model_path:
            print("[WARNING] MEDCAT2_MODEL_PATH가 설정되지 않았습니다. MedCAT2 추출을 건너뜁니다.")
            self._model = None
            return
        
        if not os.path.exists(self.model_path):
            print(f"[WARNING] MEDCAT2 모델 파일을 찾을 수 없습니다: {self.model_path}")
            self._model = None
            return
        
        self._load_model()
    
    def _load_model(self):
        """MEDCAT2 모델 로드"""
        try:
            from medcat.cat import CAT
            self._model = CAT.load_model_pack(self.model_path)
            print(f"[MedCAT2] 모델 로드 완료: {self.model_path}")
        except ImportError as e:
            print(f"[WARNING] MedCAT2 의존성 오류: {e}")
            print("[WARNING] pip install medcat peft 를 실행하세요.")
            self._model = None
        except Exception as e:
            print(f"[WARNING] MEDCAT2 모델 로드 실패: {e}")
            self._model = None
    
    def extract_entities(self, text: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        텍스트에서 의료 엔티티 추출
        
        Args:
            text: 분석할 텍스트
        
        Returns:
            {
                "conditions": [{"name": "...", "cui": "...", "confidence": 0.9}],
                "symptoms": [...],
                "medications": [...],
                "vitals": [...],
                "labs": [...]
            }
        """
        if not text or not text.strip():
            return {
                "conditions": [],
                "symptoms": [],
                "medications": [],
                "vitals": [],
                "labs": []
            }
        
        result = {
            "conditions": [],
            "symptoms": [],
            "medications": [],
            "vitals": [],
            "labs": []
        }
        
        # MedCAT2 모델이 있으면 사용
        if self._model:
            try:
                entities = self._model.get_entities(text)
                
                for entity in entities.get('entities', {}).values():
                    cui = entity.get('cui', '')
                    name = entity.get('pretty_name', entity.get('source_value', ''))
                    confidence = entity.get('acc', 0.0)
                    tui = entity.get('tui', [])
                    type_ids = entity.get('type_ids', [])
                    name_lower = name.lower() if name else ""
                    
                    # TUI 기반 분류 (UMLS 모델용)
                    if tui:
                        if any(t in tui for t in ['T047', 'T048', 'T049']):  # 질환
                            result["conditions"].append({
                                "name": name,
                                "cui": cui,
                                "confidence": confidence,
                                "source": "medcat2"
                            })
                        elif any(t in tui for t in ['T184']):  # 증상
                            result["symptoms"].append({
                                "name": name,
                                "cui": cui,
                                "confidence": confidence,
                                "source": "medcat2"
                            })
                        elif any(t in tui for t in ['T121', 'T200']):  # 약물
                            result["medications"].append({
                                "name": name,
                                "cui": cui,
                                "confidence": confidence,
                                "source": "medcat2"
                            })
                    # SNOMED 모델용: pretty_name 기반 키워드 매칭
                    elif type_ids or name:
                        # 질환 키워드
                        condition_keywords = ['diabetes', 'hypertension', 'disease', 'disorder', 'syndrome', 
                                            'failure', 'mellitus', 'family history']
                        # 증상 키워드
                        symptom_keywords = ['chest', 'tightness', 'dizziness', 'pain', 'dyspnea', 'headache',
                                          'nausea', 'vomiting', 'fever', 'cough', 'symptom', 'sign']
                        # 약물 키워드
                        medication_keywords = ['metformin', 'drug', 'medication', 'medicine', 'pharmaceutical']
                        
                        if any(kw in name_lower for kw in medication_keywords):
                            result["medications"].append({
                                "name": name,
                                "cui": cui,
                                "confidence": confidence,
                                "source": "medcat2"
                            })
                        elif any(kw in name_lower for kw in symptom_keywords):
                            result["symptoms"].append({
                                "name": name,
                                "cui": cui,
                                "confidence": confidence,
                                "source": "medcat2"
                            })
                        elif any(kw in name_lower for kw in condition_keywords):
                            result["conditions"].append({
                                "name": name,
                                "cui": cui,
                                "confidence": confidence,
                                "source": "medcat2"
                            })
                        else:
                            # 기본적으로 질환으로 분류 (의료 엔티티는 대부분 질환)
                            result["conditions"].append({
                                "name": name,
                                "cui": cui,
                                "confidence": confidence,
                                "source": "medcat2"
                            })
            except Exception as e:
                print(f"[WARNING] MedCAT2 추출 오류: {e}")
        
        # 수치 정보는 정규표현식으로 추출
        vitals_labs = _parse_vitals_labs(text)
        result["vitals"].extend(vitals_labs["vitals"])
        result["labs"].extend(vitals_labs["labs"])
        
        return result
    
    def extract_entities_multilingual(
        self, 
        text: str,
        use_neural_translation: bool = True,
        use_dict_translation: bool = True
    ) -> Dict[str, Any]:
        """
        다국어 텍스트에서 의료 엔티티 추출 (한국어 자동 번역)
        
        Args:
            text: 입력 텍스트 (한국어 또는 영어)
            use_neural_translation: Helsinki-NLP 신경망 번역 사용 여부
            use_dict_translation: 사전 기반 번역 사용 여부
        
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
        try:
            from .multilingual_medcat import MultilingualMedCAT
            
            multilingual = MultilingualMedCAT(
                use_neural_translation=use_neural_translation,
                use_dict_translation=use_dict_translation,
                medcat_model_path=self.model_path
            )
            # 싱글톤이므로 기존 모델 공유
            multilingual._medcat_adapter = self
            
            return multilingual.extract_entities(text)
        except Exception as e:
            logger.warning(f"[MedCAT2] 다국어 추출 실패, 기본 추출 사용: {e}")
            result = self.extract_entities(text)
            result["metadata"] = {
                "original_text": text,
                "translated_text": text,
                "detected_language": "unknown",
                "translation_method": "none",
                "error": str(e)
            }
            return result
    
    def get_raw_entities(self, text: str) -> Dict[str, Any]:
        """
        MedCAT2 원본 엔티티 반환 (분류 없이)
        
        Args:
            text: 분석할 텍스트
            
        Returns:
            MedCAT2 원본 결과
        """
        if not self._model:
            return {"entities": {}, "tokens": []}
        
        try:
            return self._model.get_entities(text)
        except Exception as e:
            logger.error(f"[MedCAT2] 원본 엔티티 추출 오류: {e}")
            return {"entities": {}, "tokens": []}


# 유틸리티 함수
def _dedup(items: List[Dict], key) -> List[Dict]:
    """리스트 중복 제거"""
    seen = set()
    result = []
    for item in items:
        k = key(item)
        if k and k not in seen:
            seen.add(k)
            result.append(item)
    return result


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
        pass
    except Exception:
        pass
    
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


# 편의 함수
def medcat2_extract(text: str, multilingual: bool = True) -> Dict[str, Any]:
    """
    MedCAT2 엔티티 추출 편의 함수
    
    Args:
        text: 입력 텍스트
        multilingual: 다국어 지원 여부
        
    Returns:
        추출된 엔티티 딕셔너리
    """
    adapter = MedCAT2Adapter()
    
    if multilingual:
        return adapter.extract_entities_multilingual(text)
    else:
        return adapter.extract_entities(text)


def medcat2_extract_korean(text: str) -> Dict[str, Any]:
    """
    한국어 텍스트 전용 MedCAT2 엔티티 추출
    
    Args:
        text: 한국어 텍스트
        
    Returns:
        추출된 엔티티 (한국어 이름 포함)
    """
    adapter = MedCAT2Adapter()
    return adapter.extract_entities_multilingual(
        text,
        use_neural_translation=True,
        use_dict_translation=True
    )

