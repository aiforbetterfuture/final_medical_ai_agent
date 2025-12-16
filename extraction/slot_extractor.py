"""
슬롯 추출 모듈
- MedCAT2 기반 엔티티 추출 (다국어 지원)
- 정규표현식 기반 인구통계학적 정보 추출
- 6개 슬롯 구조화

다국어 파이프라인 통합:
- 한국어 입력 자동 감지 및 번역
- Helsinki-NLP 신경망 번역 + 사전 기반 번역 하이브리드
- 추출된 엔티티에 한국어/영어 이름 동시 제공
"""

import re
import logging
from typing import Dict, Any, List, Optional

logger = logging.getLogger(__name__)

from .medcat2_adapter import MedCAT2Adapter, _detect_language


class SlotExtractor:
    """
    슬롯 추출기 (다국어 지원)
    
    사용자 질의에서 다음 정보를 추출:
    - 인구통계학적 정보 (나이, 성별)
    - 질환 (conditions)
    - 증상 (symptoms)
    - 수치 (vitals, labs)
    - 복용 약물 (medications)
    
    다국어 지원:
    - 한국어 입력 자동 감지 및 영어 번역
    - 추출된 엔티티에 한국어/영어 이름 동시 제공
    """
    
    def __init__(
        self, 
        use_medcat2: bool = True, 
        medcat2_model_path: Optional[str] = None,
        use_multilingual: bool = True,
        use_neural_translation: bool = True,
        use_dict_translation: bool = True
    ):
        """
        Args:
            use_medcat2: MedCAT2 사용 여부
            medcat2_model_path: MedCAT2 모델 경로
            use_multilingual: 다국어 파이프라인 사용 여부
            use_neural_translation: Helsinki-NLP 신경망 번역 사용 여부
            use_dict_translation: 사전 기반 번역 사용 여부
        """
        self.use_medcat2 = use_medcat2
        self.use_multilingual = use_multilingual
        self.use_neural_translation = use_neural_translation
        self.use_dict_translation = use_dict_translation
        self.medcat2_adapter = None
        
        if use_medcat2:
            try:
                self.medcat2_adapter = MedCAT2Adapter(model_path=medcat2_model_path)
            except Exception as e:
                logger.warning(f"MedCAT2 초기화 실패: {e}")
                self.medcat2_adapter = None
        
        # 기본 키워드 (MedCAT2 실패 시 대체용)
        self.condition_keywords = [
            "고혈압", "당뇨", "당뇨병", "고지혈증", "신장병", "만성콩팥병",
            "천식", "감기", "폐염", "폐렴", "diabetes", "hypertension"
        ]
        self.symptom_keywords = [
            "흉통", "가슴 통증", "호흡곤란", "어지럼", "두통", "속쓰림",
            "심계항진", "부종", "기침", "발열", "메스꺼움", "구토"
        ]
        self.medication_keywords = [
            "Metformin", "Simvastatin", "Amlodipine", "Atorvastatin",
            "메트포르민", "스타틴", "아스피린", "인슐린"
        ]
    
    def extract(self, text: str) -> Dict[str, Any]:
        """
        텍스트에서 슬롯 추출 (다국어 자동 지원)
        
        Args:
            text: 사용자 입력 텍스트 (한국어 또는 영어)
        
        Returns:
            {
                'demographics': {'age': int, 'gender': str},
                'conditions': List[Dict],
                'symptoms': List[Dict],
                'vitals': List[Dict],
                'labs': List[Dict],
                'medications': List[Dict],
                'metadata': {  # 다국어 처리 메타데이터
                    'original_text': str,
                    'translated_text': str,
                    'detected_language': str,
                    'translation_method': str
                }
            }
        """
        t = text.strip()
        slots = {
            'conditions': [],
            'symptoms': [],
            'labs': [],
            'vitals': [],
            'medications': [],
            'demographics': {},
            'pregnancy': False,
            'metadata': {}
        }
        
        # 언어 감지
        detected_lang = _detect_language(t)
        
        # 인구통계학적 정보 추출 (원본 텍스트에서)
        self._extract_demographics(t, slots)
        
        # MedCAT2로 엔티티 추출
        if self.medcat2_adapter:
            try:
                # 다국어 파이프라인 사용 여부 결정
                use_multilingual = self.use_multilingual and detected_lang in ('ko', 'mixed')
                
                if use_multilingual:
                    # 다국어 추출 (한국어 자동 번역)
                    medcat_entities = self.medcat2_adapter.extract_entities_multilingual(
                        t,
                        use_neural_translation=self.use_neural_translation,
                        use_dict_translation=self.use_dict_translation
                    )
                    # 메타데이터 저장
                    slots['metadata'] = medcat_entities.get('metadata', {})
                else:
                    # 기본 영어 추출
                    medcat_entities = self.medcat2_adapter.extract_entities(t)
                    slots['metadata'] = {
                        'original_text': t,
                        'translated_text': t,
                        'detected_language': detected_lang,
                        'translation_method': 'none'
                    }
                
                # 질환 (신뢰도 임계값 낮춤 - 다국어 번역 후 정확도 고려)
                confidence_threshold = 0.5 if use_multilingual else 0.7
                for cond in medcat_entities.get('conditions', []):
                    if cond.get('confidence', 0) >= confidence_threshold:
                        slots['conditions'].append(cond)
                
                # 증상
                symptom_threshold = 0.4 if use_multilingual else 0.6
                for symp in medcat_entities.get('symptoms', []):
                    if symp.get('confidence', 0) >= symptom_threshold:
                        slots['symptoms'].append(symp)
                
                # 약물
                med_threshold = 0.6 if use_multilingual else 0.8
                for med in medcat_entities.get('medications', []):
                    if med.get('confidence', 0) >= med_threshold:
                        slots['medications'].append(med)
                
                # 수치
                slots['vitals'].extend(medcat_entities.get('vitals', []))
                slots['labs'].extend(medcat_entities.get('labs', []))
                
            except Exception as e:
                logger.debug(f"MedCAT2 추출 오류: {e}")
                slots['metadata']['error'] = str(e)
        
        # 키워드 기반 추출 (보완)
        self._extract_by_keywords(t, slots)
        
        # 중복 제거
        slots['conditions'] = self._dedup_list(slots['conditions'], key=lambda x: x.get('name', ''))
        slots['symptoms'] = self._dedup_list(slots['symptoms'], key=lambda x: x.get('name', ''))
        slots['medications'] = self._dedup_list(slots['medications'], key=lambda x: x.get('name', ''))
        
        return slots
    
    def _extract_demographics(self, text: str, slots: Dict[str, Any]):
        """인구통계학적 정보 추출"""
        # 나이 추출
        age_patterns = [
            (r'(\d{1,2})0대', lambda m: int(m.group(1)) * 10 + 5),  # "50대" -> 55
            (r'나이\s*[:는]?\s*(\d{1,3})', lambda m: int(m.group(1))),
            (r'(?<![0-9/])(\d{1,2})\s*세(?!\s*환자)', lambda m: int(m.group(1))),
            (r'(?<![0-9/])(\d{1,2})\s*살', lambda m: int(m.group(1))),
        ]
        
        for pattern, converter in age_patterns:
            m = re.search(pattern, text)
            if m:
                age = converter(m)
                if 10 <= age <= 120:  # 유효한 나이 범위
                    slots['demographics']['age'] = age
                    break
        
        # 성별 추출
        if '남성' in text or '남자' in text or 'male' in text.lower():
            slots['demographics']['gender'] = 'male'
        elif '여성' in text or '여자' in text or 'female' in text.lower():
            slots['demographics']['gender'] = 'female'
        
        # 임신 여부
        if '임신' in text or 'pregnant' in text.lower():
            slots['pregnancy'] = True
    
    def _extract_by_keywords(self, text: str, slots: Dict[str, Any]):
        """키워드 기반 추출 (보완)"""
        text_lower = text.lower()
        
        # 질환
        for kw in self.condition_keywords:
            if kw.lower() in text_lower:
                if not any(kw.lower() in c.get('name', '').lower() for c in slots['conditions']):
                    slots['conditions'].append({
                        'name': kw,
                        'confidence': 0.5,
                        'source': 'keyword'
                    })
        
        # 증상
        for kw in self.symptom_keywords:
            if kw.lower() in text_lower:
                negated = any(n in text for n in ["없", "아니"])
                if not any(kw.lower() in s.get('name', '').lower() for s in slots['symptoms']):
                    slots['symptoms'].append({
                        'name': kw,
                        'negated': negated,
                        'confidence': 0.5,
                        'source': 'keyword'
                    })
        
        # 약물
        for kw in self.medication_keywords:
            if kw.lower() in text_lower:
                if not any(kw.lower() in m.get('name', '').lower() for m in slots['medications']):
                    slots['medications'].append({
                        'name': kw,
                        'confidence': 0.5,
                        'source': 'keyword'
                    })
    
    @staticmethod
    def _dedup_list(items: List[Dict], key) -> List[Dict]:
        """리스트 중복 제거"""
        seen = set()
        result = []
        for item in items:
            k = key(item)
            if k and k not in seen:
                seen.add(k)
                result.append(item)
        return result

