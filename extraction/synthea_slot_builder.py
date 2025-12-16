"""
Synthea 프로필 카드에서 슬롯 추출 및 요약
환자 1명의 프로필 카드 → 슬롯(truth table) 생성
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import re


class SyntheaSlotBuilder:
    """Synthea 프로필 카드에서 슬롯 추출"""
    
    def __init__(self, config_dir: str = "config/synthea"):
        self.config_dir = Path(config_dir)
        self.condition_priority = self._load_yaml("condition_priority.yaml")
        self.symptom_map = self._load_yaml("symptom_map.yaml")
        self.loinc_map = self._load_yaml("loinc_map.yaml")
    
    def _load_yaml(self, filename: str) -> Dict:
        """YAML 설정 파일 로드"""
        path = self.config_dir / filename
        if not path.exists():
            return {}
        with open(path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def build_slots(self, profile_card: Dict) -> Dict[str, Any]:
        """
        프로필 카드에서 슬롯 추출
        
        Returns:
            Dict with keys: age, sex, primary_condition, comorbidities,
            key_meds, key_vitals, key_labs, major_procedures, chief_symptom
        """
        demographics = profile_card.get('demographics', {})
        clinical = profile_card.get('clinical_summary', {})
        
        # 기본 인구통계
        age = demographics.get('age_years', 0)
        sex_code = demographics.get('sex', 'M')
        sex = "남성" if sex_code == 'M' else "여성"
        
        # 질환 선택 및 우선순위 결정
        conditions = clinical.get('conditions', [])
        primary_condition, comorbidities = self._select_primary_condition(conditions)
        
        # 약물 추출 (active만)
        medications = clinical.get('medications', [])
        key_meds = self._extract_key_medications(medications, primary_condition)
        
        # 바이탈 추출
        vitals = clinical.get('vitals_recent', [])
        key_vitals = self._extract_key_vitals(vitals, demographics)
        
        # 검사 결과 추출
        labs = clinical.get('labs_recent', [])
        key_labs = self._extract_key_labs(labs, primary_condition)
        
        # 시술/수술 추출
        procedures = clinical.get('procedures', [])  # 프로필 카드에 없을 수 있음
        major_procedures = self._extract_major_procedures(procedures)
        
        # 증상 생성 (데이터에 없으므로 룰 기반)
        chief_symptom = self._generate_chief_symptom(primary_condition)
        
        return {
            "age": age,
            "sex": sex,
            "sex_code": sex_code,
            "primary_condition": primary_condition,
            "comorbidities": comorbidities,
            "key_meds": key_meds,
            "key_vitals": key_vitals,
            "key_labs": key_labs,
            "major_procedures": major_procedures,
            "chief_symptom": chief_symptom,
        }
    
    def _select_primary_condition(self, conditions: List[Dict]) -> tuple[str, List[str]]:
        """Primary condition 선택 (우선순위 기반)"""
        if not conditions:
            return "기저질환", []
        
        # Active 상태만 필터링
        active_conditions = [
            c for c in conditions
            if c.get('status', '').lower() == 'active'
        ]
        
        if not active_conditions:
            active_conditions = conditions  # fallback
        
        # 우선순위 점수 계산
        priority_map = self.condition_priority.get('condition_priority', {})
        scored = []
        
        for c in active_conditions:
            name = c.get('name', '').lower()
            # 정확한 매칭 시도
            score = priority_map.get(name, priority_map.get('default', 10))
            
            # 부분 매칭 시도
            if score == priority_map.get('default', 10):
                for key, val in priority_map.items():
                    if key != 'default' and key in name:
                        score = val
                        break
            
            scored.append({
                "name": c.get('name', ''),
                "score": score,
                "onset_date": c.get('onset_date', '')
            })
        
        # 점수 순으로 정렬
        scored.sort(key=lambda x: (-x['score'], x['onset_date']))
        
        if not scored:
            return "기저질환", []
        
        primary = scored[0]['name']
        comorbidities = [s['name'] for s in scored[1:4]]  # 최대 3개
        
        return primary, comorbidities
    
    def _extract_key_medications(self, medications: List[Dict], primary_condition: str) -> List[str]:
        """핵심 약물 추출 (active 상태, primary condition 관련 우선)"""
        if not medications:
            return []
        
        # Active 상태만 필터링
        active_meds = [
            m for m in medications
            if m.get('status', '').lower() == 'active'
        ]
        
        if not active_meds:
            # completed도 포함 (최근 것만)
            active_meds = medications[:3]
        
        # 약물명 정리 (용량 정보 제거)
        med_names = []
        for m in active_meds[:3]:  # 최대 3개
            name = m.get('name', '')
            # "Simvastatin 10 MG Oral Tablet" -> "Simvastatin"
            if ' ' in name:
                name = name.split()[0]
            med_names.append(name)
        
        return med_names
    
    def _extract_key_vitals(self, vitals: List[Dict], demographics: Dict) -> Dict[str, Any]:
        """핵심 바이탈 추출"""
        result = {}
        
        # Demographics에서 직접 추출
        if 'height_cm' in demographics:
            result['height'] = f"{demographics['height_cm']}cm"
        if 'weight_kg' in demographics:
            result['weight'] = f"{demographics['weight_kg']}kg"
        if 'bmi' in demographics:
            result['bmi'] = f"{demographics['bmi']}"
        
        # Vitals에서 추출
        for vital in vitals:
            v_type = vital.get('type', '')
            value = vital.get('value', '')
            unit = vital.get('unit', '')
            
            if v_type == 'blood_pressure':
                # "74/131" 형식 파싱
                if '/' in str(value):
                    parts = str(value).split('/')
                    if len(parts) == 2:
                        result['bp_diastolic'] = f"{parts[0]}mmHg"
                        result['bp_systolic'] = f"{parts[1]}mmHg"
            elif v_type == 'heart_rate':
                result['heart_rate'] = f"{value}{unit}"
            elif v_type == 'respiratory_rate':
                result['respiratory_rate'] = f"{value}{unit}"
        
        return result
    
    def _extract_key_labs(self, labs: List[Dict], primary_condition: str) -> Dict[str, Any]:
        """핵심 검사 결과 추출 (primary condition 기반)"""
        if not labs:
            return {}
        
        # 최신 측정치만 선택 (중복 제거)
        latest_by_name = {}
        for lab in labs:
            name = lab.get('name', '')
            if not name:
                continue
            
            # 최신 것만 유지
            measured_at = lab.get('measured_at', '')
            if name not in latest_by_name or measured_at > latest_by_name[name].get('measured_at', ''):
                latest_by_name[name] = lab
        
        # Primary condition 기반으로 필요한 검사 선택
        condition_lower = primary_condition.lower()
        result = {}
        
        for name, lab in latest_by_name.items():
            name_lower = name.lower()
            value = lab.get('value', '')
            unit = lab.get('unit', '')
            
            # 당뇨 관련
            if ('diabetes' in condition_lower or 'prediabetes' in condition_lower):
                if 'hba1c' in name_lower or 'a1c' in name_lower:
                    result['hba1c'] = f"{value}{unit}"
                if 'glucose' in name_lower:
                    result['glucose'] = f"{value}{unit}"
            
            # 고혈압/심혈관 관련
            if ('hypertension' in condition_lower or 'heart' in condition_lower or 'ischemic' in condition_lower):
                if 'ldl' in name_lower:
                    result['ldl'] = f"{value}{unit}"
                if 'triglyceride' in name_lower or 'tg' in name_lower:
                    result['triglyceride'] = f"{value}{unit}"
            
            # 신장 관련
            if ('kidney' in condition_lower or 'ckd' in condition_lower):
                if 'creatinine' in name_lower or 'cr' in name_lower:
                    result['creatinine'] = f"{value}{unit}"
                if 'egfr' in name_lower or 'gfr' in name_lower:
                    result['egfr'] = f"{value}{unit}"
                if 'potassium' in name_lower or 'k' in name_lower:
                    result['potassium'] = f"{value}{unit}"
            
            # 빈혈 관련
            if 'anemia' in condition_lower:
                if 'hemoglobin' in name_lower or 'hb' in name_lower:
                    result['hemoglobin'] = f"{value}{unit}"
            
            # 지방간 관련
            if 'fatty liver' in condition_lower or 'nafld' in condition_lower:
                if 'alt' in name_lower:
                    result['alt'] = f"{value}{unit}"
                if 'ast' in name_lower:
                    result['ast'] = f"{value}{unit}"
        
        # Fallback: 일반적으로 중요한 검사들
        if not result:
            for name, lab in list(latest_by_name.items())[:4]:
                name_lower = name.lower()
                value = lab.get('value', '')
                unit = lab.get('unit', '')
                if 'hba1c' in name_lower:
                    result['hba1c'] = f"{value}{unit}"
                elif 'glucose' in name_lower:
                    result['glucose'] = f"{value}{unit}"
                elif 'ldl' in name_lower:
                    result['ldl'] = f"{value}{unit}"
                elif 'creatinine' in name_lower:
                    result['creatinine'] = f"{value}{unit}"
        
        return result
    
    def _extract_major_procedures(self, procedures: List[Dict]) -> List[str]:
        """주요 시술/수술 추출 (screening 제외)"""
        if not procedures:
            return []
        
        exclude_keywords = [
            'screening', 'assessment', 'reconciliation', 'counsel',
            'survey', 'questionnaire', 'consultation'
        ]
        
        major = []
        for proc in procedures[:2]:  # 최대 2개
            name = proc.get('name', '') or proc.get('display', '')
            if not name:
                continue
            
            name_lower = name.lower()
            if any(kw in name_lower for kw in exclude_keywords):
                continue
            
            major.append(name)
        
        return major
    
    def _generate_chief_symptom(self, primary_condition: str) -> str:
        """Primary condition에서 증상 생성 (룰 기반)"""
        if not primary_condition or primary_condition == "기저질환":
            return "불편감"
        
        condition_lower = primary_condition.lower()
        symptom_map = self.symptom_map.get('symptom_mapping', {})
        
        # 정확한 매칭 시도
        candidates = symptom_map.get(condition_lower, [])
        
        # 부분 매칭 시도
        if not candidates:
            for key, symptoms in symptom_map.items():
                if key != 'default' and key in condition_lower:
                    candidates = symptoms
                    break
        
        # Fallback
        if not candidates:
            candidates = symptom_map.get('default', ['불편감'])
        
        return candidates[0] if candidates else "불편감"

