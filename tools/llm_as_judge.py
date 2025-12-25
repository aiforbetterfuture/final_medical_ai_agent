from __future__ import annotations

"""LLM-as-a-judge utilities (핵심 3-메트릭: faithfulness, answer_relevance, perplexity).

이 모듈은 의도적으로 방어적으로 작성되었습니다:
- 가장 흔한 실패 모드는 스키마 드리프트 (YAML의 키와 judge 반환 키가 다름)
- 그 다음은 non-JSON 모델 출력으로 json.loads가 충돌하는 경우

핵심 메트릭 (고정):
  - faithfulness (0..1, 높을수록 좋음): TS 근거로 뒷받침되는가?
  - answer_relevance (0..1, 높을수록 좋음): 질문에 실제로 답하는가?
  - perplexity (float, 낮을수록 좋음): 로컬에서 가능하면 계산
선택적:
  - context_use (0..1, 높을수록 좋음): 환자 맥락 활용

Perplexity:
  - `transformers` + `torch`가 있으면 작은 causal LM으로 계산 (기본: 'distilgpt2')
  - 없으면 perplexity = -1.0으로 기록하고 이유를 남김
"""

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import yaml


# ---------------------------
# 헬퍼 함수
# ---------------------------

def _clamp01(x: Any) -> float:
    """0~1 범위로 제한."""
    try:
        v = float(x)
    except Exception:
        return 0.0
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _extract_first_json_object(text: str) -> Optional[dict]:
    """모델 출력에서 JSON 파싱 시도.

    1) 직접 json.loads
    2) ```json ... ``` 펜스 블록 찾기
    3) 첫 {...} 블록 찾기
    """
    s = (text or "").strip()
    if not s:
        return None

    # Try fenced code block first
    m = re.search(r"```json\s*(\{.*?\})\s*```", s, flags=re.DOTALL | re.IGNORECASE)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass

    # Try direct parse
    try:
        return json.loads(s)
    except Exception:
        pass

    # Try finding first JSON object
    m = re.search(r"(\{.*\})", s, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            return None
    return None


# ---------------------------
# Perplexity (선택적)
# ---------------------------

def _try_compute_perplexity_hf(text: str, model_name: str) -> Tuple[float, str]:
    """Returns (perplexity, source_note). 실패 시 예외 발생."""
    import torch  # type: ignore
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForCausalLM.from_pretrained(model_name)
    mdl.to(device)
    mdl.eval()

    enc = tok(text, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)

    with torch.no_grad():
        out = mdl(input_ids=input_ids, labels=input_ids)
        loss = out.loss

    ppl = float(torch.exp(loss).detach().cpu().item())
    return ppl, f"hf:{model_name}@{device}"


def compute_perplexity(text: str, model_name: Optional[str] = None) -> Dict[str, Any]:
    """Perplexity 계산 (최선 노력).

    Output:
      {"perplexity": float, "perplexity_source": str, "perplexity_ok": bool}
    """
    if model_name is None:
        model_name = os.getenv("HF_PERPLEXITY_MODEL", "distilgpt2")

    text = (text or "").strip()
    if not text:
        return {"perplexity": -1.0, "perplexity_source": "empty_text", "perplexity_ok": False}

    try:
        ppl, src = _try_compute_perplexity_hf(text, model_name=model_name)
        return {"perplexity": ppl, "perplexity_source": src, "perplexity_ok": True}
    except Exception as e:
        return {
            "perplexity": -1.0,
            "perplexity_source": f"unavailable:{type(e).__name__}",
            "perplexity_ok": False,
        }


# ---------------------------
# Judge 설정 (Rubric 기반)
# ---------------------------

@dataclass
class LLMJudgeConfig:
    """LLM Judge 설정 (eval_rubric.yaml에서 로드 가능)."""
    enabled: bool = True
    model: str = "gpt-4o-mini"
    temperature: float = 0.0
    max_tokens: int = 800
    timeout_s: int = 60
    threshold: float = 0.75

    system_prompt: str = (
        "당신은 의료 QA 품질 심사관입니다.\n"
        "근거(TS evidence)만을 신뢰하며, TL 힌트는 참고용으로만 봅니다.\n"
        "반드시 유효한 JSON만 출력하세요. (마크다운/설명/코드블록 금지)\n"
        "점수는 0~1 사이의 실수이며, 보수적으로 채점하세요.\n"
        "- faithfulness: TS 근거로 뒷받침되는가? (환각/추측이면 크게 감점)\n"
        "- answer_relevance: 사용자 질문에 실제로 답하는가?\n"
        "- context_use: 제공된 환자 맥락(나이/성별/병력/복약 등)을 적절히 활용하는가?\n"
        "※ 한국어/영어가 섞여도 언어 자체로 감점하지 말고, 의미/정확성으로만 평가하세요.\n"
    )

    @classmethod
    def from_rubric(cls, rubric_path: str) -> "LLMJudgeConfig":
        """eval_rubric.yaml에서 설정 로드."""
        try:
            with open(rubric_path, 'r', encoding='utf-8') as f:
                rubric = yaml.safe_load(f)

            llm_judge = rubric.get('llm_judge', {})

            return cls(
                enabled=llm_judge.get('enabled', True),
                model=llm_judge.get('model', 'gpt-4o-mini'),
                temperature=llm_judge.get('temperature', 0.0),
                max_tokens=llm_judge.get('max_tokens', 800),
                timeout_s=llm_judge.get('timeout_s', 60),
                threshold=llm_judge.get('threshold', 0.75),
                system_prompt=llm_judge.get('system_prompt', cls.system_prompt)
            )
        except Exception as e:
            print(f"[WARNING] Failed to load rubric from {rubric_path}: {e}")
            print("[WARNING] Using default LLMJudgeConfig")
            return cls()


def _build_judge_prompt(question: str, answer: str, evidence: str, cfg: LLMJudgeConfig) -> str:
    """Judge 프롬프트 생성."""
    ev = (evidence or "").strip()
    if len(ev) > 8000:
        ev = ev[:8000] + "\n...[TRUNCATED]..."

    q = (question or "").strip()
    a = (answer or "").strip()

    return (
        "다음 스키마로 정확히 JSON을 반환하세요:\n"
        "{\n"
        '  "scores": {"faithfulness": 0.0, "answer_relevance": 0.0, "context_use": 0.0},\n'
        '  "verdict": "pass|fail",\n'
        '  "rationale": "간단한 이유"\n'
        "}\n\n"
        f"질문:\n{q}\n\n"
        f"답변:\n{a}\n\n"
        "근거 스니펫 (TS evidence만; TL은 힌트이며 사실 근거가 아닐 수 있음):\n"
        f"{ev}\n"
    )


# ---------------------------
# OpenAI 클라이언트 어댑터
# ---------------------------

def _call_openai_chat(user_prompt: str, cfg: LLMJudgeConfig) -> str:
    """최소한의 OpenAI chat-completions 호출."""
    from openai import OpenAI  # type: ignore

    client = OpenAI()
    resp = client.chat.completions.create(
        model=cfg.model,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        messages=[
            {"role": "system", "content": cfg.system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        timeout=cfg.timeout_s,
    )
    return resp.choices[0].message.content or ""


# ---------------------------
# 공개 API
# ---------------------------

def judge_one(
    *,
    question: str,
    answer: str,
    evidence: str,
    cfg: Optional[LLMJudgeConfig] = None,
    rubric_path: Optional[str] = None,
    llm_provider: str = "openai",
) -> Dict[str, Any]:
    """단일 (question, answer, evidence) 쌍 평가.

    Args:
        question: 사용자 질문
        answer: 생성된 답변
        evidence: 검색된 근거 (TS context)
        cfg: LLMJudgeConfig (None이면 기본값 또는 rubric에서 로드)
        rubric_path: eval_rubric.yaml 경로 (cfg가 None일 때 사용)
        llm_provider: LLM 제공자 ("openai" 등)

    Returns:
        스키마 안정적인 딕셔너리:
        {
          "scores": {"faithfulness": float, "answer_relevance": float, "context_use": float},
          "perplexity": float,
          "perplexity_ok": bool,
          "perplexity_source": str,
          "verdict": "pass|fail|skip",
          "rationale": str,
          "raw_text": str
        }
    """
    # Config 로드
    if cfg is None:
        if rubric_path:
            cfg = LLMJudgeConfig.from_rubric(rubric_path)
        else:
            cfg = LLMJudgeConfig()

    # 1) Perplexity (최선 노력, 로컬)
    perplexity_model = None
    if rubric_path:
        try:
            with open(rubric_path, 'r', encoding='utf-8') as f:
                rubric = yaml.safe_load(f)
            perplexity_cfg = rubric.get('perplexity', {})
            if perplexity_cfg.get('enabled', True):
                perplexity_model = perplexity_cfg.get('model', 'distilgpt2')
        except Exception:
            pass

    ppl = compute_perplexity(answer, model_name=perplexity_model)

    # 2) LLM judge (faithfulness / answer_relevance / context_use)
    if not cfg.enabled:
        return {
            "scores": {"faithfulness": 0.0, "answer_relevance": 0.0, "context_use": 0.0},
            **ppl,
            "verdict": "skip",
            "rationale": "llm_judge disabled",
            "raw_text": "",
        }

    user_prompt = _build_judge_prompt(question, answer, evidence, cfg)

    raw = ""
    try:
        if llm_provider == "openai":
            raw = _call_openai_chat(user_prompt, cfg)
        else:
            raise RuntimeError(f"Unsupported llm_provider={llm_provider}")
    except Exception as e:
        return {
            "scores": {"faithfulness": 0.0, "answer_relevance": 0.0, "context_use": 0.0},
            **ppl,
            "verdict": "fail",
            "rationale": f"llm_call_failed:{type(e).__name__}",
            "raw_text": raw,
        }

    obj = _extract_first_json_object(raw) or {}
    scores = obj.get("scores") if isinstance(obj.get("scores"), dict) else {}

    # 하위 호환 alias 매핑 (스키마 드리프트 방지)
    alias_map = {
        # 레거시 키 → 표준 키
        "factuality": "faithfulness",
        "relevance": "answer_relevance",
        "completeness": "answer_relevance",  # 일부 모델이 이렇게 반환
        # 표준 키
        "faithfulness": "faithfulness",
        "answer_relevance": "answer_relevance",
        "context_use": "context_use",
    }

    norm_scores = {"faithfulness": 0.0, "answer_relevance": 0.0, "context_use": 0.0}

    # scores dict에서 추출
    for k, v in list(scores.items()):
        dst = alias_map.get(k)
        if dst:
            norm_scores[dst] = _clamp01(v)

    # top-level 필드에서도 추출 (일부 모델이 flat하게 출력)
    for k, dst in alias_map.items():
        if dst in norm_scores and norm_scores[dst] > 0:
            continue
        if k in obj:
            norm_scores[dst] = _clamp01(obj.get(k))

    verdict = obj.get("verdict", "pass")
    if verdict not in ("pass", "fail", "skip"):
        verdict = "pass"

    rationale = obj.get("rationale", "")
    if not isinstance(rationale, str):
        rationale = ""

    return {
        "scores": norm_scores,
        **ppl,
        "verdict": verdict,
        "rationale": rationale,
        "raw_text": raw,
    }


def load_rubric(rubric_path: str) -> Dict[str, Any]:
    """eval_rubric.yaml 로드."""
    with open(rubric_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
