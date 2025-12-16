# Claude Sonnet 4 기본 LLM 설정 가이드

## [클립보드] 개요

평가 결과에 따라 스캐폴드의 기본 LLM을 **Claude Sonnet 4**로 설정했습니다.

**변경 이유**:
- Inference Memory 27% 더 높음 (0.695 -> 0.883)
- CMR 50% 더 안전함 (0.400 -> 0.200)
- Delta P (개인화 효과) 개선 (−0.053 -> 0.023)
- 의학 정보의 정확성과 안전성이 우수

---

## [수정] 변경된 설정

### 1. 기본 LLM 설정

**파일**: `config/model_config.yaml`

```yaml
llm:
  provider: "anthropic"  # Claude를 기본 모델로 사용
  model: "claude-sonnet-4-20250514"  # Claude Sonnet 4 (최신)
  temperature: 0.2

# 대체 LLM 설정 (Claude 토큰 초과 또는 연결 불량 시 자동 전환)
llm_fallback:
  provider: "openai"  # GPT-4o-mini로 fallback
  model: "gpt-4o-mini"
  temperature: 0.2
```

### 2. LLM 우선순위 변경

**파일**: `agent/nodes/answer.py`

**변경 내용**:
```python
# 이전: OpenAI > Anthropic > Gemini
# 현재: Claude > GPT > Gemini

if os.getenv("ANTHROPIC_API_KEY"):
    client = MCPLLMClient(provider="anthropic", model="claude-sonnet-4-20250514", temperature=0.2)
elif os.getenv("OPENAI_API_KEY"):
    client = MCPLLMClient(provider="openai", model="gpt-4o-mini", temperature=0.2)
elif os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"):
    client = MCPLLMClient(provider="gemini", temperature=0.2)
```

### 3. Claude 모델 Fallback 리스트 업데이트

**파일**: `llm/mcp_client.py`

```python
available_models = [
    "claude-sonnet-4-20250514",  # Claude Sonnet 4 (최신, 우선)
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620",
    "claude-3-haiku-20240307",  # 빠른 응답용
    # ... 기타 모델
]
```

---

## [차트] 성능 비교

### GPT-4o-mini vs Claude Sonnet 4 (5개 샘플)

| 지표 | GPT-4o-mini | Claude Sonnet 4 | 차이 | 승자 |
|------|-------------|-----------------|------|------|
| **Inference Memory** | 0.695 | **0.883** | +27% | [이모지] Claude |
| **Slot F1** | 0.227 | 0.227 | 0% | 동일 |
| **CMR** | 0.400 | **0.200** | −50% | [이모지] Claude |
| **Context Retention** | 0.600 | 0.600 | 0% | 동일 |
| **Delta P** | −0.053 | **0.023** | +145% | [이모지] Claude |
| **답변 길이** | ~600자 | ~919자 | +53% | Claude |

**결론**: Claude가 4개 지표에서 우수하거나 동등

---

## [새로고침] Fallback 동작

### 자동 전환 시나리오

1. **Claude API 호출 실패** -> GPT-4o-mini로 자동 전환
2. **Claude 토큰 한도 초과** -> GPT-4o-mini 사용
3. **Claude 모델 404 오류** -> 다른 Claude 모델 시도 후 GPT로 전환
4. **모든 API 실패** -> 템플릿 기반 답변으로 fallback

### Fallback 순서

```
1차: Claude Sonnet 4 (claude-sonnet-4-20250514)
  [감소] 실패 시
2차: 다른 Claude 모델 (claude-3-5-sonnet, claude-3-haiku)
  [감소] 실패 시
3차: GPT-4o-mini (openai)
  [감소] 실패 시
4차: Gemini (google)
  [감소] 실패 시
5차: 템플릿 기반 답변
```

---

## [주의]️ 유지된 파일 (수정 제외)

다음 파일들은 **특정 LLM 용도로 구분되어 있으므로 수정하지 않음**:

### 테스트 및 비교 스크립트
- `scripts/test_api_keys.py` - 3개 LLM API 키 검증
- `scripts/test_claude_simple.py` - Claude 전용 테스트
- `scripts/test_medcat2_improved_metrics.py` - GPT 전용 평가
- `scripts/test_medcat2_claude_metrics.py` - Claude 전용 평가
- `scripts/compare_metrics_with_medcat2.py` - 비교 분석

### 예제 및 문서
- `examples/*` - 특정 LLM 사용 예제
- `docs/MEDCAT2_*` - 문서 파일 (LLM별 성능 비교 포함)

---

## [실행] 사용 방법

### 1. 기본 사용 (Claude Sonnet 4 자동 사용)

```python
from agent.graph_langgraph import AgentGraphLangGraph

# 기본 설정으로 초기화 (Claude Sonnet 4 자동 사용)
agent = AgentGraphLangGraph(
    cfg_paths={"corpus_config": "config/corpus_config.yaml"}
)

# 질의 실행
result = agent.run(
    user_text="당뇨병 환자가 주의해야 할 약물은?",
    profile_data={"conditions": ["당뇨병"]},
    mode="hybrid"
)
```

### 2. 특정 LLM 지정 (필요 시)

```python
from llm.mcp_client import MCPLLMClient

# GPT-4o-mini 사용 (명시적 지정)
llm_client = MCPLLMClient(provider="openai", model="gpt-4o-mini")

# Claude Sonnet 4 사용 (명시적 지정)
llm_client = MCPLLMClient(provider="anthropic", model="claude-sonnet-4-20250514")
```

### 3. Fallback 동작 확인

```python
from agent.nodes.answer import _get_mcp_client

# 자동 fallback 테스트
client = _get_mcp_client()
print(f"사용 중인 LLM: {client.provider.value} - {client.model}")
```

---

## [메모] 환경 변수 설정

**.env 파일**:

```env
# LLM API 키 (우선순위: Claude > GPT > Gemini)
ANTHROPIC_API_KEY=sk-ant-api...  # Claude (기본)
OPENAI_API_KEY=sk-proj-...        # GPT (fallback)
GOOGLE_API_KEY=AIza...            # Gemini (optional)

# MEDCAT2 설정
MEDCAT2_MODEL_PATH=models/v2_Snomed2025_MIMIC_IV_bbe806e192df009f
MEDCAT2_LICENSE_CODE=NLM-10000060827
MEDCAT2_API_KEY=84605af4-35bb-4292-90e7-19f906c2d38f

# LangSmith 트레이싱 (optional, 성능 향상을 위해 비활성화 권장)
LANGCHAIN_TRACING_V2=false
```

---

## [목표] 검증

### 설정 확인 스크립트

```bash
# API 키 검증
python scripts/test_api_keys.py

# Claude 간단 테스트
python scripts/test_claude_simple.py

# 5대 지표 평가 (Claude)
python scripts/test_medcat2_claude_metrics.py \
    --model-pack models/medcat2/medcat2_supervised_trained_1e0ceff2c20a0a02.zip \
    --sample-size 5
```

### 기대 결과

```
[완료] Claude Sonnet 4 API 정상 작동
[완료] Inference Memory: 0.883 (GPT 대비 +27%)
[완료] CMR: 0.200 (GPT 대비 −50%, 더 안전)
[완료] Delta P: 0.023 (개인화 효과 양수)
```

---

## [차트] 비용 고려사항

### 예상 비용 비교 (2024년 기준)

| 모델 | Input | Output | 상대 비용 |
|------|-------|--------|-----------|
| **Claude Sonnet 4** | $3/M tokens | $15/M tokens | 100% |
| **GPT-4o-mini** | $0.15/M tokens | $0.60/M tokens | ~20% |
| **Claude Haiku** | $0.25/M tokens | $1.25/M tokens | ~10% |

**권장 전략**:
- 복잡한 의학 질의 -> Claude Sonnet 4
- 간단한 질문 -> GPT-4o-mini (fallback 자동 전환)
- 대량 처리 -> Claude Haiku 또는 GPT-4o-mini

---

## [검색] 추가 최적화 방안

### 1. 답변 길이 제어

Claude가 답변이 더 긴 편이므로 토큰 비용 절감을 위해:

```python
system_prompt += """
답변은 500-800자 이내로 간결하게 작성하세요.
핵심 정보만 포함하고 불필요한 반복을 피하세요.
"""
```

### 2. 비용 모니터링

```python
# 비용 추적 (optional)
response_metadata = llm_client.get_last_response_metadata()
tokens_used = response_metadata.get("usage", {})
print(f"토큰 사용: {tokens_used}")
```

### 3. 하이브리드 전략

```python
# 복잡도에 따라 모델 자동 선택
if complexity_score > 0.7:
    model = "claude-sonnet-4-20250514"
else:
    model = "gpt-4o-mini"
```

---

## [완료] 변경 완료 체크리스트

- [x] `config/model_config.yaml` - Claude Sonnet 4 기본 설정
- [x] `agent/nodes/answer.py` - LLM 우선순위 변경 (Claude > GPT > Gemini)
- [x] `llm/mcp_client.py` - Claude 모델 fallback 리스트 업데이트
- [x] 테스트 스크립트 유지 (LLM별 용도 구분)
- [x] 문서화 완료

---

**작성일**: 2024년
**버전**: 2.0
**상태**: 설정 완료 [완료]

