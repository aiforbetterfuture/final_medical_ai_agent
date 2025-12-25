# LLM Judge & YAML: 자주 터지는 지점(방지 팁 포함)

이 패치는 아래 "자주 터지는 지점"을 코드로 흡수해서 재발을 줄이는 방향으로 구성했습니다.

## 1) 스키마 불일치 (가장 흔함)
- `configs/eval_rubric.yaml`이 SSOT입니다.
- judge 출력 키는 **faithfulness / answer_relevance / context_use**로 고정.
- 과거 키(예: factuality)가 섞여도 `faithfulness`로 자동 매핑합니다.

## 2) 모델 출력이 JSON이 아님
- system_prompt에서 "JSON만 출력"을 강제.
- 그래도 누락되면 parser가 첫 JSON 객체를 추출합니다.

## 3) Perplexity 의존성/다운로드 문제
- `transformers`+`torch`가 없거나 모델 로드 실패 시:
  - `perplexity=-1.0`, `perplexity_ok=false`로 기록하고 파이프라인은 계속 진행합니다.
- 진짜 perplexity를 쓰려면:
  - `pip install transformers torch`
  - 필요 시 `HF_PERPLEXITY_MODEL=distilgpt2` 설정

## 4) 한/영 혼용 평가
- 언어 자체로 감점하지 않도록 프롬프트에 명시.
- 의미/근거/맥락만 평가합니다.

## 5) PowerShell 줄바꿈(실행 커맨드) 실수
- PowerShell에서 줄바꿈은 `^`가 아니라 **백틱(`)** 을 쓰는 경우가 많습니다.
- 가장 안전한 건 한 줄로 실행하거나 `.ps1` 스크립트로 감싸는 방식입니다.
