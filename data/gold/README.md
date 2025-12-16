# Gold annotation format

골드 파일은 JSONL로 관리합니다.

각 라인:
```json
{"id":"ex1","text":"...","entities":[
  {"start":0,"end":3,"text":"흉통","label":"SYMPTOM","code":null},
  {"start":10,"end":14,"text":"심근경색","label":"DISEASE","code":"C0027051"}
]}
```

- start/end는 **문자 인덱스(파이썬 슬라이스 기준)** 입니다.
- KM-BERT는 NER만 수행하므로 code는 보통 null.
- MedCAT/QuickUMLS 비교(링킹 평가)를 원하면 code(UMLS CUI)를 채워주세요.
