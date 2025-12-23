from __future__ import annotations

"""
TS+TL FUSION TUNING (EVAL)
- 목표: (1) quota 모드가 "진짜로" 적용되는지, (2) TL self-hit 제거가 정상인지,
       (3) TS가 TL dom+qtype miss를 얼마나 rescue 하는지 빠르게 수치로 확인.

이 파일은 PowerShell에서 그대로 실행하는 "스크립트" 형태라서,
repo root가 sys.path에 자동으로 안 들어가는 문제가 있습니다.
-> 아래에서 sys.path에 repo root를 강제로 추가해서 import 오류를 막습니다.
"""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


# -----------------------------
# sys.path fix (script execution)
# -----------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _force_utf8_stdio() -> None:
    # Windows(cp949)에서 print가 죽는 문제 방지
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


# -----------------------------
# optional dependency: keyword_recall
# -----------------------------
try:
    from experiments.retrieval_tuning.metrics import keyword_recall  # type: ignore
except Exception:
    def keyword_recall(answer: str, ctx_texts: List[str]) -> float:  # fallback
        # 매우 단순한 fallback: 공백 토큰 overlap 비율
        a = set((answer or "").split())
        if not a:
            return 0.0
        c = set(" ".join(ctx_texts or []).split())
        return len(a & c) / max(1, len(a))


# -----------------------------
# imports from retrieval package
# -----------------------------
from retrieval.aihub_flat.config import AIHubIndexConfig  # noqa: E402
from retrieval.aihub_flat.fused_retriever import AIHubFusedRetriever  # noqa: E402


def load_evalset(path: Path, limit: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_index_dir_from_paths_yaml(paths_yaml: Path) -> Optional[Path]:
    """
    configs/aihub_71874_paths.yaml에서 output.index_dir을 읽어옵니다.
    (없으면 None)
    """
    try:
        d = yaml.safe_load(paths_yaml.read_text(encoding="utf-8")) or {}
        idx = d.get("output", {}).get("index_dir")
        return Path(idx) if idx else None
    except Exception:
        return None


def is_ts(item: Dict[str, Any]) -> bool:
    src = str(item.get("source", ""))
    if src.startswith("ts_"):
        return True
    st = str(item.get("source_type", ""))
    return st.lower().startswith("ts")


def fuse_quota(
    tl_hits: List[Dict[str, Any]],
    ts_hits: List[Dict[str, Any]],
    *,
    out_k: int,
    tl_quota: int,
    ts_quota: int,
    quota_strategy: str,
) -> List[Dict[str, Any]]:
    """
    quota fusion:
      - tl_first: TL -> TS 순으로 quota 채운 뒤, 남는 out_k는 score로 merge
      - ts_first: TS -> TL
      - interleave: TL/TS 번갈아 quota 채움
    """
    tl = list(tl_hits)
    ts = list(ts_hits)

    out: List[Dict[str, Any]] = []

    def take(src_list: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
        if n <= 0:
            return []
        return src_list[:n]

    if quota_strategy == "ts_first":
        out.extend(take(ts, ts_quota))
        out.extend(take(tl, tl_quota))
    elif quota_strategy == "interleave":
        i = j = 0
        tl_taken = ts_taken = 0
        while (tl_taken < tl_quota or ts_taken < ts_quota) and len(out) < out_k:
            if tl_taken < tl_quota and i < len(tl):
                out.append(tl[i]); i += 1; tl_taken += 1
            if ts_taken < ts_quota and j < len(ts) and len(out) < out_k:
                out.append(ts[j]); j += 1; ts_taken += 1
        # 잔여 채우기(점수 기준)
        tl_rest = tl[i:]
        ts_rest = ts[j:]
        rest = sorted(tl_rest + ts_rest, key=lambda x: float(x.get("score", 0.0)), reverse=True)
        out.extend(rest[: max(0, out_k - len(out))])
    else:  # default: tl_first
        out.extend(take(tl, tl_quota))
        out.extend(take(ts, ts_quota))

    # out_k 강제
    if len(out) > out_k:
        out = out[:out_k]
    elif len(out) < out_k:
        # 남은 자리는 점수로 merge해서 채움
        used_ids = set()
        for x in out:
            used_ids.add(id(x))
        rest = [x for x in (tl + ts) if id(x) not in used_ids]
        rest = sorted(rest, key=lambda x: float(x.get("score", 0.0)), reverse=True)
        out.extend(rest[: max(0, out_k - len(out))])

    return out


def tl_hits_self_filter(tl_raw: List[Dict[str, Any]], qa_id: Any) -> Tuple[List[Dict[str, Any]], int]:
    """qa_id self-hit 제거. 제거된 개수 리턴."""
    if qa_id is None:
        return tl_raw, 0
    removed = 0
    out = []
    for x in tl_raw:
        if x.get("qa_id") == qa_id:
            removed += 1
            continue
        out.append(x)
    return out, removed


def tl_match_domain(x: Dict[str, Any], domain_id: Any) -> bool:
    if domain_id is None:
        return False
    mid = x.get("domain_id", x.get("domain"))
    try:
        return int(mid) == int(domain_id)
    except Exception:
        return False


def tl_match_domqt(x: Dict[str, Any], domain_id: Any, q_type: Any) -> bool:
    if not tl_match_domain(x, domain_id):
        return False
    try:
        return int(x.get("q_type")) == int(q_type)
    except Exception:
        return False


def ts_support(answer: str, ts_items: List[Dict[str, Any]]) -> bool:
    ctx = [(x.get("text") or "") for x in ts_items]
    return keyword_recall(answer, ctx) > 0.0


def main() -> None:
    _force_utf8_stdio()

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/aihub_71874_paths.yaml", help="paths yaml (index_dir 포함)")
    ap.add_argument("--evalset", default="experiments/retrieval_tuning/eval_tl.jsonl")
    ap.add_argument("--limit", type=int, default=200)

    # 기본값 고정(원하신 셋)
    ap.add_argument("--kdoc", type=int, default=30)
    ap.add_argument("--kts", type=int, default=4)

    ap.add_argument("--fusion_mode", choices=["quota", "rrf"], default="quota")
    ap.add_argument("--out_k", type=int, default=22)
    ap.add_argument("--tl_quota", type=int, default=20)
    ap.add_argument("--ts_quota", type=int, default=2)
    ap.add_argument("--quota_strategy", choices=["tl_first", "ts_first", "interleave"], default="tl_first")

    ap.add_argument("--no_self_hit", action="store_true", default=True, help="TL self-hit 제거 (default: True)")
    ap.add_argument("--report", default="experiments/retrieval_tuning/fusion_tuning_report.json")
    args = ap.parse_args()

    paths_yaml = Path(args.config)
    eval_path = Path(args.evalset)

    rows = load_evalset(eval_path, args.limit)

    # config 준비
    cfg = AIHubIndexConfig.default()
    idx_dir = load_index_dir_from_paths_yaml(paths_yaml)
    if idx_dir:
        object.__setattr__(cfg, "index_dir", idx_dir)

    object.__setattr__(cfg, "topk_coarse_docs", int(args.kdoc))
    object.__setattr__(cfg, "topk_ts_final", int(args.kts))
    object.__setattr__(cfg, "topk_tl_final", int(args.kdoc))  # TL pool을 kdoc만큼 뽑아 평가

    # build retriever
    r = AIHubFusedRetriever.build(cfg)

    # header
    print("\n================ TS+TL FUSION TUNING (EVAL) ================")
    print(f"Config        : {paths_yaml.as_posix()}")
    print(f"Evalset       : {eval_path.as_posix()}")
    print(f"Rows          : {len(rows):,}")
    print(f"TS index      : ts_coarse (backend=flat)")
    print(f"TL index      : tl_coarse (backend=flat)")
    print(f"kts/kdoc      : {args.kts}/{args.kdoc}")
    print(f"out_k         : {args.out_k}")
    print(f"fusion_mode   : {args.fusion_mode}")
    print(f"tl_quota      : {args.tl_quota}")
    print(f"ts_quota      : {args.ts_quota}")
    print(f"quota_strategy: {args.quota_strategy}")
    print(f"no_self_hit(TL): {bool(args.no_self_hit)}")
    print("===========================================================\n")

    # counters
    n_total = 0
    missing_qa = 0

    ts_support_cnt = 0
    ts_in_fused_support_cnt = 0
    ts_rescue_cnt = 0

    tl_qaid_hit_cnt = 0
    tl_domain_hit_cnt = 0
    tl_domqt_hit_cnt = 0
    tl_domqt_hit_at_quota_cnt = 0

    fused_any_cnt = 0
    fused_domqt_cnt = 0

    tl_top1_self_raw = 0
    tl_top1_self_filt = 0
    self_skipped_total = 0

    per_rows: List[Dict[str, Any]] = []

    for ex in rows:
        n_total += 1
        q = ex.get("question") or ex.get("q") or ""
        ans = ex.get("answer") or ex.get("a") or ""
        qa_id = ex.get("qa_id")
        dom = ex.get("domain_id")
        qt = ex.get("q_type")

        if qa_id is None:
            missing_qa += 1

        # base pools (no fusion)
        # - ts_context: topk_fine = kts
        ts_hits = r.retrieve_ts_only(q)
        # - tl_hints: topk = kdoc
        tl_raw = r.retrieve_tl(q, use_opts=False, topk=int(args.kdoc), domain_id=dom)

        # self-hit raw
        if tl_raw and qa_id is not None and tl_raw[0].get("qa_id") == qa_id:
            tl_top1_self_raw += 1

        # self-hit filter
        tl_hits = tl_raw
        removed = 0
        if args.no_self_hit:
            tl_hits, removed = tl_hits_self_filter(tl_raw, qa_id)
            if removed > 0:
                self_skipped_total += 1

        # self-hit filt top1
        if tl_hits and qa_id is not None and tl_hits[0].get("qa_id") == qa_id:
            tl_top1_self_filt += 1

        # TL hit metrics (pool)
        tl_qaid_hit = any((qa_id is not None and x.get("qa_id") == qa_id) for x in tl_hits)
        tl_domain_hit = any(tl_match_domain(x, dom) for x in tl_hits)
        tl_domqt_hit = any(tl_match_domqt(x, dom, qt) for x in tl_hits)

        if tl_qaid_hit:
            tl_qaid_hit_cnt += 1
        if tl_domain_hit:
            tl_domain_hit_cnt += 1
        if tl_domqt_hit:
            tl_domqt_hit_cnt += 1

        # TL hit at quota (top tl_quota within TL pool)
        tl_hits_at_quota = tl_hits[: int(args.tl_quota)]
        tl_domqt_hit_at_quota = any(tl_match_domqt(x, dom, qt) for x in tl_hits_at_quota)
        if tl_domqt_hit_at_quota:
            tl_domqt_hit_at_quota_cnt += 1

        # TS support (pool)
        ts_sup = ts_support(ans, ts_hits)
        if ts_sup:
            ts_support_cnt += 1

        # fuse
        if args.fusion_mode == "quota":
            fused = fuse_quota(
                tl_hits, ts_hits,
                out_k=int(args.out_k),
                tl_quota=int(args.tl_quota),
                ts_quota=int(args.ts_quota),
                quota_strategy=str(args.quota_strategy),
            )
        else:
            # rrf 모드: 기존 코드/함수에 의존하지 않기 위해 아주 단순 RRF 구현
            def _key_fn(item: Dict[str, Any]) -> str:
                if is_ts(item):
                    return f"TS::{item.get('doc_id')}::{item.get('chunk_id')}"
                return f"TL::{item.get('qa_id')}"
            def _rrf(list_hits: List[Dict[str, Any]], k: int) -> Dict[str, float]:
                scores: Dict[str, float] = {}
                for rank, it in enumerate(list_hits, 1):
                    key = _key_fn(it)
                    scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
                return scores
            rrf_k = int(getattr(cfg, "rrf_k", 60))
            ts_map = _rrf(ts_hits, rrf_k)
            tl_map = _rrf(tl_hits, rrf_k)
            merged: Dict[str, float] = {}
            for k, v in ts_map.items():
                merged[k] = merged.get(k, 0.0) + float(getattr(cfg, "weight_ts", 1.0)) * v
            for k, v in tl_map.items():
                merged[k] = merged.get(k, 0.0) + float(getattr(cfg, "weight_tl", 0.6)) * v
            # pick items by merged score
            key2item: Dict[str, Dict[str, Any]] = {}
            for it in ts_hits + tl_hits:
                key = _key_fn(it)
                if key not in key2item:
                    key2item[key] = it
            fused_keys = sorted(merged.items(), key=lambda kv: kv[1], reverse=True)[: int(args.out_k)]
            fused = [key2item[k] for k, _ in fused_keys if k in key2item]

        # TS in fused support
        ts_in_fused = [x for x in fused if is_ts(x)]
        ts_in_fused_sup = ts_support(ans, ts_in_fused) if ts_in_fused else False
        if ts_in_fused_sup:
            ts_in_fused_support_cnt += 1

        # fused dom+qtype (TL only, but "실제로 fused에 포함된 TL" 기준)
        tl_in_fused = [x for x in fused if not is_ts(x)]
        fused_domqt = any(tl_match_domqt(x, dom, qt) for x in tl_in_fused)
        if fused_domqt:
            fused_domqt_cnt += 1

        # rescue: TL dom+qt miss + TS_in_fused_support
        rescue = (not fused_domqt) and ts_in_fused_sup
        if rescue:
            ts_rescue_cnt += 1

        # fused any: dom+qt hit OR rescue
        fused_any = fused_domqt or rescue
        if fused_any:
            fused_any_cnt += 1

        per_rows.append({
            "qa_id": qa_id,
            "domain_id": dom,
            "q_type": qt,
            "ts_support": ts_sup,
            "ts_in_fused_support": ts_in_fused_sup,
            "tl_domqt_hit_pool": tl_domqt_hit,
            "tl_domqt_hit_at_quota": tl_domqt_hit_at_quota,
            "fused_domqt": fused_domqt,
            "fused_any": fused_any,
            "rescue": rescue,
        })

    n = max(1, n_total)

    def rate(x: int) -> float:
        return x / n

    # report json
    report = {
        "args": vars(args),
        "n": n_total,
        "missing_qa_id": missing_qa,
        "metrics": {
            "ts_support_rate": rate(ts_support_cnt),
            "ts_in_fused_support_rate": rate(ts_in_fused_support_cnt),
            "ts_rescue_rate": rate(ts_rescue_cnt),
            "tl_qa_id_hit_rate": rate(tl_qaid_hit_cnt),
            "tl_domain_hit_rate": rate(tl_domain_hit_cnt),
            "tl_domain_qtype_hit_rate": rate(tl_domqt_hit_cnt),
            "tl_domqt_hit_at_quota_rate": rate(tl_domqt_hit_at_quota_cnt),
            "fused_any_rate": rate(fused_any_cnt),
            "fused_domqt_rate": rate(fused_domqt_cnt),
            "tl_top1_self_hit_raw_rate": rate(tl_top1_self_raw),
            "tl_top1_self_hit_filt_rate": rate(tl_top1_self_filt),
            "self_skipped_total": self_skipped_total,
        },
        "rows": per_rows[: min(200, len(per_rows))],  # 너무 커지는 것 방지
    }

    out_report = Path(args.report)
    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # print results (기존 출력 형식과 최대한 유사)
    print("\n================= RESULT =================")
    print(f"Rows used                 : {n_total} (missing qa_id={missing_qa})")
    print(f"TS support-rate           : {rate(ts_support_cnt):.4f} ({ts_support_cnt}/{n_total})  [base@kts={args.kts}]")
    print(f"TS in-fused support-rate  : {rate(ts_in_fused_support_cnt):.4f} ({ts_in_fused_support_cnt}/{n_total})  [in_fused@ts_quota={args.ts_quota}]")
    print(f"TS rescue-rate            : {rate(ts_rescue_cnt):.4f} ({ts_rescue_cnt}/{n_total})  [dom+qt miss & TS_in_fused_support]")
    print(f"TL qa_id hit-rate         : {rate(tl_qaid_hit_cnt):.4f} ({tl_qaid_hit_cnt}/{n_total})")
    print(f"TL domain hit-rate        : {rate(tl_domain_hit_cnt):.4f} ({tl_domain_hit_cnt}/{n_total})")
    print(f"TL domain+qtype hit-rate  : {rate(tl_domqt_hit_cnt):.4f} ({tl_domqt_hit_cnt}/{n_total})  [@kdoc={args.kdoc}]")
    print(f"TL dom+qtype hit@tl_quota : {rate(tl_domqt_hit_at_quota_cnt):.4f} ({tl_domqt_hit_at_quota_cnt}/{n_total})  [@tl_quota={args.tl_quota}]")
    print(f"FUSED any-rate            : {rate(fused_any_cnt):.4f} ({fused_any_cnt}/{n_total})")
    print(f"FUSED dom+qtype rate      : {rate(fused_domqt_cnt):.4f} ({fused_domqt_cnt}/{n_total})")
    print(f"TL top1 self-hit raw rate : {rate(tl_top1_self_raw):.4f} ({tl_top1_self_raw}/{n_total})")
    print(f"TL top1 self-hit filt rate: {rate(tl_top1_self_filt):.4f} ({tl_top1_self_filt}/{n_total})")
    print(f"self_skipped_total        : {self_skipped_total}")
    print(f"report                    : {out_report.as_posix()}")
    print("==========================================\n")


if __name__ == "__main__":
    main()
