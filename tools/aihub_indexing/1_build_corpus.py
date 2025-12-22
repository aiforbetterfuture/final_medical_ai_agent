
from __future__ import annotations
from pathlib import Path
import argparse, math, time
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import orjson
from tqdm import tqdm
from transformers import AutoTokenizer

from common import (
    load_yaml, ensure_dir, iter_json_objects_from_path,
    guess_ts_language_and_source_type, flatten_text_from_json,
    split_mcq_stem_options, format_options, extract_choice_from_answer,
    DOMAIN_NAME_TO_ID, sha1_text
)

def chunk_by_tokens(text: str, tokenizer, max_tokens: int, overlap_tokens: int) -> list[str]:
    # tokenizer 기반 토큰 길이로 분할
    if not text.strip():
        return []
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return [text.strip()]
    chunks = []
    step = max_tokens - overlap_tokens
    if step <= 0:
        step = max_tokens
    for i in range(0, len(ids), step):
        sub = ids[i:i+max_tokens]
        if not sub:
            break
        chunk_text = tokenizer.decode(sub)
        chunk_text = chunk_text.strip()
        if chunk_text:
            chunks.append(chunk_text)
        if i + max_tokens >= len(ids):
            break
    return chunks

def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    n = 0
    with path.open("ab") as f:
        for r in rows:
            f.write(orjson.dumps(r))
            f.write(b"\n")
            n += 1
    return n

def normalize_passage(text: str) -> str:
    # E5 권장: passage prefix
    return f"passage: {text.strip()}"

def infer_domain_id_from_folder(folder: str) -> Optional[int]:
    # folder like "TL_마취통증의학과"
    name = folder.split("_", 1)[-1]
    return DOMAIN_NAME_TO_ID.get(name)

def build_ts(cfg: dict, tokenizer):
    out_root = Path(cfg["output"]["processed_dir"])
    ensure_dir(out_root)
    shard_size = int(cfg["build"]["shard_size"])
    chunk_cfg = cfg["chunking"]["TS"]

    coarse_rows = []
    fine_rows = []
    shard_idx = 0
    total_written = {"ts_coarse": 0, "ts_fine": 0}

    def flush():
        nonlocal shard_idx, coarse_rows, fine_rows
        if coarse_rows:
            p = out_root / f"ts_coarse_shard{shard_idx:03d}.jsonl"
            total_written["ts_coarse"] += write_jsonl(p, coarse_rows)
            coarse_rows = []
        if fine_rows:
            p = out_root / f"ts_fine_shard{shard_idx:03d}.jsonl"
            total_written["ts_fine"] += write_jsonl(p, fine_rows)
            fine_rows = []
        shard_idx += 1

    for ts_dir in cfg["local_paths"]["TS_dirs"]:
        pdir = Path(ts_dir)
        folder_name = pdir.name
        lang, source_type = guess_ts_language_and_source_type(folder_name)

        # chunk profile
        prof = chunk_cfg["fine_by_source_type"].get(source_type, chunk_cfg["fine_by_source_type"]["other"])
        max_tok = int(prof["max_tokens"])
        ov_tok = int(prof["overlap_tokens"])

        for origin_path, obj, extra in tqdm(iter_json_objects_from_path(pdir), desc=f"TS {folder_name}", unit="file"):
            # title heuristic: try typical keys, else derive from filename
            title = ""
            if isinstance(obj, dict):
                for k in ["title", "doc_title", "paper_title", "name"]:
                    if k in obj and isinstance(obj[k], str):
                        title = obj[k].strip()
                        break

            text = flatten_text_from_json(obj)
            if not text:
                continue

            doc_id = sha1_text(origin_path)
            # coarse: title + prefix tokens
            coarse_max = int(chunk_cfg["coarse_max_tokens"])
            coarse_text = (title + "\n" + text).strip() if title else text
            coarse_chunks = chunk_by_tokens(coarse_text, tokenizer, coarse_max, 0)
            if coarse_chunks:
                coarse_rows.append({
                    "id": f"tsdoc_{doc_id}",
                    "text": normalize_passage(coarse_chunks[0]),
                    "meta": {
                        "source": "TS",
                        "language": lang,
                        "source_type": source_type,
                        "title": title,
                        "doc_id": doc_id,
                        "origin_path": origin_path,
                        **extra
                    }
                })

            # fine chunks
            chunks = chunk_by_tokens(text, tokenizer, max_tok, ov_tok)
            for j, ch in enumerate(chunks):
                fine_rows.append({
                    "id": f"tsch_{doc_id}_{j:04d}",
                    "text": normalize_passage(ch),
                    "meta": {
                        "source": "TS",
                        "language": lang,
                        "source_type": source_type,
                        "title": title,
                        "doc_id": doc_id,
                        "chunk_id": j,
                        "origin_path": origin_path,
                        **extra
                    }
                })

            if (len(coarse_rows) + len(fine_rows)) >= shard_size:
                flush()

    flush()
    return total_written

def build_tl(cfg: dict):
    out_root = Path(cfg["output"]["processed_dir"])
    ensure_dir(out_root)
    shard_size = int(cfg["build"]["shard_size"])

    rows = {"tl_stem": [], "tl_stem_opts": [], "tl_q_full": [], "tl_coarse": []}
    shard_idx = 0
    total_written = {k: 0 for k in rows.keys()}

    def flush():
        nonlocal shard_idx, rows
        for k, buf in rows.items():
            if buf:
                p = out_root / f"{k}_shard{shard_idx:03d}.jsonl"
                total_written[k] += write_jsonl(p, buf)
                rows[k] = []
        shard_idx += 1

    for tl_dir in cfg["local_paths"]["TL_dirs"]:
        pdir = Path(tl_dir)
        folder_name = pdir.name
        domain_id = infer_domain_id_from_folder(folder_name)

        for origin_path, obj, extra in tqdm(iter_json_objects_from_path(pdir), desc=f"TL {folder_name}", unit="file"):
            # 한 파일에 여러 QA가 들어있을 수도/단일 obj일 수도 있어 방어적으로 처리
            qas = []
            if isinstance(obj, list):
                qas = obj
            elif isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
                qas = obj["data"]
            elif isinstance(obj, dict):
                qas = [obj]
            else:
                continue

            for qa in qas:
                if not isinstance(qa, dict):
                    continue
                qa_id = qa.get("qa_id")
                q_type = int(qa.get("q_type", 0) or 0)
                # 파일/폴더 기반 domain_id가 더 신뢰도 높으면 덮어쓰기
                dom = int(qa.get("domain", domain_id or 0) or 0)
                if domain_id:
                    dom = domain_id
                question = str(qa.get("question", "")).strip()
                answer = str(qa.get("answer", "")).strip()
                if not question:
                    continue

                base_meta = {
                    "source": "TL",
                    "qa_id": qa_id,
                    "domain_id": dom,
                    "q_type": q_type,
                    "origin_path": origin_path,
                    **extra
                }

                # full question
                rows["tl_q_full"].append({
                    "id": f"tlq_{qa_id}",
                    "text": normalize_passage(question),
                    "meta": base_meta
                })

                # coarse: stem + short answer(앞 200자)
                short_a = answer[:200].strip()
                coarse_text = question + ("\n\n[SHORT_ANSWER]\n" + short_a if short_a else "")
                rows["tl_coarse"].append({
                    "id": f"tlc_{qa_id}",
                    "text": normalize_passage(coarse_text),
                    "meta": base_meta
                })

                # mcq split
                if cfg["chunking"]["TL"]["mcq_split"] and q_type == 1:
                    stem, opts = split_mcq_stem_options(question)
                    opts_text = format_options(opts)
                    choice = extract_choice_from_answer(answer)
                    meta2 = {**base_meta, "mcq_answer_choice": choice, "mcq_options_n": len(opts)}

                    if stem:
                        rows["tl_stem"].append({
                            "id": f"tls_{qa_id}",
                            "text": normalize_passage(stem),
                            "meta": meta2
                        })
                    if stem and opts_text:
                        rows["tl_stem_opts"].append({
                            "id": f"tlso_{qa_id}",
                            "text": normalize_passage(stem + "\n\n[OPTIONS]\n" + opts_text),
                            "meta": meta2
                        })

                if sum(len(v) for v in rows.values()) >= shard_size:
                    flush()

    flush()
    return total_written

def build_vl(cfg: dict):
    out_root = Path(cfg["output"]["processed_dir"])
    ensure_dir(out_root)
    shard_size = int(cfg["build"]["shard_size"])

    buf = []
    shard_idx = 0
    total = 0

    def flush():
        nonlocal shard_idx, buf, total
        if buf:
            p = out_root / f"vl_question_shard{shard_idx:03d}.jsonl"
            total += write_jsonl(p, buf)
            buf = []
            shard_idx += 1

    for vl_dir in cfg["local_paths"]["VL_dirs"]:
        pdir = Path(vl_dir)
        folder_name = pdir.name
        domain_id = infer_domain_id_from_folder(folder_name.replace("VL_", "TL_", 1))  # 동일 규칙 재활용

        for origin_path, obj, extra in tqdm(iter_json_objects_from_path(pdir), desc=f"VL {folder_name}", unit="file"):
            qas = []
            if isinstance(obj, list):
                qas = obj
            elif isinstance(obj, dict) and "data" in obj and isinstance(obj["data"], list):
                qas = obj["data"]
            elif isinstance(obj, dict):
                qas = [obj]
            else:
                continue

            for qa in qas:
                if not isinstance(qa, dict):
                    continue
                qa_id = qa.get("qa_id")
                q_type = int(qa.get("q_type", 0) or 0)
                dom = int(qa.get("domain", domain_id or 0) or 0)
                if domain_id:
                    dom = domain_id
                question = str(qa.get("question", "")).strip()
                if not question:
                    continue

                buf.append({
                    "id": f"vlq_{qa_id}",
                    "text": normalize_passage(question),
                    "meta": {
                        "source": "VL",
                        "qa_id": qa_id,
                        "domain_id": dom,
                        "q_type": q_type,
                        "origin_path": origin_path,
                        **extra
                    }
                })

                if len(buf) >= shard_size:
                    flush()

    flush()
    return {"vl_question": total}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--auto-extract", action="store_true",
                    help="(옵션) 현재 버전은 ZIP 스트리밍을 기본 지원합니다. 디스크 캐시 해제는 2_embed 단계에서 선택하세요.")
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))

    # tokenizer: embedding 모델과 동일한 토크나이저로 chunk 토큰 길이를 맞춤
    model_name = cfg["embedding"]["model_name"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    out_root = Path(cfg["output"]["processed_dir"])
    ensure_dir(out_root)

    print("Building TS corpus...")
    ts_stats = build_ts(cfg, tokenizer)
    print("TS done:", ts_stats)

    print("Building TL corpus...")
    tl_stats = build_tl(cfg)
    print("TL done:", tl_stats)

    print("Building VL corpus...")
    vl_stats = build_vl(cfg)
    print("VL done:", vl_stats)

    print("\nAll corpora built under:", out_root)

if __name__ == "__main__":
    main()
