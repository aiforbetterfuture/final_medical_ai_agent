from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import yaml


def repo_root_from(path: Path) -> Path:
    # .../final_medical_ai_agent/retrieval/aihub_flat/config.py
    return path.resolve().parents[2]


@dataclass
class AIHubIndexConfig:
    index_dir: Path

    # TS
    ts_coarse_npy: str = "ts_coarse_flat.npy"
    ts_coarse_meta: str = "ts_coarse_meta.jsonl"
    ts_fine_npy: str = "ts_fine_flat.npy"
    ts_fine_meta: str = "ts_fine_meta.jsonl"

    # TL
    tl_stem_npy: str = "tl_stem_flat.npy"
    tl_stem_meta: str = "tl_stem_meta.jsonl"
    tl_stem_opts_npy: str = "tl_stem_opts_flat.npy"
    tl_stem_opts_meta: str = "tl_stem_opts_meta.jsonl"

    # VL(평가셋 인덱스는 선택)
    vl_question_npy: str = "vl_question_flat.npy"
    vl_question_meta: str = "vl_question_meta.jsonl"

    # 성능 파라미터
    block_size: int = 200_000
    topk_coarse_docs: int = 30
    topk_ts_final: int = 8
    topk_tl_final: int = 5

    # RRF
    rrf_k: int = 60
    weight_ts: float = 1.0
    weight_tl: float = 0.6

    # embedding
    embed_model_name: str = "intfloat/multilingual-e5-base"

    @classmethod
    def default(cls) -> "AIHubIndexConfig":
        root = repo_root_from(Path(__file__))
        return cls(index_dir=root / "data" / "aihub_71874" / "indexes")

    def to_dict(self) -> dict:
        return {
            "index_dir": str(self.index_dir),
            "ts_coarse_npy": self.ts_coarse_npy,
            "ts_coarse_meta": self.ts_coarse_meta,
            "ts_fine_npy": self.ts_fine_npy,
            "ts_fine_meta": self.ts_fine_meta,
            "tl_stem_npy": self.tl_stem_npy,
            "tl_stem_meta": self.tl_stem_meta,
            "tl_stem_opts_npy": self.tl_stem_opts_npy,
            "tl_stem_opts_meta": self.tl_stem_opts_meta,
            "vl_question_npy": self.vl_question_npy,
            "vl_question_meta": self.vl_question_meta,
            "block_size": self.block_size,
            "topk_coarse_docs": self.topk_coarse_docs,
            "topk_ts_final": self.topk_ts_final,
            "topk_tl_final": self.topk_tl_final,
            "rrf_k": self.rrf_k,
            "weight_ts": self.weight_ts,
            "weight_tl": self.weight_tl,
            "embed_model_name": self.embed_model_name,
        }

    def save_yaml(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)

    @classmethod
    def load_yaml(cls, path: Path) -> "AIHubIndexConfig":
        with path.open("r", encoding="utf-8") as f:
            d = yaml.safe_load(f)
        d["index_dir"] = Path(d["index_dir"])
        return cls(**d)

