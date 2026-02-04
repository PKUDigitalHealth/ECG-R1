import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import evaluate
import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm

DEFAULT_SBERT_MODEL = "BAAI/bge-m3"


def compute_metrics_single(
    ts_output: str,
    img_output: str,
    bleu_metric,
    rouge_metric,
    sbert_model: SentenceTransformer,
) -> Dict[str, float]:
    """Compute BLEU/ROUGE/SBERT for a single sample."""
    bleu_score = bleu_metric.compute(predictions=[ts_output], references=[[img_output]])["bleu"]
    rouge_l = rouge_metric.compute(predictions=[ts_output], references=[img_output])["rougeL"]

    ts_emb = sbert_model.encode(ts_output, convert_to_tensor=True)
    img_emb = sbert_model.encode(img_output, convert_to_tensor=True)
    sbert_sim = util.cos_sim(ts_emb, img_emb).item()

    return {
        "BLEU": bleu_score,
        "ROUGE-L": rouge_l,
        "SBERT_Cosine": sbert_sim,
    }

def extract_response(record: Dict[str, Any]) -> str:
    """Extract assistant response text from a record."""
    return (record.get("response") or "").strip()


def derive_model_name(image_only_path: Path) -> str:
    """
    Derive model name from image-only path.

    Expected layout:
    .../training/<model_root>/<version>/<checkpoint>/infer_result/...

    Returns "<model_root>/<version>/<checkpoint>" with "/" replaced by "-".
    """
    parts = image_only_path.parts
    if "training" in parts:
        idx = parts.index("training")
        if idx + 3 < len(parts):
            model_parts = parts[idx + 1 : idx + 4]
            return "/".join(model_parts).replace("/", "-")
    # Fallback to parent directory name.
    return image_only_path.parent.name.replace("/", "-")


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data


def main():
    parser = argparse.ArgumentParser(description="Compute consistency between TS-only and image-only outputs")
    parser.add_argument(
        "--image-only-path",
        type=Path,
        default=Path("path/to/infer_result/xxxxx-xxxxx.jsonl"),
        help="JSONL output path for image-only inference",
    )
    parser.add_argument(
        "--ts-only-path",
        type=Path,
        default=Path("path/to/infer_result/xxxxx-xxxxx.jsonl"),
        help="JSONL output path for TS-only inference",
    )
    script_dir = Path(__file__).resolve().parent
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=script_dir,
        help="Output directory (default: this script directory)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help='Model name (default: derived from --image-only-path, "/" replaced by "-")',
    )
    args = parser.parse_args()

    img_path: Path = args.image_only_path
    ts_path: Path = args.ts_only_path
    output_dir: Path = args.output_dir or script_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[load] reading image-only jsonl: {img_path}")
    image_records = read_jsonl(img_path)
    print(f"[load] reading ts-only   jsonl: {ts_path}")
    ts_records = read_jsonl(ts_path)

    if len(image_records) != len(ts_records):
        raise ValueError(
            f"Result length mismatch: image_only={len(image_records)}, ts_only={len(ts_records)}"
        )

    print(f"[prep] total samples: {len(image_records)}")
    img_outputs = [extract_response(r) for r in image_records]
    ts_outputs = [extract_response(r) for r in ts_records]

    print(f"[init] loading metrics: bleu / rouge")
    repo_root = Path(__file__).resolve().parents[3]
    metrics_dir = repo_root / "scripts" / "modal_missing_evaluation" / "evaluate-main" / "metrics"
    bleu_metric = evaluate.load(str(metrics_dir / "bleu"))
    rouge_metric = evaluate.load(str(metrics_dir / "rouge"))

    print(f"[init] loading SentenceTransformer: {DEFAULT_SBERT_MODEL}")
    sbert_model = SentenceTransformer(DEFAULT_SBERT_MODEL)

    print(f"[metric] computing per-sample metrics with progress bar ...")
    per_sample: List[Dict[str, float]] = []
    for ts_out, img_out in tqdm(zip(ts_outputs, img_outputs), total=len(ts_outputs), desc="samples"):
        per_sample.append(
            compute_metrics_single(
                ts_output=ts_out,
                img_output=img_out,
                bleu_metric=bleu_metric,
                rouge_metric=rouge_metric,
                sbert_model=sbert_model,
            )
        )

    summary = {
        "BLEU": float(np.mean([x["BLEU"] for x in per_sample])),
        "ROUGE-L": float(np.mean([x["ROUGE-L"] for x in per_sample])),
        "SBERT_Cosine_Mean": float(np.mean([x["SBERT_Cosine"] for x in per_sample])),
    }
    print(f"[metric] metric computation done")

    model_tag = (args.model_name or derive_model_name(img_path)).replace("/", "-")

    summary_payload = {
        "model_name": model_tag,
        "BLEU": summary["BLEU"],
        "ROUGE-L": summary["ROUGE-L"],
        "SBERT_Cosine_Mean": summary["SBERT_Cosine_Mean"],
        "num_samples": len(image_records),
        "image_only_path": str(img_path),
        "ts_only_path": str(ts_path),
    }
    summary_df = pd.DataFrame([summary_payload])
    summary_csv = output_dir / "consistency_summary.csv"
    summary_df.to_csv(summary_csv, mode="a", index=False, header=not summary_csv.exists())

    print(f"[save] appended summary to: {summary_csv}")
    for k, v in summary.items():
        print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
