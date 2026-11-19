from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from .dataset import Batcher, SafetyDataset, collate
from .model import KucLabCneser, ModelConfig
from .tokenizer import KucTokenizer


def load_checkpoint(path: str | Path, device: torch.device, tok_dir: str | Path | None = None):
    state = torch.load(path, map_location=device, weights_only=False)
    mcfg = ModelConfig(**state["model_config"])
    model = KucLabCneser(mcfg).to(device)
    model.load_state_dict(state["model"])
    model.eval()
    if tok_dir is None:
        tok_dir = Path(path).parent / "tokenizer"
        if not (Path(tok_dir) / "tokenizer.json").exists():
            tok_dir = Path("checkpoints/tokenizer")
    tok = KucTokenizer.load(tok_dir)
    return model, tok, mcfg, state


@torch.inference_mode()
def predict_proba(model: KucLabCneser, tok: KucTokenizer, texts: list[str], max_len: int, device: torch.device, batch_size: int = 128) -> list[float]:
    """Classify texts; batches are sorted by length internally to minimise padding."""
    out: list[float] = [0.0] * len(texts)
    for start in range(0, len(texts), batch_size):
        idx = list(range(start, min(start + batch_size, len(texts))))
        ids = [tok.encode_prompt(texts[i], max_len) for i in idx]
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
        sorted_ids = [ids[i] for i in order]
        max_batch_len = len(sorted_ids[-1])
        input_ids = torch.full((len(sorted_ids), max_batch_len), tok.pad_id, dtype=torch.long)
        for i, seq in enumerate(sorted_ids):
            input_ids[i, : len(seq)] = torch.tensor(seq)
        probs = model.classify(input_ids.to(device))
        probs = probs.float().cpu().tolist()
        for pos, i in enumerate(order):
            out[idx[i]] = probs[pos]
    return out


def metrics_from(labels: list[int], probs: list[float], threshold: float = 0.5) -> dict:
    tp = fp = tn = fn = 0
    for label, prob in zip(labels, probs):
        pred = 1 if prob >= threshold else 0
        if pred == 1 and label == 1:
            tp += 1
        elif pred == 1 and label == 0:
            fp += 1
        elif pred == 0 and label == 0:
            tn += 1
        else:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    total = tp + fp + tn + fn
    return {
        "threshold": threshold,
        "accuracy": (tp + tn) / total if total else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "total": total,
    }


def best_threshold(labels: list[int], probs: list[float]) -> tuple[float, dict]:
    best = (0.5, metrics_from(labels, probs, 0.5))
    for i in range(5, 96):
        threshold = i / 100
        m = metrics_from(labels, probs, threshold)
        if m["f1"] > best[1]["f1"]:
            best = (threshold, m)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Censer.5.")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--split", default="test")
    parser.add_argument("--data", default="data/processed")
    parser.add_argument("--config", default="configs/censer5.json")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tok, mcfg, state = load_checkpoint(args.checkpoint, device)

    ds = SafetyDataset(Path(args.data) / f"{args.split}.jsonl", tok, config["max_seq_len"])
    texts = [row["text"] for row in ds.rows]
    labels = [int(row["label"]) for row in ds.rows]

    probs = predict_proba(model, tok, texts, config["max_seq_len"], device)
    m = metrics_from(labels, probs)
    threshold, m_best = best_threshold(labels, probs)

    print(f"checkpoint: {args.checkpoint}  split: {args.split}  n={len(texts)}")
    print(f"0.50 threshold : acc={m['accuracy']:.4f} f1={m['f1']:.4f} p={m['precision']:.4f} r={m['recall']:.4f}")
    print(f"best threshold : {threshold:.2f} acc={m_best['accuracy']:.4f} f1={m_best['f1']:.4f} p={m_best['precision']:.4f} r={m_best['recall']:.4f}")
    print(f"confusion @0.5: tp={m['tp']} fp={m['fp']} tn={m['tn']} fn={m['fn']}")

    by_source: dict[str, dict] = defaultdict(lambda: {"labels": [], "probs": []})
    for row, prob in zip(ds.rows, probs):
        by_source[row["source"]]["labels"].append(int(row["label"]))
        by_source[row["source"]]["probs"].append(prob)

    print("\nper source (threshold 0.50):")
    source_report = {}
    for source, data in sorted(by_source.items(), key=lambda kv: -len(kv[1]["labels"])):
        sm = metrics_from(data["labels"], data["probs"], 0.5)
        source_report[source] = sm
        print(f"  {source:18s} n={sm['total']:6d} acc={sm['accuracy']:.3f} f1={sm['f1']:.3f} p={sm['precision']:.3f} r={sm['recall']:.3f}")

    errors = [
        {"text": t, "label": l, "prob": p}
        for t, l, p in zip(texts, labels, probs)
        if (p >= 0.5) != (l == 1)
    ]
    errors.sort(key=lambda e: -abs(e["prob"] - 0.5))
    print(f"\nfalse positives: {sum(1 for e in errors if e['label'] == 0)}")
    for e in [e for e in errors if e["label"] == 0][:10]:
        print(f"  p={e['prob']:.3f} {e['text'][:100]!r}")
    print(f"false negatives: {sum(1 for e in errors if e['label'] == 1)}")
    for e in [e for e in errors if e["label"] == 1][:10]:
        print(f"  p={e['prob']:.3f} {e['text'][:100]!r}")

    from .probes import PROBES

    probe_texts = [t for t, _ in PROBES]
    probe_labels = [l for _, l in PROBES]
    probe_probs = predict_proba(model, tok, probe_texts, config["max_seq_len"], device)
    pm = metrics_from(probe_labels, probe_probs, 0.5)
    print(f"\nprobe set (hand-written, {len(PROBES)} cases): acc={pm['accuracy']:.3f} f1={pm['f1']:.3f}")
    misses = [(t, l, p) for t, l, p in zip(probe_texts, probe_labels, probe_probs) if int(p >= 0.5) != l]
    for t, l, p in misses:
        print(f"  MISS exp={l} p={p:.3f} {t!r}")
    if not misses:
        print("  all probes correct")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "checkpoint": args.checkpoint,
                    "split": args.split,
                    "at_0.5": m,
                    "best": {"threshold": threshold, **m_best},
                    "per_source": source_report,
                    "model_metrics": state.get("metrics", {}),
                },
                indent=2,
            )
        )
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
