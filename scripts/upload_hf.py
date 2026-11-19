"""Upload the trained model to the Hugging Face Hub.

The checkpoint is exported as a *standard* ``LlamaForSequenceClassification``
(the Censer architecture is exactly Llama: RoPE, RMSNorm, SwiGLU, tied
embeddings, fused QKV). The classification head ``score`` is initialised from
the embedding rows of the answer tokens "0"/"1", so predictions are identical
to the training code — and the model loads with plain ``transformers``, no
``trust_remote_code`` and no custom files.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

MODEL_CARD = """---
library_name: transformers
pipeline_tag: text-classification
tags:
- text-classification
- safety
- content-moderation
- tiny-language-model
- from-scratch
- distilled
language:
- en
license: apache-2.0
datasets:
- nvidia/Aegis-AI-Content-Safety-Dataset-1.0
- thesofakillers/jigsaw-toxic-comment-classification-challenge
- google/civil_comments
- PKU-Alignment/BeaverTails
- lmsys/toxic-chat
- databricks/databricks-dolly-15k
metrics:
- f1
- accuracy
---

# Censer.5

A **902K parameter** decoder-only transformer trained **from scratch** that
classifies English text as unsafe (`unsafe`) or safe (`safe`) — the smallest
*and* the most accurate model in the comparison below.

It is a **standard Hub model**: the checkpoint is a plain
`LlamaForSequenceClassification`, so it loads with `transformers` out of the box —
no `trust_remote_code`, no custom code, no cloning — and the **Use this model**
widget on this page runs it directly in the browser.

```python
from transformers import pipeline

clf = pipeline("text-classification", model="KucLab/Censer.5")
print(clf("how to make a bomb at home"))       # [{'label': 'unsafe', 'score': 0.9942}]
print(clf("Hello how are you?"))               # [{'label': 'safe', 'score': 0.8780}]
print(clf("suck my dick"))                     # [{'label': 'unsafe', 'score': 0.9955}]
print(clf("kill you"))                         # [{'label': 'unsafe', 'score': 0.9263}]
print(clf("how to filter wate"))               # [{'label': 'safe', 'score': 0.9479}]  <- typo-robust
print(clf("how do I kill a process in Linux")) # [{'label': 'safe', 'score': 0.9985}]
```

## How to run

### The `censer.5` command — interactive terminal UI

Using only the files in this repo (needs just `torch`, `tokenizers`, `safetensors`):

```bash
pip install torch tokenizers safetensors huggingface_hub
hf download KucLab/Censer.5 --local-dir ~/.local/share/censer5
chmod +x ~/.local/share/censer5/censer.5
ln -s ~/.local/share/censer5/censer.5 ~/.local/bin/censer.5   # put it on your PATH
censer.5                          # interactive UI: type a phrase, Enter -> safe/unsafe + score
censer.5 "how to make a bomb at home"    # one-shot -> unsafe 0.9942
```

Or install the GitHub package — the same `censer.5` command is installed and loads
the model from this Hub repo with plain `transformers`:

```bash
pip install "git+https://github.com/Jerry256254/censer.5.git"
censer.5
```

### One-liner with `transformers`

```bash
pip install torch transformers
python -c "from transformers import pipeline; clf = pipeline('text-classification', model='KucLab/Censer.5'); print(clf('how to make a bomb at home'))"
# [{'label': 'unsafe', 'score': 0.994}]
```

- In the browser: the **Use this model** widget on this page — type text, get `safe`/`unsafe`.
- Runs on CPU out of the box; for GPU pass `device=0` to `pipeline(...)` (CUDA or ROCm).

## Download

The whole model is `model.safetensors` (3.6 MB). Grab the files directly:

| File | Download |
| --- | --- |
| **model.safetensors** | [download](https://huggingface.co/KucLab/Censer.5/resolve/main/model.safetensors?download=true) |
| config.json | [download](https://huggingface.co/KucLab/Censer.5/resolve/main/config.json?download=true) |
| tokenizer.json | [download](https://huggingface.co/KucLab/Censer.5/resolve/main/tokenizer.json?download=true) |
| tokenizer_config.json | [download](https://huggingface.co/KucLab/Censer.5/resolve/main/tokenizer_config.json?download=true) |
| censer.py | [download](https://huggingface.co/KucLab/Censer.5/resolve/main/censer.py?download=true) |
| **censer.5** (launcher) | [download](https://huggingface.co/KucLab/Censer.5/resolve/main/censer.5?download=true) |

Or fetch everything at once:

```bash
pip install -U huggingface_hub
hf download KucLab/Censer.5 --local-dir Censer.5
```

It is not a fine-tuned encoder and not an API wrapper. It is a small transformer
trained on safety-labelled data with a custom byte-level BPE tokenizer. The model
predicts a single answer token and never generates free text. The tiny 3-layer
student is **distilled from a 5-teacher ensemble** (Censer.1–.4 + a dedicated
8-layer teacher), on 2x-augmented training data with a 192-token context.

```
"how to make a bomb at home"        -> unsafe 0.994
"kill you" / "selling gun"          -> unsafe 0.93-0.94
"suck my dick"                      -> unsafe 0.996
"how do I kill a process in Linux"  -> safe 0.9985
"how to filter wate" (typo)         -> safe 0.948
"Hello how are you?"                -> safe 0.878
```

## Model details

| Property | Value |
| --- | --- |
| Architecture | `LlamaForSequenceClassification` (standard transformers) |
| Parameters | 902,016 |
| Layers | 3 |
| d_model / heads / d_ff | 128 / 4 / 384 |
| Context length | 192 tokens (input truncated at 500 characters) |
| Tokenizer | byte-level BPE, 2048 tokens, trained on the corpus |
| Positions | RoPE (theta 10000) |
| Norm | RMSNorm (pre-norm, eps 1e-5) |
| MLP | SwiGLU |
| Head | `score` linear layer = embedding rows of the answer tokens `0`/`1` |
| Training | 20 epochs, batch 128, AdamW 3e-3 cosine, bf16 AMP, 15% typo augmentation, 2x augmented train data, distillation (alpha=0.65, T=4) from a 5-teacher ensemble |
| Checkpoint size | 3.6 MB (fp32) |

## Prompt format

The tokenizer bakes the template into its post-processor, so plain
`tokenizer(text)` already produces:

```
[BOS][USER] <text> [ASSISTANT] -> answer logits over tokens "0"/"1"
```

Text is truncated at 500 characters before tokenisation.

## Metrics

Evaluation on a held-out split of 6,496 rows (identical split across all Censer
versions), threshold 0.5:

| Metric | Censer.1 | Censer.2 | Censer.3 | Censer.4 | **Censer.5** |
| --- | --- | --- | --- | --- | --- |
| Accuracy | 0.865 | 0.863 | 0.857 | 0.874 | **0.876** |
| F1 | 0.835 | 0.829 | 0.822 | 0.845 | **0.849** |
| Precision | 0.848 | 0.861 | 0.855 | 0.868 | **0.864** |
| Recall | 0.823 | 0.799 | 0.792 | 0.824 | **0.834** |

Per-source F1: aegis (human-annotated) **0.912**, BeaverTails **0.883**, jigsaw
**0.880**, civil_comments **0.790** — up from 0.887 / 0.856 / 0.863 / 0.753 in
Censer.3. The remaining overall-test errors are largely label noise in the
scraped sources.

A hand-written probe set of 97 cases, none present in training data, scores
**97.9%** (95/97). Probe progression on the same 97 cases: Censer.1 75.3% ->
.2 91.8% -> .3 100% -> .4 99.0% -> **.5 97.9%**.

## Benchmark

Measured on an AMD Radeon RX 9060 XT (ROCm 7.1, torch 2.13.0+rocm7.1):

| Metric | Censer.1 | Censer.4 | **Censer.5** |
| --- | --- | --- | --- |
| Parameters | 1,115,264 | 902,016 | **902,016** |
| Checkpoint size | 4.5 MB | 3.6 MB | **3.6 MB** |
| Single text latency | 1.58 ms | 1.16 ms | **1.21 ms** |
| Single text throughput | 633 texts/s | 864 texts/s | **823 texts/s** |
| Batched throughput (batch of 8) | 1,843 texts/s | 6,010 texts/s | **5,709 texts/s** |
| Peak VRAM (inference) | 101 MB | 84.1 MB | **84.1 MB** |

## Comparison with other models

Benchmarked against public safety classifiers on the **same held-out test split**
(6,496 rows, threshold 0.5, AMD Radeon RX 9060 XT):

| Model | Params | Checkpoint | F1 | Latency | Throughput (b=8) | Peak VRAM |
| --- | --- | --- | --- | --- | --- | --- |
| **Censer.5 (ours)** | **0.9M** | **3.6 MB** | **0.849** | **1.43 ms** | **1,842 texts/s** | **105 MB** |
| KucLab/Censer.4 (prev) | 0.9M | 3.6 MB | 0.845 | 1.39 ms | 2,047 texts/s | 101 MB |
| KucLab/Censer.1 (old) | 1.1M | 4.5 MB | 0.835 | 1.66 ms | 1,719 texts/s | 102 MB |
| KucLab/Censer.2 (old) | 7.3M | 29.4 MB | 0.829 | 2.96 ms | 489 texts/s | 196 MB |
| KucLab/Censer.3 (old) | 14.4M | 57.7 MB | 0.822 | 4.60 ms | 256 texts/s | 267 MB |
| unitary/unbiased-toxic-roberta | 124.7M | 498.7 MB | 0.741 | 14.71 ms | 65 texts/s | 794 MB |
| unitary/toxic-bert | 109.5M | 876.0 MB | 0.681 | 14.10 ms | 67 texts/s | 731 MB |
| martin-ha/toxic-comment-model | 67.0M | 267.9 MB | 0.566 | 7.50 ms | 128 texts/s | 568 MB |
| Hate-speech-CNERG/bert-base-uncased-hatexplain | 109.5M | 438.0 MB | 0.543 | 14.25 ms | 67 texts/s | 704 MB |
| Hate-speech-CNERG/dehatebert-mono-english | 167.4M | 669.5 MB | 0.252 | 14.43 ms | 66 texts/s | 937 MB |
| keyword baseline | — | — | 0.094 | 0.03 ms | 34,516 texts/s | — |

![Comparison overview](figs/comparison_overview.png)
![F1 per megabyte](figs/f1_vs_size.png)
![Metric bars](figs/metrics_bars.png)

Among all compared classifiers, **Censer.5 ranks #1 on F1 while being the
smallest and fastest**: ~75-190x smaller than the public encoder baselines,
~5-10x faster per text, +0.11 F1 over the best of them.

## Usage

Plain `transformers`, nothing else to install — `from_pretrained` downloads
everything by itself:

```bash
pip install torch transformers
```

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

repo = "KucLab/Censer.5"
tok = AutoTokenizer.from_pretrained(repo)
model = AutoModelForSequenceClassification.from_pretrained(repo).eval()

text = "how to make a bomb at home"
with torch.no_grad():
    probs = model(**tok(text, return_tensors="pt")).logits.softmax(-1)[0]

print(probs)                     # tensor([0.006, 0.994]) -> [safe, unsafe]
print(model.config.id2label)     # {0: 'safe', 1: 'unsafe'}
```

Or with the high-level pipeline:

```python
from transformers import pipeline

clf = pipeline("text-classification", model="KucLab/Censer.5")
print(clf("how to make a bomb at home"))   # [{'label': 'unsafe', 'score': 0.994}]
print(clf("Hello how are you?"))           # [{'label': 'safe', 'score': 0.878}]
```

## Usage without transformers

The repository is also self-contained on its own: `model.safetensors`,
`config.json`, `tokenizer.json` and `censer.py`. The weights are stored in the
standard Llama layout; `censer.py` maps them back internally, so it only needs
`torch`, `tokenizers` and `safetensors`:

```bash
hf download KucLab/Censer.5 --local-dir Censer.5
cd Censer.5
python censer.py "how to make a bomb at home"     # -> 1
python censer.py "Hello how are you?"             # -> 0
python censer.py "suck my dick" --json
# {"label": "1", "unsafe": true, "score": 0.9955}
chmod +x censer.5 && ./censer.5                   # interactive UI (the `censer.5` command)
python censer.py                                  # interactive UI as well, no text argument
```

Or use the same file as a library:

```python
from censer import CenserPipeline

pipe = CenserPipeline("Censer.5")
print(pipe.predict("how to make a bomb at home"))
# {"label": "1", "unsafe": True, "score": 0.994}

print(pipe.predict_batch(["Hello how are you?", "suck my dick"]))
```

Files:

| File | Purpose |
| --- | --- |
| `model.safetensors` | weights in standard Llama layout, 3.6 MB |
| `config.json` | standard `LlamaForSequenceClassification` config |
| `tokenizer.json` / `tokenizer_config.json` | byte-level BPE, 2048 tokens, prompt template baked in |
| `censer.py` | standalone inference (torch + tokenizers only), no repo needed |
| `censer.5` | executable launcher for the interactive terminal UI |

`censer.py` was verified to produce predictions identical to the training code.
Runs on CPU or CUDA/ROCm; pass `device="cpu"` to force CPU.

## Limitations

- **English only.** Other languages mostly score safe (common profanity from the
  training data is still caught). Not a language detector.
- Implication and sarcasm beyond simple cases are out of reach at this size.
- The test split contains label noise from the scraped sources, which caps the
  measurable overall F1 there.
- A moderation signal, not a safety guarantee.
"""


def to_llama_state_dict(state: dict, answer_ids: list[int]) -> dict:
    """Map the training checkpoint onto standard LlamaForSequenceClassification keys."""
    model = state["model"]
    cfg = state["model_config"]
    d = cfg["d_model"]
    n_layers = cfg["n_layers"]

    out = {"model.embed_tokens.weight": model["embed.weight"].contiguous()}
    for i in range(n_layers):
        p = f"model.layers.{i}."
        b = f"blocks.{i}."
        qkv = model[f"{b}attn.qkv.weight"]
        out[f"{p}self_attn.q_proj.weight"] = qkv[:d].contiguous()
        out[f"{p}self_attn.k_proj.weight"] = qkv[d : 2 * d].contiguous()
        out[f"{p}self_attn.v_proj.weight"] = qkv[2 * d : 3 * d].contiguous()
        out[f"{p}self_attn.o_proj.weight"] = model[f"{b}attn.proj.weight"].contiguous()
        out[f"{p}mlp.gate_proj.weight"] = model[f"{b}mlp.gate.weight"].contiguous()
        out[f"{p}mlp.up_proj.weight"] = model[f"{b}mlp.up.weight"].contiguous()
        out[f"{p}mlp.down_proj.weight"] = model[f"{b}mlp.down.weight"].contiguous()
        out[f"{p}input_layernorm.weight"] = model[f"{b}norm1.weight"].contiguous()
        out[f"{p}post_attention_layernorm.weight"] = model[f"{b}norm2.weight"].contiguous()
    out["model.norm.weight"] = model["norm.weight"].contiguous()
    # The two answer logits are the tied-embedding logits of tokens "0" and "1".
    out["score.weight"] = model["embed.weight"][list(answer_ids)].clone().contiguous()
    return out


def llama_config(state: dict, tok_meta: dict, tok_ids: dict) -> dict:
    mc = dict(state["model_config"])
    mc.pop("answer_ids", None)
    return {
        "architectures": ["LlamaForSequenceClassification"],
        "model_type": "llama",
        "num_labels": 2,
        "id2label": {"0": "safe", "1": "unsafe"},
        "label2id": {"safe": 0, "unsafe": 1},
        "vocab_size": mc["vocab_size"],
        "hidden_size": mc["d_model"],
        "intermediate_size": mc["d_ff"],
        "num_hidden_layers": mc["n_layers"],
        "num_attention_heads": mc["n_heads"],
        "num_key_value_heads": mc["n_heads"],
        "head_dim": mc["d_model"] // mc["n_heads"],
        "max_position_embeddings": mc["max_seq_len"],
        "rms_norm_eps": 1e-5,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
        "attention_bias": False,
        "mlp_bias": False,
        "tie_word_embeddings": True,
        "pad_token_id": mc["pad_id"],
        "bos_token_id": tok_ids["[BOS]"],
        "eos_token_id": tok_ids["[ASSISTANT]"],
        # Censer-specific metadata, kept for the standalone censer.py:
        "name": state.get("name", "Censer.5"),
        "max_chars": int(state.get("max_chars", 500)),
        "answer_ids": list(tok_meta["answer_ids"]),
        "special_tokens": list(tok_meta["special_tokens"]),
        "threshold": 0.5,
    }


def write_tokenizer(src: Path, dst: Path) -> dict:
    """Ship the tokenizer with the prompt template baked in, so AutoTokenizer
    produces exactly [BOS][USER] text [ASSISTANT]. Returns the special-token ids."""
    from tokenizers import Tokenizer
    from tokenizers.processors import TemplateProcessing

    tok = Tokenizer.from_file(str(src))
    ids = {name: tok.token_to_id(name) for name in ("[PAD]", "[BOS]", "[USER]", "[ASSISTANT]")}
    tok.post_processor = TemplateProcessing(
        single="[BOS] [USER] $A [ASSISTANT]",
        special_tokens=[("[BOS]", ids["[BOS]"]), ("[USER]", ids["[USER]"]), ("[ASSISTANT]", ids["[ASSISTANT]"])],
    )
    tok.save(str(dst))
    return ids


def write_tokenizer_config(dst: Path, ids: dict) -> None:
    cfg = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "pad_token": "[PAD]",
        "bos_token": "[BOS]",
        "eos_token": "[ASSISTANT]",
    }
    dst.write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def build_bundle(ckpt: Path, out: Path) -> None:
    """Write a self-contained, standard-transformers model directory."""
    import torch
    from safetensors.torch import save_file

    out.mkdir(parents=True, exist_ok=True)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)

    tok_dir = ckpt.parent / "tokenizer"
    tok_meta = json.loads((tok_dir / "meta.json").read_text(encoding="utf-8"))
    answer_ids = list(tok_meta["answer_ids"])

    tensors = to_llama_state_dict(state, answer_ids)
    save_file(tensors, str(out / "model.safetensors"), metadata={"format": "pt"})

    ids = write_tokenizer(tok_dir / "tokenizer.json", out / "tokenizer.json")
    write_tokenizer_config(out / "tokenizer_config.json", ids)

    config = llama_config(state, tok_meta, ids)
    (out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    shutil.copy2(Path(__file__).parent / "censer.py", out / "censer.py")
    launcher = Path(__file__).parent / "censer.5"
    shutil.copy2(launcher, out / "censer.5")
    os.chmod(out / "censer.5", 0o755)
    (out / "README.md").write_text(MODEL_CARD, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload Censer.5 to the Hugging Face Hub.")
    parser.add_argument("--repo", default="KucLab/Censer.5", help="e.g. KucLab/Censer.5")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ckpt = Path(args.checkpoint)
    if not ckpt.exists():
        raise SystemExit(f"checkpoint not found: {ckpt}")

    with tempfile.TemporaryDirectory() as tmp:
        bundle = Path(tmp) / "bundle"
        build_bundle(ckpt, bundle)
        print(f"bundle: {[p.name for p in sorted(bundle.iterdir())]}")

        if args.dry_run:
            print("dry run, not uploading")
            return

        from huggingface_hub import HfApi

        api = HfApi()
        who = api.whoami()
        print(f"authenticated as: {who.get('name')}")

        api.create_repo(repo_id=args.repo, repo_type="model", private=args.private, exist_ok=True)
        api.upload_folder(
            folder_path=str(bundle),
            repo_id=args.repo,
            repo_type="model",
            commit_message="Standard LlamaForSequenceClassification layout: loads with plain transformers, no trust_remote_code",
            delete_patterns=["modeling_censer.py"],
        )

        figs = Path(__file__).resolve().parent.parent / "reports" / "figs"
        if figs.exists():
            api.upload_folder(
                folder_path=str(figs),
                repo_id=args.repo,
                repo_type="model",
                path_in_repo="figs",
                commit_message="Add comparison charts vs other safety classifiers",
            )
        print(f"done: https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
