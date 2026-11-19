from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .evaluate import load_checkpoint, predict_proba


@dataclass
class Verdict:
    label: str
    probability: float
    threshold: float

    @property
    def is_unsafe(self) -> bool:
        return self.label == "1"

    def as_dict(self) -> dict:
        return {"label": self.label, "unsafe": self.is_unsafe, "score": round(self.probability, 4)}


class SafetyModel:
    """Inference wrapper: text in, '0' or '1' out."""

    def __init__(self, checkpoint: str | Path = "checkpoints/best.pt", threshold: float = 0.5, device: str | None = None) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model, self.tokenizer, self.model_config, self.state = load_checkpoint(checkpoint, self.device)
        self.max_chars = int(self.state.get("max_chars", 500))
        self.max_len = self.model_config.max_seq_len
        self.threshold = threshold

    def predict(self, text: str) -> Verdict:
        text = text[: self.max_chars]
        prob = predict_proba(self.model, self.tokenizer, [text], self.max_len, self.device)[0]
        return Verdict(label="1" if prob >= self.threshold else "0", probability=prob, threshold=self.threshold)

    def predict_batch(self, texts: list[str]) -> list[Verdict]:
        trimmed = [t[: self.max_chars] for t in texts]
        probs = predict_proba(self.model, self.tokenizer, trimmed, self.max_len, self.device)
        return [
            Verdict(label="1" if p >= self.threshold else "0", probability=p, threshold=self.threshold)
            for p in probs
        ]
