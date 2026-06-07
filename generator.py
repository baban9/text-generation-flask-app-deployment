"""LSTM text generation model and inference helpers."""

from __future__ import annotations

import pickle
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

import config


class RNNModule(nn.Module):
    def __init__(self, n_vocab: int, seq_size: int, embedding_size: int, lstm_size: int):
        super().__init__()
        self.seq_size = seq_size
        self.lstm_size = lstm_size
        self.embedding = nn.Embedding(n_vocab, embedding_size)
        self.lstm = nn.LSTM(embedding_size, lstm_size, batch_first=True)
        self.dense = nn.Linear(lstm_size, n_vocab)

    def forward(self, x, prev_state):
        embed = self.embedding(x)
        output, state = self.lstm(embed, prev_state)
        logits = self.dense(output)
        return logits, state

    def zero_state(self, batch_size: int):
        return (
            torch.zeros(1, batch_size, self.lstm_size),
            torch.zeros(1, batch_size, self.lstm_size),
        )


def generate_ngrams(sequence: List[str], n: int) -> List[str]:
    return [" ".join(sequence[i : i + n]) for i in range(len(sequence) - n + 1)]


class TextGenerator:
    def __init__(self):
        self.model = RNNModule(
            config.N_VOCAB,
            config.SEQ_SIZE,
            config.EMBEDDING_SIZE,
            config.LSTM_SIZE,
        )
        self.vocab_to_int = {}
        self.int_to_vocab = {}
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return

        if not all(
            __import__("os").path.exists(path)
            for path in (
                config.MODEL_PATH,
                config.VOCAB_TO_INT_PATH,
                config.INT_TO_VOCAB_PATH,
            )
        ):
            missing = [
                path
                for path in (
                    config.MODEL_PATH,
                    config.VOCAB_TO_INT_PATH,
                    config.INT_TO_VOCAB_PATH,
                )
                if not __import__("os").path.exists(path)
            ]
            raise FileNotFoundError(
                "Missing model artifacts: "
                + ", ".join(missing)
                + ". Run `python train.py` to train and export them."
            )

        with open(config.VOCAB_TO_INT_PATH, "rb") as handle:
            self.vocab_to_int = pickle.load(handle)
        with open(config.INT_TO_VOCAB_PATH, "rb") as handle:
            self.int_to_vocab = pickle.load(handle)

        state_dict = torch.load(config.MODEL_PATH, map_location=torch.device("cpu"))
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self._loaded = True

    def generate(self, seed_text: str, steps: int | None = None) -> str:
        self.load()
        steps = steps or config.GENERATION_STEPS
        tokens = generate_ngrams(seed_text.split(), config.NGRAMS)
        if not tokens:
            raise ValueError("Seed text is too short for the configured n-gram size.")

        state_h, state_c = self.model.zero_state(1)
        for token in tokens:
            if token not in self.vocab_to_int:
                raise ValueError(f"Seed contains unknown n-gram: {token}")
            index = torch.tensor([[self.vocab_to_int[token]]], dtype=torch.long)
            output, (state_h, state_c) = self.model(index, (state_h, state_c))

        _, top_ix = torch.topk(output[0], k=config.TOP_K)
        choice = int(np.random.choice(top_ix.tolist()[0]))
        tokens.append(self.int_to_vocab[choice])

        for _ in range(steps):
            index = torch.tensor([[choice]], dtype=torch.long)
            output, (state_h, state_c) = self.model(index, (state_h, state_c))
            _, top_ix = torch.topk(output[0], k=config.TOP_K)
            choice = int(np.random.choice(top_ix.tolist()[0]))
            tokens.append(self.int_to_vocab[choice])

        return " ".join(tokens)


_generator: TextGenerator | None = None


def get_generator() -> TextGenerator:
    global _generator
    if _generator is None:
        _generator = TextGenerator()
    return _generator
