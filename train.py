"""Train the LSTM model and export artifacts for the Flask app."""

from __future__ import annotations

import pickle
import re
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn

import config
from generator import RNNModule, generate_ngrams


def prep_data(corpus_text: list[str], seq_size: int, batch_size: int):
    word_counts = Counter(corpus_text)
    sorted_vocab = sorted(word_counts, key=word_counts.get, reverse=True)
    int_to_vocab = {index: word for index, word in enumerate(sorted_vocab)}
    vocab_to_int = {word: index for index, word in int_to_vocab.items()}
    n_vocab = len(int_to_vocab)

    int_text = [vocab_to_int[word] for word in corpus_text]
    num_batches = len(int_text) // (seq_size * batch_size)
    in_text = int_text[: num_batches * batch_size * seq_size]
    out_text = np.zeros_like(in_text)
    out_text[:-1] = in_text[1:]
    out_text[-1] = int_text[0]
    in_text = np.reshape(in_text, (batch_size, -1))
    out_text = np.reshape(out_text, (batch_size, -1))
    return in_text, out_text, n_vocab, int_to_vocab, vocab_to_int


def get_batches(in_text, out_text, batch_size, seq_size):
    num_batches = np.prod(in_text.shape) // (seq_size * batch_size)
    for index in range(0, num_batches * seq_size, seq_size):
        yield in_text[:, index : index + seq_size], out_text[:, index : index + seq_size]


def train_model(
    model: RNNModule,
    in_text,
    out_text,
    batch_size: int,
    seq_size: int,
    learning_rate: float,
    n_epochs: int,
) -> RNNModule:
    device = torch.device("cpu")
    model = model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        start = time.time()
        batches = get_batches(in_text, out_text, batch_size, seq_size)
        state_h, state_c = model.zero_state(batch_size)
        losses = []

        for x_batch, y_batch in batches:
            model.train()
            optimizer.zero_grad()
            x_tensor = torch.tensor(x_batch, dtype=torch.long)
            y_tensor = torch.tensor(y_batch, dtype=torch.long)
            logits, (state_h, state_c) = model(x_tensor, (state_h, state_c))
            loss = criterion(logits.transpose(1, 2), y_tensor)
            loss.backward()
            state_h = state_h.detach()
            state_c = state_c.detach()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            losses.append(loss.item())

        elapsed = round(time.time() - start, 2)
        print(f"Epoch {epoch + 1}/{n_epochs} loss={np.mean(losses):.4f} time={elapsed}s")

    return model


def main() -> None:
    with open(config.CORPUS_PATH, "r", encoding="utf-8") as handle:
        corpus = handle.read().lower()

    clean_text = re.sub(r"[^a-zA-Z0-9\s]", " ", corpus)
    clean_text = re.sub(" +", " ", clean_text)
    clean_text = re.sub(" \n+", " ", clean_text)

    corpus_text = generate_ngrams(clean_text.split(), config.NGRAMS)
    in_text, out_text, n_vocab, int_to_vocab, vocab_to_int = prep_data(
        corpus_text, config.SEQ_SIZE, config.BATCH_SIZE
    )

    with open(config.VOCAB_TO_INT_PATH, "wb") as handle:
        pickle.dump(vocab_to_int, handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(config.INT_TO_VOCAB_PATH, "wb") as handle:
        pickle.dump(int_to_vocab, handle, protocol=pickle.HIGHEST_PROTOCOL)

    model = RNNModule(
        n_vocab, config.SEQ_SIZE, config.EMBEDDING_SIZE, config.LSTM_SIZE
    )
    model = train_model(
        model,
        in_text,
        out_text,
        config.BATCH_SIZE,
        config.SEQ_SIZE,
        learning_rate=1.0,
        n_epochs=50,
    )
    torch.save(model.state_dict(), config.MODEL_PATH)
    print(f"Saved model to {config.MODEL_PATH}")


if __name__ == "__main__":
    main()
