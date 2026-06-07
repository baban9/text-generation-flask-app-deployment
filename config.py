"""Application configuration."""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SEQ_SIZE = int(os.getenv("SEQ_SIZE", "20"))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
EMBEDDING_SIZE = int(os.getenv("EMBEDDING_SIZE", "300"))
LSTM_SIZE = int(os.getenv("LSTM_SIZE", "64"))
NGRAMS = int(os.getenv("NGRAMS", "5"))
N_VOCAB = int(os.getenv("N_VOCAB", "81977"))
GENERATION_STEPS = int(os.getenv("GENERATION_STEPS", "20"))
TOP_K = int(os.getenv("TOP_K", "10"))

MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "model_dict"))
VOCAB_TO_INT_PATH = os.getenv(
    "VOCAB_TO_INT_PATH", os.path.join(BASE_DIR, "vocab_to_int.pickle")
)
INT_TO_VOCAB_PATH = os.getenv(
    "INT_TO_VOCAB_PATH", os.path.join(BASE_DIR, "int_to_vocab.pickle")
)
CORPUS_PATH = os.getenv(
    "CORPUS_PATH", os.path.join(BASE_DIR, "Harry Potter and Sorcer's Stone.txt")
)

HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))
DEBUG = os.getenv("FLASK_DEBUG", "false").lower() == "true"
