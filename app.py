"""Flask web app for LSTM text generation."""

from flask import Flask, render_template, request

import config
from generator import get_generator

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/generate", methods=["POST"])
def generate():
    seed_text = request.form.get("seed_text", "").strip()
    error = None
    prediction = None

    if not seed_text:
        error = "Enter a short seed phrase to continue."
    else:
        try:
            prediction = get_generator().generate(seed_text)
        except FileNotFoundError as exc:
            error = str(exc)
        except ValueError as exc:
            error = str(exc)

    return render_template(
        "home.html",
        seed_text=seed_text,
        prediction_text=prediction,
        error=error,
    )


@app.route("/health")
def health():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host=config.HOST, port=config.PORT, debug=config.DEBUG)
