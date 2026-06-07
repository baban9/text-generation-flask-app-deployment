.PHONY: setup run test clean

setup:
	python3 -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt

run:
	.venv/bin/python app.py

train:
	.venv/bin/python train.py

test:
	.venv/bin/python -c "from generator import RNNModule, generate_ngrams; assert generate_ngrams(['a','b','c','d'], 2)"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
