FROM python:3.9

RUN python -m pip install -U pip
RUN pip install --no-cache-dir "fastapi[standard]"
RUN pip install --no-cache-dir scikit-learn pillow levenshtein transformers
RUN pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
COPY huggingface/ ./root/.cache/huggingface
COPY data/ ./data
COPY src/ ./src

EXPOSE 8000

CMD ["uvicorn", "--host", "0.0.0.0", "src.main:app"]