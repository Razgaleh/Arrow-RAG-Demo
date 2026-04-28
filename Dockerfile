# Python 3.10: PyPI faiss-gpu wheels are published for cp310 only (not 3.11+).
# Run with GPU: docker run --gpus all ... (requires NVIDIA Container Toolkit on the host).
FROM python:3.10-slim-bookworm

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    libmagic1 \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# 1) Install everything (NumPy 2.x gets pulled by torch/scipy/etc.).
# 2) Swap numpy to 1.x -- faiss-gpu's C extension requires the NumPy 1.x ABI.
#    --no-deps avoids re-resolving scipy/torch and the massive backtracking that causes.
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir --no-deps --force-reinstall "numpy>=1.26.0,<2.0.0"

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "rag.py"]
