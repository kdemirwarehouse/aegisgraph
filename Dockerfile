FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# LightGBM için gerekli OpenMP sistem kütüphanesi
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml README.md ./
COPY config/ ./config/
COPY src/ ./src/
COPY app.py ./

# Veri ve artefakt dizinlerini oluştur; sentetik veriyi üret ve modeli build aşamasında eğit
RUN mkdir -p /app/data /app/artifacts \
    && python -m src.generator.synthetic_data \
    && python -m src.models.train \
    && chmod -R a+rwX /app/data /app/artifacts

EXPOSE 7860

# HF Spaces ile aynı giriş noktası: FastAPI (API + SSE + Dashboard) + Gradio, port 7860
CMD ["python", "app.py"]
