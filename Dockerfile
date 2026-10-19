
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    MPLBACKEND=Agg \
    YOLO_CONFIG_DIR=/app/.config/Ultralytics \
    PYTHONPATH=/app

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    git \
    curl \
    libglib2.0-0 \
    libgl1 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -r requirements.txt \
    && python -m pip check

COPY . /app

RUN mkdir -p \
    /app/data/raw \
    /app/data/processed \
    /app/models/classification \
    /app/models/detection \
    /app/reports/figures \
    /app/reports/metrics \
    /app/reports/predictions \
    /app/.config/Ultralytics

EXPOSE 8501

CMD ["bash"]
