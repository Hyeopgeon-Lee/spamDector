# ---- 기본: 경량 CPU 이미지 (Python 3.10) ----
FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# TensorFlow 런타임에 필요한 런타임 라이브러리
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# === 빌드 시 TensorFlow 버전 선택 (기본: 2.17.0) ===
ARG TF_VERSION=2.17.0

# 파이썬 패키지 설치
RUN pip install --no-cache-dir \
    "tensorflow==${TF_VERSION}" \
    fastapi==0.112.2 \
    uvicorn==0.30.6

WORKDIR /app
COPY serve.py /app/serve.py
COPY model /app/model

# 기본 환경변수
ENV MODEL_PATH=/app/model/mySpamModel.keras \
    EVAL_META_PATH=/app/model/eval_meta.json \
    THRESHOLD=0.8 \
    SCORE_DECIMALS=6 \
    TEXT_MAX_CHARS=5000 \
    BATCH_MAX_ITEMS=256 \
    CORS_ALLOW_ORIGINS=* \
    HOST=0.0.0.0 \
    PORT=8000 \
    WORKERS=2

EXPOSE 8000

# ✅ HEALTHCHECK: heredoc 대신 exec-form + python -c one-liner
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD ["python","-c","import urllib.request,json,sys; \
try: \
    r=urllib.request.urlopen('http://localhost:8000/health', timeout=3); \
    ok=json.load(r).get('ok', True); \
    sys.exit(0 if ok else 1) \
except Exception: \
    sys.exit(1)"]

# 실행
CMD ["sh", "-c", "uvicorn serve:app --host ${HOST:-0.0.0.0} --port ${PORT:-8000} --workers ${WORKERS:-2}"]
