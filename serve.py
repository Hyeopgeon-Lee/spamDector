# -*- coding: utf-8 -*-
"""
===============================================================================
Spam Detector API (FastAPI) — 상세 주석판
-------------------------------------------------------------------------------
목적
- Keras 저장 포맷(.keras)로 학습된 텍스트 스팸 분류 모델을 로드하여 REST API 제공
- 단건/배치 예측, 건강상태(health), 임계값/메타 노출
- 예외 처리를 일관된 JSON 포맷으로 통일
- CORS 허용으로 FE/다른 도메인에서 호출 가능

핵심 포인트
- 모델에는 TextVectorization이 포함되어 있으므로, 로드시 custom_standardize가 "등록"되어 있어야 함
- Windows(cp949) 인코딩 이슈를 피하기 위해, 가능한 UTF-8 환경(도커/리눅스/WSL 혹은 -X utf8) 권장
- 학습/서빙 모두 동일한 custom_standardize 이름/시그니처/로직 사용

환경변수(선택)
- MODEL_PATH        : 기본 model/mySpamModel.keras
- EVAL_META_PATH    : 기본 model/eval_meta.json (여기서 th_best/threshold 읽어 임계값 결정)
- THRESHOLD         : 명시하면 이 값을 우선 사용 (float). 없으면 meta→0.5 순서
- SCORE_DECIMALS    : 점수 문자열 소수자리수 (기본 6)
- HOST / PORT       : Uvicorn 바인딩(기본 0.0.0.0:8000)
- CORS_ALLOW_ORIGINS: CORS 허용 도메인 (* 또는 "https://a.com,https://b.com")
- TEXT_MAX_CHARS    : 단건 입력 최대 길이(기본 5000)
- BATCH_MAX_ITEMS   : 배치 요청 최대 아이템 수(기본 256)

운영 팁(선택)
- CPU 전용 컨테이너에서는 tensorflow-cpu 패키지 사용으로 이미지 슬림화 가능
- 로그 잡음 감소: TF_CPP_MIN_LOG_LEVEL=2
- 스레드 제한(자원 제한 환경): OMP_NUM_THREADS=1, TF_NUM_INTRAOP_THREADS=1, TF_NUM_INTEROP_THREADS=1
===============================================================================
"""

from typing import List, Literal, Optional, Any, Dict
import os
import json
import logging

import numpy as np
import tensorflow as tf
from tensorflow import keras
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# -----------------------------------------------------------------------------
# 로깅 설정
# - logging.basicConfig는 모듈 임포트 시 1회만 설정됨
# - Uvicorn이 별도 로거를 쓰므로, 워커를 여러 개 띄우면 동일 로그가 중복 보일 수 있음
# -----------------------------------------------------------------------------
logger = logging.getLogger("spam-api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# -----------------------------------------------------------------------------
# Keras 2/3 호환: 커스텀 직렬화 함수 등록
# - 저장/로드 시 커스텀 오브젝트를 식별하기 위해 동일한 이름으로 등록 필요
# - 이름/시그니처/로직이 학습 코드와 "완전히 동일"할수록 안전
# -----------------------------------------------------------------------------
try:
    register = keras.saving.register_keras_serializable   # Keras 3 (TF 2.20+)
except AttributeError:
    from tensorflow.keras.utils import register_keras_serializable as register  # Keras 2 (TF 2.17)

@register(package="preproc", name="custom_standardize")
def custom_standardize(x: tf.Tensor) -> tf.Tensor:
    """
    TextVectorization 표준화 단계:
    1) 소문자화
    2) URL / 이메일 / 전화번호 → 토큰 (<url>, <email>, <phone>)
    3) 허용 문자만 유지: 유니코드 '문자(L)/숫자(N)' + 공백/일부 구두점
       - RE2(텐서플로우 정규식 엔진)에서 \p{L}, \p{N} 사용 가능
       - 한글/영문/다국어를 자연스럽게 수용 (cp949 비호환 문자 포함 가능)
    4) 다중 공백 압축
    """
    x = tf.strings.lower(x)
    x = tf.strings.regex_replace(x, r"(https?://\S+|www\.\S+)", " <url> ")
    x = tf.strings.regex_replace(x, r"\S+@\S+\.\S+", " <email> ")
    x = tf.strings.regex_replace(x, r"\b\d{2,4}-\d{3,4}-\d{4}\b", " <phone> ")
    x = tf.strings.regex_replace(x, r"[^\p{L}\p{N}\s\.,!?%:/@_-]", " ")
    x = tf.strings.regex_replace(x, r"\s+", " ")
    return tf.strings.strip(x)

# (대안) 과거 학습이 '한글 범위 직접 지정'이었다면 아래로 교체하세요.
# @register(package="preproc", name="custom_standardize")
# def custom_standardize(x: tf.Tensor) -> tf.Tensor:
#     x = tf.strings.lower(x)
#     x = tf.strings.regex_replace(x, r"(https?://\S+|www\.\S+)", " <url> ")
#     x = tf.strings.regex_replace(x, r"\S+@\S+\.\S+", " <email> ")
#     x = tf.strings.regex_replace(x, r"\b\d{2,4}-\d{3,4}-\d{4}\b", " <phone> ")
#     x = tf.strings.regex_replace(x, r"[^a-z0-9\uac00-\ud7a3\s\.,!?%:/@_-]", " ")
#     x = tf.strings.regex_replace(x, r"\s+", " ")
#     return tf.strings.strip(x)

# -----------------------------------------------------------------------------
# 설정값 로딩(환경변수 → 기본값)
# -----------------------------------------------------------------------------
MODEL_PATH      = os.getenv("MODEL_PATH", "model/mySpamModel.keras")
EVAL_META_PATH  = os.getenv("EVAL_META_PATH", "model/eval_meta.json")
SCORE_DECIMALS  = int(os.getenv("SCORE_DECIMALS", "6"))
TEXT_MAX_CHARS  = int(os.getenv("TEXT_MAX_CHARS", "5000"))
BATCH_MAX_ITEMS = int(os.getenv("BATCH_MAX_ITEMS", "256"))

def _load_threshold() -> float:
    """
    임계값 로딩 우선순위:
    1) 환경변수 THRESHOLD (유효한 float일 때)
    2) EVAL_META_PATH(JSON)의 th_best 또는 threshold
    3) 기본값 0.5
    """
    env_th = os.getenv("THRESHOLD")
    if env_th:
        try:
            return float(env_th)
        except ValueError:
            logger.warning("Invalid THRESHOLD env var; falling back to meta/0.5.")
    try:
        with open(EVAL_META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
            for k in ("th_best", "threshold"):
                if k in meta:
                    return float(meta[k])
    except FileNotFoundError:
        pass
    return 0.5

THRESHOLD = _load_threshold()

# -----------------------------------------------------------------------------
# 모델 로드
# - 로드 실패 시에도 FastAPI 앱은 기동(health에서 에러 노출, /predict는 503)
# - 컨테이너/리눅스는 기본 UTF-8이라 윈도우 cp949 문제 재현 가능성 낮음
# -----------------------------------------------------------------------------
model = None
_model_load_error: Optional[str] = None
try:
    model = keras.models.load_model(
        MODEL_PATH,
        compile=False,  # 추론만 하므로 재컴파일 불필요
        custom_objects={"custom_standardize": custom_standardize},  # 커스텀 등록
        safe_mode=False,  # Keras 3: 커스텀 오브젝트 허용
    )
    logger.info("Model loaded: %s", MODEL_PATH)
except Exception as e:
    _model_load_error = f"{type(e).__name__}: {e}"
    # stacktrace는 logger.exception이 기록
    logger.exception("Failed to load model: %s", _model_load_error)

# -----------------------------------------------------------------------------
# Pydantic 스키마 (요청/응답 모델)
#  - OpenAPI 스펙 문서화 및 요청 검증
# -----------------------------------------------------------------------------
class PredictRequest(BaseModel):
    text: str = Field(..., description="분류할 문장")

class BatchPredictRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, description="분류할 문장 리스트")

class PredictResponse(BaseModel):
    label: Literal["spam", "ham"]
    score: float = Field(..., ge=0.0, le=1.0)
    score_str: Optional[str] = None  # 표시용 소수 포맷(프론트 편의)

class BatchItem(BaseModel):
    index: int          # 요청 순서(원위치 매핑)
    label: Literal["spam", "ham"]
    score: float
    score_str: Optional[str] = None

def error_body(kind: str, message: str, details: Optional[Any] = None) -> Dict[str, Any]:
    """
    모든 오류 응답을 동일한 JSON 포맷으로 반환하기 위한 헬퍼
    {
      "ok": false,
      "error": { "type": "<Kind>", "message": "<Message>", "details": ... }
    }
    """
    body = {"ok": False, "error": {"type": kind, "message": message}}
    if details is not None:
        body["error"]["details"] = details
    return body

# -----------------------------------------------------------------------------
# FastAPI 앱 + CORS
# -----------------------------------------------------------------------------
app = FastAPI(
    title="Spam Detector API",
    version="1.1.0",
    description="TensorFlow/Keras 텍스트 스팸 분류기(OpenAPI). 에러 표준화+CORS 포함.",
)

# CORS 허용 도메인 구성
_allow = os.getenv("CORS_ALLOW_ORIGINS", "*")
if _allow.strip() == "*":
    allow_origins = ["*"]
else:
    allow_origins = [o.strip() for o in _allow.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allow_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------------------------------------------------
# 전역 예외 핸들러
# - FastAPI 기본 핸들러 대신, 우리가 정의한 일관 포맷으로 응답
# -----------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def req_validation_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=error_body("RequestValidationError", "Invalid request body.", details=exc.errors()),
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content=error_body("HTTPException", exc.detail))

@app.exception_handler(Exception)
async def catch_all_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error")
    return JSONResponse(
        status_code=500,
        content=error_body("InternalServerError", f"{type(exc).__name__}: {exc}"),
    )

# -----------------------------------------------------------------------------
# 유틸 함수
# -----------------------------------------------------------------------------
def fmt_score(p: float) -> str:
    """표시용 점수 문자열(소수 자리수 환경변수로 제어)"""
    return f"{p:.{SCORE_DECIMALS}f}"

def ensure_ready():
    """모델이 메모리에 없으면 503으로 응답 (헬스는 ok=false+에러 메시지 노출)"""
    if model is None:
        raise HTTPException(status_code=503, detail=f"Model not loaded: {_model_load_error or 'unknown error'}")

def validate_text(s: str) -> str:
    """
    단건 입력 검증:
    - 타입 체크, 공백 제거, 비어있음/최대 길이 초과 검증
    """
    if s is None or not isinstance(s, str):
        raise HTTPException(status_code=400, detail="Field 'text' must be a string.")
    s2 = s.strip()
    if not s2:
        raise HTTPException(status_code=400, detail="Field 'text' must not be empty.")
    if len(s2) > TEXT_MAX_CHARS:
        raise HTTPException(status_code=400, detail=f"Text too long (>{TEXT_MAX_CHARS} chars).")
    return s2

def validate_texts(lst: List[str]) -> List[str]:
    """
    배치 입력 검증:
    - 리스트 여부/최소 길이/최대 개수/개별 항목 오류 위치 보고
    """
    if not isinstance(lst, list) or len(lst) == 0:
        raise HTTPException(status_code=400, detail="Field 'texts' must be a non-empty list.")
    if len(lst) > BATCH_MAX_ITEMS:
        raise HTTPException(status_code=400, detail=f"Too many items (>{BATCH_MAX_ITEMS}).")
    out = []
    for i, s in enumerate(lst):
        try:
            out.append(validate_text(str(s)))
        except HTTPException as e:
            # 어떤 인덱스에서 실패했는지 알리기
            raise HTTPException(status_code=e.status_code, detail=f"[index {i}] {e.detail}")
    return out

def predict_probs(texts: List[str]) -> np.ndarray:
    """
    모델 추론:
    - TextVectorization이 모델 내부에 포함되어 있으므로 입력은 문자열 그대로 전달
    - 배치 처리는 넘파이/TF 텐서 모두 가능 (여기서는 TF 상수로)
    """
    inputs = tf.constant(texts, dtype=tf.string)
    probs = model(inputs).numpy().ravel()
    return probs

# -----------------------------------------------------------------------------
# 라우트
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    """
    서버/모델 상태 점검:
    - ok: 모델 로드 여부
    - threshold: 현재 운영 임계값
    - eval_meta(가능 시): ROC/PR AUC, th_best 등 핵심 메타 노출
    """
    info = {
        "ok": True if model is not None else False,
        "model_path": MODEL_PATH,
        "threshold": THRESHOLD,
        "text_max_chars": TEXT_MAX_CHARS,
        "batch_max_items": BATCH_MAX_ITEMS,
        "cors_allow_origins": allow_origins,
    }
    if model is None:
        info["error"] = _model_load_error
    # 평가 메타 파일이 있으면 일부만 노출(민감정보 최소화)
    try:
        with open(EVAL_META_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        info["eval_meta"] = {k: meta.get(k) for k in ("roc_auc", "pr_auc", "th_best", "threshold")}
    except FileNotFoundError:
        pass
    return info

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """
    단건 예측:
    - 입력 검증 후 확률 계산
    - 임계값 기준으로 spam/ham 라벨 결정
    - score_str은 프론트 표시 편의용(고정 소수점 문자열)
    """
    ensure_ready()
    text = validate_text(req.text)
    try:
        prob = float(predict_probs([text])[0])
        label = "spam" if prob >= THRESHOLD else "ham"
        return {"label": label, "score": prob, "score_str": fmt_score(prob)}
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {type(e).__name__}: {e}")

@app.post("/predict/batch", response_model=List[BatchItem])
def predict_batch(req: BatchPredictRequest):
    """
    배치 예측:
    - 입력 리스트 검증 → 일괄 추론 → index 유지하여 결과 매핑
    """
    ensure_ready()
    texts = validate_texts(req.texts)
    try:
        probs = predict_probs(texts)
        out: List[BatchItem] = []
        for i, p in enumerate(probs):
            p = float(p)
            out.append({
                "index": i,
                "label": ("spam" if p >= THRESHOLD else "ham"),
                "score": p,
                "score_str": fmt_score(p),
            })
        return out
    except Exception as e:
        logger.exception("Batch prediction failed")
        raise HTTPException(status_code=500, detail=f"Prediction error: {type(e).__name__}: {e}")

# (옵션) 단건/배치 겸용 엔드포인트 — 프론트에서 payload가 가변일 때 편의용
class PredictFlexible(BaseModel):
    text: Optional[str] = None
    texts: Optional[List[str]] = None

@app.post("/predict-flex")
def predict_flex(req: PredictFlexible):
    """
    'text' 또는 'texts' 중 하나만 보내면 자동 분기
    - 잘못된 조합/빈 입력은 422 처리
    """
    ensure_ready()
    if req.text is None and not req.texts:
        raise HTTPException(status_code=422, detail="Provide 'text' or 'texts'.")
    if req.text is not None:
        return predict(PredictRequest(text=req.text))
    else:
        return predict_batch(BatchPredictRequest(texts=req.texts))  # type: ignore

# -----------------------------------------------------------------------------
# 로컬 직접 실행 (uvicorn)
# - 실제 배포에서는: uvicorn serve:app --host 0.0.0.0 --port 8000 --workers 1(또는 N)
# - 윈도우 로컬에서 cp949 회피: python -X utf8 serve.py
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host=host, port=port)
