# Spam Detector (Korean/English) — TF/Keras + FastAPI

한국어/영어 혼합 SNS/메시지 텍스트를 스팸/일상으로 분류하는 프로젝트입니다.  
TensorFlow/Keras(TextVectorization 포함)로 학습한 모델을 FastAPI로 서빙합니다.

---

## ✨ 주요 특징

- **엔드투엔드 파이프라인**: 학습 → 평가 → API 서빙
- **인코딩 안전**: Windows(cp949) 환경에서도 안전하도록 UTF-8 모드/폴백 처리
- **전처리 일관성**: 학습/평가/서빙 모두 동일한 `custom_standardize` 등록
- **운영 친화**: CORS, 일관된 오류 응답, 환경변수 기반 임계값(threshold)
- **컨테이너화**: Docker 이미지 제공 (`hyeopgeonlee/spam-dector`)

---

## 📦 포함된 구성요소

- **학습 스크립트**: `spam_model.py`  
  - 데이터 로드(UTF-8 우선), 학습/검증 분할(8:2), TextVectorization **학습 분할만 adapt**  
  - TextCNN 멀티브랜치(커널 3/4/5) + AdamW + CosineDecay  
  - AUC 기준 조기종료/체크포인트, 최종 모델 `.keras` 저장  
  - 검증에서 **ROC-AUC/PR-AUC**, **F1 최대 임계값(th_best)** 계산 → `model/eval_meta.json`

- **평가 스크립트**: `eval_spam.py`  
  - 저장된 `.keras` 모델을 로드해 **독립 분할**로 재평가  
  - **세그먼트별 성능**(URL/EMAIL/한글 포함), 혼동행렬, best/0.5 임계값 지표 저장  
  - **윈도우 인코딩(cp949) vocabulary 자동 교정 로더** 포함

- **API 서버**: `serve.py` (FastAPI)  
  - 엔드포인트: `/health`, `/predict`, `/predict/batch`, `/predict-flex`  
  - CORS, 일관된 오류 포맷, 임계값/메타 노출  
  - 학습 시 사용한 `custom_standardize`를 **동일 이름/로직**으로 등록

---

## 🗂 폴더 구조

```
.
├─ data/
│  └─ spam_SNS.csv                # 'content','class' 컬럼
├─ model/
│  ├─ mySpamModel.keras           # 최종 모델
│  ├─ model_checkpoint.keras      # val_auc 기준 베스트
│  └─ eval_meta.json              # roc_auc, pr_auc, th_best 등
├─ spam_model.py                  # 학습
├─ eval_spam.py                   # 평가
├─ serve.py                       # FastAPI 서버
└─ README.md
```

---

## 🧾 데이터셋 형식

- CSV 컬럼:  
  - `content` : 텍스트  
  - `class`   : `일상대화`(ham) 또는 `스팸문자`(spam)
- 인코딩: **`utf-8-sig` 권장** (실패 시 cp949로 폴백 처리)

---

## 🧪 학습(Training)

> **Windows라면** 파이썬을 **UTF-8 모드**로 실행하세요. (Keras가 vocabulary 자산을 UTF-8로 기록)

```bash
# 권장 (Windows/PowerShell)
python -X utf8 spam_model.py

# macOS/Linux
python spam_model.py
```

완료 후 생성물:
- `model/mySpamModel.keras`  
- `model/model_checkpoint.keras`  
- `model/eval_meta.json` (예: `{"roc_auc":0.99,"pr_auc":0.99,"th_best":0.63,...}`)

---

## ✅ 평가(Evaluation)

```bash
# 동일/다른 머신에서 모델을 점검
python eval_spam.py --data data/spam_SNS.csv --model model/mySpamModel.keras --out model/eval_meta.json
```

기능:
- Accuracy/Precision/Recall/F1/ROC-AUC/PR-AUC
- best(=F1 최대 임계값) & 0.5 기준 지표, 혼동행렬
- 세그먼트(contains URL/email/한글)별 지표

---

## 🚀 API 서버 실행

### 로컬(Python)

```bash
# Windows 권장: UTF-8 모드
python -X utf8 serve.py

# 또는 uvicorn
uvicorn serve:app --host 0.0.0.0 --port 8000
```

### 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `MODEL_PATH` | `model/mySpamModel.keras` | 로드할 Keras 모델 경로 |
| `EVAL_META_PATH` | `model/eval_meta.json` | `th_best` 등 메타 읽기 |
| `THRESHOLD` | (meta의 `th_best` \|\| `0.5`) | 분류 임계값(오버라이드용) |
| `SCORE_DECIMALS` | `6` | 점수 문자열 소수 자리수 |
| `HOST` / `PORT` | `0.0.0.0` / `8000` | 바인딩 주소/포트 |
| `CORS_ALLOW_ORIGINS` | `*` | CORS 허용(콤마 구분) |
| `TEXT_MAX_CHARS` | `5000` | 한 샘플 최대 길이 |
| `BATCH_MAX_ITEMS` | `256` | 배치 최대 건수 |

---

## 📡 API 사용법

FastAPI 문서:
- Swagger UI: `http://<host>:8000/docs`  
- ReDoc: `http://<host>:8000/redoc`

### 공통
- Content-Type: `application/json`
- 라벨 기준: `prob >= THRESHOLD` → `"spam"` else `"ham"`

### 1) Health

`GET /health`

응답 예:
```json
{
  "ok": true,
  "model_path": "model/mySpamModel.keras",
  "threshold": 0.63,
  "text_max_chars": 5000,
  "batch_max_items": 256,
  "cors_allow_origins": ["*"],
  "eval_meta": { "roc_auc": 0.991, "pr_auc": 0.990, "th_best": 0.63 }
}
```

### 2) 단건 예측

`POST /predict`

요청:
```json
{ "text": "무료 쿠폰 링크 클릭하세요!" }
```

응답:
```json
{ "label": "spam", "score": 0.874321, "score_str": "0.874321" }
```

### 3) 배치 예측

`POST /predict/batch`

요청:
```json
{ "texts": ["무료 쿠폰 링크 클릭", "오늘 회의 일정 공유드립니다."] }
```

응답:
```json
[
  { "index": 0, "label": "spam", "score": 0.91, "score_str": "0.910000" },
  { "index": 1, "label": "ham",  "score": 0.08, "score_str": "0.080000" }
]
```

### 4) Flexible

`POST /predict-flex`

- `{"text": "..."} | {"texts": [...]}` 둘 중 하나를 보내면 `/predict` 또는 `/predict/batch`로 위임

### 오류 응답 포맷(공통)

```json
{
  "ok": false,
  "error": {
    "type": "RequestValidationError",
    "message": "Invalid request body.",
    "details": []
  }
}
```

기타 에러 타입 예: `HTTPException`, `InternalServerError`

---

## 🐳 Docker 이미지

### 이미지 경로
- **Docker Hub:** `docker.io/hyeopgeonlee/spam-dector:latest`  
  (권장: 운영에서는 `:latest` 대신 고정 태그 사용 예: `:v0.1.0`)

### 빠른 시작 (Quick start)

```bash
# 1) 이미지 받기
docker pull hyeopgeonlee/spam-dector:latest

# 2) 실행 (호스트 8000 → 컨테이너 8000)
docker run --rm -p 8000:8000   --name spam-api   hyeopgeonlee/spam-dector:latest
```

- API 문서: http://localhost:8000/docs  
- 헬스체크: http://localhost:8000/health

### 모델/메타 파일 외부 마운트 (선택)

이미지에 모델이 포함되어 있지 않거나 바꿔서 쓰고 싶다면, 호스트의 `./model`을 컨테이너 `/app/model`로 마운트하세요.

```bash
# macOS/Linux
docker run -d -p 8000:8000   -v "$(pwd)/model:/app/model"   -e MODEL_PATH=/app/model/mySpamModel.keras   -e EVAL_META_PATH=/app/model/eval_meta.json   --name spam-api   hyeopgeonlee/spam-dector:latest

# Windows PowerShell
docker run -d -p 8000:8000 `
  -v "${PWD}\model:/app/model" `
  -e MODEL_PATH=/app/model/mySpamModel.keras `
  -e EVAL_META_PATH=/app/model/eval_meta.json `
  --name spam-api `
  hyeopgeonlee/spam-dector:latest
```

### Docker Compose 예시

```yaml
version: "3.9"
services:
  spam-api:
    image: hyeopgeonlee/spam-dector:latest
    container_name: spam-api
    ports:
      - "8000:8000"
    environment:
      MODEL_PATH: /app/model/mySpamModel.keras
      EVAL_META_PATH: /app/model/eval_meta.json
      # THRESHOLD: "0.63"   # 필요 시 오버라이드
      CORS_ALLOW_ORIGINS: "https://your-frontend.example.com"
      TF_CPP_MIN_LOG_LEVEL: "2"
      OMP_NUM_THREADS: "1"
      TF_NUM_INTRAOP_THREADS: "1"
      TF_NUM_INTEROP_THREADS: "1"
    volumes:
      - ./model:/app/model:ro
    restart: unless-stopped
```

```bash
docker compose up -d
```

### Kubernetes 배포 예시

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: spam-api
  # namespace: kopo-trainee1   # 네임스페이스 쓰면 주석 해제
spec:
  replicas: 1
  selector:
    matchLabels:
      app: spam-api
  template:
    metadata:
      labels:
        app: spam-api
    spec:
      containers:
        - name: spam-api
          image: hyeopgeonlee/spam-dector:latest
          imagePullPolicy: IfNotPresent
          ports:
            - containerPort: 8000
          env:
            - name: MODEL_PATH
              value: /app/model/mySpamModel.keras
            - name: EVAL_META_PATH
              value: /app/model/eval_meta.json
            - name: TF_CPP_MIN_LOG_LEVEL
              value: "2"
          readinessProbe:
            httpGet: { path: /health, port: 8000 }
            initialDelaySeconds: 5
            periodSeconds: 10
          livenessProbe:
            httpGet: { path: /health, port: 8000 }
            initialDelaySeconds: 10
            periodSeconds: 20
          resources:
            requests:
              cpu: "100m"
              memory: "256Mi"
            limits:
              cpu: "500m"
              memory: "512Mi"
---
apiVersion: v1
kind: Service
metadata:
  name: spam-api
  # namespace: kopo-trainee1
spec:
  type: ClusterIP
  selector:
    app: spam-api
  ports:
    - port: 8000
      targetPort: 8000

```

> 프로덕션에서는 Ingress/HTTPS, 인증·레이트리밋 등을 반드시 고려하세요.

### 태그 전략

- 최신: `hyeopgeonlee/spam-dector:latest`  
- 고정 버전 예: `hyeopgeonlee/spam-dector:v0.1.0`  
  → **운영에선 고정 태그를 권장**합니다.

### 트러블슈팅

- **포트가 열리지 않음**: 컨테이너 실행 시 `-p 8000:8000` 매핑 확인  
  `docker port spam-api` 결과에 `0.0.0.0:8000->8000/tcp`가 보여야 합니다.
- **모델 로드 실패**: `MODEL_PATH`/`EVAL_META_PATH` 경로 및 `custom_standardize` 일치 여부 점검  
- **한글/인코딩 문제**: 컨테이너는 기본 UTF-8로 동작(윈도우 cp949 이슈 없음)
- **메모리 경고**: 컨테이너 메모리 리미트 상향 또는 스레드/워커 수 줄이기

---

## 🔐 라이선스

이 프로젝트는 **Apache License 2.0**을 따릅니다. 

---

## 👤 작성자

- 한국폴리텍대학 서울강서캠퍼스 **빅데이터소프트웨어과**  
- **이협건 교수**  
- ✉️ hglee67@kopo.ac.kr  
- 🔗 빅데이터소프트웨어과 입학 상담 **오픈채팅방**: (https://open.kakao.com/o/gEd0JIad)
