# Spam Detector (Korean/English) — TF/Keras + FastAPI

한국어/영어 혼합 SNS/메시지 텍스트를 스팸/일상으로 분류하는 프로젝트입니다.  
TensorFlow/Keras(TextVectorization 포함)로 학습한 모델을 FastAPI로 서빙합니다.

---

## 무엇이 들어있나요?

- **학습 스크립트**: `spam_model.py`  
  - 데이터 로드(UTF-8 우선), 학습/검증 분할(8:2), TextVectorization **학습 분할만 adapt**  
  - TextCNN 멀티브랜치(커널 3/4/5) + AdamW + CosineDecay  
  - AUC 기준 조기종료/체크포인트, 최종 모델 `.keras` 저장  
  - 검증셋에서 **ROC-AUC/PR-AUC** 및 **F1 최대 임계값(th_best)** 계산 → `model/eval_meta.json`

- **평가 스크립트**: `eval_spam.py`  
  - 저장된 `.keras` 모델을 로드해 **독립 분할**로 다시 평가  
  - **세그먼트별 성능**(URL/EMAIL/한글 포함), 혼동행렬, best/0.5 임계값 지표 저장  
  - **윈도우 인코딩(cp949) vocabulary 자동 교정 로더** 포함

- **API 서버**: `serve.py` (FastAPI)  
  - 엔드포인트: `/health`, `/predict`, `/predict/batch`, `/predict-flex`  
  - CORS, 일관된 오류 포맷, 임계값/메타 노출  
  - 학습 시 사용한 `custom_standardize`를 **동일 이름/로직**으로 등록

---

## 폴더 구조

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

## 데이터셋 형식

- CSV 컬럼:  
  - `content` : 텍스트  
  - `class`   : `일상대화`(ham) 또는 `스팸문자`(spam)
- 인코딩: **`utf-8-sig` 권장** (실패 시 cp949로 폴백 처리)

---

## 학습(Training)

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

## 평가(Evaluation)

```bash
# 동일/다른 머신에서 모델을 점검
python eval_spam.py --data data/spam_SNS.csv --model model/mySpamModel.keras --out model/eval_meta.json
```

기능:
- Accuracy/Precision/Recall/F1/ROC-AUC/PR-AUC
- best(=F1 최대 임계값) & 0.5 기준 지표, 혼동행렬
- 세그먼트(contains URL/email/한글)별 지표

---

## API 서버 실행

### 로컬(Python)

```bash
# Windows 권장: UTF-8 모드
python -X utf8 serve.py

# 또는 uvicorn
uvicorn serve:app --host 0.0.0.0 --port 8000
```

### Docker (예시)

```bash
# 빌드
docker build -t spam-api:latest .

# 실행 (포트 공개)
docker run -d --name spam-api -p 8000:8000   -e MODEL_PATH=/app/model/mySpamModel.keras   -e EVAL_META_PATH=/app/model/eval_meta.json   spam-api:latest
```

> Docker/Linux는 기본 UTF-8이므로 윈도우 cp949 문제는 재발하지 않습니다.

### 환경변수

- `MODEL_PATH` (기본: `model/mySpamModel.keras`)  
- `EVAL_META_PATH` (기본: `model/eval_meta.json`)  
- `THRESHOLD` (기본: meta의 `th_best` → 없으면 `0.5`)  
- `SCORE_DECIMALS` (기본: `6`)  
- `HOST` / `PORT` (기본: `0.0.0.0` / `8000`)  
- `CORS_ALLOW_ORIGINS` (기본: `*` / 예: `https://a.com,https://b.com`)  
- `TEXT_MAX_CHARS` (기본: `5000`) / `BATCH_MAX_ITEMS` (기본: `256`)

---

## API 사용법

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

## 운영/배포 팁

- **Windows/파이참**: Run Configuration에 `-X utf8`(Interpreter options) 또는 `PYTHONUTF8=1`(Env) 설정  
- **Threshold 운용**: 기본은 `eval_meta.json`의 `th_best`; 실시간 민감도 조정이 필요하면 `THRESHOLD` 환경변수로 오버라이드  
- **CORS**: `CORS_ALLOW_ORIGINS`에 배포 도메인을 콤마로 나열  
- **리소스**: 컨테이너 메모리 낮으면 워커=1 권장, 스레드 제한(예: `OMP_NUM_THREADS=1`)  
- **보안**: 퍼블릭으로 노출 시 반드시 리버스 프록시(HTTPS)·API 키·레이트리밋 고려

---

## 트러블슈팅

- **UnicodeEncodeError / cp949**  
  - 증상: 모델 저장/로드 중 `'cp949' codec can't encode/decode ...`  
  - 해결: **파이썬/서버를 UTF-8 모드로 실행** (`-X utf8` 또는 `PYTHONUTF8=1`)  
  - 데이터 CSV는 `utf-8-sig` 권장

- **모델 로드 실패 (TextVectorization/StringLookup)**  
  - 원인: 학습/서빙의 `custom_standardize` **이름/로직 불일치** 혹은 인코딩  
  - 조치: `serve.py`의 `custom_standardize`를 **학습과 동일**하게 맞춤

---

## 라이선스

프로젝트 목적에 맞는 라이선스를 선택해 `LICENSE` 파일을 추가해 주세요. (예: MIT/Apache-2.0)

---

## 📚 작성자

- 한국폴리텍대학 서울강서캠퍼스 **빅데이터과**
- **이협건 교수**
- ✉️ hglee67@kopo.ac.kr
- 🔗 빅데이터학과 입학 상담 **오픈채팅방**: (링크 추가)
