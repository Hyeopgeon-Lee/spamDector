# -*- coding: utf-8 -*-
"""
[학습 스크립트: spam_model.py — 상세 주석판]

목표
- 한국어/영어 혼합 스팸 텍스트를 TF/Keras 기반으로 학습하고, 모델(.keras)과 평가 메타(JSON)를 저장합니다.
- Windows(cp949) 환경에서도 **읽기/저장/정규식**으로 인한 인코딩 문제를 피합니다.

필수 실행 팁(Windows 권장)
- 파이썬 프로세스를 **UTF-8 모드**로 실행하세요. (저장 시 Keras가 vocab 텍스트를 UTF-8로 기록)
  * PowerShell:  python -X utf8 spam_model.py
  * 또는 Run Config env에  PYTHONUTF8=1  설정

데이터 전제
- CSV: 컬럼명 'content', 'class' (class ∈ {"일상대화", "스팸문자"})
- 파일 경로: data/spam_SNS.csv  (필요 시 아래 csv_path 수정)

모델 개요
- 입력: 문자열 1개(배치)
- 전처리: TextVectorization(표준화 함수 포함, 학습 분할로만 adapt → 검증 누수 방지)
- 네트워크: TextCNN 멀티브랜치(커널 3/4/5) + GlobalMaxPool + Dense
- 옵티마이저: AdamW + CosineDecay 스케줄
- 조기종료: val_auc 모니터링
- 산출물:
  * model/mySpamModel.keras              : 최종 모델(직렬화 포함)
  * model/model_checkpoint.keras         : val_auc 기준 베스트 스냅샷
  * model/eval_meta.json                 : ROC-AUC, PR-AUC, th_best(F1 최대 임계값)

주의사항
- 학습/평가/서빙에서 **custom_standardize** 구현이 반드시 동일해야 안전합니다.
- 정규식은 RE2 엔진 제약을 고려하여 유니코드 범주(\p{L}, \p{N})를 사용했습니다.
"""

import sys
# --- (콘솔) 출력 인코딩을 UTF-8로 강제 ---
#  - 표준 출력/에러 스트림이 UTF-8이 아니면, 한글 로그가 깨질 수 있습니다.
#  - 일부 환경에선 reconfigure 미지원 → 예외 발생 시 무시합니다.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import os
# 주의: PYTHONUTF8/PYTHONIOENCODING은 "인터프리터 시작 시"만 반영됩니다.
# 코드 중간에 os.environ으로 바꿔도 저장단 인코딩 이슈는 완전히 해결되지 않습니다.
# UTF-8 모드로 프로세스를 시작하세요. (위 실행 팁 참고)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pathlib
import json

# --- 난수 고정(완전 재현은 아님) ---
#  - TF/CUDA/병렬성 등 외적 요인으로 완벽한 재현은 어렵지만, 기본적인 일관성을 위해 설정합니다.
tf.keras.utils.set_random_seed(123)

# ==============================
# Keras 2/3 호환: 직렬화 등록 헬퍼
#  - Keras 3: keras.saving.register_keras_serializable
#  - Keras 2: tensorflow.keras.utils.register_keras_serializable
# ==============================
try:
    register = keras.saving.register_keras_serializable  # Keras 3 (TF 2.20+)
except AttributeError:
    from tensorflow.keras.utils import register_keras_serializable as register  # Keras 2 (TF 2.17)

# ==============================
# 경로/출력 디렉토리
# ==============================
csv_path = pathlib.Path("data/spam_SNS.csv")   # 입력 CSV (content,class)
save_dir = pathlib.Path("model")                # 산출물 저장 폴더
save_dir.mkdir(parents=True, exist_ok=True)

# ==============================
# 데이터 로드 + 라벨 인코딩 (UTF-8 안전)
#  - UTF-8-SIG 우선 → 실패 시 cp949 폴백
#  - 실무에서 엑셀 저장본은 BOM이 섞이는 경우가 많아 'utf-8-sig'가 안전합니다.
# ==============================
def read_csv_utf8_first_then_cp949(path: pathlib.Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949")

df = read_csv_utf8_first_then_cp949(csv_path)

# 라벨 맵핑: '일상대화'→0 (ham), '스팸문자'→1 (spam)
label_map = {"일상대화": 0, "스팸문자": 1}
df["label"] = df["class"].map(label_map).astype("int32")

# 문자열/라벨 배열로 분리
texts  = df["content"].astype(str).values
labels = df["label"].values

N = len(texts)
print(f"총 샘플: {N}  (spam={labels.sum()}, ham={N - labels.sum()})")

# ==============================
# TensorFlow 파이프라인으로 8:2 분할
#  - shuffle(seed 고정) → take/skip으로 학습/검증 분할
#  - cache+prefetch로 I/O 병목 완화
# ==============================
batch_size = 128
AUTOTUNE = tf.data.AUTOTUNE

full_ds = tf.data.Dataset.from_tensor_slices((texts, labels))
full_ds = full_ds.shuffle(buffer_size=N, seed=123, reshuffle_each_iteration=False)

train_size = int(0.8 * N)
train_raw = full_ds.take(train_size)   # (text, label)
val_raw   = full_ds.skip(train_size)

train_ds = train_raw.batch(batch_size).cache().prefetch(AUTOTUNE)
val_ds   = val_raw.batch(batch_size).cache().prefetch(AUTOTUNE)

# ==============================
# 전처리/벡터화 (TextVectorization)
#  - 표준화: custom_standardize (URL/EMAIL/PHONE 토큰화 + 문자 필터)
#  - 토큰 범위: 유니코드 범주(\p{L}:문자, \p{N}:숫자) + 공백/일부 기호만 허용
#  - 길이: 최대 220 토큰으로 패딩/잘라내기
#  - 누수 방지: adapt()는 **학습 분할 텍스트만** 사용
# ==============================
MAX_TOKENS = 40000
MAX_LEN    = 220

@register(package="preproc", name="custom_standardize")
def custom_standardize(x: tf.Tensor) -> tf.Tensor:
    """
    표준화 단계에서 텍스트를 정리:
    - 소문자화
    - URL/EMAIL/전화번호를 특수 토큰으로 치환(<url>, <email>, <phone>)
    - 허용 문자(문자/숫자/공백/일부 구두점) 외는 공백으로 대체
    - 과도한 공백 압축
    주의: TextVectorization의 regex는 RE2 규칙을 따릅니다.
    """
    x = tf.strings.lower(x)
    x = tf.strings.regex_replace(x, r"(https?://\S+|www\.\S+)", " <url> ")
    x = tf.strings.regex_replace(x, r"\S+@\S+\.\S+", " <email> ")
    x = tf.strings.regex_replace(x, r"\b\d{2,4}-\d{3,4}-\d{4}\b", " <phone> ")
    # RE2 유니코드 범주: L(letters), N(numbers)
    x = tf.strings.regex_replace(x, r"[^\p{L}\p{N}\s\.,!?%:/@_-]", " ")
    x = tf.strings.regex_replace(x, r"\s+", " ")
    return tf.strings.strip(x)

vectorize = layers.TextVectorization(
    max_tokens=MAX_TOKENS,                 # 어휘 사전 상한(빈도순 상위만 유지)
    output_mode="int",                     # 정수 시퀀스 출력
    output_sequence_length=MAX_LEN,        # 고정 길이 시퀀스(패딩/트렁케이션)
    standardize=custom_standardize,        # 위에서 등록한 동일 함수
    split="whitespace",                    # 공백 기준 토큰화(표준화에서 기호 정리됨)
)

# --- 누수 방지: adapt는 학습 분할 텍스트로만 ---
train_texts_ds = train_raw.map(lambda x, y: x).batch(256)
vectorize.adapt(train_texts_ds)

# ==============================
# 모델 구성 (TextCNN 멀티브랜치)
#  - 임베딩 → Conv1D(커널 3/4/5) 병렬 → GlobalMaxPool → Concatenate
#  - Dropout + L2로 과적합 억제
# ==============================
EMBED_DIM = 128
L2 = keras.regularizers.l2(1e-6)

inp = layers.Input(shape=(1,), dtype=tf.string, name="text")
x = vectorize(inp)                                         # [B, MAX_LEN]
x = layers.Embedding(input_dim=MAX_TOKENS, output_dim=EMBED_DIM)(x)  # [B, L, D]
x = layers.SpatialDropout1D(0.2)(x)                        # 임베딩 레벨 드롭아웃

branches = []
for k in (3, 4, 5):                                       # n-gram 감지용 커널
    b = layers.Conv1D(128, k, padding="same", activation="relu", kernel_regularizer=L2)(x)
    b = layers.GlobalMaxPooling1D()(b)                    # 채널별 최대값(핵심 특징 추출)
    branches.append(b)

x = layers.Concatenate()(branches)                         # [B, 128*3]
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation="relu", kernel_regularizer=L2)(x)
out = layers.Dense(1, activation="sigmoid")(x)             # 이진확률(스팸일 확률)

model = models.Model(inp, out)

# ==============================
# 옵티마이저 & 러닝레이트 스케줄
#  - CosineDecay: 초반 빠르게 학습, 말미에 서서히 감소
#  - AdamW: Adam + weight decay(일반 L2와 달리 업데이트 경로 분리)
#  - clipnorm: 그라디언트 폭주 방지
# ==============================
epochs = 15
steps_per_epoch = max(1, train_size // batch_size)
decay_steps = steps_per_epoch * epochs

lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=decay_steps,
    alpha=1e-2,  # 최저 lr ≈ 1e-5
)
optimizer = keras.optimizers.AdamW(
    learning_rate=lr_schedule,
    weight_decay=1e-4,
    clipnorm=1.0,
)

# ==============================
# 컴파일(메트릭)
#  - AUC 이름을 'auc'로 고정 → 콜백 monitor='val_auc'와 일치시킴
#  - loss는 시그모이드에 맞춰 BinaryCrossentropy(from_logits=False)
# ==============================
model.compile(
    optimizer=optimizer,
    loss=keras.losses.BinaryCrossentropy(from_logits=False),
    metrics=["accuracy", keras.metrics.AUC(name="auc", curve="ROC")],
)

# ==============================
# 콜백
#  - EarlyStopping: val_auc 기준, 2 epoch 정체 시 중단(최고 가중치 복원)
#  - ModelCheckpoint: val_auc 최고 모델을 .keras로 저장
#    * 주의: Windows cp949 기본 인코딩이면 저장 시 vocab 텍스트 쓰기에서 에러 가능
#            → 반드시 UTF-8 모드로 파이썬 시작(상단 실행 팁).
# ==============================
early_stopping = keras.callbacks.EarlyStopping(
    monitor="val_auc", mode="max", patience=2, restore_best_weights=True
)
ckpt_cb = keras.callbacks.ModelCheckpoint(
    filepath=str(save_dir / "model_checkpoint.keras"),
    save_best_only=True,
    monitor="val_auc",
    mode="max",
)

# ==============================
# 학습
# ==============================
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    callbacks=[early_stopping, ckpt_cb],
)

# ==============================
# 모델 저장
#  - 최종 전체 모델(전처리 포함)을 .keras로 저장
#  - custom_standardize가 등록되어 직렬화 안전
# ==============================
model_path = save_dir / "mySpamModel.keras"
model.save(str(model_path))
print("✅ Training done.")
print(f"✅ Saved model: {model_path}")
print(f"✅ Best checkpoint: {save_dir / 'model_checkpoint.keras'}")

# ==============================
# 검증셋 평가 & 임계값 저장 (서빙 활용)
#  - 검증셋에서 ROC-AUC/PR-AUC 측정
#  - F1 최대가 되는 임계값(th_best) 탐색 → eval_meta.json 저장
#  - 서빙 시 이 임계값을 사용하면 운영 라벨링 품질을 쉽게 맞출 수 있습니다.
# ==============================
def collect_probs(ds):
    probs, y_true = [], []
    for xb, yb in ds:
        p = model(xb).numpy().ravel()
        probs.append(p)
        y_true.append(yb.numpy())
    return np.concatenate(probs), np.concatenate(y_true)

probs, y_val = collect_probs(val_ds)

# AUC 계산 (TF 메트릭 사용)
m_roc = keras.metrics.AUC(curve="ROC"); m_roc.update_state(y_val, probs)
m_pr  = keras.metrics.AUC(curve="PR");  m_pr.update_state(y_val, probs)
roc_auc = float(m_roc.result().numpy())
pr_auc  = float(m_pr.result().numpy())

# --- F1 최대 임계값 탐색 ---
def metrics_at(y_true, y_prob, th: float):
    y_pred = (y_prob >= th).astype(np.int32)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    acc  = (tp + tn) / len(y_true)
    return {"th": th, "acc": acc, "prec": prec, "rec": rec, "f1": f1, "tp": tp, "fp": fp, "tn": tn, "fn": fn}

ths = np.linspace(0.05, 0.95, 19)                  # 0.05 간격 그리드 검색
f1s = [metrics_at(y_val, probs, t)["f1"] for t in ths]
th_best = float(ths[int(np.argmax(f1s))])

# --- 메타 저장(JSON, UTF-8 보장) ---
meta = {
    "roc_auc": roc_auc,
    "pr_auc": pr_auc,
    "th_best": th_best,
    "note": "Scores computed on validation split (20%).",
}
with open(save_dir / "eval_meta.json", "w", encoding="utf-8") as f:
    json.dump(meta, f, ensure_ascii=False, indent=2)
print("✅ Saved eval meta:", save_dir / "eval_meta.json")

# ==============================
# 학습 곡선 시각화(옵션)
#  - 노트북/개발 환경에서 즉시 확인용
#  - 서버 환경(headless)라면 주석 처리하거나 저장하도록 변경하세요.
# ==============================
acc = history.history.get("accuracy", [])
val_acc = history.history.get("val_accuracy", [])
loss = history.history.get("loss", [])
val_loss = history.history.get("val_loss", [])
epochs_range = range(len(acc))

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")

plt.tight_layout()
plt.show()
