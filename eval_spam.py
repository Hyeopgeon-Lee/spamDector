# -*- coding: utf-8 -*-
r"""
eval_spam.py
- 학습된 모델(.keras)을 로드하여 검증셋에서 정량 평가를 수행합니다.
- Accuracy / Precision / Recall / F1 / ROC-AUC / PR-AUC / Confusion Matrix
- 세그먼트 평가: URL/EMAIL/한글 포함 여부
- 최적 임계값(F1 최대) 탐색 및 eval_meta.json 저장

TensorFlow 2.17 (Keras 2.x) ~ 2.20 (Keras 3) 호환
- .keras 내부 assets/*vocab* 파일이 cp949로 저장된 경우 자동 UTF-8 재패킹 후 로드
"""

import sys
# 콘솔 출력 UTF-8 강제 (일부 환경에서 reconfigure 미지원일 수 있음)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import argparse
import json
import re
import os
import zipfile
import tempfile
from dataclasses import asdict, dataclass
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


# ------------------------------------------------------------
# 0) Keras 2/3 호환: register_keras_serializable 헬퍼
# ------------------------------------------------------------
try:
    _register = keras.saving.register_keras_serializable    # Keras 3 (TF 2.20+)
except AttributeError:
    from tensorflow.keras.utils import register_keras_serializable as _register  # Keras 2.x (TF 2.17)


# ------------------------------------------------------------
# 1) 학습 시 사용한 표준화 함수 등록
#    - 주의: 학습 스크립트의 custom_standardize와 "동일"한 이름/시그니처여야 함
#    - RE2 유니코드 카테고리(\p{L}, \p{N}) 사용 버전 (권장)
#      * 만약 과거에 \uac00-\ud7a3 범위 버전으로 학습했다면 아래 주석의 대안으로 교체 가능
# ------------------------------------------------------------
@_register(package="preproc", name="custom_standardize")
def custom_standardize(x: tf.Tensor) -> tf.Tensor:
    x = tf.strings.lower(x)
    x = tf.strings.regex_replace(x, r"(https?://\S+|www\.\S+)", " <url> ")
    x = tf.strings.regex_replace(x, r"\S+@\S+\.\S+", " <email> ")
    x = tf.strings.regex_replace(x, r"\b\d{2,4}-\d{3,4}-\d{4}\b", " <phone> ")
    # RE2 유니코드 범주: 글자(L), 숫자(N)만 허용 (+ 공백/일부기호)
    x = tf.strings.regex_replace(x, r"[^\p{L}\p{N}\s\.,!?%:/@_-]", " ")
    x = tf.strings.regex_replace(x, r"\s+", " ")
    return tf.strings.strip(x)

# # (대안) 과거 학습이 한글 범위 직접 지정이었다면 아래 버전을 custom_standardize로 사용
# @_register(package="preproc", name="custom_standardize")
# def custom_standardize(x: tf.Tensor) -> tf.Tensor:
#     x = tf.strings.lower(x)
#     x = tf.strings.regex_replace(x, r"(https?://\S+|www\.\S+)", " <url> ")
#     x = tf.strings.regex_replace(x, r"\S+@\S+\.\S+", " <email> ")
#     x = tf.strings.regex_replace(x, r"\b\d{2,4}-\d{3,4}-\d{4}\b", " <phone> ")
#     x = tf.strings.regex_replace(x, r"[^a-z0-9\uac00-\ud7a3\s\.,!?%:/@_-]", " ")
#     x = tf.strings.regex_replace(x, r"\s+", " ")
#     return tf.strings.strip(x)


# ------------------------------------------------------------
# 2) UTF-8 자동 교정 로더 (.keras zip 내부 vocab cp949 → utf-8)
# ------------------------------------------------------------
def _repack_keras_zip_utf8(src_path: str) -> str:
    """
    assets/*vocab* 류 텍스트를 cp949→utf-8로 재인코딩해 임시 .keras 경로 반환
    """
    tmpdir = tempfile.mkdtemp(prefix="utf8fix_")
    base = os.path.basename(src_path)
    if not base.endswith(".keras"):
        base += ".keras"
    dst = os.path.join(tmpdir, base.replace(".keras", "_utf8fix.keras"))

    with zipfile.ZipFile(src_path, "r") as zin, zipfile.ZipFile(dst, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            needs = info.filename.startswith("assets/") and any(
                k in info.filename.lower() for k in ("vocab", "vocabulary", "string_lookup")
            )
            if needs:
                # 이미 UTF-8이면 통과, 아니면 cp949로 디코딩 후 UTF-8로 재인코딩
                try:
                    data.decode("utf-8")
                except UnicodeDecodeError:
                    data = data.decode("cp949").encode("utf-8")
            zout.writestr(info, data)
    return dst


def _load_model_with_utf8_fallback(path: str) -> Tuple[keras.Model, str]:
    """
    모델 로드 시 UTF-8 디코딩 문제 발생하면 cp949→utf-8로 재패킹 후 재시도
    반환: (model, effective_model_path)
    """
    try:
        m = keras.models.load_model(
            path,
            compile=False,
            custom_objects={"custom_standardize": custom_standardize},
            safe_mode=False,
        )
        return m, path
    except Exception as e:
        msg = str(e).lower()
        if ("codec can't decode byte" in msg) or ("invalid start byte" in msg) or ("utf-8" in msg and "decode" in msg):
            fixed = _repack_keras_zip_utf8(path)
            m = keras.models.load_model(
                fixed,
                compile=False,
                custom_objects={"custom_standardize": custom_standardize},
                safe_mode=False,
            )
            return m, fixed
        raise


# ------------------------------------------------------------
# 3) 유틸 함수들
# ------------------------------------------------------------
def read_csv_utf8_first_then_cp949(path: str) -> pd.DataFrame:
    try:
        # BOM 포함 UTF-8 우선
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949")


def metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, th: float) -> Dict[str, Any]:
    y_pred = (y_prob >= th).astype(np.int32)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    prec = tp / (tp + fp + 1e-9)
    rec  = tp / (tp + fn + 1e-9)
    f1   = 2 * prec * rec / (prec + rec + 1e-9)
    acc  = (tp + tn) / len(y_true)
    return {"th": th, "acc": acc, "prec": prec, "rec": rec, "f1": f1, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


def confusion(y_true: np.ndarray, y_prob: np.ndarray, th: float) -> List[List[int]]:
    y_pred = (y_prob >= th).astype(np.int32)
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    return [[tn, fp], [fn, tp]]


def predict_probs_in_batches(model: keras.Model, texts: List[str], batch: int = 4096) -> np.ndarray:
    """
    메모리 절약을 위해 배치 단위로 추론
    """
    out = []
    for i in range(0, len(texts), batch):
        bx = tf.constant(texts[i:i + batch], dtype=tf.string)
        out.append(model(bx).numpy().ravel())
    return np.concatenate(out) if out else np.array([], dtype=np.float32)


@dataclass
class EvalSummary:
    roc_auc: float
    pr_auc: float
    th_best: float
    f1_best: float
    acc_best: float
    prec_best: float
    rec_best: float
    cm_best: Any
    th_050: float
    acc_050: float
    prec_050: float
    rec_050: float
    f1_050: float
    cm_050: Any


# ------------------------------------------------------------
# 4) 메인
# ------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/spam_SNS.csv", help="CSV 경로 (content,class)")
    p.add_argument("--model", type=str, default="model/mySpamModel.keras", help="학습 모델(.keras) 경로")
    p.add_argument("--seed", type=int, default=123, help="검증 분할 시드")
    p.add_argument("--val_ratio", type=float, default=0.2, help="검증 비율")
    p.add_argument("--out", type=str, default="model/eval_meta.json", help="결과 저장 경로(JSON)")
    p.add_argument("--batch", type=int, default=4096, help="배치 추론 크기")
    args = p.parse_args()

    # 4-1) 데이터 로드/라벨 인코딩 (UTF-8 우선, 실패 시 cp949)
    df = read_csv_utf8_first_then_cp949(args.data)
    if not {"content", "class"}.issubset(df.columns):
        raise ValueError("CSV에 'content','class' 컬럼이 필요합니다.")
    df = df.copy()
    df["label"] = df["class"].map({"일상대화": 0, "스팸문자": 1}).astype("int32")
    texts = df["content"].astype(str).values
    labels = df["label"].values
    N = len(df)
    print(f"총 샘플: {N}  (spam={labels.sum()}, ham={N - labels.sum()})")

    # 4-2) 8:2 분할(NumPy permutation; 학습 검증셋과 동일하진 않을 수 있음)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(N)
    val_size = int(round(args.val_ratio * N))
    val_idx = perm[-val_size:]
    X_val = texts[val_idx]
    y_val = labels[val_idx]

    # 4-3) 모델 로드 (UTF-8 자동 교정 로더 사용)
    model, eff_model_path = _load_model_with_utf8_fallback(args.model)
    print(f"Loaded model: {args.model}\nEffective path: {eff_model_path}")

    # 4-4) 예측 확률 (배치 추론)
    probs = predict_probs_in_batches(model, list(X_val), batch=args.batch)

    # 4-5) 임계값 탐색 (F1 최대)
    ths = np.linspace(0.05, 0.95, 19)
    f1s = [metrics_at_threshold(y_val, probs, th)["f1"] for th in ths]
    best_idx = int(np.argmax(f1s))
    th_best = float(ths[best_idx])

    m050 = metrics_at_threshold(y_val, probs, 0.5)
    mbest = metrics_at_threshold(y_val, probs, th_best)

    # 4-6) AUC들 (TensorFlow 메트릭 사용)
    m_roc = keras.metrics.AUC(curve="ROC"); m_roc.update_state(y_val, probs)
    m_pr  = keras.metrics.AUC(curve="PR");  m_pr.update_state(y_val, probs)
    roc_auc = float(m_roc.result().numpy())
    pr_auc  = float(m_pr.result().numpy())

    # 4-7) 혼동행렬
    cm050 = confusion(y_val, probs, 0.5)
    cmbest = confusion(y_val, probs, th_best)

    # 4-8) 세그먼트 평가 (URL / EMAIL / 한글 포함)
    url_re   = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
    email_re = re.compile(r"\S+@\S+\.\S+")
    hangul_re = re.compile(r"[\uac00-\ud7a3]")  # Python re는 \p{L} 미지원 → 유니코드 범위 사용

    def seg_mask(texts_arr, pat):
        return np.array([1 if pat.search(t) else 0 for t in texts_arr], dtype=np.int32)

    segs = {
        "has_url": seg_mask(X_val, url_re),
        "has_email": seg_mask(X_val, email_re),
        "has_hangul": seg_mask(X_val, hangul_re),
    }

    def seg_metrics(name, mask_arr, th):
        idx = np.where(mask_arr == 1)[0]
        if len(idx) == 0:
            return name, None
        m = metrics_at_threshold(y_val[idx], probs[idx], th)
        m["count"] = int(len(idx))
        return name, m

    print("\n=== Global metrics ===")
    print(f"ROC-AUC: {roc_auc:.6f}  PR-AUC: {pr_auc:.6f}")
    print(f"[th=0.50] acc={m050['acc']:.6f}  prec={m050['prec']:.6f}  rec={m050['rec']:.6f}  f1={m050['f1']:.6f}  "
          f"TN={m050['tn']} FP={m050['fp']} FN={m050['fn']} TP={m050['tp']}")
    print(f"[best]   th={th_best:.3f}  acc={mbest['acc']:.6f}  prec={mbest['prec']:.6f}  rec={mbest['rec']:.6f}  f1={mbest['f1']:.6f}  "
          f"TN={mbest['tn']} FP={mbest['fp']} FN={mbest['fn']} TP={mbest['tp']}")

    print("\n=== Segment metrics (count, acc/prec/rec/f1) ===")
    def fmt(m): return f"n={m['count']}, acc={m['acc']:.5f}, P={m['prec']:.5f}, R={m['rec']:.5f}, F1={m['f1']:.5f}"
    print("[th=0.50]")
    for k, v in segs.items():
        name, res = seg_metrics(k, v, 0.5)
        if res: print(f"- {name}: {fmt(res)}")
    print(f"[best th={th_best:.3f}]")
    for k, v in segs.items():
        name, res = seg_metrics(k, v, th_best)
        if res: print(f"- {name}: {fmt(res)}")

    # 4-9) 결과 저장(JSON)
    summary = EvalSummary(
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        th_best=th_best,
        f1_best=mbest["f1"],
        acc_best=mbest["acc"],
        prec_best=mbest["prec"],
        rec_best=mbest["rec"],
        cm_best=cmbest,
        th_050=0.5,
        acc_050=m050["acc"],
        prec_050=m050["prec"],
        rec_050=m050["rec"],
        f1_050=m050["f1"],
        cm_050=cm050,
    )
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, ensure_ascii=False, indent=2)
    print(f"\nSaved eval to {args.out}")


if __name__ == "__main__":
    main()
