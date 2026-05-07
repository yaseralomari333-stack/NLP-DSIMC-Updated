"""
RUN.py
======
Single entry-point for the project.

Sub-commands
------------
  eda             Run EDA + write plots                  -> artifacts/eda/
  train_baseline  Train TF-IDF + classical ML baselines  -> artifacts/*.joblib
  train_marbert   Fine-tune MARBERT                       -> artifacts/transformer/best/
  compare         Compare every trained model on the same test split
  predict         Inference on text / file / dataset
  all             eda -> train_baseline -> train_marbert -> compare

Examples
--------
  # Local
  python RUN.py eda             --data ./data/dataset.xlsx
  python RUN.py train_baseline  --data ./data/dataset.xlsx
  python RUN.py train_marbert   --data ./data/dataset.xlsx --fp16
  python RUN.py compare         --data ./data/dataset.xlsx
  python RUN.py predict         --text "حسيت اني تعبت من كل شي"
  python RUN.py all             --data ./data/dataset.xlsx --fp16

  # Colab (after mounting drive at /content/drive)
  !python RUN.py all --data "/content/drive/MyDrive/NLP/Arabic Suicidal Dataset.xlsx" --fp16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---- DAL
from DAL import (
    ARTIFACTS_DIR,
    clean_text,
    load_dataset,
    plot_comparison_bars,
    prepare_splits,
    run_eda,
)

# ---- Models
from Models.BaselineModel import BASELINE_MODELS
from Models.BaselineOps   import BaselineOps
from Models.MARBERTOps    import MARBERTConfig, MARBERTOps


# ===========================================================================
# Sub-commands
# ===========================================================================
def cmd_eda(args) -> None:
    out = run_eda(args.data, args.text_col, args.label_col)
    print(json.dumps(out, indent=2, ensure_ascii=False))


def cmd_train_baseline(args) -> None:
    df, t, l = load_dataset(args.data, args.text_col, args.label_col)
    train_df, val_df, test_df, classes = prepare_splits(
        df, t, l,
        test_size=args.test_size, val_size=args.val_size, seed=args.seed,
    )
    print(f"Train={len(train_df)}  Val={len(val_df)}  Test={len(test_df)}  "
          f"Classes={classes}")

    # baselines train on train+val (no separate val needed)
    X_tr = pd.concat([train_df, val_df])["text_clean"].tolist()
    y_tr = pd.concat([train_df, val_df])["label_id"].values
    X_te = test_df["text_clean"].tolist()
    y_te = test_df["label_id"].values

    all_metrics = []
    for key in args.models:
        print(f"\n>>> Training baseline: {key}")
        ops = BaselineOps(model_key=key)
        ops.train(X_tr, y_tr)
        metrics = ops.evaluate(X_te, y_te, classes, ARTIFACTS_DIR)
        ops.save(ARTIFACTS_DIR / f"{key}.joblib")
        all_metrics.append(metrics)
        print(json.dumps(metrics, indent=2))

    out = ARTIFACTS_DIR / "baseline_metrics.json"
    out.write_text(json.dumps(all_metrics, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    print(f"\nSaved -> {out}")


def cmd_train_marbert(args) -> None:
    df, t, l = load_dataset(args.data, args.text_col, args.label_col)
    train_df, val_df, test_df, classes = prepare_splits(
        df, t, l,
        test_size=args.test_size, val_size=args.val_size, seed=args.seed,
    )
    print(f"Train={len(train_df)}  Val={len(val_df)}  Test={len(test_df)}  "
          f"Classes={classes}")

    cfg = MARBERTConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        fp16=args.fp16,
        seed=args.seed,
    )
    ops = MARBERTOps(cfg)
    ops.train(train_df, val_df, classes, ARTIFACTS_DIR)
    metrics = ops.evaluate(test_df, ARTIFACTS_DIR)
    ops.save(ARTIFACTS_DIR / "transformer" / "best")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


def cmd_compare(args) -> None:
    """Re-build the SAME test split, score every available model, plot."""
    import joblib
    from sklearn.metrics import (
        f1_score, precision_score, recall_score,
    )

    le_path = ARTIFACTS_DIR / "label_encoder.joblib"
    if not le_path.exists():
        sys.exit("Run `train_baseline` first to produce label_encoder.joblib")
    le = joblib.load(le_path)

    df, t, l = load_dataset(args.data, args.text_col, args.label_col)
    df = df.copy()
    df["label_id"] = le.transform(df[l].astype(str))

    from sklearn.model_selection import train_test_split
    _, test_df = train_test_split(
        df, test_size=args.test_size, random_state=args.seed,
        stratify=df["label_id"],
    )
    X_te = test_df["text_clean"].tolist()
    y_te = test_df["label_id"].values
    classes = list(le.classes_)
    print(f"Test size: {len(y_te)}")

    rows: list[dict] = []

    # Classical baselines
    for key in BASELINE_MODELS:
        p = ARTIFACTS_DIR / f"{key}.joblib"
        if not p.exists():
            print(f"[skip] {p} not found")
            continue
        ops = BaselineOps.load(p)
        pred = ops.predict(X_te)
        rows.append({
            "model": key,
            "accuracy":         float((pred == y_te).mean()),
            "precision_macro":  precision_score(y_te, pred, average="macro",
                                                zero_division=0),
            "recall_macro":     recall_score(y_te, pred, average="macro",
                                             zero_division=0),
            "f1_macro":         f1_score(y_te, pred, average="macro"),
            "f1_weighted":      f1_score(y_te, pred, average="weighted"),
        })
        print(f"{key:8s}  F1={rows[-1]['f1_macro']:.4f}  "
              f"Recall={rows[-1]['recall_macro']:.4f}")

    # MARBERT
    marbert_dir = ARTIFACTS_DIR / "transformer" / "best"
    if marbert_dir.exists():
        ops = MARBERTOps.load(marbert_dir)
        pred_ids, _ = ops.predict_with_confidence(X_te)
        rows.append({
            "model": "MARBERT",
            "accuracy":         float((pred_ids == y_te).mean()),
            "precision_macro":  precision_score(y_te, pred_ids, average="macro",
                                                zero_division=0),
            "recall_macro":     recall_score(y_te, pred_ids, average="macro",
                                             zero_division=0),
            "f1_macro":         f1_score(y_te, pred_ids, average="macro"),
            "f1_weighted":      f1_score(y_te, pred_ids, average="weighted"),
        })
        print(f"MARBERT   F1={rows[-1]['f1_macro']:.4f}  "
              f"Recall={rows[-1]['recall_macro']:.4f}")
    else:
        print(f"[skip] {marbert_dir} not found - run train_marbert first")

    if not rows:
        sys.exit("No models available to compare.")

    df_out = pd.DataFrame(rows)
    df_out.to_csv(ARTIFACTS_DIR / "comparison_table.csv", index=False)
    (ARTIFACTS_DIR / "comparison.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    plot_comparison_bars(rows, ARTIFACTS_DIR / "plots" / "comparison_bars.png")

    print("\n=== Comparison ===")
    print(df_out.to_string(index=False, float_format="%.4f"))


def cmd_predict(args) -> None:
    if args.text:
        raw = [args.text]
    elif args.file:
        with open(args.file, "r", encoding="utf-8") as fh:
            raw = [ln.strip() for ln in fh if ln.strip()]
    elif args.data:
        suf = Path(args.data).suffix.lower()
        df = (pd.read_excel(args.data)
              if suf in {".xlsx", ".xls"} else pd.read_csv(args.data))
        if args.text_col not in df.columns:
            sys.exit(f"--text-col {args.text_col!r} not in {list(df.columns)}")
        raw = df[args.text_col].astype(str).tolist()
    else:
        sys.exit("Provide one of: --text / --file / --data")

    cleaned = [clean_text(t) for t in raw]

    if args.backend == "baseline":
        import joblib

        ops = BaselineOps.load(ARTIFACTS_DIR / f"{args.model}.joblib")
        ids, conf = ops.predict_with_confidence(cleaned)
        le = joblib.load(ARTIFACTS_DIR / "label_encoder.joblib")
        labels = le.inverse_transform(ids)
    else:
        ops = MARBERTOps.load(ARTIFACTS_DIR / "transformer" / "best")
        ids, conf = ops.predict_with_confidence(cleaned)
        labels = np.array([ops.classes[int(i)] for i in ids])

    decisions = ["REVIEW" if c < args.threshold else "AUTO" for c in conf]

    out_df = pd.DataFrame({
        "text":       raw,
        "prediction": labels,
        "confidence": np.round(conf, 4),
        "decision":   decisions,
    })
    print(out_df.to_string(index=False, max_colwidth=80))

    if args.out:
        out_df.to_csv(args.out, index=False)
        print(f"\nWrote -> {args.out}")


def cmd_all(args) -> None:
    print("\n========== STEP 1/4 : EDA ==========")
    cmd_eda(args)
    print("\n========== STEP 2/4 : BASELINES ==========")
    cmd_train_baseline(args)
    print("\n========== STEP 3/4 : MARBERT ==========")
    cmd_train_marbert(args)
    print("\n========== STEP 4/4 : COMPARE ==========")
    cmd_compare(args)


# ===========================================================================
# CLI
# ===========================================================================
def _add_data_args(p: argparse.ArgumentParser, *, require_data: bool = True):
    p.add_argument("--data", required=require_data,
                   help="Path to .xlsx / .csv (Drive path on Colab)")
    p.add_argument("--text-col",  default=None)
    p.add_argument("--label-col", default=None)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="RUN.py")
    sub = p.add_subparsers(dest="cmd", required=True)

    # eda
    p_eda = sub.add_parser("eda", help="Exploratory data analysis with plots")
    _add_data_args(p_eda)
    p_eda.set_defaults(func=cmd_eda)

    # train_baseline
    p_b = sub.add_parser("train_baseline",
                         help="Train TF-IDF + classical ML baselines")
    _add_data_args(p_b)
    p_b.add_argument("--models", nargs="+", default=list(BASELINE_MODELS),
                     choices=list(BASELINE_MODELS))
    p_b.add_argument("--test-size", type=float, default=0.2)
    p_b.add_argument("--val-size",  type=float, default=0.1)
    p_b.add_argument("--seed",      type=int,   default=42)
    p_b.set_defaults(func=cmd_train_baseline)

    # train_marbert
    p_m = sub.add_parser("train_marbert", help="Fine-tune MARBERT")
    _add_data_args(p_m)
    p_m.add_argument("--model-name", default="UBC-NLP/MARBERT")
    p_m.add_argument("--max-length", type=int, default=128)
    p_m.add_argument("--batch-size", type=int, default=16)
    p_m.add_argument("--epochs",     type=int, default=3)
    p_m.add_argument("--lr",         type=float, default=2e-5)
    p_m.add_argument("--weight-decay", type=float, default=0.01)
    p_m.add_argument("--test-size",  type=float, default=0.2)
    p_m.add_argument("--val-size",   type=float, default=0.1)
    p_m.add_argument("--seed",       type=int,   default=42)
    p_m.add_argument("--fp16", action="store_true")
    p_m.set_defaults(func=cmd_train_marbert)

    # compare
    p_c = sub.add_parser("compare",
                         help="Compare every trained model on the same split")
    _add_data_args(p_c)
    p_c.add_argument("--test-size", type=float, default=0.2)
    p_c.add_argument("--seed",      type=int,   default=42)
    p_c.set_defaults(func=cmd_compare)

    # predict
    p_p = sub.add_parser("predict", help="Inference on text / file / dataset")
    p_p.add_argument("--text")
    p_p.add_argument("--file")
    p_p.add_argument("--data")
    p_p.add_argument("--text-col", default="text")
    p_p.add_argument("--backend",  choices=["transformer", "baseline"],
                     default="transformer")
    p_p.add_argument("--model",    default="svm",
                     help="baseline model when --backend baseline")
    p_p.add_argument("--threshold", type=float, default=0.85,
                     help="below this confidence -> flagged REVIEW")
    p_p.add_argument("--out", default=None)
    p_p.set_defaults(func=cmd_predict)

    # all
    p_all = sub.add_parser(
        "all", help="EDA -> train_baseline -> train_marbert -> compare"
    )
    _add_data_args(p_all)
    p_all.add_argument("--models", nargs="+", default=list(BASELINE_MODELS),
                       choices=list(BASELINE_MODELS))
    p_all.add_argument("--model-name", default="UBC-NLP/MARBERT")
    p_all.add_argument("--max-length", type=int, default=128)
    p_all.add_argument("--batch-size", type=int, default=16)
    p_all.add_argument("--epochs",     type=int, default=3)
    p_all.add_argument("--lr",         type=float, default=2e-5)
    p_all.add_argument("--weight-decay", type=float, default=0.01)
    p_all.add_argument("--test-size",  type=float, default=0.2)
    p_all.add_argument("--val-size",   type=float, default=0.1)
    p_all.add_argument("--seed",       type=int,   default=42)
    p_all.add_argument("--fp16", action="store_true")
    p_all.set_defaults(func=cmd_all)

    return p


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
