"""Smoke test - end-to-end on a tiny synthetic dataset."""
from pathlib import Path
import os, tempfile, json
import pandas as pd

# Use a fresh artifacts dir so the test doesn't pollute the real one
os.environ["NLP_ARTIFACTS_DIR"] = tempfile.mkdtemp(prefix="art_test_")

from DAL import clean_text, run_eda, ARTIFACTS_DIR
from DAL.data_loader import load_dataset, prepare_splits
from Models.BaselineOps import BaselineOps

# 1. cleaning
samples = [
    "حسّيت إنّي تعبان مرّةً @user http://x #حزن",
    "أحبّ الحياةَ ولكنّني خائف",
    "أريد أن أنتهي من كل شيء الآن",
    "اليوم جميل ومشمس والحمدلله",
    "كل الأمل ضاع منّي ولا أرى مخرجاً",
]
print("== preprocessing ==")
for s in samples:
    print(f"  IN : {s}\n  OUT: {clean_text(s)!r}")

# 2. tiny synthetic dataset -> dataframe -> CSV
df = pd.DataFrame({
    "tweet": (samples * 8)[:40],
    "label": (["suicidal", "non", "suicidal", "non", "suicidal"] * 8),
})
csv = Path(ARTIFACTS_DIR) / "_tiny.csv"
df.to_csv(csv, index=False)
print(f"\n== loading {csv}")

dfl, t, l = load_dataset(str(csv))
print(f"  rows={len(dfl)}  text_col={t!r}  label_col={l!r}")
print(dfl.head())

# 3. EDA
print("\n== EDA ==")
print(json.dumps(run_eda(str(csv)), indent=2, ensure_ascii=False))

# 4. baseline train + eval round-trip
print("\n== baseline train/eval ==")
train_df, val_df, test_df, classes = prepare_splits(
    dfl, t, l, test_size=0.25, val_size=0.2, seed=0,
)
X_tr = pd.concat([train_df, val_df])["text_clean"].tolist()
y_tr = pd.concat([train_df, val_df])["label_id"].values
X_te = test_df["text_clean"].tolist()
y_te = test_df["label_id"].values

ops = BaselineOps("logreg")
ops.train(X_tr, y_tr)
m = ops.evaluate(X_te, y_te, classes, ARTIFACTS_DIR)
print(json.dumps(m, indent=2))

# 5. save+load roundtrip
p = ops.save(ARTIFACTS_DIR / "logreg.joblib")
ops2 = BaselineOps.load(p)
labels, conf = ops2.predict_with_confidence(["اريد ان انتهي من كل شيء"])
print("loaded predict:", labels.tolist(), conf.tolist())

print("\nALL OK")
