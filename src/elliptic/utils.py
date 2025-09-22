import json, os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def metrics_dict(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_proba),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

def metrics_table(results):
    rows = []
    for name, m in results.items():
        rows.append([name, m["accuracy"], m["precision"], m["recall"], m["f1"], m["roc_auc"]])
    return pd.DataFrame(rows, columns=["model","accuracy","precision","recall","f1","roc_auc"]).sort_values("f1", ascending=False)

def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)