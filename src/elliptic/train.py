import argparse

import joblib
import yaml
from sklearn.model_selection import train_test_split

from .data import load_elliptic, make_xy
from .pipeline import make_pipelines
from .utils import ensure_dir, ensure_subdir, metrics_dict, metrics_table, save_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    paths = cfg["paths"]
    ensure_dir(paths["registry_dir"])
    ensure_dir(paths["artifacts_dir"])
    ensure_dir(paths["reports_dir"])
    metrics_dir = ensure_subdir(paths["reports_dir"], "metrics")

    df = load_elliptic(paths)
    X, y = make_xy(df, cfg["features"]["drop"])

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=cfg["test_size"], stratify=y, random_state=cfg["random_state"]
    )

    pipes = make_pipelines(cfg["random_state"], cfg["imbalance"]["smote"])
    use = cfg["models"]
    results = {}
    best_name, best_f1 = None, -1
    best_pipe = None

    for name, pipe in pipes.items():
        if not use.get(f"use_{name.lower()}", False):
            continue
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        m = results[name] = metrics_dict(y_te, proba, threshold=0.5)
        if m["f1"] > best_f1:
            best_name, best_f1, best_pipe = name, m["f1"], pipe

    # persist best
    best_path = f'{paths["registry_dir"]}/best_pipeline.joblib'
    joblib.dump(best_pipe, best_path)

    # save metrics
    table = metrics_table(results)
    table.to_csv(f"{metrics_dir}/{best_name}_comparison.csv", index=False)
    save_json(results, f"{metrics_dir}/all_metrics.json")

    print("\n=== Results ===")
    print(table.to_string(index=False))
    print(f"\nSaved best model: {best_name} -> {best_path}")


if __name__ == "__main__":
    main()
