import argparse, joblib, yaml
from sklearn.model_selection import train_test_split
from .data import load_elliptic, make_xy
from .utils import metrics_dict


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--config", default="config/train.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    df = load_elliptic(cfg["paths"])
    X, y = make_xy(df, cfg["features"]["drop"])
    _, X_te, _, y_te = train_test_split(
        X, y, test_size=cfg["test_size"], stratify=y, random_state=cfg["random_state"]
    )

    pipe = joblib.load(args.model_path)
    proba = pipe.predict_proba(X_te)[:, 1]
    m = metrics_dict(y_te, proba, 0.5)
    print(m)


if __name__ == "__main__":
    main()
