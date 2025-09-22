import argparse

import joblib
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--input", required=True)  # CSV with same feature columns used in training
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    pipe = joblib.load(args.model_path)
    X_new = pd.read_csv(args.input)
    scores = pipe.predict_proba(X_new)[:, 1]
    pd.DataFrame({"score_illicit": scores}).to_csv(args.output, index=False)
    print(f"Saved scores -> {args.output}")


if __name__ == "__main__":
    main()
