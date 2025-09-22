import pandas as pd

def load_elliptic(paths):
    feats = pd.read_csv(paths["features"], header=None)
    num_features = feats.shape[1] - 2
    feats.columns = ["txId", "time_step"] + [f"feature{i}" for i in range(1, num_features + 1)]
    labels = pd.read_csv(paths["classes"])
    labels = labels[labels["class"] != "unknown"].copy()
    df = feats.merge(labels, on="txId", how="inner")
    return df

def make_xy(df, drops):
    X = df.drop(columns=drops).copy()
    y = df["class"].map({"1": 1, "2": 0}).astype(int)
    return X, y