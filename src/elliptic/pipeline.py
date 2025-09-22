from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def make_pipelines(random_state=42, use_smote=True):
    steps = [("scaler", StandardScaler())]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=random_state)))

    pipes = {
        "LogReg": ImbPipeline(
            steps=steps
            + [
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        solver="lbfgs",
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                )
            ]
        ),
        "DecisionTree": ImbPipeline(
            steps=steps + [("clf", DecisionTreeClassifier(random_state=random_state))]
        ),
        "RandomForest": ImbPipeline(
            steps=steps
            + [("clf", RandomForestClassifier(n_estimators=100, random_state=random_state))]
        ),
        "XGBoost": ImbPipeline(
            steps=steps
            + [
                (
                    "clf",
                    XGBClassifier(
                        random_state=random_state,
                        use_label_encoder=False,
                        eval_metric="logloss",
                        n_jobs=-1,
                    ),
                )
            ]
        ),
    }
    return pipes
