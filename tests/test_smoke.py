import joblib
import tempfile
import os
import numpy as np
import pandas as pd
from src.elliptic.pipeline import make_pipelines

def test_pipeline_construction():
    pipes = make_pipelines(random_state=0, use_smote=False)  # quick: disable SMOTE for speed in test
    assert "LogReg" in pipes and "XGBoost" in pipes

def test_inference_api_smoke():
    pipes = make_pipelines(random_state=0, use_smote=False)
    # pick the lightweight LogReg pipe and check predict_proba shape
    pipe = pipes["LogReg"]
    # make tiny fake data: 5 rows, same number of features as the project expects
    # We'll create 10 synthetic features for the test; pipeline will accept it
    X_fake = pd.DataFrame(np.random.randn(5, 10), columns=[f"feature{i}" for i in range(1,11)])
    # The primary purpose is to exercise pipeline.fit/predict_proba shape
    y_fake = pd.Series([0,0,1,0,1])
    pipe.fit(X_fake, y_fake)
    probs = pipe.predict_proba(X_fake)
    assert probs.shape == (5, 2)