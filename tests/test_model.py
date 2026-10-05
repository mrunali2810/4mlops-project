import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from preprocess import preprocess  # noqa: E402
from train import train  # noqa: E402
from predict import predict  # noqa: E402


def test_pipeline_end_to_end(tmp_path):
    proc, models = str(tmp_path / "proc"), str(tmp_path / "models")
    preprocess("data/data.csv", proc)
    acc = train(proc, models)
    assert acc >= 0.85, "Model accuracy too low"
    pred = predict([5.1, 3.5, 1.4, 0.2], os.path.join(models, "model.joblib"))
    assert len(pred) == 1


def test_preprocess_has_no_nulls(tmp_path):
    train_df, test_df = preprocess("data/data.csv", str(tmp_path))
    assert train_df.isnull().sum().sum() == 0
    assert len(train_df) > len(test_df)
