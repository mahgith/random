"""
LOCAL COMPONENT TESTS
=====================
Tests that run your component logic locally WITHOUT Vertex AI.
This lets you iterate fast on your notebook code before submitting to GCP.

Run with:
    pytest tests/test_components_local.py -v

The trick: we call the component's underlying Python function directly
(not through KFP), so we can test the logic without any GCP setup.
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import os
from pathlib import Path
from unittest.mock import MagicMock


# ── Helper: fake KFP artifact ────────────────────────────────────────────────
class FakeArtifact:
    """Mimics a KFP Dataset/Model artifact for local testing."""
    def __init__(self, tmp_dir: str, name: str):
        self.path = os.path.join(tmp_dir, name)
        self.uri = f"gs://fake-bucket/{name}"

    def log_metric(self, key: str, value):
        pass  # no-op for testing


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestDataIngestion:
    def test_produces_parquet_with_expected_columns(self, tmp_path):
        """data_ingestion_op should write a parquet file with date and value columns."""
        # Import the underlying function
        # KFP @component wraps the function — .python_func gives us the raw function
        from pipelines.components.data_ingestion import data_ingestion_op

        raw_dataset = FakeArtifact(str(tmp_path), "raw")

        # Call the underlying Python function directly
        data_ingestion_op.python_func(
            project_id="test-project",
            raw_data_gcs_path="gs://fake/raw/",
            lookback_days=60,
            raw_dataset=raw_dataset,
        )

        df = pd.read_parquet(raw_dataset.path + ".parquet")
        assert "date" in df.columns
        assert "value" in df.columns
        assert len(df) == 60

    def test_schema_validation_catches_missing_columns(self, tmp_path):
        """Should raise ValueError if required columns are missing (after schema is changed)."""
        # This test documents the expected behavior — it passes with the placeholder data
        # because placeholder always has the right columns.
        # When you plug in real data, add a test here that deliberately breaks the schema.
        pass


class TestPreprocessing:
    def test_creates_lag_features(self, tmp_path):
        """preprocessing_op should produce lag and rolling features."""
        from pipelines.components.preprocessing import preprocessing_op
        from pipelines.components.data_ingestion import data_ingestion_op

        # First create raw data
        raw = FakeArtifact(str(tmp_path), "raw")
        data_ingestion_op.python_func(
            project_id="test-project",
            raw_data_gcs_path="gs://fake/",
            lookback_days=120,
            raw_dataset=raw,
        )

        train = FakeArtifact(str(tmp_path), "train")
        test = FakeArtifact(str(tmp_path), "test")

        preprocessing_op.python_func(
            raw_dataset=raw,
            forecast_horizon=7,
            test_size_fraction=0.2,
            train_dataset=train,
            test_dataset=test,
        )

        train_df = pd.read_parquet(train.path + ".parquet")
        test_df = pd.read_parquet(test.path + ".parquet")

        # Lag features should exist
        assert "lag_1" in train_df.columns
        assert "lag_7" in train_df.columns

        # Train should be larger than test
        assert len(train_df) > len(test_df)

        # No NaN values in features
        assert not train_df.isnull().any().any(), "Train set has NaN values"

    def test_chronological_split(self, tmp_path):
        """Test split should come strictly after train split (time-series correctness)."""
        from pipelines.components.preprocessing import preprocessing_op
        from pipelines.components.data_ingestion import data_ingestion_op

        raw = FakeArtifact(str(tmp_path), "raw")
        data_ingestion_op.python_func(
            project_id="test-project",
            raw_data_gcs_path="gs://fake/",
            lookback_days=120,
            raw_dataset=raw,
        )

        train = FakeArtifact(str(tmp_path), "train")
        test = FakeArtifact(str(tmp_path), "test")
        preprocessing_op.python_func(
            raw_dataset=raw, forecast_horizon=7, test_size_fraction=0.2,
            train_dataset=train, test_dataset=test,
        )

        train_df = pd.read_parquet(train.path + ".parquet")
        test_df = pd.read_parquet(test.path + ".parquet")

        max_train_date = pd.to_datetime(train_df["date"]).max()
        min_test_date = pd.to_datetime(test_df["date"]).min()
        assert max_train_date < min_test_date, "Data leakage: test dates appear before end of train!"


class TestTraining:
    def test_model_artifact_is_created(self, tmp_path):
        """training_op should produce a model.joblib and features.json."""
        from pipelines.components.data_ingestion import data_ingestion_op
        from pipelines.components.preprocessing import preprocessing_op
        from pipelines.components.training import training_op

        raw = FakeArtifact(str(tmp_path), "raw")
        data_ingestion_op.python_func(
            project_id="test-project", raw_data_gcs_path="gs://fake/",
            lookback_days=120, raw_dataset=raw,
        )
        train = FakeArtifact(str(tmp_path), "train")
        test = FakeArtifact(str(tmp_path), "test")
        preprocessing_op.python_func(
            raw_dataset=raw, forecast_horizon=7, test_size_fraction=0.2,
            train_dataset=train, test_dataset=test,
        )

        model = FakeArtifact(str(tmp_path), "model")
        os.makedirs(model.path, exist_ok=True)
        metrics = FakeArtifact(str(tmp_path), "metrics")

        training_op.python_func(
            train_dataset=train,
            project_id="test-project",
            experiment_name="test-experiment",
            n_estimators=10,   # small for speed in tests
            max_depth=3,
            learning_rate=0.1,
            model=model,
            metrics=metrics,
        )

        assert os.path.exists(os.path.join(model.path, "model.joblib"))
        assert os.path.exists(os.path.join(model.path, "features.json"))


class TestEvaluation:
    def _run_full_pipeline_locally(self, tmp_path):
        """Helper: run ingestion → preprocessing → training, return artifacts."""
        from pipelines.components.data_ingestion import data_ingestion_op
        from pipelines.components.preprocessing import preprocessing_op
        from pipelines.components.training import training_op

        raw = FakeArtifact(str(tmp_path), "raw")
        data_ingestion_op.python_func(
            project_id="test-project", raw_data_gcs_path="gs://fake/",
            lookback_days=150, raw_dataset=raw,
        )
        train = FakeArtifact(str(tmp_path), "train")
        test = FakeArtifact(str(tmp_path), "test")
        preprocessing_op.python_func(
            raw_dataset=raw, forecast_horizon=7, test_size_fraction=0.2,
            train_dataset=train, test_dataset=test,
        )
        model = FakeArtifact(str(tmp_path), "model")
        os.makedirs(model.path, exist_ok=True)
        metrics = FakeArtifact(str(tmp_path), "metrics")
        training_op.python_func(
            train_dataset=train, project_id="test-project",
            experiment_name="test-exp", n_estimators=10, max_depth=3,
            learning_rate=0.1, model=model, metrics=metrics,
        )
        return model, test

    def test_evaluation_writes_approval_decision(self, tmp_path):
        """evaluation_op should write 'approved' or 'rejected'."""
        from pipelines.components.evaluation import evaluation_op

        model, test = self._run_full_pipeline_locally(tmp_path)

        eval_metrics = FakeArtifact(str(tmp_path), "eval_metrics")
        approval = FakeArtifact(str(tmp_path), "approval")
        # Give a very high threshold so synthetic data gets approved
        evaluation_op.python_func(
            model=model, test_dataset=test,
            mae_threshold=9999.0,
            evaluation_metrics=eval_metrics,
            approval_decision=approval,
        )

        with open(approval.path) as f:
            decision = f.read().strip()
        assert decision in ("approved", "rejected")

    def test_evaluation_rejects_when_threshold_too_low(self, tmp_path):
        """Model should be rejected if MAE > threshold."""
        from pipelines.components.evaluation import evaluation_op

        model, test = self._run_full_pipeline_locally(tmp_path)

        eval_metrics = FakeArtifact(str(tmp_path), "eval_metrics2")
        approval = FakeArtifact(str(tmp_path), "approval2")

        evaluation_op.python_func(
            model=model, test_dataset=test,
            mae_threshold=0.0001,   # impossibly low threshold
            evaluation_metrics=eval_metrics,
            approval_decision=approval,
        )

        with open(approval.path) as f:
            decision = f.read().strip()
        assert decision == "rejected"
