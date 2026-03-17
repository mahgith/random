"""
LOCAL COMPONENT TESTS
=====================
Tests that run component logic locally WITHOUT Vertex AI or BigQuery.

Run with:
    pytest tests/test_components_local.py -v

Strategy
--------
- data_ingestion_op requires BQ, so we test the post-BQ logic
  (weekend folding, dedup aggregation) by calling helper functions
  extracted inline.
- All other components are tested end-to-end using synthetic daily data
  that mimics the output of data_ingestion_op.
- KFP artifacts are mocked with FakeArtifact, matching how components
  use .path / .uri / log_metric().
"""

import json
import os
import tempfile

import numpy as np
import pandas as pd
import pytest


# ── Helpers ───────────────────────────────────────────────────────────────────

class FakeArtifact:
    """Mimics a KFP Dataset / Model / Metrics artifact for local testing."""
    def __init__(self, tmp_dir: str, name: str):
        self.path = os.path.join(tmp_dir, name)
        self.uri  = f"gs://fake-bucket/{name}"
        self._metrics: dict = {}

    def log_metric(self, key: str, value):
        self._metrics[key] = value


def _make_synthetic_daily(n_days: int = 300, seed: int = 42) -> pd.DataFrame:
    """
    Synthetic workday series that looks like warehouse package volumes.
    Mimics the output of data_ingestion_op (columns: ds, y).
    """
    rng = np.random.default_rng(seed)
    # Start on a Monday
    start = pd.Timestamp("2023-01-02")
    dates = pd.bdate_range(start=start, periods=n_days, freq="B")
    trend = np.linspace(1000, 1500, n_days)
    noise = rng.normal(0, 80, n_days)
    weekly = 50 * np.sin(2 * np.pi * np.arange(n_days) / 5)
    y = np.maximum(trend + noise + weekly, 100)
    return pd.DataFrame({"ds": dates, "y": y})


def _run_preprocessing(tmp_path: str, df_raw: pd.DataFrame) -> FakeArtifact:
    """Runs preprocessing_op on synthetic data, returns processed_data artifact."""
    from pipelines.components.preprocessing import preprocessing_op

    raw = FakeArtifact(tmp_path, "raw")
    df_raw.to_parquet(raw.path + ".parquet", index=False)

    processed = FakeArtifact(tmp_path, "processed")
    preprocessing_op.python_func(
        direction="inbound",
        raw_dataset=raw,
        processed_data=processed,
    )
    return processed


def _run_training(tmp_path: str, processed: FakeArtifact) -> FakeArtifact:
    """Runs training_op on processed data, returns model artifact."""
    from pipelines.components.training import training_op

    model   = FakeArtifact(tmp_path, "model")
    os.makedirs(model.path, exist_ok=True)
    metrics = FakeArtifact(tmp_path, "train_metrics")

    training_op.python_func(
        direction="inbound",
        processed_data=processed,
        lookback_days=30,           # smaller for speed
        half_life_days=10,
        prophet_changepoint_prior_scale=0.1,
        lgbm_n_estimators=50,       # small for speed
        lgbm_learning_rate=0.1,
        lgbm_num_leaves=15,
        model=model,
        metrics=metrics,
    )
    return model


# ── Weekend-folding logic (from data_ingestion_op) ────────────────────────────

class TestWeekendFolding:
    def test_saturday_folds_to_friday(self):
        """Saturday packages should be added to Friday's total."""
        df = pd.DataFrame({
            "ds": pd.to_datetime(["2024-01-05", "2024-01-06"]),  # Fri, Sat
            "y":  [100.0, 50.0],
        })
        dow = df["ds"].dt.dayofweek
        df.loc[dow == 5, "ds"] -= pd.Timedelta(days=1)   # Sat → Fri
        df.loc[dow == 6, "ds"] += pd.Timedelta(days=1)   # Sun → Mon
        result = df.groupby("ds")["y"].sum().reset_index()
        assert len(result) == 1, "Saturday should merge into Friday"
        assert result.iloc[0]["y"] == 150.0

    def test_sunday_folds_to_monday(self):
        df = pd.DataFrame({
            "ds": pd.to_datetime(["2024-01-07", "2024-01-08"]),  # Sun, Mon
            "y":  [30.0, 200.0],
        })
        dow = df["ds"].dt.dayofweek
        df.loc[dow == 5, "ds"] -= pd.Timedelta(days=1)
        df.loc[dow == 6, "ds"] += pd.Timedelta(days=1)
        result = df.groupby("ds")["y"].sum().reset_index()
        assert len(result) == 1
        assert result.iloc[0]["y"] == 230.0

    def test_weekdays_unchanged(self):
        df = pd.DataFrame({
            "ds": pd.bdate_range("2024-01-08", periods=5),  # Mon-Fri
            "y":  [100.0] * 5,
        })
        original_len = len(df)
        dow = df["ds"].dt.dayofweek
        df.loc[dow == 5, "ds"] -= pd.Timedelta(days=1)
        df.loc[dow == 6, "ds"] += pd.Timedelta(days=1)
        result = df.groupby("ds")["y"].sum().reset_index()
        assert len(result) == original_len


# ── Preprocessing ─────────────────────────────────────────────────────────────

class TestPreprocessing:
    def test_feature_columns_present(self, tmp_path):
        df_raw = _make_synthetic_daily(200)
        processed = _run_preprocessing(str(tmp_path), df_raw)

        df = pd.read_parquet(processed.path + ".parquet")
        expected_cols = [
            "ds", "y",
            "is_holiday", "is_pre_holiday", "is_post_holiday",
            "day_of_week", "week_of_year", "month", "iso_year",
            "rolling_10", "rolling_20", "rolling_30",
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_holiday_flags_are_binary(self, tmp_path):
        df_raw = _make_synthetic_daily(200)
        processed = _run_preprocessing(str(tmp_path), df_raw)
        df = pd.read_parquet(processed.path + ".parquet")

        for flag in ("is_holiday", "is_pre_holiday", "is_post_holiday"):
            assert set(df[flag].unique()).issubset({0, 1}), f"{flag} must be 0/1"

    def test_no_weekends_in_output(self, tmp_path):
        df_raw = _make_synthetic_daily(200)
        processed = _run_preprocessing(str(tmp_path), df_raw)
        df = pd.read_parquet(processed.path + ".parquet")
        assert (df["day_of_week"] < 5).all(), "Weekends must not appear in output"

    def test_rolling_stats_are_lagged(self, tmp_path):
        """Rolling stats must be computed on shifted y to avoid leakage."""
        df_raw = _make_synthetic_daily(200)
        processed = _run_preprocessing(str(tmp_path), df_raw)
        df = pd.read_parquet(processed.path + ".parquet")

        # rolling_10 at row i should equal mean of y[i-10 : i], not include y[i]
        # We verify it doesn't equal y[i] (would be suspicious if it did)
        # This is a heuristic check, not a strict proof
        mismatch = (df["rolling_10"] != df["y"]).sum()
        assert mismatch > len(df) * 0.9, "rolling_10 looks like it includes current y (leakage?)"


# ── Training ──────────────────────────────────────────────────────────────────

class TestTraining:
    def test_model_bundle_files_exist(self, tmp_path):
        df_raw = _make_synthetic_daily(300)
        processed = _run_preprocessing(str(tmp_path), df_raw)
        model = _run_training(str(tmp_path), processed)

        expected_files = [
            "config.json",
            "multiplier_table.parquet",
            "prophet_model.pkl",
            "lgbm_model.joblib",
            "lgbm_features.json",
        ]
        for fname in expected_files:
            assert os.path.exists(os.path.join(model.path, fname)), (
                f"Missing model file: {fname}"
            )

    def test_config_has_expected_keys(self, tmp_path):
        df_raw = _make_synthetic_daily(300)
        processed = _run_preprocessing(str(tmp_path), df_raw)
        model = _run_training(str(tmp_path), processed)

        with open(os.path.join(model.path, "config.json")) as f:
            cfg = json.load(f)

        for key in ("direction", "training_cutoff", "lookback_days", "half_life_days"):
            assert key in cfg, f"config.json missing key: {key}"

    def test_multiplier_table_covers_weekdays(self, tmp_path):
        df_raw = _make_synthetic_daily(300)
        processed = _run_preprocessing(str(tmp_path), df_raw)
        model = _run_training(str(tmp_path), processed)

        tbl = pd.read_parquet(os.path.join(model.path, "multiplier_table.parquet"))
        # Should have entries for each day of week (0-4)
        assert set(tbl["day_of_week"].unique()).issubset({0, 1, 2, 3, 4})
        assert "multiplier" in tbl.columns

    def test_lgbm_features_match_model(self, tmp_path):
        df_raw = _make_synthetic_daily(300)
        processed = _run_preprocessing(str(tmp_path), df_raw)
        model = _run_training(str(tmp_path), processed)

        import joblib
        lgbm = joblib.load(os.path.join(model.path, "lgbm_model.joblib"))
        features = json.load(open(os.path.join(model.path, "lgbm_features.json")))

        assert lgbm.n_features_in_ == len(features), (
            "LightGBM model was trained on a different number of features than lgbm_features.json"
        )


# ── Evaluation ────────────────────────────────────────────────────────────────

class TestEvaluation:
    def test_approval_decision_is_valid(self, tmp_path):
        from pipelines.components.evaluation import evaluation_op

        df_raw = _make_synthetic_daily(400)
        processed = _run_preprocessing(str(tmp_path), df_raw)
        model = _run_training(str(tmp_path), processed)

        eval_metrics = FakeArtifact(str(tmp_path), "eval_metrics")
        approval     = FakeArtifact(str(tmp_path), "approval")

        evaluation_op.python_func(
            direction="inbound",
            processed_data=processed,
            model=model,
            evaluation_start_date="2023-09-01",
            forecast_horizon=20,
            backtest_step_days=10,
            mape_threshold=0.99,    # very permissive — synthetic data should pass
            evaluation_metrics=eval_metrics,
            approval_decision=approval,
        )

        with open(approval.path) as f:
            decision = f.read().strip()
        assert decision in ("approved", "rejected"), f"Unexpected decision: {decision!r}"

    def test_tight_threshold_rejects(self, tmp_path):
        from pipelines.components.evaluation import evaluation_op

        df_raw = _make_synthetic_daily(400)
        processed = _run_preprocessing(str(tmp_path), df_raw)
        model = _run_training(str(tmp_path), processed)

        eval_metrics = FakeArtifact(str(tmp_path), "eval_metrics2")
        approval     = FakeArtifact(str(tmp_path), "approval2")

        evaluation_op.python_func(
            direction="inbound",
            processed_data=processed,
            model=model,
            evaluation_start_date="2023-09-01",
            forecast_horizon=20,
            backtest_step_days=10,
            mape_threshold=0.000001,   # impossibly tight
            evaluation_metrics=eval_metrics,
            approval_decision=approval,
        )

        with open(approval.path) as f:
            decision = f.read().strip()
        assert decision == "rejected"

    def test_metrics_logged(self, tmp_path):
        from pipelines.components.evaluation import evaluation_op

        df_raw = _make_synthetic_daily(400)
        processed = _run_preprocessing(str(tmp_path), df_raw)
        model = _run_training(str(tmp_path), processed)

        eval_metrics = FakeArtifact(str(tmp_path), "eval_metrics3")
        approval     = FakeArtifact(str(tmp_path), "approval3")

        evaluation_op.python_func(
            direction="inbound",
            processed_data=processed,
            model=model,
            evaluation_start_date="2023-09-01",
            forecast_horizon=20,
            backtest_step_days=10,
            mape_threshold=0.99,
            evaluation_metrics=eval_metrics,
            approval_decision=approval,
        )

        assert "mean_mape" in eval_metrics._metrics
        assert eval_metrics._metrics["mean_mape"] >= 0
