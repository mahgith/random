"""
SPLIT COMPONENT
===============
Step 2b — Splits the fully preprocessed dataset into a training portion
and preserves the full dataset for evaluation.

Why this split is necessary
---------------------------
The evaluate_op runs a rolling backtest starting from evaluation_start_date.
Without this split, train_op would train on ALL data — including dates that
fall inside the evaluation window — causing the model to have already "seen"
those actuals during training, producing overly optimistic metrics.

Outputs
-------
train_data   : rows where ds < train_end_date (fed to train_op)

Note: evaluate_op continues to receive the full processed_data so that it
has sufficient history (lookback_days) before the first evaluation cutoff.
"""

from kfp.dsl import component, Input, Output, Dataset

_ML_TRAINING_IMAGE = "europe-west3-docker.pkg.dev/prj-ceva-gr-wkfc-adapt-dev/gr-forecast/ml-training:latest"


@component(base_image=_ML_TRAINING_IMAGE)
def split_op(
    processed_data: Input[Dataset],
    train_end_date: str,
    train_data: Output[Dataset],
):
    """Filter processed data to rows strictly before train_end_date for leak-free training."""
    import logging
    import pandas as pd

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    df = pd.read_parquet(processed_data.path + ".parquet")
    df["ds"] = pd.to_datetime(df["ds"])

    cutoff = pd.Timestamp(train_end_date)
    train = df[df["ds"] < cutoff].copy()

    logger.info(
        "Split at %s: %d total rows → %d train rows (%s – %s), %d eval rows held out",
        train_end_date,
        len(df),
        len(train),
        train["ds"].min().date() if len(train) else "n/a",
        train["ds"].max().date() if len(train) else "n/a",
        len(df) - len(train),
    )

    if len(train) == 0:
        raise ValueError(
            f"No training rows found before train_end_date='{train_end_date}'. "
            "Check that evaluation_start_date leaves enough data for training."
        )

    train.to_parquet(train_data.path + ".parquet", index=False)
