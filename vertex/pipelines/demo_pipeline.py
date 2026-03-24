from kfp import dsl

from components.demo_read_gcs import demo_read_gcs_op
from components.demo_preprocess import demo_preprocess_op
from components.demo_train import demo_train_op


@dsl.pipeline(name="demo-pipeline")
def demo_pipeline(
    project_id: str,
    gcs_uri: str,
    target_column: str,
):
    read_task = demo_read_gcs_op(
        project_id=project_id,
        gcs_uri=gcs_uri,
    )
    read_task.set_display_name("1 — Read GCS")

    preprocess_task = demo_preprocess_op(
        raw_data=read_task.outputs["raw_data"],
    )
    preprocess_task.set_display_name("2 — Preprocess (simulated)")

    train_task = demo_train_op(
        processed_data=preprocess_task.outputs["processed_data"],
        target_column=target_column,
    )
    train_task.set_display_name("3 — Train (constant mean)")
