# Expose all components so the pipeline can import them cleanly:
#   from pipelines.components import data_ingestion, preprocessing, training, evaluation, model_registration

from .data_ingestion import data_ingestion_op
from .preprocessing import preprocessing_op
from .training import training_op
from .evaluation import evaluation_op
from .model_registration import model_registration_op
