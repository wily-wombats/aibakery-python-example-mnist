from typing import List
from pydantic import BaseModel

import joblib
import numpy as np

from aibakery.serving.model_prediction_service import ResultCapture, \
    ModelPredictionService

model_prediction_service = ModelPredictionService()


class MnistFeature(BaseModel):
    x: List[int]


def load_joblib(model_location):
    return joblib.load(f'{model_location}/mnist_svc.joblib')


@model_prediction_service.prediction(feature_schema=MnistFeature,
                                     model_loader=load_joblib)
def predict(model, feature: MnistFeature, results: ResultCapture):
    x = np.array(feature.x).reshape(1, -1)

    results.add_result(
            key='number',
            value=model.predict(x)[0],
    )
