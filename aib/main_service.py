from typing import List
from pydantic import BaseModel

import joblib
import numpy as np

from aibakery.aibakery_service import ResultCapture, AIBakeryService

aibakery_service = AIBakeryService()


class MnistFeature(BaseModel):
    x: List[int]


def load_joblib(model_location):
    return joblib.load(f'{model_location}/mnist_svc.joblib')


@aibakery_service.prediction(feature_schema=MnistFeature,
                             model_loader=load_joblib)
def predict(model, feature: MnistFeature, results: ResultCapture):
    x = np.array(feature.x).reshape(1, -1)

    probabilities = model.predict_proba(x)[0]
    predicted_number = probabilities.argmax()
    predicted_probability = probabilities[predicted_number]

    results.add_result(
            key='predicted_number',
            value=predicted_number,
            meta={
                'predicted_probability': predicted_probability
            }
    )
