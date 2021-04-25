from typing import List

import joblib
import numpy as np
from aibakery import logger
from aibakery.aibakery_service import ResultCapture, AIBakeryService
from pydantic import BaseModel

aibakery_service = AIBakeryService()


class MnistFeature(BaseModel):
    x: List[int]


def load_model(model_directory):
    return joblib.load(f'{model_directory}/mnist_svc.joblib')


@aibakery_service.prediction(model_loader=load_model,
                             feature_schema=MnistFeature)
def predict(model, feature: MnistFeature, results: ResultCapture):
    logger.info('Starting model prediction')

    x = np.array(feature.x).reshape(1, -1)
    logger.info(f'Model input {x}')

    probabilities = model.predict_proba(x)[0]
    logger.info(f'Model prediction output {probabilities}')
    predicted_number = probabilities.argmax()
    predicted_probability = probabilities[predicted_number]
    logger.info(f'Maximum probability index::{predicted_number} probability::{predicted_probability}')

    results.add_result(
            key='predicted_number',
            value=predicted_number,
            meta={
                'predicted_probability': predicted_probability
            }
    )

    logger.info('Model prediction complete')


if __name__ == '__main__':
    sample_input = {
        "x": [1, 2, 9, 8, 9, 4, 4, 4, 4, 14, 5, 14, 8,
              7, 1, 14, 4, 12, 5, 2, 4, 9, 4, 4, 4, 7,
              5, 4, 1, 14, 5, 4, 4, 4, 14, 4, 4, 2, 14,
              4, 4, 1, 14, 4, 4, 1, 12, 4, 4, 4, 5, 14,
              5, 9, 14, 4, 4, 4, 4, 9, 16, 16, 7, 4]
    }

    print(predict(feature=sample_input))
