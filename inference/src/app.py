import os
from flask import request, Flask

from model_handler import model_fn, input_fn, predict_fn, output_fn


app = Flask(__name__)
MODEL_DIR = os.getenv('MODEL_DIR', '/opt/ml/model')


class ModelService:

    _instance = None
    _model = model_fn(MODEL_DIR)

    def __init__(self):
        raise Exception('Call instance()')

    @classmethod
    def instance(cls):
        if cls._instance is None:
            cls._instance = cls.__new__(cls)
        return cls._instance

    def transform(self, df):
        return predict_fn(df, self._model)

ModelService.instance()


@app.route("/")
def index():

    app_name = os.getenv("APP_NAME")

    if app_name:
        return f"Hello from {app_name}!"

    return "Hello from Flask"


@app.route("/predict", methods = ['POST'])
def predict():
    if 'application/json' in request.content_type:
        input_data = request.json
    elif 'text/csv' in request.content_type:
        input_data = request.stream.read().decode('utf-8')
    elif '*/*' in request.content_type:
        input_data = request.json
    else:
        raise ValueError(f'Unknown content-type {request.content_type}')
    df = input_fn(input_data, request.content_type)
    predictions = ModelService.instance().transform(df)
    flask_accept = request.headers.getlist('accept')
    if 'application/json' in flask_accept:
        accept = 'application/json'
    elif 'text/csv' in flask_accept:
        accept = 'text/csv'
    else:
        accept = 'application/json'
    response = output_fn(predictions, accept)
    return response
