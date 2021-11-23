import os
from flask import request, Flask

from entry_point import model_fn, input_fn, predict_fn, output_fn

MODEL_DIR = '../../model'


app = Flask(__name__)
model = model_fn(MODEL_DIR)


@app.route("/")
def index():

    app_name = os.getenv("APP_NAME")

    if app_name:
        return f"Hello from {app_name} running in a Docker container behind Nginx!"

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
    predictions = predict_fn(df, model)
    flask_accept = request.headers.getlist('accept')
    if 'application/json' in flask_accept:
        accept = 'application/json'
    elif 'text/csv' in flask_accept:
        accept = 'text/csv'
    else:
        accept = 'application/json'
    response = output_fn(predictions, accept)
    return response


if __name__ == '__main__':
    app.run()
