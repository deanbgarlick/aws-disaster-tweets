import time
import sys
from io import StringIO
import os
import shutil

import argparse
import csv
import json
import numpy as np
import pandas as pd
import torch

from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
feature_columns_names = ['keyword', 'location', 'text']


def input_fn(input_data, content_type):
    """Parse input data payload

    We currently only take csv input. Since we need to process both labelled
    and unlabelled data we first determine whether the label column is present
    by looking at how many columns were provided.
    """

    if content_type == 'text/csv':

        # Read the raw input data as CSV.
        df = pd.read_csv(StringIO(input_data), header=None, delimiter=',', index_col=0)
        df.columns = feature_columns_names
        return df

    if content_type == 'application/json':
        df = pd.DataFrame(json.loads(input_data), columns=feature_columns_names)
        return df

    else:
        raise ValueError("{} not supported by script!".format(content_type))


def output_fn(predictions, accept):
    """Format prediction output

    The default accept/content-type between containers for serial inference is JSON.
    We also want to set the ContentType or mimetype as the same value as accept so the next
    container can read the response payload correctly.
    """
    if accept == "application/json":
        instances = []
        for row in predictions:
            instances.append({"prediction": row})
        json_output = {"instances": instances}
        return json.dumps(json_output)

    elif accept == 'text/csv':
        predictions_array = np.array(predictions)
        stream = StringIO()
        np.savetxt(stream, predictions_array, delimiter=",", fmt="%s")
        return stream.getvalue()

    else:
        raise RuntimeException("{} accept type is not supported by this script.".format(accept))


class ModelWrapper:

    def __init__(self, tokenizer, classifier):
        self._tokenizer = tokenizer
        self._classifier = classifier
        self._device = self._classifier.device

    def __call__(self, text):
        inputs = self._tokenizer.encode_plus(text, return_tensors = "pt")
        with torch.no_grad():
            outputs = self._classifier(**inputs)
            output = torch.argmax(outputs.logits).item()
        return output


def predict_fn(df, model):

    def preprocess_row(x):
        return x.text + ' [KEYWORD] ' + str(x.keyword) + ' [LOCATION] ' + str(x.location)

    tweets_series = df.apply(preprocess_row, axis=1)
    tweets_list = tweets_series.astype(str).tolist()
    predictions = []
    for tweet in tweets_list:
        prediction = model(tweet)
        predictions.append(prediction)
    return predictions


def model_fn(model_dir):
    """Deserialize fitted model"""
    tokenizer = DistilBertTokenizerFast.from_pretrained(os.path.join(model_dir, 'tokenizer'), local_files_only=True)
    classifier = DistilBertForSequenceClassification.from_pretrained(os.path.join(model_dir, 'model'), local_files_only=True)
    model = ModelWrapper(tokenizer, classifier)
    return model
