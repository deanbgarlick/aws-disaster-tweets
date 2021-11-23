# https://aws.amazon.com/blogs/machine-learning/preprocess-input-data-before-making-predictions-using-amazon-sagemaker-inference-pipelines-and-scikit-learn/
# https://github.com/aws/sagemaker-inference-toolkit/blob/master/src/sagemaker_inference/encoder.py

from io import StringIO

import boto3
import pandas as pd
import sagemaker

from sagemaker.serializers import SimpleBaseSerializer
from sagemaker.deserializers import SimpleBaseDeserializer
from sagemaker.pytorch.model import PyTorchModel

# from sagemaker import get_execution_role
# from sagemaker.local import LocalSession

# role = get_execution_role()
# sagemaker_session = LocalSession()
# sagemaker_session.config = {'local': {'local_code': True}}

sagemaker_session = sagemaker.Session(boto3.session.Session(region_name='us-west-2'))

class Serializer(SimpleBaseSerializer):

    def __init__(self, content_type="application/json"):
        super().__init__(content_type=content_type)

    def serialize(self, data):
        return data.to_json()


class Deserializer(SimpleBaseDeserializer):

    def __init__(self, encoding="utf-8", accept="text/csv"):
        super().__init__(accept=accept)
        self.encoding = encoding

    def deserialize(stream, content_type):
        if content_type != self.ACCEPT():
            raise ValueError(f'Content-type was {content_type}, not {self.ACCEPT()}')
        data = stream.read().decode(self.encoding)
        return pd.read_csv(StringIO(data), header=None, delimiter=',', index_col=0)


def predictor_cls(name,session):
    return PyTorchPredictor(name, session, Serializer(), Deserializer())

print('made callback')

sagemaker_model = PyTorchModel(
    model_data='s3://disaster-tweets-example-remote-storage/sagemaker-output/pytorch-training-2021-11-14-07-33-43-391/output/model.tar.gz',
    role='arn:aws:iam::257018485161:role/sagemaker-disaster-tweets',
    entry_point='sgmkr/inference/entry_point.py',
    dependencies=['sgmkr/inference/requirements.txt'],
    py_version='py38',
    # sagemaker_session=sagemaker_session,
    framework_version='1.9',
    predictor_cls=predictor_cls,
    name='distilbert-disaster-tweets'
    )

print('made model')

my_predictor = sagemaker_model.deploy(initial_instance_count=1, instance_type='ml.m5.xlarge')

print('predictor deployed')

