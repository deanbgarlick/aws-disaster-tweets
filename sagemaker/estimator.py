# https://sagemaker.readthedocs.io/en/stable/overview.html
# https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/using_pytorch.html#serve-a-pytorch-model

import boto3
import sagemaker
from sagemaker.pytorch import PyTorch
from sagemaker.estimator import Estimator
from sagemaker.session import TrainingInput


estimator = PyTorch(
    entry_point='sagemaker/train/entry_point.py',
    role="arn:aws:iam::257018485161:role/sagemaker-disaster-tweets",
    instance_count=1,
    instance_type="ml.m5.xlarge", # instance_type='local',
    # hyperparamters={'epocs':1},
    py_version='py38',
    framework_version='1.9',
    volume_size=5,
    output_path='s3://disaster-tweets-example-remote-storage/sagemaker-output',
    sagemaker_session=sagemaker.Session(boto3.session.Session(region_name='us-west-2')), # sagemaker_session=sagemaker.LocalSession(),
    dependencies=['src/train/requirements.txt', 'src/train/dataset.py']
    )

estimator.fit({"train": "s3://disaster-tweets-example-remote-storage/data"}) #, wait=True)

estimator.model_data # Prints info on the model trained such as the binaries location
