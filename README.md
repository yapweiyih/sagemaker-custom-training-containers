# Amazon SageMaker custom training containers

Implementations of Amazon SageMaker-compatible custom containers for training.

# XGB container

- <https://github.com/awslabs/amazon-sagemaker-examples/blob/master/introduction_to_amazon_algorithms/xgboost_abalone/xgboost_abalone_dist_script_mode.ipynb>
- <https://sagemaker.readthedocs.io/en/stable/frameworks/xgboost/using_xgboost.html>
- sagemaker-container and sagemaker-inference have been installed, refer to requirements.txt file in Dockerfile
- <https://github.com/aws/sagemaker-xgboost-container/blob/master/docker/1.0-1/final/Dockerfile.cpu>

## How it works

Step 1: Run locally
Run train.py locally using debug mode so that launch.json has the key arguments to run successfully.

Step 2: Local mode with built in containers
Run `script-mode-xgb/notebook/sagemaker-script-mode-xgb.ipynb` to train with sagemaker local mode with build in xgboost container
Output is created at environment variable `module_dir`

Step 3: Run Sagamake with HPO and custom metrics
This is easier than running multiple job locally using different parameters

## data type

- content_type='application/x-parquet'
- content_type='text/csv'

# PyTorch container

- <https://github.com/aws/sagemaker-pytorch-training-toolkit>

# Helper container

scp -i ~/Documents/awskey/sg-aws-sandbox-keypair.pem  ec2-user@ec2a2:/home/ec2-user/efs_project/audio-classification-aws/notebooks/build_and_push.sh  notebook/

# Custom metrics

<https://docs.aws.amazon.com/sagemaker/latest/dg/training-metrics.html#training-metrics-sample-notebooks>

Hyperparameter tuning:
metric_definitions = [{'Name': 'average test loss',
                       'Regex': 'Test set: Average loss: ([0-9\\.]+)'}]

from sagemaker.analytics import TrainingJobAnalytics

metric_name = 'validation:rmse'

metrics_dataframe = TrainingJobAnalytics(training_job_name=job_name, metric_names=[metric_name]).dataframe()
